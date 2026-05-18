#%% Import files

import os
import joblib
import sys
import json
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from playwright.async_api import async_playwright
from urllib.parse import urlparse
from collections import Counter
from typing import Literal
#%%Set the directory

file_directory = os.path.abspath(__file__)
current_dir = os.path.dirname(file_directory)
project_root = os.path.dirname(current_dir)

os.chdir(project_root)

if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except:
        pass
nest_asyncio.apply()

#%%Import json file of the selected model
with open('./model_config.json', 'r') as f:
    model_config = json.load(f)

#Import model parameters

model = joblib.load(f"./Model_params/{model_config['Model_name']}.pkl")

# %%1: Find the name of features used

col_names = model_config["Column_names"]

# %%2: Go through all the features and get the thresholds
def get_weighted_average_thresholds(model, feature_names):
    # Stack up the threshold info on the list: data
    data = []
    
    for estimator in model.estimators_:
        tree = estimator.tree_
        
        for node in range(tree.node_count):
            # Check if a node diverges and has child node
            if tree.children_left[node] != tree.children_right[node]:
                f_idx = tree.feature[node]
                f_name = feature_names[f_idx]
                threshold = tree.threshold[node]
                weight = tree.n_node_samples[node] # Use number of samples reached to the node as weight
                
                data.append({
                    'feature': f_name,
                    'threshold': threshold,
                    'weight': weight
                })
    
    df = pd.DataFrame(data)
    #Store the weighted average for each feature
    weighted_avg_res = {}
    for feat in df['feature'].unique():
        subset = df[df['feature'] == feat]
        weighted_avg = np.average(subset['threshold'], weights=subset['weight'])
        weighted_avg_res[feat] = round(weighted_avg, 3)
        
    return weighted_avg_res

# Get thresholds
final_thresholds = get_weighted_average_thresholds(model, col_names)

#%% Check thresholds for the variables whose names start with Num

Num_Threshold = {}

for feat, val in final_thresholds.items():
    #Only filter variables with names starting as "Num"
    if feat[:3] == 'Num':
        Num_Threshold[feat] = val

# %%Make a first_filter function that determines if a given url is benign or not based on variables starting with "Num"

def first_filter(url:str, mode: Literal['l','d'] = 'l'):
    """
    url: type in the url of interest \n
    mode: set the operation mode \n
    mode = 'l': Only tells if the given url is suspicious or not \n
    mode = 'd': Returns the number of specific letters/words in the given url
    """
    if len(url) == 0:
        raise ValueError("The length of url should be greater than 0")
    if url.startswith("http") == False:
        raise ValueError("Type in a valid url")

    #Make a dictionary that takes counts from URL

    Num_counts = {}

    #Counts of dashes
    Num_counts['NumDash'] = url.count('-')

    #Counts of dots
    Num_counts['NumDots'] = url.count('.')

    #Counts of sensitive words
    Num_sensitive_words = 0

    sensitive_words = [
        'login', 'verify', 'update', 'secure', 'account', 
        'bank', 'webscr', 'signin', 'mail', 'confirm'
    ]

    lower_url = url.lower()
    for word in sensitive_words:
        if word in lower_url:
            Num_sensitive_words += 1

    Num_counts['NumSensitiveWords'] = Num_sensitive_words

    if mode == 'l':

        is_suspicious = any(Num_counts[k] > Num_Threshold[k] for k in list(Num_counts.keys()))
        
        if is_suspicious:
            return "1- The webpage may be suspicious, needs deeper inspection"
        return "0 - The webpage seems safe"
        
    elif mode == 'd':

        return(Num_counts)

# %%Create in-depth data collector from the given url

async def in_depth_filter(url:str, threshold=0.5):
    async with async_playwright() as p:
        # Launch browser (Set headless=False if you want to see the browser window)
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            current_domain = urlparse(url).netloc
            
            # --- 1. (PctNullSelfRedirect & PctExtHyperlinks) ---
            hrefs = await page.eval_on_selector_all("a", "elements => elements.map(e => e.getAttribute('href'))")
            total_links = len(hrefs)
            
            null_self_count = 0
            valid_http_links = []
            
            for href in hrefs:
                if not href:
                    null_self_count += 1
                    continue
                target = href.strip().lower()
                # Null/Self Redirect 
                if target in ["#", "#none", "javascript:void(0)", "javascript:void(0);", ""] or target == url.lower() or target == urlparse(url).path:
                    null_self_count += 1
                # Check for external links
                if target.startswith('http'):
                    valid_http_links.append(target)

            PctNullSelfRedirectHyperlinks = null_self_count / total_links if total_links > 0 else 0
            PctExtNullSelfRedirectHyperlinksRT = 1 if PctNullSelfRedirectHyperlinks > threshold else 0
            
            ext_count = sum(1 for h in valid_http_links if urlparse(h).netloc != current_domain)
            PctExtHyperlinks = ext_count / len(valid_http_links) if len(valid_http_links) > 0 else 0

            # --- 2. FrequentDomainNameMismatch ---
            resource_urls = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a, img, link, script'))
                            .map(el => el.href || el.src)
                            .filter(src => src && src.startsWith('http'));
            }""")
            domain_list = [urlparse(u).netloc for u in resource_urls if urlparse(u).netloc]
            if not domain_list:
                FrequentDomainNameMismatch = -1
            else:
                most_frequent = Counter(domain_list).most_common(1)[0][0]
                FrequentDomainNameMismatch = 1 if most_frequent != current_domain else 0

            # --- 3. SubmitInfoToEmail & InsecureForms ---
            form_actions = await page.eval_on_selector_all("form", "elements => elements.map(e => e.getAttribute('action'))")
            page_content = await page.content()
            
            SubmitInfoToEmail = 1 if "mailto:" in page_content.lower() or any("mailto:" in (a or "").lower() for a in form_actions) else 0
            
            InsecureForms = 0
            if form_actions:
                for action in form_actions:
                    if not action or action.strip().lower().startswith("http://") or action.strip().lower() in ["", "#", "about:blank"]:
                        InsecureForms = 1
                        break

            # --- 4. Frame existence checker ---
            frame_count = await page.evaluate("() => document.querySelectorAll('iframe, frame').length")
            IframeOrFrame = 1 if frame_count > 0 else 0
        
            # --- 5. Numeric features ---
            num_features = first_filter(url=url, mode='d')

            # Results
            result = {
                **num_features,
                "PctNullSelfRedirectHyperlinks": round(PctNullSelfRedirectHyperlinks, 4),
                "PctExtNullSelfRedirectHyperlinksRT": PctExtNullSelfRedirectHyperlinksRT,
                "FrequentDomainNameMismatch": FrequentDomainNameMismatch,
                "SubmitInfoToEmail": SubmitInfoToEmail,
                "InsecureForms": InsecureForms,
                "PctExtHyperlinks": round(PctExtHyperlinks, 4),
                "IframeOrFrame": IframeOrFrame
            }

            final_result = {k: result[k] for k in col_names}
            
            return final_result

        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            # Ensure the browser is closed to free up resources
            await browser.close()

#%%