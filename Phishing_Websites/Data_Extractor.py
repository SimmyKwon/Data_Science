#%% Import files

import os
import joblib
import json
import asyncio
import pandas as pd
import numpy as np
from playwright.async_api import async_playwright
from urllib.parse import urlparse
from collections import Counter
from typing import Literal
#%%Set the directory

file_directory = os.path.abspath(__file__)
os.chdir(os.path.dirname(file_directory))

#%%Import json file of the selected model
with open('model_config.json', 'r') as f:
    model_config = json.load(f)

#%%Import model

model = joblib.load(f"./model_params/{model_config['Model_name']}.pkl")

# %%1: Find the name of features used

if hasattr(model, 'feature_names_in_'):
    features = model.feature_names_in_
    print("Name of the features used:", features)
else:
    print("No information about the variables found.")

#Find the column names as well

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
final_thresholds = get_weighted_average_thresholds(model, list(features))

# Check thresholds for the variables whose names start with Num
print("--- Numerical Variables' thresholds ---")
for feat, val in final_thresholds.items():
    #Only 
    if feat[:3] == 'Num':
        print(f"{feat}: {val}")
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

    #Counts of dashes
    Num_dashes = url.count('-')

    #Counts of dots
    Num_dots = url.count('.')

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

    if mode == 'l':

        if Num_dashes > 2 or Num_dots > 2 or Num_sensitive_words >= 1:
            return("1- The webpage m be suspicious, needs deeper inspection")    
        else:
            return("0 - The webpage seems safe")
        
    elif mode == 'd':

        return({"NumDash":Num_dashes, "NumDots":Num_dots, "NumSensitiveWords": Num_sensitive_words})

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
            # --- 1. Calculate PctNullSelfRedirectHyperlinks & PctNullSelfRedirectHyperlinksRT ---
            # Navigate to the page
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Extract all 'href' attributes from <a> tags
            hrefs = await page.eval_on_selector_all("a", "elements => elements.map(e => e.getAttribute('href'))")
            
            total_links = len(hrefs)
            if total_links == 0:
                # Return -1 (Unknown) if no hyperlinks are found
                PctNullSelfRedirectHyperlinks = 0

            null_self_count = 0
            for href in hrefs:
                if not href:
                    null_self_count += 1
                    continue
                
                target = href.strip().lower()
                # Define null or self-redirecting patterns
                is_null = target in ["#", "#none", "javascript:void(0)", "javascript:void(0);", ""]
                is_self = target == url.lower() or target == urlparse(url).path
                
                if is_null or is_self:
                    null_self_count += 1

            # Calculate the final ratio
            PctNullSelfRedirectHyperlinks = null_self_count / total_links
            
            # --- Relation Threshold (RT) Logic ---
            # Usually: 1 (Phishing/Danger), 0 (Legitimate/Safe)
            # If the ratio exceeds the threshold, it is flagged as suspicious
            PctNullSelfRedirectHyperlinksRT = 1 if PctNullSelfRedirectHyperlinks > threshold else 0

            # --- 2. Calculate FrequentDomainNameMismatch ---
            # Inspect all resource-loading tags: a(href), img(src), link(href), script(src)
            # Extract current domain from the input URL
            current_domain = urlparse(url).netloc
            
            # Extract all external resource URLs using JavaScript
            # Covers hyperlinks, images, external CSS, and scripts
            resource_urls = await page.evaluate("""() => {
                const urls = [];
                document.querySelectorAll('a, img, link, script').forEach(el => {
                    const src = el.href || el.src;
                    if (src && src.startsWith('http')) urls.push(src);
                });
                return urls;
            }""")

            # Extract only the network location (domain) from each URL
            domain_list = [urlparse(u).netloc for u in resource_urls if urlparse(u).netloc]
            
            if not domain_list:
                # Return -1 (Unknown) if no external resources are found
                FrequentDomainNameMismatch = -1

            # Identify the most frequent domain among all extracted resources
            most_frequent_domain = Counter(domain_list).most_common(1)[0][0]
            
            # --- Mismatch Detection Logic ---
            # Flag as 1 if the dominant domain is NOT the current domain
            FrequentDomainNameMismatch = 1 if most_frequent_domain != current_domain else 0

            # --- 3. Check SubmitInfoToEmail ---
            # Scan for 'mailto:' in form actions or specific email input fields
            # Also check if form submission leads to an external email-handling script
            submit_to_email = 0
            
            # Check <form action="mailto:...">
            form_actions = await page.eval_on_selector_all("form", "elements => elements.map(e => e.action)")
            if any("mailto:" in action.lower() for action in form_actions):
                SubmitInfoToEmail = 1
            
            # Secondary check: Look for email-related keywords in form or input elements
            if submit_to_email == 0:
                page_content = await page.content()
                if "mailto:" in page_content.lower():
                    SubmitInfoToEmail = 1

            # --- 4. Check InsecureForms ---
            # Extract all 'action' attributes from <form> tags
            form_actions = await page.eval_on_selector_all("form", "elements => elements.map(e => e.getAttribute('action'))")
            
            if not form_actions:
                # Return 0 (Safe) if there are no forms on the page
                InsecureForms =  0

            insecure_flag = 0
            for action in form_actions:
                if action is None:
                    # Case where <form> has no action attribute (submits to self, often suspicious in phishing)
                    InsecureForms = 1
                    break
                
                clean_action = action.strip().lower()
                
                # Check for insecure submission patterns
                # 1. Submitting via insecure 'http'
                # 2. Empty action or placeholder patterns
                # 3. 'about:blank' or 'javascript:void(0)'
                if clean_action.startswith("http://"):
                    InsecureForms = 1
                    break
                elif clean_action in ["", "#", "about:blank", "javascript:void(0);", "javascript:void(0)"]:
                    InsecureForms = 1
                    break

            # --- 5. Check PctExtHyperlinks ---
            # Filter out empty or non-URL links (e.g., javascript:, tel:, mailto:)
            valid_links = [h for h in hrefs if h.startswith('http')]
            total_links = len(valid_links)
            
            if total_links == 0:
                # Return 0 if no valid hyperlinks are found to avoid division by zero
                PctExtHyperlinks = 0

            ext_link_count = 0
            for href in valid_links:
                link_domain = urlparse(href).netloc
                
                # Check if the link's domain is different from the current domain
                if link_domain and link_domain != current_domain:
                    ext_link_count += 1

            # Calculate the ratio of external links
            PctExtHyperlinks = ext_link_count / total_links

            # --- 6. Check IframeOrFrame ---

            # Count <iframe> and <frame> elements using JavaScript
            # This covers both standard iframes and older frameset structures
            frame_count = await page.evaluate("""() => {
                const iframes = document.getElementsByTagName('iframe').length;
                const frames = document.getElementsByTagName('frame').length;
                return iframes + frames;
            }""")

            # --- Detection Logic ---
            # Flag as 1 if any frame or iframe exists, else 0
            IframeOrFrame = 1 if frame_count > 0 else 0
        
            # --- 7. Check other numerical features ---
            num_features = first_filter(url=url, mode='d')

        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            # Ensure the browser is closed to free up resources
            await browser.close()
    



# %%
