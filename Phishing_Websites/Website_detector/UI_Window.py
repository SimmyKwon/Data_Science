#%% Import libraries needed

import uvicorn
import joblib
import os
import sys
import json
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from Data_Extractor import first_filter, in_depth_filter
#%%Set directory
file_directory = os.path.abspath(__file__)
current_dir = os.path.dirname(file_directory)
project_root = os.path.dirname(current_dir)
os.chdir(project_root)

#Append the directory name if it is not in system's path
if project_root not in sys.path:
    sys.path.append(project_root)

#%%Load the backbone model

with open('model_config.json', 'r') as f:
    model_config = json.load(f)

#Import model

model = joblib.load(f"./Model_params/{model_config['Model_name']}.pkl")

#%%UI creation

# 1. Instance creation
app = FastAPI(title = "Phishing Website detector", 
              description= "Uses Machine Learning model to detect if a given url is phishing website or not",
              version= "1.0.0",
              swagger_ui_parameters={"defaultModelsExpandDepth": -1})


'''
#from fastapi.responses import RedirectResponse

# 2. Set the root directory
#This root directory won't be exposed when used, and redirects to the /docs straight away for convenient use of website detection

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
'''
#2. Set the root directory
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <html>
        <head>
            <title>Phishing Detector</title>
            <style>
                body { font-family: Arial, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; background-color: #f5f5f5; }
                .container { text-align: center; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                input { width: 300px; padding: 10px; margin-right: 10px; border: 1px solid #ccc; border-radius: 5px; font-size: 1em; }
                button { padding: 10px 20px; border: none; background-color: #007bff; color: white; border-radius: 5px; cursor: pointer; font-size: 1em; }
                button:hover { background-color: #0056b3; }
                #result { margin-top: 25px; font-weight: bold; font-size: 1.2em; color: #333; white-space: pre-line; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🛡️ Real-Time Phishing Website Detector</h2>
                <p>Enter URL you want to analyse.</p>
                <input type="text" id="urlInput" placeholder="https://example.com">
                <button onclick="checkUrl()">Examine</button>
                <div id="result"></div>
            </div>

            <script>
                // Edited to function() for JS compatibility
                async function checkUrl() {
                    const url = document.getElementById('urlInput').value;
                    const resultDiv = document.getElementById('result');
                    if(!url) return alert('Enter URL please!');
                    
                    resultDiv.style.color = "#333";
                    resultDiv.innerText = "🔍 Wait for a while, analysis is ongoing.\\n(Playwright is running the browser)";
                    
                    try {
                        const response = await fetch(`/predict?url=${encodeURIComponent(url)}`);
                        const data = await response.json();
                        
                        // Different colours assigned based on the result
                        if (data.message.includes("Phishing")) {
                            resultDiv.style.color = "#dc3545"; // red
                        } else if (data.message.includes("suspicious")) {
                            resultDiv.style.color = "#ffc107"; // yellow
                        } else {
                            resultDiv.style.color = "#28a745"; // green
                        }
                        
                        resultDiv.innerText = data.message;
                    } catch (error) {
                        resultDiv.style.color = "#dc3545";
                        resultDiv.innerText = "❌ Error while communicating with the backend server.";
                    }
                }
            </script>
        </body>
    </html>
    """

# 3. Set the prediction page
@app.get("/predict", summary="Analysis of URL being suspicious or not", tags=["🛡️ Detection API"],
         responses={
        200: {"description": "Successful Response", "content": None},
        422: {"description": "Validation Error", "content": None}})
async def predict(url: str):

    #First, use the first_filter to check the status
    initial_data = first_filter(url=url)

    #Deep inspection goes on if the website seems suspicious
    if initial_data.startswith("1"):
        print("Performing deep analysis")

        full_data = await in_depth_filter(url=url)

        if full_data is None:
            return {"input_url": url, "status": "error", "message": "Failed to retrieve the data from the webpage."}
        
        else:

            input_data = pd.DataFrame(full_data)

            classification_result = model.predict(input_data)
            classification_probability = max(model.predict_proba(input_data)[0])
            #Label websites based on the classification_result
            result_summary = "Phishing Website" if classification_result == 1 else "Benign Website"
            
            return {"input_url": url, "status": "Data retrieved", 
                    "message": f"The given url is {result_summary} with the probability of {round(100 * classification_probability,1)}%."}

    else:
        return {"input_url": url, "status": "Data retrieved", "message": initial_data}
    

#%%Execution
if __name__ == "__main__":
    # Use UI_Window:app to correctly load the script onto the internet
    uvicorn.run("UI_Window:app", host="127.0.0.1", port=8080, reload=True)
# %%
