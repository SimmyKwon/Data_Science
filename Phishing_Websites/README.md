# 🚀 Real-Time Phishing Website Detector
- Developed the ML model that detects if a given website is benign or not, with interactive UI made with FastAPI 😁

---

## Features
- Trained and tested a randomforestclassifier from scikit-learn library for phishing website detection
- Developed a data parser that extracts information from the given url for analysis
- Created an interactive html webpage to use the data parser and ML model, by simply copying and pasting the urls that need to be examined

## 🛠️ Tech Stack ![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
- **Framework:** Scikit-Learn / PlayWright / FastAPI
- **DevOps:** Docker (To be uploaded!)

---
## 📁 폴더 구조 (Project Structure)
```text
Phishing_Websites/
├── Model_params/              # All model parameters are stored here
├── Model_training_testing/    # Code for training and testing model is here
├── Website_detector/          # Data parser and Web UI are here
├── model_config.json          # JSON file containing the information of key model
├── setup.py                   # Script for installing key packages of the project
└── README.md                  # Explanation of the whole project
```

## 🏃‍♂️ Getting Started

### 1. Prerequisites
- Python >= 3.11.15

### 2. Installation

```bash
# 1. How to clone the project
git clone [https://github.com/SimmyKwon/Data_Science/Phishing_Websites.git](https://github.com/SimmyKwon/Data_Science/Phishing_Websites.git)

# 2. Change to the current directory
cd .

# 3. Create virtual environment 
python -m venv [Name of your choice for the project]

#For Windows
# .\[Name of your choice for the project]\Scripts\activate
#For Mac/Linux
# source [Name of your choice for the project]/bin/activate

# 4. Install dependencies
pip install -e .
