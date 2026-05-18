#%%Import Packages

import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.model_selection import KFold as KF, cross_validate as cv
import matplotlib.pyplot as plt
import os
import joblib
import json
#%%Set the directory for analysis

file_directory = os.path.abspath(__file__)
current_dir = os.path.dirname(file_directory)
os.chdir(current_dir)

# %%Import files

data = pd.read_csv('./dataset/Phishing_Legitimate_full.csv')
data.describe()
# %%More info gathering on the dataset

data.info() #There are no null values in all of the rows

# %% Have a look at numerical features of the dataset

for col in data.columns:
    print(f'The variance for column {col} is: {data[col].var()}')
    print(f'The mean for column {col} is: {data[col].mean()}')

# %% Determine the correlation between variables

data_corr = data.iloc[:,1:].corr()
data_corr

# %%Select the top 10 independent variables with the most correlation with the dependent variable
data_corr['inf_on_res'] = data_corr['CLASS_LABEL'].abs()
data_corr

#Sort the top 10 most influential variables
data_corr.sort_values(by='inf_on_res', ascending=False, inplace=True)
data_corr_sorted = data_corr.iloc[:11,:]

#Leave all other variables out
data_corr_sorted = data_corr_sorted[list(data_corr_sorted.index)]
data_corr_sorted

# %%Make a visualisation of the correlation of graphs

fig, ax = plt.subplots(figsize = (10.5,10.5))

mask = np.tril(np.ones_like(data_corr_sorted, dtype=bool))
corr_masked = np.ma.masked_where(mask, data_corr_sorted)

im = ax.imshow(corr_masked, cmap='plasma')
cbar = fig.colorbar(im)

cols = list(data_corr_sorted.columns)

ax.set_xticks(range(len(cols)), labels = cols,
              rotation=45, ha ="right", rotation_mode='anchor')
ax.set_yticks(range(len(cols)), labels = cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        if i < j:
            text = ax.text(j,i, round(data_corr_sorted.iloc[i,j],3), ha = 'center', va='center', color = 'w')

ax.set_title("Correlation between major variables")
plt.tight_layout()
plt.show()

# %%Filter out the data 

#Divide into 2 scenarios
#1: Use all 10 variables for model training and testing
#2: Use 5 variables of choice based on correlation with other variables

#Filter out the data columns for analysis
data = data[cols]
#Separate dependent and independent variables

#Data for y variable
df_y = data['CLASS_LABEL']

#Extract the columns of independent variables
x_cols = list(data.columns)
x_cols.remove('CLASS_LABEL')

#Data for x variable
df_x = data[x_cols]

# %%Make the folds of data

Folds = KF(5,shuffle=True, random_state=25)

#Change to the directory above

project_root = os.path.dirname(current_dir)
os.chdir(project_root)
#Make a folder storing model parameters

os.makedirs('./Model_params',exist_ok=True)

#Make a list to store all information of models 

model_info = []

#%% Scenrario 1: Use all 10 variables for training and testing

#Define model first
RandForest1 = skl.ensemble.RandomForestClassifier(random_state=25) 

#Train and evaluate

score_all_var = cv(RandForest1, df_x, df_y, cv=Folds, scoring=['accuracy', 'f1'], return_estimator=True)
print(f"Accuracy for each fold: {score_all_var['test_accuracy']}")
print(f"F1 score for each fold: {score_all_var['test_f1']}")

#Save model parameters

for k, model in enumerate(score_all_var['estimator']):

    model_name = f'Fold_all_var_model_{k+1}'

    joblib.dump(model, f'./Model_params/{model_name}.pkl')
    print(f"Fold no.{k+1} accuracy: {score_all_var['test_accuracy'][k]}")
    print(f"Fold no.{k+1} f1 score: {score_all_var['test_f1'][k]}")

    #Make dictionary to store all information needed for the model selection
    temp_dict = {"Model_name":model_name, "Accuracy": score_all_var['test_accuracy'][k], "F1_score": score_all_var['test_f1'][k]}    

    model_info.append(temp_dict)

    print(f'Fold no.{k+1} saved!')
# %% Scenario 2: Use the top 5 most important variables for training and testing

#Define model
RandForest2 = skl.ensemble.RandomForestClassifier(random_state=25) 

#Using RFE from scikit learn

feature_selector = skl.feature_selection.RFE(estimator = RandForest2, n_features_to_select = 5, step = 1)
feature_selector.fit(df_x, df_y)

#Show the selected features
print(f"Selected features: {df_x.columns[feature_selector.support_]}")

selected_vars = list(df_x.columns[feature_selector.support_])
print(selected_vars)

#Filter out the most needed variables
df_x_selected = df_x[selected_vars]

#%%Train and evaluate
score_5_var = cv(RandForest2, df_x_selected, df_y, cv=Folds, scoring=['accuracy', 'f1'], return_estimator=True)
print(f"Accuracy for each fold: {score_5_var['test_accuracy']}")
print(f"F1 score for each fold: {score_5_var['test_f1']}")

#Save model parameters
for l, model in enumerate(score_5_var['estimator']):

    model_name = f'Fold_5_var_model_{l+1}'

    joblib.dump(model, f'./Model_params/{model_name}.pkl')
    print(f"Fold no.{l+1} accuracy: {score_5_var['test_accuracy'][l]}")
    print(f"Fold no.{l+1} f1 score: {score_5_var['test_f1'][l]}")

    #Make dictionary to store all information needed for the model selection
    temp_dict = {"Model_name":model_name, "Accuracy": score_5_var['test_accuracy'][k], "F1_score": score_5_var['test_f1'][k]}    

    model_info.append(temp_dict)

    print(f'Fold no.{l+1} saved!')
# %%Select the most effective model

#Based on the results, the model performs better when all 10 variables are used, and the model tested on the first fold was the best
#Sort the list of models according to the best accuracy
model_info.sort(key=lambda x: x['Accuracy'], reverse= True)

# %%Select the best model

print("The best model is:",model_info[0]['Model_name'])

#%%Save the model name for future references

#Save the column names for the referring purpose

model_info[0]["Column_names"] = x_cols

#Save the model parameters into the designated directory
with open('model_config.json', 'w') as f:
    json.dump(model_info[0], f, indent= 4)

# %%
