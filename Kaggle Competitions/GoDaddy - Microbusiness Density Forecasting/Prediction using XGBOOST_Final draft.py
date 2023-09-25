
#%%Import Packages

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.impute import KNNImputer as KImp
from sklearn.metrics import mean_squared_error

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['figure.figsize'] = (18,18)
plt.rc('axes', labelsize =13)
plt.rc('font', size = 11)
#%%Import Files we need

os.chdir(r'./godaddy-microbusiness-density-forecasting')
#Import the given datasets
train_df = pd.read_csv("train.csv")
census_df = pd.read_csv("census_starter.csv")

#Import the external datasets
os.chdir("./External Datasets")

rent_df = pd.read_csv("county_rent_estimates.csv", encoding = "utf-8")

employment_df = pd.read_csv("state_employment.csv")

popul_df = pd.read_csv("US County population dataset_2016_2021.csv")

corp_tax_df = pd.read_csv("state_corporate_tax_rates.csv") 
corp_tax_DC_df = pd.read_excel("corporate_tax_rates_DC.xlsx")

trsr_df = pd.read_csv("10yr_treasury_rate.csv")

county_ngbr_df = pd.read_csv("county_neighbours.csv")

#%%Dataset Initial Pre-processing

#Combine 2 different corp tax data

corp_tax_df = pd.concat([corp_tax_df, corp_tax_DC_df]).drop("tax_rate_diff", axis = 1).rename(columns = {"tax_rate": "Corporate Tax Rate"})

#Extract County and State name separately in population dataset and rearrange the dataset
popul_county = popul_df["county"].str.split(", ", expand = True).rename(columns= {0: "County", 1: "State"})

popul_df = pd.concat([popul_county, popul_df], axis = 1).drop("county", axis = 1).rename(columns = {"2016": "Population_2018", "2017": "Population_2019", "2018": "Population_2020", "2019": "Population_2021", "2020": "Population_2022", "2021": "Population_2023"})
#Changed the year columns into Population_{year} format, assuming the population dataset follows the same trend with census_df

#Make a separate column in county neighbor dataset
county_ngbr_df["neighbours"] = county_ngbr_df["neighbours"].apply(lambda x: x.split())

county_ngbr_df["Number of Neighbouring Counties"] = county_ngbr_df["neighbours"].apply(lambda x: len(x))
county_ngbr_df = county_ngbr_df.drop("neighbours", axis = 1)

#%%Dataset Preview: census dataset
census_df.dtypes

census_df.describe()

census_df_dscrb = pd.DataFrame(census_df.describe()) #There is one null value in some of the columns

#Shall take a closer look at cfips
census_df_nunique = pd.DataFrame(census_df.nunique(), copy = False)

#Now, will try to reorder the columns of census_df
census_df_cols_no_cfips = census_df.columns.to_list() #making list of columns
census_df_cols_no_cfips.remove("cfips") #will remove cfips for making a new list later
cfips_list = ["cfips"]

cfips_list.extend(census_df_cols_no_cfips)

census_df = census_df[cfips_list] #Changing order of columns so that cfips comes to the left-most side

census_df_corr = census_df.corr().iloc[1:,1:] #correlation analysis
masker_census = np.triu(np.ones_like(census_df_corr)) #masking matrix for correlation matrix

census_df_corr_sns = sns.heatmap(data = census_df_corr, annot = True, mask = masker_census, cmap = "Greens") #Using Seaborn to plot heatmap of correlation matrix

#%%Dataset Preview: corp_tax_df

#%%Dataset Preview: employment_df

#%%Dataset Preview: popul_df

#%%Dataset Preview: trsr_df

#%%Dataset Preview: train_df

train_df.dtypes
train_df_dscrb = train_df.describe()

train_df.count()
train_df.nunique()

train_df_nunq = pd.DataFrame(train_df.nunique()).reset_index(names = "columns") #reset_index prevents 1st column to be index 

train_df_corr = train_df.iloc[:,-2:].corr() #Checking how density and number of microbusinesses are relevant

plt.rcParams['figure.figsize'] = (12,12)
plt.rc('font', size = 13)
train_df_corr_sns = sns.heatmap(data = train_df_corr, annot = True)

#%%Dataset Merge

#1: Extract year info from train_df
train_df["first_day_of_month"].dtypes

train_df["first_day"] = pd.to_datetime(train_df["first_day_of_month"], format = '%Y-%m-%d')

train_df["Measured Year"] = train_df["first_day"].dt.year #can extract wanted measure of time using dt.(time measure) 
train_df["Measured Month"] = train_df["first_day"].dt.month

train_df["Measured Year"].value_counts() #Year 2019 to 2022
train_df["Measured Month"].value_counts() #Jan to Dec

#Let's figure out where the difference came from

# Get the unique values of the columns in each dataset
unique_values_1 = set(train_df['cfips'].unique())
unique_values_2 = set(census_df['cfips'].unique())

# Find the difference between the two sets of unique values
difference = unique_values_1.symmetric_difference(unique_values_2)

# Print the difference, if any exists
if difference:
    print('The two columns have different unique values:')
    print(difference)
else:
    print('The two columns have identical unique values')
    
#Will take the difference into account, but the differences won't be reflected to train_df

#2: Combine two datasets: train_df & census_df

train_df_xg = pd.merge(left = train_df, right=census_df, left_on= "cfips", right_on = "cfips")
#Now, filter the columns based on the measured year

for a in range(len(train_df_xg)):
    
    if train_df_xg.loc[a,"Measured Year"] == 2019:
        
        train_df_xg.loc[a,"pct_bb"] = train_df_xg.loc[a,"pct_bb_2017"]
        train_df_xg.loc[a,"pct_college"] = train_df_xg.loc[a,"pct_college_2017"]
        train_df_xg.loc[a,"pct_foreign_born"] = train_df_xg.loc[a,"pct_foreign_born_2017"]
        train_df_xg.loc[a,"pct_it_workers"] = train_df_xg.loc[a,"pct_it_workers_2017"]
        train_df_xg.loc[a, "median_hh_inc"] = train_df_xg.loc[a,"median_hh_inc_2017"]
        
    elif train_df_xg.loc[a,"Measured Year"] == 2020:
        
        train_df_xg.loc[a,"pct_bb"] = train_df_xg.loc[a,"pct_bb_2018"]
        train_df_xg.loc[a,"pct_college"] = train_df_xg.loc[a,"pct_college_2018"]
        train_df_xg.loc[a,"pct_foreign_born"] = train_df_xg.loc[a,"pct_foreign_born_2018"]
        train_df_xg.loc[a,"pct_it_workers"] = train_df_xg.loc[a,"pct_it_workers_2018"]
        train_df_xg.loc[a, "median_hh_inc"] = train_df_xg.loc[a,"median_hh_inc_2018"]
        
    elif train_df_xg.loc[a,"Measured Year"] == 2021:
        
        train_df_xg.loc[a,"pct_bb"] = train_df_xg.loc[a,"pct_bb_2019"]
        train_df_xg.loc[a,"pct_college"] = train_df_xg.loc[a,"pct_college_2019"]
        train_df_xg.loc[a,"pct_foreign_born"] = train_df_xg.loc[a,"pct_foreign_born_2019"]
        train_df_xg.loc[a,"pct_it_workers"] = train_df_xg.loc[a,"pct_it_workers_2019"]
        train_df_xg.loc[a, "median_hh_inc"] = train_df_xg.loc[a,"median_hh_inc_2019"]
        
    else: 
        
        train_df_xg.loc[a,"pct_bb"] = train_df_xg.loc[a,"pct_bb_2020"]
        train_df_xg.loc[a,"pct_college"] = train_df_xg.loc[a,"pct_college_2020"]
        train_df_xg.loc[a,"pct_foreign_born"] = train_df_xg.loc[a,"pct_foreign_born_2020"]
        train_df_xg.loc[a,"pct_it_workers"] = train_df_xg.loc[a,"pct_it_workers_2020"]
        train_df_xg.loc[a, "median_hh_inc"] = train_df_xg.loc[a,"median_hh_inc_2020"]
 
new_columns = train_df.columns.to_list() #As this is the first merged dataset, the list of columns from this function and 
new_columns.extend(["pct_bb", "pct_college", "pct_foreign_born", "pct_it_workers", "median_hh_inc"]) #Removing redundant columns for each row

train_df_xg = train_df_xg[new_columns]

#3: Combine train data & corporate tax data

#Combine DC and other states together

train_df_xg = pd.merge(left= train_df_xg, right = corp_tax_df, left_on = ["state", "Measured Year"], right_on = ["state", "year"]).drop("year", axis = 1)

#4: Combine train data & employment data

train_df_xg = pd.merge(left = train_df_xg, right = employment_df[["state","first_day_of_month","pct_non_inst_pop", "pct_employed", "pct_unemployed"]], left_on = ["state", "first_day_of_month"], right_on= ["state", "first_day_of_month"])

#5: Combine train data & rent data

train_df_xg = pd.merge(left = train_df_xg, right = rent_df, left_on = ["cfips", "Measured Year"], right_on = ["cfips", "year"]).drop("year", axis = 1)

#6: Combine train data & population data

new_columns_2 =train_df_xg.columns.to_list()

train_df_xg = pd.merge(left = train_df_xg, right = popul_df, left_on = ["county", "state"], right_on = ["County", "State"], how = 'left').drop(["County", "State"], axis = 1)

train_df_xg_zeros = train_df_xg[train_df_xg["active"] == 0].reset_index(drop = True)

train_df_xg_nonzeros = train_df_xg[train_df_xg["active"] != 0].reset_index(drop = True)

for a in range(len(train_df_xg_zeros)): #Estimation of Population using external data
    
    if train_df_xg_zeros.loc[a,"Measured Year"] == 2019:
        
        train_df_xg_zeros.loc[a,"Population"] = train_df_xg_zeros.loc[a,"Population_2019"]
        
    elif train_df_xg_zeros.loc[a,"Measured Year"] == 2020:
 
        train_df_xg_zeros.loc[a,"Population"] = train_df_xg_zeros.loc[a,"Population_2020"]
        
    elif train_df_xg_zeros.loc[a,"Measured Year"] == 2021:
 
        train_df_xg_zeros.loc[a,"Population"] = train_df_xg_zeros.loc[a,"Population_2021"]        

    else:
 
        train_df_xg_zeros.loc[a,"Population"] = train_df_xg_zeros.loc[a,"Population_2022"]
        
train_df_xg_nonzeros["Population"] = round(100 * train_df_xg_nonzeros["active"] / train_df_xg_nonzeros["microbusiness_density"],0) #Computing Population using microbusiness density and active column

# 100 : density = Popul : active so, Popul = 100 * active / density)

train_df_xg = pd.concat([train_df_xg_nonzeros, train_df_xg_zeros]).reset_index(drop = True)

new_columns_2.append("Population")

train_df_xg = train_df_xg[new_columns_2]

#7: Combine train data & treasury data

train_df_xg = pd.merge(left = train_df_xg , right = trsr_df, left_on = "first_day_of_month", right_on = "first_day_of_month", how = 'left')

#8: Combine train data & counties' neighbour data

train_df_xg = pd.merge(left = train_df_xg, right = county_ngbr_df[["cfips", "Number of Neighbouring Counties"]], left_on = "cfips", right_on= "cfips", how = 'left')
#Merging is now complete

#Now, will rearrange the columns to make it handy to analyse

qual_col_1 = list(train_df_xg.columns[0:5])
qual_col_2 = (train_df_xg.columns[7])
quant_col_1 = list(train_df_xg.columns[5:7])
quant_col_2 = list(train_df_xg.columns[8:])

new_columns_xg = []
new_columns_xg.extend(qual_col_1);new_columns_xg.append(qual_col_2);new_columns_xg.extend(quant_col_1);new_columns_xg.extend(quant_col_2)

new_columns_xg.remove("microbusiness_density"); new_columns_xg.append("microbusiness_density")

train_df_xg = train_df_xg[new_columns_xg]

#%%EDA on combined dataset

train_df_xg_dscrb = train_df_xg.describe()

train_df_xg_corr = train_df_xg.iloc[:,6:25].corr() #Exclude qualitative-ish data and our target variable

masker_xg = np.triu(np.ones_like(train_df_xg_corr))

plt.rcParams['figure.figsize'] = (18,18)
plt.rc('axes', labelsize =11)
plt.rc('font', size = 8.5)

train_df_xg_corr_heatmap = sns.heatmap(data = train_df_xg_corr,mask = masker_xg,  annot = True).set(title = "Correlation Matrix of training data")

train_df_xg.isnull().sum()

train_df_xg.columns

#Make the list of columns with null values

columns_with_nulls = []

for col in list(train_df_xg.columns):
    
    if train_df_xg[col].isnull().sum() > 0:
        
        columns_with_nulls.append(col)

#Fill in the null values using ffill method()

train_df_xg[columns_with_nulls] = train_df_xg[columns_with_nulls].ffill()

train_df_xg.isnull().sum() #No null values detected

#%%Split train dataset into train and validation data

#Categorical Data are transferred into numbers (dummified), so only numeric data will be used instead
#Also, as the test data already exists in separate dataframe, I will split the dataset into train and validation data

X_train, X_val, Y_train, Y_val = train_test_split(train_df_xg.iloc[:,6:25], train_df_xg.iloc[:,25], test_size = 0.2, random_state= 125)

#%%Shall also make a test grid for the best hyperparameter selection

'''
Please use squarederror, not squaredlogerror
'''

#Make parameter grid with random numbers' list

param_grid = {
    'learning_rate' : [0.05, 0.1, 0.125],
    'max_depth' : [8,9,10],
    'n_estimators' : [100, 125],
    }

rgr = xgb.XGBRegressor(objective = "reg:squarederror") #To ensure all predictions are positive

num_cores = os.cpu_count() #Got 8

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rgr, param_grid=param_grid, cv=5, error_score= 'raise', n_jobs = num_cores - 2) #n_job set to 5 so that each cpu core can be used on each cv process
grid_search.fit(X_train, Y_train)
grid_search.best_params_

print(grid_search.best_params_)
#%%Make XGBoost Regression models to perform prediction and compare them
# Use the best hyperparameters to train the model
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, Y_train)

feat_imp_1 = best_xgb.feature_importances_

#1st threshold: Importance Score More than average only
threshold = feat_imp_1.mean() #Remove all features with importance less than threshold

importants_1 = []
importants_1_idx = []
for idx, feat in enumerate(feat_imp_1):

    if feat_imp_1[idx] > threshold:
        
        importants_1.append(feat)
        importants_1_idx.append(idx)

X_train_mod = X_train.iloc[:,importants_1_idx]

best_xgb_avgplus = grid_search.best_estimator_.fit(X_train_mod, Y_train) #Train on new dataset

#2nd threshold: Top Importance Score Based

temp = []

for idx, feat in enumerate(feat_imp_1):
    
    temp.append([idx, feat])
    temp.sort(key = lambda x: x[1], reverse = True)

Score_base = np.array(temp)

#Making a threshold: Top 5 variables based on importance score
importants_2 = Score_base[:5,0]
X_train_top5 = X_train.iloc[:, importants_2[:]]

best_xgb_top5 = grid_search.best_estimator_.fit(X_train_top5, Y_train) #Train on new dataset

importants_3 = Score_base[:10,0]
X_train_top10 = X_train.iloc[:, importants_3[:]]

best_xgb_top10 = grid_search.best_estimator_.fit(X_train_top10, Y_train)#Train on new dataset

#Now will use cross validation score on these 4 models

from sklearn.metrics import get_scorer_names

get_scorer_names()

scores_1 = cross_val_score(best_xgb, X_train[:30000], Y_train[:30000], cv =5, scoring='neg_mean_squared_error', n_jobs= num_cores -2)
rmse_1 = (-scores_1.mean())**0.5
mae_1 = (-scores_1.mean())

scores_2 = cross_val_score(best_xgb_avgplus, X_train[30000:60000], Y_train[30000:60000], cv =5, scoring='neg_mean_squared_error', n_jobs= num_cores -2)
rmse_2 = (-scores_2.mean())**0.5
mae_2 = (-scores_2.mean())

scores_3 = cross_val_score(best_xgb_top5, X_train[60000:90000], Y_train[60000:90000], cv =5, scoring='neg_mean_squared_error', n_jobs= num_cores -2)
rmse_3 = (-scores_3.mean())**0.5
mae_3 = (-scores_3.mean())

scores_4 = cross_val_score(best_xgb_top10, X_train[90000:], Y_train[90000:], cv =5, scoring='neg_mean_squared_error', n_jobs= num_cores -2)
rmse_4 = (-scores_4.mean())**0.5
mae_4 = (-scores_4.mean())

min(rmse_1, rmse_2, rmse_3, rmse_4)
min(mae_1, mae_2, mae_3, mae_4)

#Visualise later

#The first model is the best

#%%Prediction Pt.1 - Predict on validation set

models_cols_order = best_xgb.get_booster().feature_names #The whole columns of train dataset

Y_val_pred = best_xgb.predict(X_val[models_cols_order])

rmse_val = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
print(rmse_val) #Less than 0.15

#%%Now, import test dataset

#Import test data
os.chdir("../")

test_df = pd.read_csv("test.csv")

test_df["first_day_of_month"].unique()
test_df["cfips"].nunique()

train_df["cfips"].nunique()

test_months = list(test_df["first_day_of_month"].unique())

#Check if there are any differences in cfips
unique_values_train = set(train_df['cfips'].unique())
unique_values_test = set(test_df['cfips'].unique())

# Find the difference between the two sets of unique values
difference_test = unique_values_train.symmetric_difference(unique_values_test)

if difference_test:
    print('The two columns have different unique values:')
    print(difference_test)
else:
    print('The two columns have identical unique values')
    
#We found they have identical set of cfips, which makes the prediction more handy as below    


#%%Prediction Pt.2 - Perform prediction on test dataset

#Reduce the dimension of train dataset

train_df_xg_final = train_df_xg.drop(["county", "state", "first_day", "microbusiness_density"], axis = 1)


#Group train dataframe by cfips
train_df_bycfips = train_df_xg_final.groupby(train_df["cfips"])

#Group test dataframe by cfips 
test_df_bycfips = test_df.groupby(test_df["cfips"])

#KNNImputer to fill out the missing values: Considering data a year ahead
Imp = KImp(n_neighbors= 6, weights= "distance") #To fill out the data values based on the previous 6 months

#Now the prediction begins based on windowing mechanism, with divide and conquer

test_result = pd.DataFrame()

for (name1, group1), (name2, group2) in zip(train_df_bycfips, test_df_bycfips):
    
    window_size = 7
    
    columns_filt = list(range(len(models_cols_order)))
    
    if name1 == name2: #Connecting same cfips 
        
        token = pd.concat([group1, group2]).reset_index(drop = True)
        
        token_cat = token[["row_id", "cfips", "first_day_of_month"]]
        token_num = token.loc[:, models_cols_order]
        token_arr = np.array(token_num)
        
        for i in range(token_arr.shape[0]-window_size+1):
            
            window_data = token_arr[i:i+window_size, columns_filt]
            imputed_data = Imp.fit_transform(window_data)
            imputed_data = imputed_data.round(0)
            token_arr[i:i+window_size, columns_filt] = imputed_data
            
        token_num.iloc[:,:] = token_arr[:]

        token['microbusiness_density'] = best_xgb.predict(token_num).round(3)

        test_result = pd.concat([test_result, token[token["row_id"].isin(list(group2["row_id"]))][["row_id","microbusiness_density"]]]).reset_index(drop = True) #Append test_df with prediction made

for l in range(len(test_result)):

    if test_result.loc[l, "microbusiness_density"] < 0:
        
        test_result.loc[l, "microbusiness_density"] = 0 #Negative value makes no sense
#%%Make the Prediction into Dataset

test_result.to_csv("submission.csv", index = False)
