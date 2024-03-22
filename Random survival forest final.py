# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:34:08 2024

@author: Hanneke
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score
from sksurv.metrics import cumulative_dynamic_auc

# this is the dataframe imported after preprocessing from the discete time survival model:
file_path = 'final_df.csv'
df = pd.read_csv(file_path)

def drop_columns(df,columns):
    return df.drop(columns=columns)

def CS_data(df):
     return df.groupby('policy_nr_hashed').last().reset_index()

def get_dummies(dataframe, columns):
    result = dataframe.copy()
    for col in columns:
        result = pd.get_dummies(result, columns=[col])
    return result

def postcode_digit(dataframe, i):
    dataframe['postcode_range'] = dataframe['postcode'].astype(str).str[i]
    dataframe['postcode_range'] = dataframe['postcode_range'].astype('category')
    return dataframe


remove = ['Unnamed: 0','year_initiation_policy', 'year_initiation_policy_version', 'year_end_policy','d_churn_cancellation',
       'd_churn_between_prolongations', 'd_churn_around_prolongation', 'premium_main_coverages', 'premium_supplementary_coverages', 'welcome_discount_control_group'
       , 'postcode','premium_mutations','type','weight','brand','postcode_new']

dataframe2 = drop_columns(df,remove)
dataframe2 = CS_data(dataframe2)
dataframe2['initial premium']=dataframe2['total_premium']
dataframe2=dataframe2.drop(['total_premium'],axis=1)
dataframe = dataframe2[~((dataframe2['years_since_policy_started'] == 3) & (dataframe2['WD_receive_group'] == 1))]
dataframe = get_dummies(dataframe, [ "fuel_type", "sales_channel"])

y = Surv.from_dataframe('d_churn', 'years_since_policy_started', dataframe)
X = dataframe.drop(['d_churn','years_since_policy_started','policy_nr_hashed'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,stratify=X['WD_receive_group'])

# save to use for the discete-time model:
X_train_policynumbers=X_train['policy_nr_hashed']
X_test_policynumbers=X_test['policy_nr_hashed']
X_train_policynumbers.to_csv('X_train_policynumbers1.csv', index=False)
X_test_policynumbers.to_csv('X_test_policynumbers1.csv', index=False)

## implement for ourselfs:
X_train_policynumbers = pd.read_csv('X_train_policynumbers1.csv')
X_test_policynumbers = pd.read_csv('X_test_policynumbers1.csv')
X_train = dataframe[dataframe['policy_nr_hashed'].isin(X_train_policynumbers['policy_nr_hashed'])]
X_test = dataframe[dataframe['policy_nr_hashed'].isin(X_test_policynumbers['policy_nr_hashed'])]
## random survival forest format when downloading csv:
y_train = Surv.from_dataframe('d_churn', 'years_since_policy_started', X_train)
y_test = Surv.from_dataframe('d_churn', 'years_since_policy_started', X_test)
X_train = X_train.drop(['d_churn', 'years_since_policy_started', 'policy_nr_hashed'], axis=1)
X_test = X_test.drop(['d_churn', 'years_since_policy_started', 'policy_nr_hashed'], axis=1)


## Hyper parameter optimization:

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50, 60],
    'min_samples_split': [2, 5, 10, 20, 30, 40],
     'min_samples_leaf': [2, 5, 6, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None],
}

def c_index_scorer(estimator, X, y):
    prediction = estimator.predict(X)
    event_indicator = y['d_churn']
    event_time = y['years_since_policy_started']
    c_index, _, _, _, _ = concordance_index_censored(event_indicator, event_time, prediction)
    return c_index

rsf = RandomSurvivalForest(random_state=42)

grid_search = GridSearchCV(estimator=rsf, param_grid=param_grid, scoring=c_index_scorer, n_jobs=-1, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best C-index:", grid_search.best_score_)


c_index_results = {}
for n in range(1, 201):
    rsf = RandomSurvivalForest(n_estimators=n, random_state=42, n_jobs=-1)
    rsf.fit(X_train, y_train)
    c_index = c_index_scorer(rsf, X_test, y_test)
    c_index_results[n] = c_index
    print(f"n_estimators={n}, C-index: {c_index}")
    

c_index_results = []


# using the optimal parameters for prediction:
rsf = RandomSurvivalForest(n_estimators=100, random_state=42,min_samples_split=10,min_samples_leaf=5,max_depth=30, n_jobs=1,oob_score=True)
rsf.fit(X_train, y_train)

pred_survival = rsf.predict_survival_function(X_test)
output=pd.Series(pred_survival)

## brier accuracy score:
max_time = max(y_test['years_since_policy_started'])
times = np.linspace(0, max_time, 6)
surv_arrays = np.asarray([fn(times) for fn in pred_survival])
times = np.linspace(0, max_time, 6)[:-1]
surv_arrays = surv_arrays[:, :-1]
ibs = integrated_brier_score(y_train, y_test,surv_arrays,times)
bs = brier_score(y_train, y_test,surv_arrays,times)
print("Brier Score:", bs)

# AUC:
risk_scores = rsf.predict(X_test)
auc_values2, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
print(f"Mean AUC up to max time: {mean_auc}")
print(f"AUC values: {auc_values2}")


## feature importance:
perm_importance = permutation_importance(rsf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
feature_importances = perm_importance.importances_mean
features_used = X_test.columns
importance_df = pd.DataFrame({'Feature': features_used, 'Importance': feature_importances})
importance_df1 = importance_df.sort_values(by='Importance', ascending=True)


plt.figure(figsize=(12, 10))
plt.barh(importance_df1['Feature'], importance_df1['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.grid(axis='x')

plt.show()

# make predictions for discount group and non discount group:
X_train_received_WD = X_train[X_train['WD_receive_group'] == 1]
X_train_not_received_WD = X_train[X_train['WD_receive_group'] == 0]

y_train_received_WD = y_train[X_train['WD_receive_group'] == 1]
y_train_not_received_WD = y_train[X_train['WD_receive_group'] == 0]

X_test_received_WD = X_test[X_test['WD_receive_group'] == 1]
X_test_not_received_WD = X_test[X_test['WD_receive_group'] == 0]

y_test_received_WD = y_test[X_test['WD_receive_group'] == 1]
y_test_not_received_WD = y_test[X_test['WD_receive_group'] == 0]

rsf.fit(X_train_received_WD,y_train_received_WD)
rsf.fit(X_train_not_received_WD,y_train_not_received_WD)

pred_survival_group5 = rsf.predict_survival_function(X_test_received_WD,return_array=True)
pred_survival_group6 = rsf.predict_survival_function(X_test_not_received_WD,return_array=True)
