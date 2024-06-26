# import the package
import numpy as np
import pandas as pd
import preprocessing as prep
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pygam import LogisticGAM, s, f
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler


def remove_starting_one(data):
    get_wrongly_assigned = data[data['years_since_policy_started'] == 0]
    to_unique = get_wrongly_assigned['policy_nr_hashed'].unique()
    data = data[data['policy_nr_hashed'].isin(to_unique)]
    
    return data

# read data and combine clustering
data = pd.read_csv('20240117_churn_data.csv')
clustering = pd.read_table('Aegon_data_only_clusters_2019_with_IDs_3', sep=',')
cluster = clustering[['policy_nr_hashed','cluster_k_prototypes_2']]
cluster = cluster.rename(columns={'cluster_k_prototypes_2': 'cluster'})
merged_df = pd.merge(data, cluster, on='policy_nr_hashed', how='left')

# data cleaning
data = prep.drop_older_policies(merged_df, 2019)
data = remove_starting_one(data)
data = prep.minor_edits_for_survival(data)

data.drop(["mutation_1", "mutation_2", "mutation_3", "mutation_4", "mutation_5", "mutation_6",'n_main_coverages', 
           "mutation_7", "mutation_8", "mutation_9", "mutation_10", "mutation_11", "mutation_12",
           "data_collection_date", "product", "n_coverages", 'wettelijke aansprakelijkheid'],
          axis=1, inplace=True)

data = prep.sum_cols(data, [
    "premium_change_mutation_1", "premium_change_mutation_2",
    "premium_change_mutation_3", "premium_change_mutation_4",
    "premium_change_mutation_5", "premium_change_mutation_6",
    "premium_change_mutation_7", "premium_change_mutation_8",
    "premium_change_mutation_9", "premium_change_mutation_10",
    "premium_change_mutation_11", "premium_change_mutation_12"], 'premium_mutations')
data = prep.add_treatment_vars(data)
def postcode_digit(dataframe, i):
    dataframe['postcode_range'] = dataframe['postcode'].astype(str).str[i]
    dataframe['postcode_range'] = dataframe['postcode_range'].astype('category')
    return dataframe

#data=postcode_digit(data, 0)
# data['postcode_new'] = data['postcode'].str[:2]

# impute the inital price and the total premium 
# get values
def fill_premium(data):
    
    df = data.sort_values(by=['policy_nr_hashed', 'years_since_policy_started'])

    get_ini_pri = df[['policy_nr_hashed','total_premium','cluster']]
    get_ini_pri = get_ini_pri.rename(columns={'total_premium': 'inital_price'})

    # Create the model that we can use to create the model
    get_ini_pri = pd.DataFrame(get_ini_pri.drop_duplicates(subset=['policy_nr_hashed'], keep='first'))
    get_ini_pri['inital_price'] = get_ini_pri['inital_price'].astype(float)

    # can be mean or median, we use mean in our case
    def replace_zero_with_group_mean(row, group_means):
        if row['inital_price'] == 0:
            return group_means[row['cluster']]
        else:
            return row['inital_price']

    # Calculate means excluding zeros
    group_means = get_ini_pri[get_ini_pri['inital_price'] != 0].groupby('cluster')['inital_price'].mean()

    # Apply Cluster median to replace zeros
    get_ini_pri['inital_price'] = get_ini_pri.apply(replace_zero_with_group_mean, axis=1, group_means=group_means)

    # df['total_premium'] = df['total_premium'].replace(0, np.nan)
    # df['total_premium'].fillna(method='ffill', inplace=True)
    df = pd.merge(df,get_ini_pri, on='policy_nr_hashed', how='left')
    df = df.drop(['cluster_y'],axis=1)
    df.loc[(df['years_since_policy_started'] == 0) & (df['total_premium'] == 0), 'total_premium'] = df['inital_price']
    df_sorted = df.sort_values(by=['policy_nr_hashed', 'years_since_policy_started'])
    df_sorted['total_premium'] = df_sorted['total_premium'].mask(df_sorted['total_premium'] == 0).ffill()
    df_sorted = df_sorted.rename(columns={'cluster_x': 'cluster'})
    df_sorted['total_premium'] = np.log(df_sorted['total_premium'])
    return df_sorted

df = fill_premium(data)
#df = df.drop('inital_price', axis=1)

remove = ['inital_price','year_initiation_policy', 'year_initiation_policy_version', 'year_end_policy','d_churn_cancellation',
       'd_churn_between_prolongations', 'd_churn_around_prolongation', 'premium_main_coverages', 'premium_supplementary_coverages', 'welcome_discount_control_group'
       , 'postcode','premium_mutations','type','weight','brand', 'WD_level','eligibility_cat']

data_survival = prep.convert_person_period(df, remove)

indexes_nur = [6]

# Convert column indexes to column names
columns_nur = data_survival.columns[indexes_nur]
# df[columns_nur].dtypes
scaler = StandardScaler()
df_scaled_numerical = scaler.fit_transform(data_survival[columns_nur])
data_survival[columns_nur] = df_scaled_numerical
data_survival = data_survival.drop(['cluster_0.0','fuel_type_anders','sales_channel_Other'], axis=1)
data_survival['welcome_discount'] = 1 - data_survival['welcome_discount']

working_columns = ['total_premium','accident_free_years', 'car_value', 'customer_age', 'age_car'
                   ,'n_supplementary_coverages', 'WD_eligible','LPA_eligible', 'cluster_1.0', 'fuel_type_benzine',
                   'fuel_type_diesel', 'fuel_type_electro', 'fuel_type_gas', 'fuel_type_hybride', 'sales_channel_Aegon.nl', 'sales_channel_Independer','allrisk basis', 'allrisk compleet', 'allrisk royaal', 'wa-extra']
newdf = prep.survival_analysis_spe1(data_survival, working_columns)
print(data_survival)
# get the test and train data
# in the case we used the data set splited by the command in the RSF file and import the split directly
policy_nr_train_df = pd.read_csv('X_train_policynumbers1.csv')
policy_nr_train = policy_nr_train_df['policy_nr_hashed']
# Split the dataset into training and testing sets, stratifying by the target variable
# policy_nr_train, policy_nr_test, wd_train, wd_test = train_test_split(
# unique_units['policy_nr_hashed'], unique_units['WD_receive_group'], test_size=0.2, stratify=unique_units['WD_receive_group'])
train_df = newdf[newdf['policy_nr_hashed'].isin(policy_nr_train)]
test_df = newdf[~newdf['policy_nr_hashed'].isin(policy_nr_train)]
y_train = train_df['d_churn']
y_test = test_df['d_churn']
X_train = train_df.drop(['policy_nr_hashed','d_churn','WD_received'], axis=1)
X_test = test_df.drop(['policy_nr_hashed','d_churn','WD_received'], axis=1)
# train_df_rec = train_df[train_df['cluster_1.0'] == 1]
# train_df_nrec = train_df[train_df['cluster_1.0'] == 0]
# test_df_rec = test_df[test_df['cluster_1.0'] == 1]
# test_df_nrec = test_df[test_df['cluster_1.0'] == 0]
# y_train_rec = train_df_rec['d_churn']
# y_train_nrec = train_df_nrec['d_churn']
# y_test_rec = test_df_rec['d_churn']
# y_test_nrec = test_df_nrec['d_churn']
# X_trainrec = train_df_rec.drop(['policy_nr_hashed','d_churn','cluster_1.0','WD_receive_group'], axis=1)
# X_train_norec = train_df_nrec.drop(['policy_nr_hashed','d_churn', 'cluster_1.0','WD_receive_group'], axis=1)
# X_test_rec = test_df_rec.drop(['policy_nr_hashed','d_churn', 'cluster_1.0','WD_receive_group'], axis=1)
# X_test_nrec = test_df_nrec.drop(['policy_nr_hashed','d_churn', 'cluster_1.0','WD_receive_group'], axis=1)

# run the model and get the results
logit_model = sm.Logit(y_train, X_train.astype(float))
result = logit_model.fit()
ame = result.get_margeff().summary()

# Print the summary and the average marginal effect
print(result.summary())
print(ame)
# get the in-sample fitted values and out-of-sample predictions
train_df['h_hat'] = result.predict(X_train)
test_df['h_hat'] = result.predict(X_test)

# get back the years-since-policy-started column for comparision 

columns = [
    'years_since_policy_started_0',
    'years_since_policy_started_1',
    'years_since_policy_started_2',
    'years_since_policy_started_3',
    'years_since_policy_started_4',
    'years_since_policy_started_5',
    #'years_since_policy_started_6',
    #'years_since_policy_started_7',
    #'years_since_policy_started_8',
    #'years_since_policy_started_9',
    #'years_since_policy_started_10',
    #'years_since_policy_started_11',
    #'years_since_policy_started_12',
    #'years_since_policy_started_13',
    #'years_since_policy_started_14',
    #'years_since_policy_started_15',
    #'years_since_policy_started_16',
    #'years_since_policy_started_17',
    #'years_since_policy_started_18',
    #'years_since_policy_started_19',
    #'years_since_policy_started_20',
    #'years_since_policy_started_21',
]
def determine_year(row):
    for i, col in enumerate(columns, start=0):
        if row[col] == 1:
            return i
    return 0  # Return 0 or an appropriate value if none of the conditions are met

# Apply the function to each row to create the new column
train_df['year_since_started'] = train_df.apply(determine_year, axis=1)
test_df['year_since_started'] = test_df.apply(determine_year, axis=1)
train_df['WD'] = train_df.WD_WD_eligible + train_df.WD_eligible
test_df['WD'] = test_df.WD_WD_eligible + test_df.WD_eligible
train_df['LPA'] = train_df.WD_LPA_eligible + train_df.LPA_eligible
test_df['LPA'] = test_df.WD_LPA_eligible + test_df.LPA_eligible


# Sort values (important for cumulative calculations)
test_df.sort_values(by=['policy_nr_hashed', 'year_since_started'], inplace=True)

# Calculate survival probabilities
test_df['survival_probability_per_year'] = 1 - test_df['h_hat']
test_df['survival_probability'] = test_df.groupby('policy_nr_hashed')['survival_probability_per_year'].cumprod()

# Create dictionaries to save the results
def Brier_and_AUC(data):
    brier_scores = {}
    auc_scores = {}
    # Calculate Brier scores and AUC for each period
    for period in data['year_since_started'].unique():

        period_data = data[data['year_since_started'] == period]
        
        #Brier score
        brier_score = brier_score_loss(1 - period_data['d_churn'], period_data['survival_probability'])
        brier_scores[period] = brier_score
        # print(len(np.unique(period_data['d_churn']))) debug
        # AUC - compare true events to predicted survival probabilities
        if len(np.unique(period_data['d_churn'])) > 1:  # AUC is undefined for cases with one unique value
            auc_scores[period] = roc_auc_score(1 - period_data['d_churn'], period_data['survival_probability'])
    
    return brier_scores, auc_scores
    
brier_scores_logit, auc_scores_logit  = Brier_and_AUC(test_df)
print(brier_scores_logit)
print(auc_scores_logit)

drawsur_df = test_df.groupby(['year_since_started', 'WD_received'])['survival_probability'].agg(average_value='median').reset_index()
grouped = drawsur_df.groupby(['WD_received'])
# Create the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

for name, group in grouped:
    plt.plot(group['year_since_started'], group['average_value'], marker='o', linestyle='-', label=f'WD_received = {name}')

plt.ylim(0.4, 1)  
plt.xlabel('Year Since Started')  
plt.ylabel('Mean Survival Prob Probability')  
# plt.title('Mean Survival Prob by Year Since Policy Started')  
plt.legend(title='Group', loc='best')

plt.grid(True)  
plt.tight_layout()  

# Show the plot
plt.show()


# calculate the Brier score for RSF
# prepare the data
def prepare_Brier_for_RSF(da1, da2, da3):
    new_columns_df = pd.DataFrame(da1, columns=['0', '1', '2', '3', '4', '5'])
    # Concatenate the new columns to the original DataFrame
    df = pd.concat([da2.reset_index(drop=True), new_columns_df.reset_index(drop=True),da3.reset_index(drop=True)], axis=1)
    df = df.drop(['d_churn', 'years_since_policy_started'], axis = 1)
    long_df = pd.melt(df, id_vars=['policy_nr_hashed'], var_name='year_since_started', value_name='survival_probability')
    long_df['year_since_started'] = long_df['year_since_started'].astype(int)
    test = test_df.copy()
    test = test.drop('survival_probability', axis = 1)
    tt_df = pd.merge(test, long_df, on=['policy_nr_hashed', 'year_since_started'], how='left')
    tt_df = tt_df[~(tt_df['survival_probability'].isna())]
    return tt_df
    
np_pred_prop = np.load('array.npy') # 'array.npy' is a numpy array of saved by 
df = pd.read_csv('X_test_policynumbers1.csv')
dff = pd.read_csv('Y_test.csv')
tt_df = prepare_Brier_for_RSF(np_pred_prop, df, dff)
brier_scores_rsf = {}
# Calculate Brier scores and AUC for each period
for period in tt_df['year_since_started'].unique():
    period_data = tt_df[tt_df['year_since_started'] == period]
    #Brier score
    brier_score = brier_score_loss(1 - period_data['d_churn'], period_data['survival_probability'])
    brier_scores_rsf[period] = brier_score
print(brier_scores_rsf)

    

# the next part will be gam model
# train_df_1 = data_survival[data_survival['policy_nr_hashed'].isin(policy_nr_train)]
# test_df_1 = data_survival[~data_survival['policy_nr_hashed'].isin(policy_nr_train)]
# y_train_1 = train_df_1['d_churn']
# y_test_1 = test_df_1['d_churn']
# X_train_1 = train_df_1.drop(['policy_nr_hashed','d_churn','WD_received'], axis=1)
# X_test_1 = test_df_1.drop(['policy_nr_hashed','d_churn','WD_received'], axis=1)
# print(X_train_1)
# gam = LogisticGAM(s(0) + s(1) + s(2)+ s(3)+ s(4)+ s(5)+ s(6)+ s(7)+ s(8)+ s(9)+ s(10)+ s(11)+ s(12)+ s(13)+ s(14)+ s(15)+ s(16)
#                  + s(17)+ s(18)+ s(19)+ s(20)+ s(21)+ s(22) + s(23)+ s(24)+ s(25)).fit(X_train_1, y_train_1)
# gam.summary()
"""
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        
        plt.figure()
        plt.plot(XX[:, i], pdep)
        #plt.plot(XX[:, i], confi, c='r', ls='--')
        plt.title(f'Feature {i}')
        plt.show()
"""

