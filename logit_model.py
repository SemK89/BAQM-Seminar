import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pygam import LogisticGAM, s, f
from sklearn.metrics import roc_auc_score
import preprocessing as prep
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss

def remove_starting_one(data):
    a = data[data['years_since_policy_started'] == 0]
    b = a['policy_nr_hashed'].unique()
    data = data[data['policy_nr_hashed'].isin(b)]
    
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
data = prep.minor_edits(data)

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
data['postcode_new'] = data['postcode'].str[:2]

# impute the data
def fill_premium(data):
    
    df = data.sort_values(by=['policy_nr_hashed', 'years_since_policy_started'])

    get_ini_pri = df[['policy_nr_hashed','total_premium','cluster']]
    get_ini_pri = get_ini_pri.rename(columns={'total_premium': 'inital_price'})

    get_ini_pri = pd.DataFrame(get_ini_pri.drop_duplicates(subset=['policy_nr_hashed'], keep='first'))
    get_ini_pri['inital_price'] = get_ini_pri['inital_price'].astype(float)

    
    def replace_zero_with_group_mean(row, group_means):
        if row['inital_price'] == 0:
            return group_means[row['cluster']]
        else:
            return row['inital_price']

    # Calculate means excluding zeros
    group_means = get_ini_pri[get_ini_pri['inital_price'] != 0].groupby('cluster')['inital_price'].median()

    # Apply the function to replace zeros
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
       , 'postcode','premium_mutations','type','weight','brand']

data_survival = prep.convert_person_period(df, remove)

indexes_nur = [6]

# Convert column indexes to column names
columns_nur = data_survival.columns[indexes_nur]
# df[columns_nur].dtypes
scaler = StandardScaler()
df_scaled_numerical = scaler.fit_transform(data_survival[columns_nur])
data_survival[columns_nur] = df_scaled_numerical

data_survival = data_survival.drop(['cluster_0.0','fuel_type_anders','sales_channel_Other','postcode_new'], axis=1)
# data_survival['welcome_discount'] = 1 - data_survival['welcome_discount']

working_columns = ['total_premium','accident_free_years', 'car_value', 'customer_age'
                   ,'n_supplementary_coverages', 'WD_eligible','LPA_eligible', 'cluster_1.0', 'fuel_type_benzine',
                   'fuel_type_diesel', 'fuel_type_electro', 'fuel_type_gas', 'fuel_type_hybride', 'sales_channel_Aegon.nl', 'sales_channel_Independer','allrisk basis', 'allrisk compleet', 'allrisk royaal', 'wa-extra']
newdf = prep. survival_analysis_spe1(data_survival, working_columns)

# get the test and train data

policy_nr_train_df = pd.read_csv('X_train_policynumbers1.csv')
policy_nr_train = policy_nr_train_df['policy_nr_hashed']
# Split the dataset into training and testing sets, stratifying by the target variable
#policy_nr_train, policy_nr_test, wd_train, wd_test = train_test_split(
#    unique_units['policy_nr_hashed'], unique_units['WD_receive_group'], test_size=0.2, stratify=unique_units['WD_receive_group'])
train_df = newdf[newdf['policy_nr_hashed'].isin(policy_nr_train)]
test_df = newdf[~newdf['policy_nr_hashed'].isin(policy_nr_train)]
test_df = test_df[~((test_df['policy_nr_hashed'].isin(['0WKoK0m','0WKobZ5']))& (test_df['welcome_discount'] != 0))]
train_df_rec = train_df[train_df['cluster_1.0'] == 1]
train_df_nrec = train_df[train_df['cluster_1.0'] == 0]
test_df_rec = test_df[test_df['cluster_1.0'] == 1]
test_df_nrec = test_df[test_df['cluster_1.0'] == 0]
y_train = train_df['d_churn']
y_test = test_df['d_churn']
y_train_rec = train_df_rec['d_churn']
y_train_nrec = train_df_nrec['d_churn']
y_test_rec = test_df_rec['d_churn']
y_test_nrec = test_df_nrec['d_churn']

X_train = train_df.drop(['policy_nr_hashed','d_churn','WD_receive_group', 'cluster_1.0'], axis=1)
X_test = test_df.drop(['policy_nr_hashed','d_churn','WD_receive_group', 'cluster_1.0'], axis=1)
X_trainrec = train_df_rec.drop(['policy_nr_hashed','d_churn','cluster_1.0','WD_receive_group'], axis=1)
X_train_norec = train_df_nrec.drop(['policy_nr_hashed','d_churn', 'cluster_1.0','WD_receive_group'], axis=1)
X_test_rec = test_df_rec.drop(['policy_nr_hashed','d_churn', 'cluster_1.0','WD_receive_group'], axis=1)
X_test_nrec = test_df_nrec.drop(['policy_nr_hashed','d_churn', 'cluster_1.0','WD_receive_group'], axis=1)

logit_model = sm.Logit(y_train, X_train.astype(float))
result = logit_model.fit()

# Print the summary
print(result.summary())

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
test_df['survival_probability'] = 1 - test_df['h_hat']
test_df['cumulative_survival'] = test_df.groupby('policy_nr_hashed')['survival_probability'].cumprod()

# Initialize a dictionary to hold Brier scores for each period
brier_scores = {}
auc_scores = {}
# Calculate Brier scores for each period
for period in test_df['year_since_started'].unique():
    # Filter data for the current period
    period_data = test_df[test_df['year_since_started'] == period]
    
    # Calculate Brier score
    brier_score = brier_score_loss(period_data['d_churn'], 1-period_data['cumulative_survival'])
    brier_scores[period] = brier_score
    # print(len(np.unique(period_data['d_churn'])))
    
    if len(np.unique(period_data['d_churn'])) > 1:  # AUC is undefined for cases with one label!!!!
        auc_scores[period] = roc_auc_score(1-period_data['d_churn'], period_data['cumulative_survival'])
        print()

# Convert Brier scores dictionary to a DataFrame for easier viewing/manipulation
brier_scores_df = pd.DataFrame(list(brier_scores.items()), columns=['year_since_started', 'Brier Score'])
print(brier_scores_df)
print(auc_scores)

drawsur_df = test_df.groupby(['year_since_started', 'WD_receive_group'])['cumulative_survival'].agg(average_value='median').reset_index()
grouped = drawsur_df.groupby(['WD_receive_group'])
# Create the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

for name, group in grouped:
    plt.plot(group['year_since_started'], group['average_value'], marker='o', linestyle='-', label=f'WD_receive_group = {name}')

# Customize the plot
plt.ylim(0.4, 1)  # Set the limits for the y-axis
plt.xlabel('Year Since Started')  # x-axis label
plt.ylabel('Median Survival Prob Probability')  # y-axis label
plt.title('Median Survival Prob by Year Since Policy Started')  # Plot title
plt.legend(title='Group', loc='best')  # Show legend with a title, adjust location as needed

plt.grid(True)  # Add a grid for better readability
plt.tight_layout()  # Adjust subplots to fit into figure area.

# Show the plot
plt.show()

auc = {
    'time': [0, 1, 2, 3, 4],
    'AUC_RSF': [0.77457805, 0.80593139, 0.81364628, 0.79631005, 0.65836781],
    'AUC_D': [0.7498240618543209, 0.6923026052838748, 0.6315181425868449, 0.6101605621076793, 0.5855144032921811],
    'AUC_Bayes': [0.513148, 0.475947, 0.416001, None, None]
}

df = pd.DataFrame(auc)

plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['AUC_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['AUC_D'], marker='o', linestyle='-', color='blue', label='Discete survival model')
#plt.title('AUC over Time')
plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('Time (Years)')
plt.ylabel('AUC')
plt.legend()
plt.grid(False)
plt.show()



bs = {
    'time': [0, 1, 2, 3, 4, 5],
    'Brier_score_RSF': [0.0665457, 0.113371, 0.136691, 0.194780, 0.208777, 0.224739],
    'Brier_score_D': [0.110983, 0.141267, 0.148436, 0.176767, 0.209872, 0.222184],
    'Brier_score_Bayes': [0.149836, 0.159919, 0.128879, None, None]
}
# Creating a DataFrame
df = pd.DataFrame(bs)


plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['Brier_score_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['Brier_score_D'], marker='o', linestyle='-', color='blue', label='Discete survival model')
#plt.title('Brier Score over Time')
plt.xlabel('Time (Years)')
plt.ylabel('Brier Score')
plt.legend()
plt.grid(False)
plt.show()

# comp wd not receeived group
data = {
    'time': [0, 1, 2, 3, 4, 5],
    'nodiscount_D': [0.9071953846, 0.8060586875, 0.7352967617, 0.6632053886,0.6148284882,0.5789819005],
    'nodiscount_RSF': [0.886757, 0.779244, 0.703352, 0.649414, 0.61716, 0.616383],
    'nodiscount_Bayes': [0.857971, 0.857971*0.830846, 0.857971*0.830846*0.982453, None, None]
    }

# Creating a DataFrame
df = pd.DataFrame(data)


plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['nodiscount_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['nodiscount_D'], marker='o', linestyle='-', color='blue', label='Discete survival model')
#plt.title('Survival function Non Discount customers')
plt.xlabel('Time (Years)')
plt.ylim(0.5,1)
plt.ylabel('mean survival probability')
plt.legend()
plt.grid(False)
plt.show()

# wd received group
data = {
    'time': [0, 1, 2],
    'discount_D': [0.8056584128, 0.6290134692, 0.5723574354],
    'discount_RSF': [0.799636,0.677937,0.648282],
    'discount_Bayes': [0.853999, 0.853999*0.841527, 0.853999*0.841527*0.852274]
    }

# Creating a DataFrame
df = pd.DataFrame(data)


plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['discount_RSF'], marker='o', linestyle='-', color='red', label='Random Survival Forest')
plt.plot(df['time'], df['discount_D'], marker='o', linestyle='-', color='blue', label='Discete survival model')
#plt.title('Survival function Discount customers')
plt.xlabel('Time (Years)')
plt.ylabel('mean survival probability')
plt.legend()
plt.ylim(0.5,1)
plt.grid(False)
plt.show()