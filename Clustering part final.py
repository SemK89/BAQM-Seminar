# Clustering part finall
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from kmodes.kprototypes import KPrototypes

from sklearn.metrics import silhouette_score
from collections import Counter

# Load CSV
df = pd.read_csv('20240117_churn_data.csv')

# Preprocessing functions
def data_modification(df):
    # Remove mutations with Nan values
    df_dropped = df.drop(["mutation_1", "premium_change_mutation_1", "mutation_2", "premium_change_mutation_2",
                          "mutation_3", "premium_change_mutation_3", "mutation_4", "premium_change_mutation_4",
                          "mutation_5", "premium_change_mutation_5", "mutation_6", "premium_change_mutation_6",
                          "mutation_7", "premium_change_mutation_7", "mutation_8", "premium_change_mutation_8",
                          "mutation_9", "premium_change_mutation_9", "mutation_10", "premium_change_mutation_10",
                          "mutation_11", "premium_change_mutation_11", "mutation_12", "premium_change_mutation_12"],
                         axis=1, inplace=False)

    # Change NaN values of mutations to zero's
    df_mutation = (df[["mutation_1", "premium_change_mutation_1", "mutation_2", "premium_change_mutation_2",
                       "mutation_3", "premium_change_mutation_3", "mutation_4", "premium_change_mutation_4",
                       "mutation_5", "premium_change_mutation_5", "mutation_6", "premium_change_mutation_6",
                       "mutation_7", "premium_change_mutation_7", "mutation_8", "premium_change_mutation_8",
                       "mutation_9", "premium_change_mutation_9", "mutation_10", "premium_change_mutation_10",
                       "mutation_11", "premium_change_mutation_11", "mutation_12", "premium_change_mutation_12"]]
                   .fillna(0))

    # Add back the non-NaN valued mutations
    merged_df = pd.concat([df_dropped, df_mutation], axis=1)
    df = merged_df

    # The label encoder makes sure that categorical variables (object variables) are transformed to labels (numbers)
    label_encoder = LabelEncoder()
    df['welcome_discount_control_group_label'] = label_encoder.fit_transform(df['welcome_discount_control_group'])
    df['brand_label'] = label_encoder.fit_transform(df['brand'])
    df['type_label'] = label_encoder.fit_transform(df['type'])
    df['fuel_type_label'] = label_encoder.fit_transform(df['fuel_type'])
    df['product_label'] = label_encoder.fit_transform(df['product'])
    df['sales_channel_label'] = label_encoder.fit_transform(df['sales_channel'])
    df['policy_nr_hashed_label'] = label_encoder.fit_transform(df['policy_nr_hashed'])
    
    # Convert all elements to strings
    df['postcode'] = df['postcode'].astype(str)

    # Remove quotations if present
    df['postcode'] = df['postcode'].str.replace("'", "")
    
    df['postcode_label'] = label_encoder.fit_transform(df['postcode'])

    # Change NaN values of year-end policy to zero.
    # Note: this variable is might not usable in general for ML/ clustering
    # since 0 now will be associated with not churning.
    # This gives the variable incorrect predictive power.
    df["year_end_policy"] = df["year_end_policy"].fillna(0)

    return df


def drop_older_policies(df, year):
    return (df[df['year_initiation_policy'] >= year]
            .sort_values(by=['policy_nr_hashed', 'year_initiation_policy_version']))


def sum_cols(df, cols_to_sum, new_col_name):
    df[new_col_name] = df[cols_to_sum].sum(axis=1, min_count=1).fillna(0)
    df.drop(cols_to_sum, axis=1, inplace=True)
    return df


def add_treatment_vars(df):
    df['WD_eligible'] = 1
    df.loc[df['welcome_discount_control_group'].str.contains('no WD'), 'WD_eligible'] = 0
    df['LPA_eligible'] = 1
    df.loc[df['welcome_discount_control_group'].str.contains('no LPA'), 'LPA_eligible'] = 0
    df['eligibility_cat'] = (df['WD_eligible'] + 2*df['LPA_eligible']).astype(int)

    df_cs = df.groupby('policy_nr_hashed').agg(WD_level=('welcome_discount', 'min'))
    df = df.merge(df_cs, how='left', on='policy_nr_hashed')

    df['WD_received'] = 0
    df.loc[df['WD_level'] < 1, 'WD_received'] = 1

    return df


def shorten_postal_code(df, digits):
    for i in range(len(df['postcode'])):
        code = df.iloc[i, 10]
        try:
            df.iloc[i, 10] = float(code) // (10**(4-digits))
        except ValueError:
            df.iloc[i, 10] = 0

    df['postcode'].astype(int)
    return df


def minor_edits(df):
    # Apparently some rows are exactly identical, Luca said it was safe to drop these
    df = df.drop_duplicates(subset=['policy_nr_hashed', 'years_since_policy_started'], keep='last')

    # Recompute time periods as some seem to contain errors
    df['years_since_policy_started'] = (df['year_initiation_policy_version'] - df['year_initiation_policy']).astype(int)

    # Removes just over 200 datapoints that ended their policy in 2018 but still show up as a 2019 policy.
    df = df[df['year_end_policy'] != 2018]

    # Some of the postcode values are between quotations (e.g. 2045 is written as '2045').
    # We want to remove this so 2045 and '2045' are treated as the same postcode.
    # df['postcode'] = df['postcode'].str.replace("'", '')

    # Make d_churn=1 for a few cases where it is zero even though the year_end_policy clearly states they churn
    df.loc[df['year_initiation_policy_version'] == df['year_end_policy'], 'd_churn'] = 1

    # WD set to 1 if it is not the first year of the policy
    df.loc[df['years_since_policy_started'] != 0, 'welcome_discount'] = 1

    # Reduce accident-free years by 5 if it is mentioned in the mutation, changeNcbmData
    # rows added by Luca have nan values for premium_main_coverages and premium_supplementary_coverages
    df.loc[((df['premium_main_coverages'].isnull()) &
           ((df['mutation_1'] == 'changeNcbmData') | (df['mutation_2'] == 'changeNcbmData') |
            (df['mutation_3'] == 'changeNcbmData') | (df['mutation_4'] == 'changeNcbmData') |
            (df['mutation_5'] == 'changeNcbmData') | (df['mutation_6'] == 'changeNcbmData') |
            (df['mutation_7'] == 'changeNcbmData') | (df['mutation_8'] == 'changeNcbmData') |
            (df['mutation_9'] == 'changeNcbmData') | (df['mutation_10'] == 'changeNcbmData') |
            (df['mutation_11'] == 'changeNcbmData') |
            (df['mutation_12'] == 'changeNcbmData'))), 'accident_free_years'] -= 5

    # drop rows with churn in 2019 but 0 premium
    df = df[~((df['total_premium'] == 0) & (df['d_churn'] == 1) & (df['year_initiation_policy_version'] == 2019))]

    return df


# Apply Preprocessing for 2019 (Note: Haven't used sum_cols and add_treatment_vars functions!!!!!!!)
df_2019_and_beyond = drop_older_policies(df,2019)
df_2019_and_beyond = data_modification(df_2019_and_beyond)
df_2019_and_beyond = minor_edits(df_2019_and_beyond)
df_2019_and_beyond.reset_index(drop=True, inplace=True)

# Assign df
df = df_2019_and_beyond

# Create a modified dataframe with only unique costumers and their first year observations
# There are some mistakes in the data, causing years_since_policy_started to be 1 or 2 for 
# 2021 iniciated plicies this is not possible so they are removed
df_unique = df[df.years_since_policy_started == 0]
df = df_unique

# Select certain variables of the dataframe for the clustering process
# Welscome_discount not used since we might want to use this later for comparing discount rates
# same goes for welcome_discount_control_group_label
df_cluster = df[["customer_age","accident_free_years","car_value","age_car","weight","allrisk basis","allrisk compleet","allrisk royaal",
                "wa-extra","n_supplementary_coverages", "brand_label", "type_label", "fuel_type_label", "product_label", "sales_channel_label", 
                "postcode_label"]]

################################################################################################################################
################################################################################################################################
# DBSCAN Clustering

# test on smaller dataset
# df_cluster_reduced = df_cluster.sample(n=200000 , random_state=42)
df_cluster_reduced = df_cluster

# Standardize
scaler = StandardScaler()
df_scaled_reduced = scaler.fit_transform(df_cluster_reduced)

start_time = time.time()

# Create DBSCAN object and fit the data
# eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples is the number of samples in a neighborhood for a point to be considered as a core point
dbscan_1 = DBSCAN(eps=3, min_samples=5000)  
dbscan_1.fit(df_scaled_reduced)

end_time = time.time()
print(f"Excecution time: {end_time - start_time} seconds" )


# ANother fit with different epsilon
dbscan_2 = DBSCAN(eps=2.5, min_samples=5000)  
dbscan_2.fit(df_scaled_reduced)

end_time = time.time()

# Add cluster labels to the original DataFrame
df['cluster_dbscan_4'] = dbscan_1.labels_
df['cluster_dbscan_3'] = dbscan_2.labels_


print(f"Excecution time: {end_time - start_time} seconds")

################################################################################################################################
################################################################################################################################
# K-prototypes Clustering

# Specify the categorical columns
categorical_columns = ['allrisk basis', 'allrisk compleet', 'allrisk royaal', 'wa-extra',
                       'brand_label', 'type_label', 'fuel_type_label',
                       'product_label', 'sales_channel_label', "postcode_label"]

# Start timer to check computing time
start_time = time.time()

# Standardize the numerical data
numerical_columns = df_cluster.columns.difference(categorical_columns)
scaler = StandardScaler()
df_scaled_numerical = scaler.fit_transform(df_cluster[numerical_columns])

# Convert back to DataFrame
df_scaled_numerical_df = pd.DataFrame(df_scaled_numerical, columns=numerical_columns)

# Select categorical values
df_cluster_cat = df_cluster[categorical_columns]

# Reset indices of both DataFrames
df_scaled_numerical_df.reset_index(drop=True, inplace=True)
df_cluster_cat.reset_index(drop=True, inplace=True)

# Combine the scaled numerical data with the original categorical data along columns
df_scaled = pd.concat([df_scaled_numerical_df, df_cluster_cat], axis=1)

# i relates to the amount of segments
for i in [2,3,4,5,6,7]:
# for i in [5,6,7]:


    # Specify the number of clusters (you need to decide this based on your problem)
    num_clusters = i

    # Create KPrototypes object and fit the data
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=2, max_iter=14, n_init = 4)
    clusters = kproto.fit_predict(df_scaled, categorical=list(range(len(numerical_columns), len(df_scaled.columns))))

    # End timer
    end_time = time.time()

    # Check computing time
    print(f"Excecution time: {end_time - start_time} seconds" )
    # Add cluster labels to the original DataFrame
    df[f'cluster_k_prototypes_{i}'] = clusters


###############################################################################################################################
###############################################################################################################################    
# Save the new DataFrame as a CSV file
df.to_csv("Aegon_data_with_clusters_2019_v3", index=False)

# Save a file for the clusters only
clusters_df_with_ID = df[["policy_nr_hashed","cluster_k_prototypes_2","cluster_k_prototypes_3","cluster_k_prototypes_4", 
                  "cluster_k_prototypes_5","cluster_k_prototypes_6","cluster_k_prototypes_7",
                  "cluster_dbscan_3", "cluster_dbscan_4"]]
clusters_df_with_ID.to_csv("Aegon_data_only_clusters_2019_with_IDs_v?", index=False)


###############################################################################################################################
############################################################################################################################### 
# Cluster evaluation
df = pd.read_csv('Aegon_data_with_clusters_2019_v3.csv')

# First select all variables used for clustering so the clusters can be assesed on these variables
# No wettelijke aanspraakelijkheid!!!
df = df[["customer_age","accident_free_years","car_value","age_car","weight","allrisk basis","allrisk compleet","allrisk royaal",
         "wa-extra","n_supplementary_coverages", "brand_label", "type_label", "fuel_type_label", "product_label", "sales_channel_label", 
         "postcode_label",
         "cluster_k_prototypes_2","cluster_k_prototypes_3","cluster_k_prototypes_4","cluster_k_prototypes_5", 
         "cluster_k_prototypes_6","cluster_k_prototypes_7","cluster_dbscan_3", "cluster_dbscan_4"]]

# Separate the variables and the optained clusters
df_non_cluster = df.drop(["cluster_k_prototypes_2","cluster_k_prototypes_3","cluster_k_prototypes_4","cluster_k_prototypes_5", 
                          "cluster_k_prototypes_6","cluster_k_prototypes_7","cluster_dbscan_3", "cluster_dbscan_4"], axis = 1, inplace = False)
df_cluster_k_prototypes = df[["cluster_k_prototypes_2","cluster_k_prototypes_3","cluster_k_prototypes_4","cluster_k_prototypes_5", 
                              "cluster_k_prototypes_6","cluster_k_prototypes_7"]]
df_cluster_dbscan = df[["cluster_dbscan_3", "cluster_dbscan_4"]]

# Standardize the non-cluster data (only numerical variables)
categorical_columns = ['allrisk basis', 'allrisk compleet', 'allrisk royaal', 'wa-extra',
                       'n_supplementary_coverages','brand_label', 'type_label', 'fuel_type_label',
                       'product_label', 'sales_channel_label', "postcode_label"]

# Standardize the numerical data
numerical_columns_3 = df_non_cluster.columns.difference(categorical_columns)
numerical_columns = numerical_columns_3

scaler = StandardScaler()
df_scaled_numerical = scaler.fit_transform(df_non_cluster[numerical_columns])

# Combine the scaled numerical data with the original categorical data
df_non_cluster_scaled = pd.concat([pd.DataFrame(df_scaled_numerical, columns=numerical_columns), df_non_cluster[categorical_columns]], axis=1)

###############################################################################################################################
############################################################################################################################### 
# Apply Silhouette Score for K-prototypes
# Score:
# Higher is beter, range: -1 to 1

for i in range(6):
    silhouette_avg_k_prototypes = silhouette_score(df_non_cluster_scaled, df_cluster_k_prototypes.iloc[:,i])
    print(f"Silhouette Score with {i+2} clusters for K-prototypes: {silhouette_avg_k_prototypes}")

# Apply Silhouette Score for DBSCAN
for i in range(2):
    silhouette_avg_dbscan = silhouette_score(df_non_cluster_scaled, df_cluster_dbscan.iloc[:,i])
    print(f"Silhouette Score with DBSCAN {i+3}: {silhouette_avg_dbscan}")

###############################################################################################################################
############################################################################################################################### 
    
from sklearn.metrics import davies_bouldin_score

# Apply davies_bouldin Score for K-prototypes
# Score:
# Lower is beter, minimum is 0
for i in range(6):
    davies_bouldin_k_prototypes = davies_bouldin_score(df_non_cluster_scaled, df_cluster_k_prototypes.iloc[:,i])
    print(f"Davies-Bouldin Index for {i+2} clusters for K-prototypes: {davies_bouldin_k_prototypes}")


# Apply davies_bouldin Score for DBSCAN
for i in range(2):
    davies_bouldin_dbscan = davies_bouldin_score(df_non_cluster_scaled, df_cluster_dbscan.iloc[:,i])
    print(f"Davies-Bouldin Index for DBSCAN {i+3}: {davies_bouldin_dbscan}")


###############################################################################################################################
############################################################################################################################### 
from sklearn.metrics import calinski_harabasz_score

# Apply calinski_harabasz Score for K-prototypes
# Score:
# Higher is beter
for i in range(6):
    calinski_harabasz_k_prototypes = calinski_harabasz_score(df_non_cluster_scaled, df_cluster_k_prototypes.iloc[:,i])
    print(f"calinski_harabasz Index for {i+2} clusters (K-prototype): {calinski_harabasz_k_prototypes}")

# Apply calinski_harabasz Score for DBSCAN
for i in range(2):
    calinski_harabasz_dbscan = calinski_harabasz_score(df_non_cluster_scaled, df_cluster_dbscan.iloc[:,i])
    print(f"calinski_harabasz Index for DBSCAN {i+3}: {calinski_harabasz_dbscan}")


# Original data
# Silhouette Scores

silhouette_scores = [0.013985597736489289, -0.021207547492660406, -0.04047699042565607, -0.04767343028835916,-0.06369579243563082, -0.06874793040935237, -0.018139837452503662, -0.0458069673875922]
davies_bouldin_index = [11.585211159696387, 19.11875058942078, 34.54395203271346, 76.25020255169322,34.875990250331625, 112.48492064080438, 25.7349424042242, 28.981523603040802]
calinski_harabasz_index = [522.6155487067361, 350.2385679270941, 271.26111553191146, 288.2959763709064,279.83489885839197, 233.72711715782532, 161.45274234977157, 324.6776723722957]

# Take inverse of Davies-bouldin score
davies_bouldin_scores_inverse = []
for i in range(len(davies_bouldin_index)):
    davies_bouldin_scores_inverse. append(1/davies_bouldin_index[i])

# Normalization
def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

normalized_silhouette_scores = normalize(silhouette_scores)
normalized_davies_bouldin_scores = normalize(davies_bouldin_index)
normalized_calinski_harabasz_scores = normalize(calinski_harabasz_index)

normalized_davies_bouldin_scores_mirrored = [(x * -1) + 1 for x in normalized_davies_bouldin_scores]

# Printing normalized scores
print("Normalized Silhouette Scores:", normalized_silhouette_scores)
print("Normalized Davies-Bouldin Scores mirrored:", normalized_davies_bouldin_scores_mirrored)
print("Normalized Calinski-Harabasz Scores:", normalized_calinski_harabasz_scores)  

# Fuel types distribution per cluster
print(df[df.cluster_k_prototypes_2 == 0].fuel_type_label.value_counts()/ len(df[df.cluster_k_prototypes_2 == 0]))
print(df[df.cluster_k_prototypes_2 == 1].fuel_type_label.value_counts()/ len(df[df.cluster_k_prototypes_2 == 1]))
print(df.fuel_type_label.value_counts()/ len(df.cluster_k_prototypes_2))

print(df_2019_and_beyond.fuel_type.unique())
print(df_2019_and_beyond.fuel_type_label.unique())

# Means of nummerical variables
cluster_avg_2 = df.groupby('cluster_k_prototypes_2').agg({'customer_age': 'mean',
                                                        'accident_free_years': 'mean',
                                                        'car_value': 'mean',
                                                        'age_car': 'mean',
                                                        'weight': 'mean',
                                                        'allrisk basis': 'mean',
                                                        'allrisk compleet': 'mean',
                                                        'allrisk royaal': 'mean',
                                                        'wa-extra': 'mean',
                                                        "n_supplementary_coverages": 'mean'})

cluster_avg_2.columns = ["customer_age","accident_free_years", "car_value", "age_car", "weight", "allrisk basis", "allrisk compleet",
                        "allrisk royaal", "wa-extra","n_supplementary_coverages"]

cluster_avg = cluster_avg_2

average_variables = df[[ "customer_age","accident_free_years", "car_value", "age_car", "weight", "allrisk basis", "allrisk compleet",
                        "allrisk royaal", "wa-extra","n_supplementary_coverages"]].mean()


for column in cluster_avg:
    cluster_avg[column] = cluster_avg[column] / average_variables[column]


# # subtract 1 to get the relative average of each varaible of the clusters centred around 0.
relative_cluster_avg = cluster_avg - 1

###############################################################################################################################
############################################################################################################################### 
# Make plots
# Get the variables and their values
variables = relative_cluster_avg.columns
cluster_0_values = relative_cluster_avg.loc[0]
cluster_1_values = relative_cluster_avg.loc[1]

# Set the width of the bars
bar_width = 0.35

# Set the x-axis positions for the bars
x = np.arange(len(variables))

# Plot the bars
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, cluster_0_values, width=bar_width, color='blue', label='Cluster 0')
plt.bar(x + bar_width/2, cluster_1_values, width=bar_width, color='red', label='Cluster 1')

# Set the labels and title
plt.xlabel('Variables')
plt.ylabel('Relative Average of centred arround 0')
plt.title('Relative Average of Variables for Each Cluster centred arround 0')

# Set the x-axis tick labels
plt.xticks(x, variables, rotation=90)

# Add legend
plt.legend()

# Another plot

import matplotlib.pyplot as plt

# Separate the dataframe based on the cluster
cluster_0 = df[df['cluster_k_prototypes_2'] == 0]
cluster_1 = df[df['cluster_k_prototypes_2'] == 1]

# Plot the values of accident_free_years against car_value
plt.figure(figsize=(5, 3))
plt.scatter(cluster_0['car_value'], cluster_0['accident_free_years'], label='Cluster 0',alpha = 1, marker = '.', color = 'red') 
plt.scatter(cluster_1['car_value'], cluster_1['accident_free_years'], label='Cluster 1',alpha = 1, marker = '.', color = 'blue')
plt.xlabel('Car Value')
plt.ylabel('Accident Free Years')
plt.legend()
plt.show()
