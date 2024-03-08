import numpy as np
import pandas as pd
import preprocessing as prep

data = pd.read_csv('20240117_churn_data.csv')
data = prep.drop_older_policies(data, 2021)
data = prep.minor_edits(data)

# Initial preprocessing: cleaning the dataset to only contain potentially useful data and variables
data.drop(["mutation_1", "mutation_2", "mutation_3", "mutation_4", "mutation_5", "mutation_6",
           "mutation_7", "mutation_8", "mutation_9", "mutation_10", "mutation_11", "mutation_12",
           "data_collection_date", "product", "sales_channel"],
          axis=1, inplace=True)
data = prep.sum_cols(data, [
    "premium_change_mutation_1", "premium_change_mutation_2",
    "premium_change_mutation_3", "premium_change_mutation_4",
    "premium_change_mutation_5", "premium_change_mutation_6",
    "premium_change_mutation_7", "premium_change_mutation_8",
    "premium_change_mutation_9", "premium_change_mutation_10",
    "premium_change_mutation_11", "premium_change_mutation_12"], 'premium_mutations')
data = prep.add_treatment_vars(data)

# Secondary preprocessing: (dropping)/modifying variables (not) included in causal model, and adding cluster labels
data.drop(["d_churn_cancellation", "d_churn_between_prolongations", "d_churn_around_prolongation",
           "welcome_discount", "welcome_discount_control_group",
           "premium_main_coverages", "premium_supplementary_coverages", "total_premium", "premium_mutations",
           "brand", "type", "weight", "fuel_type", "wettelijke aansprakelijkheid", "n_main_coverages", "n_coverages"],
          axis=1, inplace=True)
data = prep.shorten_postal_code(data, 2)  # SLOW, takes about a minute, comment out when debugging

clustering = pd.read_csv('Aegon_data_only_clusters_2021_with_id',
                         usecols=["policy_nr_hashed", "cluster_k_prototypes_4"])
data = data.merge(clustering, how='left', on='policy_nr_hashed')
data['cluster_k_prototypes_4'] = data['cluster_k_prototypes_4'].fillna(4).astype(int)

data.rename(columns={'postcode': 'post2', 'cluster_k_prototypes_4': 'cluster'}, inplace=True)

print(data.describe())
