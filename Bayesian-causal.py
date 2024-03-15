import numpy as np
import pandas as pd
import stan
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

# clustering = pd.read_csv('Aegon_data_only_clusters_2021_with_id',
#                          usecols=["policy_nr_hashed", "cluster_k_prototypes_4"])
# data = data.merge(clustering, how='left', on='policy_nr_hashed')
# data['cluster_k_prototypes_4'] = data['cluster_k_prototypes_4'].fillna(-1).astype(int) + 1
#
# data = prep.shorten_postal_code(data, 2)  # SLOW, takes about a minute, comment out when debugging
# data.rename(columns={'postcode': 'post2', 'cluster_k_prototypes_4': 'cluster'}, inplace=True)
# data.to_csv('churn_data_cleaned_causal.csv', index=False)

bayes_data = {'N': len(data['d_churn']),
              'T': max(data['eligibility_cat']),
              'c': data.loc[:, 'd_churn'].to_numpy(),
              'I_1': data.loc[:, 'customer_age'].to_numpy(),
              'I_2': data.loc[:, 'accident_free_years'].to_numpy(),
              'I_3': data.loc[:, 'car_value'].to_numpy(),
              'I_4': data.loc[:, 'age_car'].to_numpy(),
              'K_1': data.loc[:, 'allrisk basis'].to_numpy(),
              'K_2': data.loc[:, 'allrisk compleet'].to_numpy(),
              'K_3': data.loc[:, 'allrisk royaal'].to_numpy(),
              'K_4': data.loc[:, 'wa-extra'].to_numpy(),
              'K_5': data.loc[:, 'n_supplementary_coverages'].to_numpy(),
              'R': data.loc[:, 'eligibility_cat'].to_numpy()}

simple_flat = """
data {
  int<lower=0> N;
  int<lower=0> T;  
  vector[N] i_1;                // customer age
  vector[N] k_5;                // suppl. insurance
  array[N] int<lower=1,upper=T> r; // treatment factor
  array[N] int<lower=0,upper=1> c; // churn outcomes
}
parameters {
  vector[T] alpha_t;            // treatment intercept
  vector[T] beta_1t;            // treatment slopes I
  real gamma_5;                 // slope K_5
}
model {
  alpha_t ~ normal(0, 100);
  
  for (n in 1:N)
    c[n] ~ bernoulli_logit(alpha_t[r[n]] + beta_1t[r[n]] * i_1[n] + gamma_5 * k_5[n]);
}
"""

logit = stan.build(simple_flat, bayes_data, verbose=True)
fit = logit.sample(data=bayes_data, iter_warmup=1000, iter_sampling=10000, chains=1, show_progress=True)
print(fit)
