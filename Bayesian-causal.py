import sys
import multiprocessing
import pandas as pd
import numpy as np
import stan  # ensure cython and numpy are installed, as well as a gcc compiler using homebrew
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing as prep  # own class

multiprocessing.set_start_method("fork")
pd.options.mode.chained_assignment = None
f = open("Bayes_output.txt", 'w')
sys.stdout = f
seed = sum([ord(letter) for i, letter in enumerate('Herstad')])
np.random.seed(seed)

data = pd.read_csv('20240117_churn_data.csv')
data = prep.drop_older_policies(data, 2021)
data = prep.drop_new_policies(data)
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

# Secondary preprocessing: (dropping)/modifying variables (not) included in causal model,
# and adding cluster + train/test labels
data.drop(["d_churn_cancellation", "d_churn_between_prolongations", "d_churn_around_prolongation",
           "welcome_discount", "welcome_discount_control_group", "postcode",
           "premium_main_coverages", "premium_supplementary_coverages", "total_premium", "premium_mutations",
           "brand", "type", "weight", "fuel_type", "wettelijke aansprakelijkheid", "n_main_coverages", "n_coverages"],
          axis=1, inplace=True)
clustering = pd.read_csv('Aegon_data_only_clusters_2019_with_IDs_3',
                         usecols=["policy_nr_hashed", "cluster_k_prototypes_2"])
data = data.merge(clustering, how='left', on='policy_nr_hashed')
data['cluster_k_prototypes_2'] = data['cluster_k_prototypes_2'].fillna(-1).astype(int) + 1
data.rename(columns={'cluster_k_prototypes_2': 'cluster'}, inplace=True)

test_sample = pd.read_csv('X_test_policynumbers1.csv')['policy_nr_hashed'].to_list()
data['tt_split'] = 0
data.loc[data['policy_nr_hashed'].isin(test_sample), 'tt_split'] = 1


# Utility functions for computing results similarly across model configurations
def bayes_data(data_train, data_test):
    return {'N': len(data_train['d_churn']),
            'P': len(data_test['d_churn']),
            'E': max(data_train['eligibility_cat']),
            'T': max(data_train['years_since_policy_started']),
            'c': data_train.loc[:, 'd_churn'].to_list(),
            't': data_train.loc[:, 'years_since_policy_started'].to_list(),
            'e': data_train.loc[:, 'eligibility_cat'].to_list(),
            'clu': data_train.loc[:, 'cluster'].to_list(),
            'i_1': data_train.loc[:, 'customer_age'].to_list(),
            'i_2': data_train.loc[:, 'accident_free_years'].to_list(),
            'i_3': data_train.loc[:, 'car_value'].to_list(),
            'i_4': data_train.loc[:, 'age_car'].to_list(),
            'k_1': data_train.loc[:, 'allrisk basis'].to_list(),
            'k_2': data_train.loc[:, 'allrisk compleet'].to_list(),
            'k_3': data_train.loc[:, 'allrisk royaal'].to_list(),
            'k_4': data_train.loc[:, 'wa-extra'].to_list(),
            'k_5': data_train.loc[:, 'n_supplementary_coverages'].to_list(),
            'p_c': data_test.loc[:, 'd_churn'].to_list(),
            'p_t': data_test.loc[:, 'years_since_policy_started'].to_list(),
            'p_e': data_test.loc[:, 'eligibility_cat'].to_list(),
            'p_clu': data_test.loc[:, 'cluster'].to_list(),
            'p_i_1': data_test.loc[:, 'customer_age'].to_list(),
            'p_i_2': data_test.loc[:, 'accident_free_years'].to_list(),
            'p_i_3': data_test.loc[:, 'car_value'].to_list(),
            'p_i_4': data_test.loc[:, 'age_car'].to_list(),
            'p_k_1': data_test.loc[:, 'allrisk basis'].to_list(),
            'p_k_2': data_test.loc[:, 'allrisk compleet'].to_list(),
            'p_k_3': data_test.loc[:, 'allrisk royaal'].to_list(),
            'p_k_4': data_test.loc[:, 'wa-extra'].to_list(),
            'p_k_5': data_test.loc[:, 'n_supplementary_coverages'].to_list()}


def accuracy_measures(df):
    tp = len(df.loc[(df['d_churn'] == 1) & (df['pred_churn'] == 1), :])
    fp = len(df.loc[(df['d_churn'] == 0) & (df['pred_churn'] == 1), :])
    fn = len(df.loc[(df['d_churn'] == 1) & (df['pred_churn'] == 0), :])
    tn = len(df.loc[(df['d_churn'] == 0) & (df['pred_churn'] == 0), :])
    n = len(df['d_churn'])
    sq_diff = sum(np.square(df['pred_p_churn'] - df['d_churn']))

    acc = (tp+tn)/n
    f1 = 2*tp / (2*tp + fp + fn)
    brier = sq_diff / n
    return acc, f1, brier


def brier_auc_per_cat(df):
    brier_scores = {}
    auc_scores = {}
    for period in df['years_since_policy_started'].unique():
        # Filter data for the current period
        period_data = df[df['years_since_policy_started'] == period]

        diff = np.square(period_data['pred_p_churn'] - period_data['d_churn'])
        tot_diff = sum(diff)
        n = len(period_data['d_churn'])
        brier_scores[period] = tot_diff/n

        if len(np.unique(period_data['d_churn'])) > 1:  # AUC is undefined for cases with one label!!!!
            auc_scores[period] = roc_auc_score(1 - period_data['d_churn'], 1 - period_data['pred_p_churn'])

    return brier_scores, auc_scores


def output_results(fit, vals_to_fit, data_label, give_draws=False):
    results = fit.to_frame()

    if give_draws:
        param_draws = results.loc[:, f'alpha{"" if max(vals_to_fit["years_since_policy_started"]) == 1 else ".1"}':f'gamma_5{"" if max(vals_to_fit["years_since_policy_started"]) == 1 else "."+str(max(vals_to_fit["years_since_policy_started"]))}']
        param_draws.to_csv(f'param_draws_{data_label}.csv')

        fit_phi_agg = np.array(results.loc[:, 'phi_sim.1':f'phi_sim.{len(vals_to_fit["d_churn"])}'].mean(axis=0))
        vals_to_fit['pred_treat_eff'] = fit_phi_agg
        phi_table_cat = pd.pivot_table(vals_to_fit, index='years_since_policy_started', columns='eligibility_cat',
                                       values='pred_treat_eff', aggfunc=[np.mean, np.std])
        phi_table_cat.to_csv(f'treat_eff_{data_label}.csv')

    fit_p_agg = np.array(results.loc[:, 'p_sim.1':f'p_sim.{len(vals_to_fit["d_churn"])}'].mean(axis=0))
    vals_to_fit['pred_p_churn'] = fit_p_agg
    vals_to_fit['pred_churn'] = np.random.binomial(1, fit_p_agg)
    vals_to_fit.to_csv(f'output_data_{data_label}.csv')

    p_table_cat = 1 - pd.pivot_table(vals_to_fit, index='years_since_policy_started', columns='eligibility_cat',
                                     values='pred_p_churn', aggfunc=np.mean)
    p_table_discount = 1 - pd.pivot_table(vals_to_fit, index='years_since_policy_started', columns='WD_received',
                                          values='pred_p_churn', aggfunc=np.mean)

    accuracy, f1_score, brier_score_overall = np.round(accuracy_measures(vals_to_fit), 6)
    brier_scores, auc_scores = brier_auc_per_cat(vals_to_fit)
    print(p_table_cat)
    print(p_table_discount)
    print(f'AUC: \n {auc_scores}')
    print(f'Brier: \n {brier_scores}')
    print(f'Overall Brier-score: {brier_score_overall}')
    print(f'Accuracy: {accuracy}')
    print(f'F1-score: {f1_score}')


# Different versions of used STAN models
general_model_weak = """
data {
  int<lower=0> N;                   // n_datapoints train
  int<lower=0> P;                   // n_datapoints for prediction
  int<lower=0> T;                   // time periods
  int<lower=0> E;                   // treatment categories
  
  vector[N] i_1;                    // customer age
  vector[N] i_2;                    // accident free years
  vector[N] i_3;                    // car value
  vector[N] i_4;                    // car age
  vector[N] k_1;                    // allrisk basis
  vector[N] k_2;                    // allrisk compleet
  vector[N] k_3;                    // allrisk. royaal
  vector[N] k_4;                    // wa-extra
  vector[N] k_5;                    // suppl. insurance
  array[N] int<lower=1,upper=T> t;  // time factor
  array[N] int<lower=1,upper=E> e;  // treatment factor
  array[N] int<lower=0,upper=1> c;  // churn outcomes
  
  vector[P] p_i_1;                  // test sample analogues
  vector[P] p_i_2;                    
  vector[P] p_i_3;                    
  vector[P] p_i_4;                    
  vector[P] p_k_1;                    
  vector[P] p_k_2;                    
  vector[P] p_k_3;                    
  vector[P] p_k_4;                    
  vector[P] p_k_5;                    
  array[P] int<lower=1,upper=T> p_t;  
  array[P] int<lower=1,upper=E> p_e;  
  array[P] int<lower=0,upper=1> p_c;  
}
parameters {
  vector[T] alpha;                  // general intercept
  matrix[T, E] alpha_e;             // treatment intercept
  vector[T] beta_1;                 // general slopes I
  vector[T] beta_2;
  vector[T] beta_3;
  vector[T] beta_4;
  matrix[T, E] beta_1e;             // treatment slopes I
  matrix[T, E] beta_2e;
  matrix[T, E] beta_3e;
  matrix[T, E] beta_4e;
  vector[T] gamma_1;                // slopes K
  vector[T] gamma_2;
  vector[T] gamma_3;
  vector[T] gamma_4;
  vector[T] gamma_5;
  real<lower=0> scale_alpha;        // variances
  real<lower=0> scale_other;
  cholesky_factor_corr[T] R;
  cholesky_factor_corr[T-1] R_restr;
}
transformed parameters {
  vector[T] Zero = rep_vector(0, T);
  row_vector[T] row_Zero = rep_row_vector(0, T);
  vector[T] near_Zero = rep_vector(0.0001, T);

  cholesky_factor_cov[T] Sigma_alpha;
  cholesky_factor_cov[T] Sigma_alpha_e;
  cholesky_factor_cov[T] Sigma_beta_e;
  cholesky_factor_cov[T] Sigma_other;

  Sigma_alpha = diag_pre_multiply(rep_vector(scale_alpha, T), R);
  Sigma_other = diag_pre_multiply(rep_vector(scale_other, T), R);

  Sigma_alpha_e[1,:] = row_Zero;    // treat. cat 1 params should equal 0, but must be estimable
  Sigma_alpha_e[:,1] = near_Zero;
  Sigma_beta_e[1,:] = row_Zero;
  Sigma_beta_e[:,1] = near_Zero;

  Sigma_alpha_e[2:,2:] = diag_pre_multiply(rep_vector(scale_alpha, T-1), R_restr);
  Sigma_beta_e[2:,2:] = diag_pre_multiply(rep_vector(scale_other, T-1), R_restr);
}
model {
  scale_alpha ~ cauchy(0, 25);                               // variance priors
  scale_other ~ cauchy(0, 10);
  R ~ lkj_corr_cholesky(1);
  R_restr ~ lkj_corr_cholesky(1);

  alpha ~ multi_normal_cholesky(Zero, Sigma_alpha);         // coefficient priors
  beta_1 ~ multi_normal_cholesky(Zero, Sigma_other);
  beta_2 ~ multi_normal_cholesky(Zero, Sigma_other);
  beta_3 ~ multi_normal_cholesky(Zero, Sigma_other);
  beta_4 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_1 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_2 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_3 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_4 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_5 ~ multi_normal_cholesky(Zero, Sigma_other);

  for (d in 1:E) {
  alpha_e[:,d] ~ multi_normal_cholesky(Zero, Sigma_alpha_e);
  beta_1e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  beta_2e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  beta_3e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  beta_4e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  }

  for (n in 1:N)
    c[n] ~ bernoulli_logit(alpha[t[n]] + alpha_e[t[n], e[n]] + 
    beta_1e[t[n], e[n]] * i_1[n] + beta_1[t[n]] * i_1[n] +
    beta_2e[t[n], e[n]] * i_2[n] + beta_2[t[n]] * i_2[n] + 
    beta_3e[t[n], e[n]] * i_3[n] + beta_3[t[n]] * i_3[n] + 
    beta_4e[t[n], e[n]] * i_4[n] + beta_4[t[n]] * i_4[n] +  
    gamma_1[t[n]] * k_1[n] + 
    gamma_2[t[n]] * k_2[n] + 
    gamma_3[t[n]] * k_3[n] + 
    gamma_4[t[n]] * k_4[n] + 
    gamma_5[t[n]] * k_5[n]);
}
generated quantities {
  vector[P] phi_sim;
  vector[P] p_sim;
  
  for(p in 1:P) {
    phi_sim[p] = alpha[p_t[p]] + alpha_e[p_t[p], p_e[p]] + 
    beta_1e[p_t[p], p_e[p]] * p_i_1[p] + beta_1[p_t[p]] * p_i_1[p] +
    beta_2e[p_t[p], p_e[p]] * p_i_2[p] + beta_2[p_t[p]] * p_i_2[p] + 
    beta_3e[p_t[p], p_e[p]] * p_i_3[p] + beta_3[p_t[p]] * p_i_3[p] + 
    beta_4e[p_t[p], p_e[p]] * p_i_4[p] + beta_4[p_t[p]] * p_i_4[p];
    
    p_sim[p] = inv_logit(phi_sim[p] +
    gamma_1[p_t[p]] * p_k_1[p] + 
    gamma_2[p_t[p]] * p_k_2[p] + 
    gamma_3[p_t[p]] * p_k_3[p] + 
    gamma_4[p_t[p]] * p_k_4[p] + 
    gamma_5[p_t[p]] * p_k_5[p]);
  }
}
"""

dummy_model_weak = """
data {
  int<lower=0> N;                   // n_datapoints train
  int<lower=0> P;                   // n_datapoints for prediction
  int<lower=0> T;                   // time periods
  int<lower=0> E;                   // treatment categories

  vector[N] i_1;                    // customer age
  vector[N] i_2;                    // accident free years
  vector[N] i_3;                    // car value
  vector[N] i_4;                    // car age
  vector[N] k_1;                    // allrisk basis
  vector[N] k_2;                    // allrisk compleet
  vector[N] k_3;                    // allrisk. royaal
  vector[N] k_4;                    // wa-extra
  vector[N] k_5;                    // suppl. insurance
  vector[N] clu;                    // cluster dummy
  array[N] int<lower=1,upper=T> t;  // time factor
  array[N] int<lower=1,upper=E> e;  // treatment factor
  array[N] int<lower=0,upper=1> c;  // churn outcomes

  vector[P] p_i_1;                  // test sample analogues
  vector[P] p_i_2;                    
  vector[P] p_i_3;                    
  vector[P] p_i_4;                    
  vector[P] p_k_1;                    
  vector[P] p_k_2;                    
  vector[P] p_k_3;                    
  vector[P] p_k_4;                    
  vector[P] p_k_5;
  vector[P] p_clu;                    
  array[P] int<lower=1,upper=T> p_t;  
  array[P] int<lower=1,upper=E> p_e;  
  array[P] int<lower=0,upper=1> p_c;  
}
parameters {
  vector[T] alpha;                  // general intercept
  matrix[T, E] alpha_e;             // treatment intercept
  vector[T] beta_1;                 // general slopes I
  vector[T] beta_2;
  vector[T] beta_3;
  vector[T] beta_4;
  matrix[T, E] beta_1e;             // treatment slopes I
  matrix[T, E] beta_2e;
  matrix[T, E] beta_3e;
  matrix[T, E] beta_4e;
  real eta;                         // cluster dummy coefficient
  vector[T] gamma_1;                // slopes K
  vector[T] gamma_2;
  vector[T] gamma_3;
  vector[T] gamma_4;
  vector[T] gamma_5;
  real<lower=0> scale_alpha;        // variances
  real<lower=0> scale_other;
  cholesky_factor_corr[T] R;
  cholesky_factor_corr[T-1] R_restr;
}
transformed parameters {
  vector[T] Zero = rep_vector(0, T);
  row_vector[T] row_Zero = rep_row_vector(0, T);
  vector[T] near_Zero = rep_vector(0.0001, T);

  cholesky_factor_cov[T] Sigma_alpha;
  cholesky_factor_cov[T] Sigma_alpha_e;
  cholesky_factor_cov[T] Sigma_beta_e;
  cholesky_factor_cov[T] Sigma_other;

  Sigma_alpha = diag_pre_multiply(rep_vector(scale_alpha, T), R);
  Sigma_other = diag_pre_multiply(rep_vector(scale_other, T), R);

  Sigma_alpha_e[1,:] = row_Zero;    // treat. cat 1 params should equal 0, but must be estimable
  Sigma_alpha_e[:,1] = near_Zero;
  Sigma_beta_e[1,:] = row_Zero;
  Sigma_beta_e[:,1] = near_Zero;

  Sigma_alpha_e[2:,2:] = diag_pre_multiply(rep_vector(scale_alpha, T-1), R_restr);
  Sigma_beta_e[2:,2:] = diag_pre_multiply(rep_vector(scale_other, T-1), R_restr);
}
model {
  scale_alpha ~ cauchy(0, 25);                               // variance priors
  scale_other ~ cauchy(0, 10);
  R ~ lkj_corr_cholesky(1);
  R_restr ~ lkj_corr_cholesky(1);

  eta ~ normal(0, scale_alpha);
  alpha ~ multi_normal_cholesky(Zero, Sigma_alpha);         // coefficient priors
  beta_1 ~ multi_normal_cholesky(Zero, Sigma_other);
  beta_2 ~ multi_normal_cholesky(Zero, Sigma_other);
  beta_3 ~ multi_normal_cholesky(Zero, Sigma_other);
  beta_4 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_1 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_2 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_3 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_4 ~ multi_normal_cholesky(Zero, Sigma_other);
  gamma_5 ~ multi_normal_cholesky(Zero, Sigma_other);

  for (d in 1:E) {
  alpha_e[:,d] ~ multi_normal_cholesky(Zero, Sigma_alpha_e);
  beta_1e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  beta_2e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  beta_3e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  beta_4e[:,d] ~ multi_normal_cholesky(Zero, Sigma_beta_e);
  }

  for (n in 1:N)
    c[n] ~ bernoulli_logit(alpha[t[n]] + alpha_e[t[n], e[n]] + eta * clu[n] + 
    beta_1e[t[n], e[n]] * i_1[n] + beta_1[t[n]] * i_1[n] +
    beta_2e[t[n], e[n]] * i_2[n] + beta_2[t[n]] * i_2[n] + 
    beta_3e[t[n], e[n]] * i_3[n] + beta_3[t[n]] * i_3[n] + 
    beta_4e[t[n], e[n]] * i_4[n] + beta_4[t[n]] * i_4[n] +  
    gamma_1[t[n]] * k_1[n] + 
    gamma_2[t[n]] * k_2[n] + 
    gamma_3[t[n]] * k_3[n] + 
    gamma_4[t[n]] * k_4[n] + 
    gamma_5[t[n]] * k_5[n]);
}
generated quantities {
  vector[P] phi_sim;
  vector[P] p_sim;

  for(p in 1:P) {
    phi_sim[p] = alpha[p_t[p]] + alpha_e[p_t[p], p_e[p]] + 
    beta_1e[p_t[p], p_e[p]] * p_i_1[p] + beta_1[p_t[p]] * p_i_1[p] +
    beta_2e[p_t[p], p_e[p]] * p_i_2[p] + beta_2[p_t[p]] * p_i_2[p] + 
    beta_3e[p_t[p], p_e[p]] * p_i_3[p] + beta_3[p_t[p]] * p_i_3[p] + 
    beta_4e[p_t[p], p_e[p]] * p_i_4[p] + beta_4[p_t[p]] * p_i_4[p];

    p_sim[p] = inv_logit(phi_sim[p] + eta * p_clu[p] +
    gamma_1[p_t[p]] * p_k_1[p] + 
    gamma_2[p_t[p]] * p_k_2[p] + 
    gamma_3[p_t[p]] * p_k_3[p] + 
    gamma_4[p_t[p]] * p_k_4[p] + 
    gamma_5[p_t[p]] * p_k_5[p]);
  }
}
"""

one_period_model_weak = """
data {
  int<lower=0> N;                   // n_datapoints train
  int<lower=0> P;                   // n_datapoints for prediction
  int<lower=0> E;                   // treatment categories

  vector[N] i_1;                    // customer age
  vector[N] i_2;                    // accident free years
  vector[N] i_3;                    // car value
  vector[N] i_4;                    // car age
  vector[N] k_1;                    // allrisk basis
  vector[N] k_2;                    // allrisk compleet
  vector[N] k_3;                    // allrisk. royaal
  vector[N] k_4;                    // wa-extra
  vector[N] k_5;                    // suppl. insurance
  array[N] int<lower=1,upper=E> e;  // treatment factor
  array[N] int<lower=0,upper=1> c;  // churn outcomes

  vector[P] p_i_1;                  // test sample analogues
  vector[P] p_i_2;                    
  vector[P] p_i_3;                    
  vector[P] p_i_4;                    
  vector[P] p_k_1;                    
  vector[P] p_k_2;                    
  vector[P] p_k_3;                    
  vector[P] p_k_4;                    
  vector[P] p_k_5;                    
  array[P] int<lower=1,upper=E> p_e;  
  array[P] int<lower=0,upper=1> p_c;  
}
parameters {
  real alpha;                  // general intercept
  vector[E] alpha_e;             // treatment intercept
  real beta_1;                 // general slopes I
  real beta_2;
  real beta_3;
  real beta_4;
  vector[E] beta_1e;             // treatment slopes I
  vector[E] beta_2e;
  vector[E] beta_3e;
  vector[E] beta_4e;
  real gamma_1;                // slopes K
  real gamma_2;
  real gamma_3;
  real gamma_4;
  real gamma_5;
  real<lower=0> scale_alpha;        // variances
  real<lower=0> scale_other;
}
transformed parameters {
  vector[E] Zero = rep_vector(0, E);
}
model {
  scale_alpha ~ cauchy(0, 25);                               // variance priors
  scale_other ~ cauchy(0, 10);

  alpha ~ normal(0, scale_alpha);         // coefficient priors
  beta_1 ~ normal(0, scale_other);
  beta_2 ~ normal(0, scale_other);
  beta_3 ~ normal(0, scale_other);
  beta_4 ~ normal(0, scale_other);
  gamma_1 ~ normal(0, scale_other);
  gamma_2 ~ normal(0, scale_other);
  gamma_3 ~ normal(0, scale_other);
  gamma_4 ~ normal(0, scale_other);

  alpha_e ~ normal(Zero, scale_alpha);
  beta_1e ~ normal(Zero, scale_other);
  beta_2e ~ normal(Zero, scale_other);
  beta_3e ~ normal(Zero, scale_other);
  beta_4e ~ normal(Zero, scale_other);

  for (n in 1:N)
    c[n] ~ bernoulli_logit(alpha + alpha_e[e[n]] + 
    beta_1e[e[n]] * i_1[n] + beta_1 * i_1[n] +
    beta_2e[e[n]] * i_2[n] + beta_2 * i_2[n] + 
    beta_3e[e[n]] * i_3[n] + beta_3 * i_3[n] + 
    beta_4e[e[n]] * i_4[n] + beta_4 * i_4[n] +  
    gamma_1 * k_1[n] + 
    gamma_2 * k_2[n] + 
    gamma_3 * k_3[n] + 
    gamma_4 * k_4[n] + 
    gamma_5 * k_5[n]);
}
generated quantities {
  vector[P] phi_sim;
  vector[P] p_sim;

  for(p in 1:P) {
    phi_sim[p] = alpha + alpha_e[p_e[p]] + 
    beta_1e[p_e[p]] * p_i_1[p] + beta_1 * p_i_1[p] +
    beta_2e[p_e[p]] * p_i_2[p] + beta_2 * p_i_2[p] + 
    beta_3e[p_e[p]] * p_i_3[p] + beta_3 * p_i_3[p] + 
    beta_4e[p_e[p]] * p_i_4[p] + beta_4 * p_i_4[p];

    p_sim[p] = inv_logit(phi_sim[p] +
    gamma_1 * p_k_1[p] + 
    gamma_2 * p_k_2[p] + 
    gamma_3 * p_k_3[p] + 
    gamma_4 * p_k_4[p] + 
    gamma_5 * p_k_5[p]);
  }
}
"""

# Splitting data for different models
data1 = data.loc[data['cluster'] == 1, :]
data2 = data.loc[data['cluster'] == 2, :]
data_in = data.loc[data['tt_split'] == 0, :]
data_out = data.loc[data['tt_split'] == 1, :]
data_eval_t1 = data.loc[(data['year_initiation_policy_version'] < 2023) &
                        (data['years_since_policy_started'] <= 1), :]
data_eval_t2 = data.loc[(data['year_initiation_policy_version'] < 2023) &
                        (data['years_since_policy_started'] <= 2), :]
data_pred_t1 = data.loc[(data['year_initiation_policy_version'] == 2023) &
                        (data['years_since_policy_started'] == 1), :]
data_pred_t2 = data.loc[(data['year_initiation_policy_version'] == 2023) &
                        (data['years_since_policy_started'] == 2), :]

# Generating data dictionaries to be understood by STAN
full = bayes_data(data, data)
cluster1 = bayes_data(data1, data1)
cluster2 = bayes_data(data2, data2)
predictive_tt = bayes_data(data_in, data_out)
predictive_t1 = bayes_data(data_eval_t1, data_pred_t1)
predictive_t2 = bayes_data(data_eval_t2, data_pred_t2)


# Setting parameters of HMC
hyperparams = {'num_warmup': 250, 'num_samples': 750, 'num_chains': 16,
               'delta': 0.8, 'stepsize_jitter': 0.1, 'refresh': 250}

# Run models and store core outputs in csv and txt files
print('Full model (in-sample), weak priors:')
logit_weak_full = stan.build(general_model_weak, full, random_seed=seed)
fit_weak_full = logit_weak_full.sample(**hyperparams)
output_results(fit_weak_full, data, 'weak_full', give_draws=True)

print('Full model (train/test split), weak priors:')
logit_weak_predictive_tt = stan.build(general_model_weak, predictive_tt, random_seed=seed)
fit_weak_predictive_tt = logit_weak_predictive_tt.sample(**hyperparams)
output_results(fit_weak_predictive_tt, data_out, 'tt')

print('Full model with cluster dummy (in-sample), weak priors:')
logit_weak_dummy = stan.build(dummy_model_weak, full, random_seed=seed)
fit_weak_dummy = logit_weak_dummy.sample(**hyperparams)
output_results(fit_weak_dummy, data, 'weak_dummy', give_draws=True)

print('Cluster 1 model (in-sample), weak priors:')
logit_weak_cluster1 = stan.build(general_model_weak, cluster1, random_seed=seed)
fit_weak_cluster1 = logit_weak_cluster1.sample(**hyperparams)
output_results(fit_weak_cluster1, data1, 'weak_cluster1', give_draws=True)

print('Cluster 2 model (in-sample), weak priors:')
logit_weak_cluster2 = stan.build(general_model_weak, cluster2, random_seed=seed)
fit_weak_cluster2 = logit_weak_cluster2.sample(**hyperparams)
output_results(fit_weak_cluster2, data2, 'weak_cluster2', give_draws=True)

print('2023 prediction for t=1, weak priors:')
logit_weak_predictive_t1 = stan.build(one_period_model_weak, predictive_t1, random_seed=seed)
fit_weak_predictive_t1 = logit_weak_predictive_t1.sample(**hyperparams)
output_results(fit_weak_predictive_t1, data_pred_t1, 't1')

print('2023 prediction for t=2, weak priors:')
logit_weak_predictive_t2 = stan.build(general_model_weak, predictive_t2, random_seed=seed)
fit_weak_predictive_t2 = logit_weak_predictive_t2.sample(**hyperparams)
output_results(fit_weak_predictive_t2, data_pred_t2, 't2')

f.close()


# Plotting
plt_data = pd.read_csv('output_data_weak_full.csv')
churn_probs = plt_data.loc[:, ['years_since_policy_started', 'eligibility_cat', 'pred_p_churn']]
churn_probs['eligibility_cat'] = churn_probs['eligibility_cat'] - 1
churn_probs0 = churn_probs.loc[churn_probs['years_since_policy_started'] == 1, :]
churn_probs1 = churn_probs.loc[churn_probs['years_since_policy_started'] == 2, :]
churn_probs2 = churn_probs.loc[churn_probs['years_since_policy_started'] == 3, :]

sns.set_theme(style="whitegrid")
p1 = sns.kdeplot(data=churn_probs0, x='pred_p_churn', hue='eligibility_cat',
                 cut=3, bw_adjust=2, fill=True, alpha=0.3, common_norm=False, palette='colorblind')
p1.set_xlim(0, 0.8)
plt.show()

p2 = sns.kdeplot(data=churn_probs1, x='pred_p_churn', hue='eligibility_cat',
                 cut=3, bw_adjust=2, fill=True, alpha=0.3, common_norm=False, palette='colorblind')
p2.set_xlim(0, 0.8)
plt.show()

param_draws = pd.read_csv('param_draws_weak_full.csv')
for param in param_draws:
    sns.kdeplot(data=param_draws, x=param, cut=2, bw_adjust=3)
    plt.show()
