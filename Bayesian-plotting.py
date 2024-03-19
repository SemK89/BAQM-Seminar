import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

data = pd.read_csv('output_data_weak_full.csv')
churn_probs = data.loc[:, ['years_since_policy_started','eligibility_cat','pred_p_churn']]
churn_probs['eligibility_cat'] = churn_probs['eligibility_cat'] - 1

churn_probs0 = churn_probs.loc[churn_probs['years_since_policy_started'] == 1, :]
churn_probs1 = churn_probs.loc[churn_probs['years_since_policy_started'] == 2, :]
churn_probs2 = churn_probs.loc[churn_probs['years_since_policy_started'] == 3, :]

plt.xlim=([0, 1])
sns.kdeplot(data=churn_probs, x='pred_p_churn', hue='eligibility_cat',
            cut=0, fill=True, alpha=0.3, common_norm=False, palette='colorblind')
plt.show()
sns.kdeplot(data=churn_probs0, x='pred_p_churn', hue='eligibility_cat',
            cut=0, fill=True, alpha=0.3, common_norm=False, palette='colorblind')
plt.show()
sns.kdeplot(data=churn_probs1, x='pred_p_churn', hue='eligibility_cat',
            cut=0, fill=True, alpha=0.3, common_norm=False, palette='colorblind')
plt.show()
sns.kdeplot(data=churn_probs2, x='pred_p_churn', hue='eligibility_cat',
            cut=0, fill=True, alpha=0.3, common_norm=False, palette='colorblind')

plt.show()


param_draws = pd.read_csv('param_draws_weak_full.csv')
for param in param_draws:
    sns.kdeplot(data=param_draws, x=param)
    plt.show()
