import pandas as pd
import preprocessing as prep

data = pd.read_csv('20240117_churn_data.csv')

data = prep.drop_older_policies(data, 2021)
data = prep.remove_vars(data, ["premium_change_mutation_1", "premium_change_mutation_2", 
                        "premium_change_mutation_3", "premium_change_mutation_4", 
                        "premium_change_mutation_5", "premium_change_mutation_6", 
                        "premium_change_mutation_7", "premium_change_mutation_8", 
                        "premium_change_mutation_9", "premium_change_mutation_10", 
                        "premium_change_mutation_11", "premium_change_mutation_12", 
                        "data_collection_date"])
data = prep.sum_cols(data, ["mutation_1", "mutation_2", "mutation_3", "mutation_4",
                     "mutation_5", "mutation_6", "mutation_7", "mutation_8", 
                     "mutation_9", "mutation_10", "mutation_11", "mutation_12"],
              'mutation')

print(data.describe())




