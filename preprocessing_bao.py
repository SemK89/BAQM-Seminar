import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.discriminant_analysis import StandardScaler

# for my part of preprocessing i used following code
# data = prep.drop_older_policies(merged_df, 2018)
# data = prep.minor_edits(data)
"""
data.drop(["mutation_1", "mutation_2", "mutation_3", "mutation_4", "mutation_5", "mutation_6",
           "mutation_7", "mutation_8", "mutation_9", "mutation_10", "mutation_11", "mutation_12",
           "data_collection_date", "product", "n_coverages", 'allrisk basis',
           'allrisk compleet', 'allrisk royaal', 'wa-extra','wettelijke aansprakelijkheid'],
          axis=1, inplace=True)

data = prep.sum_cols(data, [
    "premium_change_mutation_1", "premium_change_mutation_2",
    "premium_change_mutation_3", "premium_change_mutation_4",
    "premium_change_mutation_5", "premium_change_mutation_6",
    "premium_change_mutation_7", "premium_change_mutation_8",
    "premium_change_mutation_9", "premium_change_mutation_10",
    "premium_change_mutation_11", "premium_change_mutation_12"], 'premium_mutations')
    
data = prep.add_treatment_vars(data)

based on this to remove zero years use the remove_zero_year (df)


"""

 
def drop_older_policies(df, year):
    return (df[df['year_initiation_policy'] >= year]
            .sort_values(by=['policy_nr_hashed', 'year_initiation_policy', 'year_initiation_policy_version']))

def sum_cols(df, cols_to_sum, new_col_name):
    df[new_col_name] = df[cols_to_sum].sum(axis=1, min_count=1).fillna(0)
    df.drop(cols_to_sum, axis=1, inplace=True)
    return df

def minor_edits(df):
    # Apparently some rows are exactly identical, Luca said it was safe to drop these
    df.drop_duplicates()

    # Some of the postcode values are between quotations (e.g. 2045 is written as '2045').
    # We want to remove this so 2045 and '2045' are treated as the same postcode.
    df['postcode'] = df['postcode'].str.replace(" ' ", '')

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
    #df = df[~((df['total_premium'] == 0) & (df['d_churn'] == 1) & (df['year_initiation_policy_version'] == 2019))]
    # I was not sure why only 2019

    return df

def add_treatment_vars(df):
    df['WD_eligible'] = 1
    df.loc[df['welcome_discount_control_group'].str.contains('no WD'), 'WD_eligible'] = 0
    df['LPA_eligible'] = 1
    df.loc[df['welcome_discount_control_group'].str.contains('no LPA'), 'LPA_eligible'] = 0

    df_cs = df.copy().groupby('policy_nr_hashed')['welcome_discount'].min()
    df_cs.rename({'welcome_discount': 'WD_level'})
    df.merge(df_cs, how='left', on='policy_nr_hashed')
    unique_values = df[df['welcome_discount'] < 1]['policy_nr_hashed'].unique()

    # Group the dataset based on those unique values
    df['WD_receive_group'] = df['policy_nr_hashed'].isin(unique_values).astype(int)
    
    return df

def remove_zero_year (df):
    df['years_since_policy_started'] = df['years_since_policy_started'].replace(0, 1)

    # To select the data that they are wrong, we check the duplicates after replacement 
    columns_to_check = ['policy_nr_hashed', 'years_since_policy_started']
    
    # removw the ones show the first (the orginal zeros)
    df = df.loc[~df.duplicated(subset=columns_to_check, keep='last')]
    
    return df
    

def convert_person_period(df, remove_columns):
    
    df = df.copy()
    df = remove_zero_year(df)
    
    df_person_period = df.drop(remove_columns, axis = 1)
    
    # prepare for generating dummies
    df_person_period['years_since_policy_started'] = df_person_period['years_since_policy_started'].astype('category')
    df_person_period['cluster'] = df_person_period['cluster'].astype('category')
    
    # Get dummies without specifying prefix
    df_dummies = pd.get_dummies(df_person_period[['years_since_policy_started', 'cluster']])
    
    # Concatenate the dummy variables with the original DataFrame
    df_person_period = pd.concat([df_person_period, df_dummies], axis=1)

    # Drop the original categorical columns for identification
    df_person_period = df_person_period.drop(['years_since_policy_started' , 'cluster'], axis=1)
  
    # replace all boolin columns with dummies
    bool_columns = df_person_period.select_dtypes(include='bool').columns
    df_person_period[bool_columns] = df_person_period[bool_columns].astype(int)

    return df_person_period

def survival_analysis_spe1(df, working_columns):
    
    # Ensure working_columns is a list to avoid indexing issues
    if isinstance(working_columns, str):
        working_columns = [working_columns]
    
    # Make a copy of the DataFrame to avoid modifying the original
    df_w_1 = df.copy()
    
    # Create new columns in the copy based on 'WD_receive_group'
    for col in working_columns:
        wd_col = 'WD' + col
        # Create modified column directly in df_w_1
        df_w_1[wd_col] = df_w_1[col] * df_w_1['WD_receive_group']
        # Modify original column in-place
        df_w_1[col] = df_w_1[col] * (1 - df_w_1['WD_receive_group'])
        
    # No need to merge df_w_1_c since we directly modify df_w_1
    
    return df_w_1


def preprocessing_clustering(data):
    
    df_all = drop_older_policies(data, 2018)
    df_edit = minor_edits(df_all)
    df = df_edit.drop_duplicates(subset='policy_nr_hashed', keep='first')
    
    df.drop(["mutation_1", "mutation_2", "mutation_3", "mutation_4", "mutation_5", "mutation_6",
           "mutation_7", "mutation_8", "mutation_9", "mutation_10", "mutation_11", "mutation_12",
           "data_collection_date", "n_coverages", 'allrisk basis',
           'allrisk compleet', 'allrisk royaal', 'wa-extra','wettelijke aansprakelijkheid'],
          axis=1, inplace=True)
    
    indexes_to_drop = list(range(1, 13))

    # Convert column indexes to column names
    columns_to_drop = df.columns[indexes_to_drop]

    # Drop the columns
    df_dropped = df.drop(columns=columns_to_drop)
    
    df_kept = df_dropped.iloc[:, :-12]
    
    indexes_nur = [2,3,4,5,8,12,13]

    # Convert column indexes to column names
    columns_nur = df_kept.columns[indexes_nur]
    # df[columns_nur].dtypes
    scaler = StandardScaler()
    df_scaled_numerical = scaler.fit_transform(df_kept[columns_nur])
    df_kept[columns_nur] = df_scaled_numerical
    
    df_clus = df_kept.drop('policy_nr_hashed',axis=1)

    # combine the postcode with the other data set and fill na, now just for easy fill zero
    df_clus['postcode'] = df_clus['postcode'].fillna("0000")
    
    return df_clus