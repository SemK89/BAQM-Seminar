import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('20240117_churn_data.csv')
print(dataset.describe())


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


def drop_new_policies(df):
    return df[df['year_initiation_policy_version'] != 2024]


def sum_cols(df, cols_to_sum, new_col_name):
    df[new_col_name] = df[cols_to_sum].sum(axis=1, min_count=1).fillna(0)
    df.drop(cols_to_sum, axis=1, inplace=True)
    return df


def add_treatment_vars(df):
    df['WD_eligible'] = 1
    df.loc[df['welcome_discount_control_group'].str.contains('no WD'), 'WD_eligible'] = 0
    df['LPA_eligible'] = 1
    df.loc[df['welcome_discount_control_group'].str.contains('no LPA'), 'LPA_eligible'] = 0
    df['eligibility_cat'] = (1 + df['WD_eligible'] + 2*df['LPA_eligible']).astype(int)

    df_cs = df.groupby('policy_nr_hashed').agg(WD_level=('welcome_discount', 'min'))
    df = df.merge(df_cs, how='left', on='policy_nr_hashed')

    df['WD_received'] = 0
    df.loc[df['WD_level'] < 1, 'WD_received'] = 1

    return df


def shorten_postal_code(df, digits):
    for i in range(len(df['postcode'])):
        code = df.iloc[i, 10]
        try:
            df.iloc[i, 10] = int(float(code) // (10**(4-digits)))
        except ValueError:
            df.iloc[i, 10] = int(0)

    return df


def minor_edits(df):
    # Apparently some rows are exactly identical, Luca said it was safe to drop these
    df = df.drop_duplicates(subset=['policy_nr_hashed', 'years_since_policy_started'], keep='last')

    # Recompute time periods as some seem to contain errors, and make them start as period 1 instead of 0.
    df['years_since_policy_started'] = (df['year_initiation_policy_version'] - df['year_initiation_policy'] + 1).astype(int)
    df['year_initiation_policy_version'] = df['year_initiation_policy_version'].astype(int)

    # Removes just over 200 datapoints that ended their policy in 2018 but still show up as a 2019 policy.
    df = df[df['year_end_policy'] != 2018]

    # Some of the postcode values are between quotations (e.g. 2045 is written as '2045').
    # We want to remove this so 2045 and '2045' are treated as the same postcode.
    # Following line is commented because it was causing issues.
    # df['postcode'] = df['postcode'].str.replace("'", '')

    # Make d_churn=1 for a few cases where it is zero even though the year_end_policy clearly states they churn
    df.loc[df['year_initiation_policy_version'] == df['year_end_policy'], 'd_churn'] = 1

    # WD set to 1 if it is not the first year of the policy
    df.loc[df['years_since_policy_started'] != 1, 'welcome_discount'] = 1

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


# this is used in the survival analysis, the orginal version
def minor_edits_for_survival(df):
    # Apparently some rows are exactly identical, Luca said it was safe to drop these
    df.drop_duplicates()

    # Some of the postcode values are between quotations (e.g. 2045 is written as '2045').
    # We want to remove this so 2045 and '2045' are treated as the same postcode.
    df['postcode'] = df['postcode'].astype(str)

    # Remove quotations if present
    df['postcode'] = df['postcode'].str.replace("'", "")

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
    # I was not sure why only 2019

    return df



def convert_person_period(df, remove_columns):
    
    df = df.copy()
    #df = remove_zero_year(df)
    
    df_person_period = df.drop(remove_columns, axis = 1)
    
    # prepare for generating dummies
    df_person_period['years_since_policy_started'] = df_person_period['years_since_policy_started'].astype('category')
    df_person_period['cluster'] = df_person_period['cluster'].astype('category')
    
    # Get dummies without specifying prefix
    df_dummies = pd.get_dummies(df_person_period[['years_since_policy_started', 'cluster', 'fuel_type', 'sales_channel']])
    
    # Concatenate the dummy variables with the original DataFrame
    df_person_period = pd.concat([df_person_period, df_dummies], axis=1)

    # Drop the original categorical columns for identification
    df_person_period = df_person_period.drop(['years_since_policy_started' , 'cluster', 'fuel_type', 'sales_channel'], axis=1)
  
    # replace all boolin columns with dummies
    bool_columns = df_person_period.select_dtypes(include='bool').columns
    df_person_period[bool_columns] = df_person_period[bool_columns].astype(int)

    return df_person_period


def survival_analysis_spe1(df, working_columns):


    df_w_1 = df.copy()
    wd_received = df_w_1['WD_received']
    # Create new columns in the copy based on 'WD_receive_group'
    for col in working_columns:
        wd_col = 'WD_' + col
        # Create modified column directly in df_w_1
        df_w_1[wd_col] = df_w_1[col] * wd_received
        df_w_1[col] = df_w_1[col] * (1 - wd_received)
        
    # No need to merge df_w_1_c since we directly modify df_w_1
    
    return df_w_1