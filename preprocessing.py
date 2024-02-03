import pandas as pd
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
    df['postcode_label'] = label_encoder.fit_transform(df['postcode'])

    # Change NaN values of year-end policy to zero.
    # Note: this variable is might not usable in general for ML/ clustering
    # since 0 now will be associated with not churning.
    # This gives the variable incorrect predictive power.
    df["year_end_policy"] = df["year_end_policy"].fillna(0)

    return df


def drop_older_policies(df, year):
    return (df[df['year_initiation_policy'] >= year]
            .sort_values(by=['year_initiation_policy', 'policy_nr_hashed', 'year_initiation_policy_version']))


def sum_cols(df, cols_to_sum, new_col_name):
    df[new_col_name] = df[cols_to_sum].sum(axis=1, min_count=1).fillna(0)
    df.drop(cols_to_sum, axis=1, inplace=True)
    return df


def minor_edits(df):
    df.drop_duplicates()
    # Some of the postcode values are between quotations (e.g. 2045 is written as '2045').
    # We want to remove this so 2045 and '2045' are treated as the same postcode.
    df['postcode'] = df['postcode'].str.replace(" ' ", '')

    # Make d_churn=1 for a few cases where it is zero even though the year_end_policy clearly states they churn
    df.loc[df['year_initiation_policy_version'] == df['year_end_policy'], 'd_churn'] = 1

    # WD set to 1 if it is not the first year of the policy
    df.loc[df['years_since_policy_started'] != 0, 'welcome_discount'] = 1

    # Reduce accident-free years by 5 if it is mentioned in the mutation, changeNcbmData
    # rows added by Luca have nan values for premium_main_coverage and premium_supplementary_coverage
    df.loc[(df['premium_main_coverage'].isnull()) and
           ((df['mutation_1'] == 'changeNcbmData') or (df['mutation_2'] == 'changeNcbmData') or
            (df['mutation_3'] == 'changeNcbmData') or (df['mutation_4'] == 'changeNcbmData') or
            (df['mutation_5'] == 'changeNcbmData') or (df['mutation_6'] == 'changeNcbmData') or
            (df['mutation_7'] == 'changeNcbmData') or (df['mutation_8'] == 'changeNcbmData') or
            (df['mutation_9'] == 'changeNcbmData') or (df['mutation_10'] == 'changeNcbmData') or
            (df['mutation_11'] == 'changeNcbmData') or
            (df['mutation_12'] == 'changeNcbmData')), 'accident_free_years'] -= 5

    # drop rows with churn in 2019 but 0 premium
    df = df[not ((df['total_premium'] == 0) and (df['d_churn'] == 1) and (df['year_policy_version'] == 2019))]

    return df
