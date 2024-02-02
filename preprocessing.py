def data_modification(df):
    # Remove mutations with Nan values
    df_dropped = df.drop(["mutation_1","premium_change_mutation_1","mutation_2","premium_change_mutation_2","mutation_3","premium_change_mutation_3",
       "mutation_4","premium_change_mutation_4","mutation_5","premium_change_mutation_5","mutation_6","premium_change_mutation_6",
       "mutation_7","premium_change_mutation_7","mutation_8","premium_change_mutation_8","mutation_9","premium_change_mutation_9",
       "mutation_10","premium_change_mutation_10","mutation_11","premium_change_mutation_11","mutation_12",
       "premium_change_mutation_12"], axis = 1, inplace = False)

    # Change NaN values of mutations to zero's
    df_mutation = df[["mutation_1","premium_change_mutation_1","mutation_2","premium_change_mutation_2","mutation_3","premium_change_mutation_3",
       "mutation_4","premium_change_mutation_4","mutation_5","premium_change_mutation_5","mutation_6","premium_change_mutation_6",
       "mutation_7","premium_change_mutation_7","mutation_8","premium_change_mutation_8","mutation_9","premium_change_mutation_9",
       "mutation_10","premium_change_mutation_10","mutation_11","premium_change_mutation_11","mutation_12",
       "premium_change_mutation_12"]].fillna(0)

    # Add back the non-NaN valued mutations
    merged_df = pd.concat([df_dropped, df_mutation], axis = 1)
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

    # Some of the postcode values are between quotations (e.g 2045 is written as '2045'). We want to remove this so 2045 and '2045'
    # are treated as the same postcode
    df['postcode'] = df['postcode'].str.replace(" ' ", '')

    # Label encode postcode as well
    df['postcode_label'] = label_encoder.fit_transform(df['postcode'])

    # Change NaN values of year end policy to zero. Note: this variable is might not usable in general for ML/ clustering since 
    # 0 now will be associated with not churning. This gives the variable incorrect predictive power.
    df["year_end_policy"] = df["year_end_policy"].fillna(0)

    return df

data_modification(df)
