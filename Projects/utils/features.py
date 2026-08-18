import numpy as np


def create_features(df):

    #Age
    df["AGE"] = (abs(df["DAYS_BIRTH"])/365).round(0)

    #Employment Years
    df["EMPLOYMENT_YEARS"] = abs(df["DAYS_EMPLOYED"])/365

    #Credit Income Ratio
    df["CREDIT_INCOME_RATIO"] = (df["AMT_CREDIT"]/df["AMT_INCOME_TOTAL"])

    # Annuity Income Ratio
    df["ANNUITY_INCOME_RATIO"] = (df["AMT_ANNUITY"]/df["AMT_INCOME_TOTAL"])

    # Credit Goods Ratio
    df["CREDIT_GOODS_RATIO"] = (df["AMT_CREDIT"]/df["AMT_GOODS_PRICE"])

    # Average External Score
    
    ext_cols = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]
    print(df.columns[df.columns.str.contains("EXT")])
    df["AVG_EXT_SCORE"] = df[ext_cols].mean(axis=1)

    return df


# usage

# from utils.features import create_features

# df = create_features(df)

