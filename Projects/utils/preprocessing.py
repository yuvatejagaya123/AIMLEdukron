import numpy as np

def clean_data(df):
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243,np.nan)
    return df

## Usage
###from utils.preprocessing import clean_data
###df = clean_data(df)