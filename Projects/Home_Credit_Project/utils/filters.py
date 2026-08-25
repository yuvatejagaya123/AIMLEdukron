import streamlit as st

def sidebar_filters(df):

    st.sidebar.header("Filters")

    if "AGE" not in df.columns:
        df["AGE"] = (abs(df["DAYS_BIRTH"]) / 365).round(0)
    age_min = int(df["AGE"].min())
    age_max = int(df["AGE"].max())

    age_range = st.sidebar.slider(
        "Age Range",
        age_min,
        age_max,
        (age_min, age_max))    
    
    target = st.sidebar.multiselect("Target",df["TARGET"].unique(),default=df["TARGET"].unique())
    gender = st.sidebar.multiselect("Gender",df["CODE_GENDER"].dropna().unique(),default=df["CODE_GENDER"].dropna().unique())
    education = st.sidebar.multiselect("Education",df["NAME_EDUCATION_TYPE"].dropna().unique(),
                                       default = df["NAME_EDUCATION_TYPE"].dropna().unique())
    contract = st.sidebar.multiselect("Contract Type",df["NAME_CONTRACT_TYPE"].dropna().unique(),
                                      default=df["NAME_CONTRACT_TYPE"].dropna().unique())    
    
    filtered_df = df[(df["TARGET"].isin(target))&
                     (df["CODE_GENDER"].isin(gender))&
                     (df["NAME_EDUCATION_TYPE"].isin(education))&
                     (df["NAME_CONTRACT_TYPE"].isin(contract))&
                     (df["AGE"].between(age_range[0],age_range[1]))]
    
    return filtered_df

## Usage
# from utils.filters import sidebar_filters

# filtered_df = sidebar_filters(df)