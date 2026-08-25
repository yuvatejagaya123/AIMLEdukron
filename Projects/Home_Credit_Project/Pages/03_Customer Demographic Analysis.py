import streamlit as st
import pandas as pd

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features

from utils.charts import (
    pie_chart,
    bar_chart
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Demographic Analysis",
    page_icon="👥",
    layout="wide"
)


# =====================================================
# PAGE HEADER
# =====================================================

st.title("👥 Customer Demographic Analysis")

st.markdown(
    """
    This page helps understand the demographic characteristics of Home Credit applicants,
    including gender, age group, family status, education level, housing type,
    and demographic default risk patterns.
    """
)


# =====================================================
# LOAD DATA
# =====================================================

try:
    df = load_data("Data/application_train.csv")

    df = clean_data(df)

    df = create_features(df)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# =====================================================
# CREATE AGE GROUP
# =====================================================

df["AGE_GROUP"] = pd.cut(
    df["AGE"],
    bins=[18, 25, 35, 45, 55, 65, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "65+"
    ]
)


# =====================================================
# PAGE SPECIFIC FILTERS
# =====================================================

st.sidebar.header("Demographic Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["CODE_GENDER"].dropna().unique(),
    default=df["CODE_GENDER"].dropna().unique()
)

age_filter = st.sidebar.multiselect(
    "Age Group",
    options=df["AGE_GROUP"].dropna().unique(),
    default=df["AGE_GROUP"].dropna().unique()
)

family_filter = st.sidebar.multiselect(
    "Family Status",
    options=df["NAME_FAMILY_STATUS"].dropna().unique(),
    default=df["NAME_FAMILY_STATUS"].dropna().unique()
)

education_filter = st.sidebar.multiselect(
    "Education",
    options=df["NAME_EDUCATION_TYPE"].dropna().unique(),
    default=df["NAME_EDUCATION_TYPE"].dropna().unique()
)

housing_filter = st.sidebar.multiselect(
    "Housing Type",
    options=df["NAME_HOUSING_TYPE"].dropna().unique(),
    default=df["NAME_HOUSING_TYPE"].dropna().unique()
)


# =====================================================
# APPLY FILTERS
# =====================================================

filtered_df = df[
    (df["CODE_GENDER"].isin(gender_filter))
    & (df["AGE_GROUP"].isin(age_filter))
    & (df["NAME_FAMILY_STATUS"].isin(family_filter))
    & (df["NAME_EDUCATION_TYPE"].isin(education_filter))
    & (df["NAME_HOUSING_TYPE"].isin(housing_filter))
]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()


# =====================================================
# KPI CALCULATIONS
# =====================================================

total_customers = len(filtered_df)

average_age = filtered_df["AGE"].mean()

male_customers = (
    filtered_df["CODE_GENDER"] == "M"
).sum()

female_customers = (
    filtered_df["CODE_GENDER"] == "F"
).sum()

average_family_size = filtered_df["CNT_FAM_MEMBERS"].mean()


# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Average Age",
    f"{average_age:.0f}"
)

col3.metric(
    "Male Customers",
    f"{male_customers:,}"
)

col4.metric(
    "Female Customers",
    f"{female_customers:,}"
)

col5.metric(
    "Average Family Size",
    f"{average_family_size:.1f}"
)

st.divider()


# =====================================================
# CUSTOMERS BY GENDER AND AGE GROUP
# =====================================================

st.subheader("👤 Customer Distribution by Gender and Age")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        pie_chart(
            filtered_df,
            "CODE_GENDER",
            "Customers by Gender"
        ),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        bar_chart(
            filtered_df,
            "AGE_GROUP",
            None,
            "Customers by Age Group"
        ),
        use_container_width=True
    )

st.divider()


# =====================================================
# CUSTOMERS BY FAMILY STATUS AND EDUCATION
# =====================================================

st.subheader("👨‍👩‍👧 Family and Education Profile")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        bar_chart(
            filtered_df,
            "NAME_FAMILY_STATUS",
            None,
            "Customers by Family Status"
        ),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        bar_chart(
            filtered_df,
            "NAME_EDUCATION_TYPE",
            None,
            "Customers by Education"
        ),
        use_container_width=True
    )

st.divider()


# =====================================================
# CUSTOMERS BY HOUSING TYPE
# =====================================================

st.subheader("🏠 Housing Type Distribution")

st.plotly_chart(
    bar_chart(
        filtered_df,
        "NAME_HOUSING_TYPE",
        None,
        "Customers by Housing Type"
    ),
    use_container_width=True
)

st.divider()


# =====================================================
# DEFAULT RATE BY DEMOGRAPHIC GROUP
# =====================================================

st.subheader("⚠ Default Rate by Demographic Group")


# -------------------------------
# Default Rate by Age Group
# -------------------------------

age_default_df = (
    filtered_df
    .groupby("AGE_GROUP")["TARGET"]
    .mean()
    .reset_index()
)

age_default_df["Default Rate %"] = (
    age_default_df["TARGET"] * 100
)


# -------------------------------
# Default Rate by Family Status
# -------------------------------

family_default_df = (
    filtered_df
    .groupby("NAME_FAMILY_STATUS")["TARGET"]
    .mean()
    .reset_index()
)

family_default_df["Default Rate %"] = (
    family_default_df["TARGET"] * 100
)


col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        bar_chart(
            age_default_df,
            "AGE_GROUP",
            "Default Rate %",
            "Default Rate by Age Group"
        ),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        bar_chart(
            family_default_df,
            "NAME_FAMILY_STATUS",
            "Default Rate %",
            "Default Rate by Family Status"
        ),
        use_container_width=True
    )

st.divider()


# =====================================================
# ADDITIONAL DEFAULT RATE BY EDUCATION AND HOUSING
# =====================================================

st.subheader("📊 Default Rate by Education and Housing")


# -------------------------------
# Default Rate by Education
# -------------------------------

education_default_df = (
    filtered_df
    .groupby("NAME_EDUCATION_TYPE")["TARGET"]
    .mean()
    .reset_index()
)

education_default_df["Default Rate %"] = (
    education_default_df["TARGET"] * 100
)


# -------------------------------
# Default Rate by Housing Type
# -------------------------------

housing_default_df = (
    filtered_df
    .groupby("NAME_HOUSING_TYPE")["TARGET"]
    .mean()
    .reset_index()
)

housing_default_df["Default Rate %"] = (
    housing_default_df["TARGET"] * 100
)


col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        bar_chart(
            education_default_df,
            "NAME_EDUCATION_TYPE",
            "Default Rate %",
            "Default Rate by Education"
        ),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        bar_chart(
            housing_default_df,
            "NAME_HOUSING_TYPE",
            "Default Rate %",
            "Default Rate by Housing Type"
        ),
        use_container_width=True
    )

st.divider()


# =====================================================
# DEMOGRAPHIC INSIGHTS
# =====================================================

st.subheader("💡 Demographic Insights")


def get_mode_value(dataframe, column_name):
    if column_name in dataframe.columns and not dataframe[column_name].dropna().empty:
        return dataframe[column_name].mode()[0]
    return "Not Available"


def get_highest_risk_group(dataframe, group_col):
    risk_table = (
        dataframe
        .groupby(group_col)["TARGET"]
        .mean()
        .dropna()
    )

    if not risk_table.empty:
        group_name = risk_table.idxmax()
        risk_rate = risk_table.max() * 100
        return group_name, risk_rate

    return "Not Available", 0


most_common_age_group = get_mode_value(
    filtered_df,
    "AGE_GROUP"
)

most_common_family_status = get_mode_value(
    filtered_df,
    "NAME_FAMILY_STATUS"
)

most_common_education = get_mode_value(
    filtered_df,
    "NAME_EDUCATION_TYPE"
)

most_common_housing = get_mode_value(
    filtered_df,
    "NAME_HOUSING_TYPE"
)

highest_risk_age_group, highest_risk_age_rate = get_highest_risk_group(
    filtered_df,
    "AGE_GROUP"
)

highest_risk_family_status, highest_risk_family_rate = get_highest_risk_group(
    filtered_df,
    "NAME_FAMILY_STATUS"
)


col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
### Customer Profile Summary

- Most common age group: **{most_common_age_group}**
- Most common family status: **{most_common_family_status}**
- Most common education level: **{most_common_education}**
- Most common housing type: **{most_common_housing}**
"""
    )

with col2:
    st.warning(
        f"""
### Demographic Risk Summary

- Highest risk age group: **{highest_risk_age_group}**
- Age group default rate: **{highest_risk_age_rate:.2f}%**
- Highest risk family status: **{highest_risk_family_status}**
- Family status default rate: **{highest_risk_family_rate:.2f}%**
"""
    )


# =====================================================
# FILTERED DATA PREVIEW
# =====================================================

with st.expander("🔍 View Filtered Demographic Data"):

    display_columns = [
        "SK_ID_CURR",
        "TARGET",
        "CODE_GENDER",
        "AGE",
        "AGE_GROUP",
        "CNT_FAM_MEMBERS",
        "NAME_FAMILY_STATUS",
        "NAME_EDUCATION_TYPE",
        "NAME_HOUSING_TYPE",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT"
    ]

    available_columns = [
        col for col in display_columns
        if col in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[available_columns].head(100),
        use_container_width=True
    )
