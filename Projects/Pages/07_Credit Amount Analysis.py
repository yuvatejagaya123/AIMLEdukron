# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features

from utils.charts import (
    histogram,
    bar_chart
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Credit Amount Analysis",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Amount Analysis")

st.markdown("""
Analyze the amount of credit requested by applicants
and understand its relationship with default behavior.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# CREDIT GROUPS
# =====================================================

df["CREDIT_GROUP"] = pd.cut(
    df["AMT_CREDIT"],
    bins=[
        0,
        100000,
        300000,
        500000,
        700000,
        1000000,
        float("inf")
    ],
    labels=[
        "Below 100K",
        "100K-300K",
        "300K-500K",
        "500K-700K",
        "700K-1M",
        "Above 1M"
    ]
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Credit Analysis Filters")

credit_group_filter = st.sidebar.multiselect(
    "Credit Group",
    options=df["CREDIT_GROUP"].dropna().unique(),
    default=df["CREDIT_GROUP"].dropna().unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["CODE_GENDER"].dropna().unique(),
    default=df["CODE_GENDER"].dropna().unique()
)

education_filter = st.sidebar.multiselect(
    "Education",
    options=df["NAME_EDUCATION_TYPE"].dropna().unique(),
    default=df["NAME_EDUCATION_TYPE"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["CREDIT_GROUP"].isin(credit_group_filter))
    &
    (df["CODE_GENDER"].isin(gender_filter))
    &
    (df["NAME_EDUCATION_TYPE"].isin(education_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CARDS
# =====================================================

total_credit = filtered_df["AMT_CREDIT"].sum()

average_credit = filtered_df["AMT_CREDIT"].mean()

median_credit = filtered_df["AMT_CREDIT"].median()

maximum_credit = filtered_df["AMT_CREDIT"].max()

minimum_credit = filtered_df["AMT_CREDIT"].min()

st.subheader("📌 KPI Cards")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Credit", f"{total_credit:,.0f}")
col2.metric("Average Credit", f"{average_credit:,.0f}")
col3.metric("Median Credit", f"{median_credit:,.0f}")
col4.metric("Maximum Credit", f"{maximum_credit:,.0f}")
col5.metric("Minimum Credit", f"{minimum_credit:,.0f}")

st.divider()

# =====================================================
# CREDIT DISTRIBUTION
# =====================================================

st.subheader("📊 Credit Amount Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "AMT_CREDIT",
        "Credit Amount Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT AMOUNT BY TARGET
# =====================================================

credit_target_df = (
    filtered_df.groupby("TARGET")["AMT_CREDIT"]
    .mean()
    .reset_index()
)

fig_credit_target = px.bar(
    credit_target_df,
    x="TARGET",
    y="AMT_CREDIT",
    title="Credit Amount by TARGET",
    text="AMT_CREDIT"
)

st.plotly_chart(
    fig_credit_target,
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT BY GENDER / INCOME TYPE
# =====================================================

col1, col2 = st.columns(2)

with col1:

    gender_credit_df = (
        filtered_df.groupby("CODE_GENDER")
        ["AMT_CREDIT"]
        .mean()
        .reset_index()
    )

    fig_gender_credit = px.bar(
        gender_credit_df,
        x="CODE_GENDER",
        y="AMT_CREDIT",
        title="Average Credit by Gender"
    )

    st.plotly_chart(
        fig_gender_credit,
        use_container_width=True
    )

with col2:

    income_credit_df = (
        filtered_df.groupby("NAME_INCOME_TYPE")
        ["AMT_CREDIT"]
        .mean()
        .reset_index()
    )

    fig_income_credit = px.bar(
        income_credit_df,
        x="NAME_INCOME_TYPE",
        y="AMT_CREDIT",
        title="Credit by Income Type"
    )

    st.plotly_chart(
        fig_income_credit,
        use_container_width=True
    )

st.divider()

# =====================================================
# CREDIT BY EDUCATION / CONTRACT TYPE
# =====================================================

col1, col2 = st.columns(2)

with col1:

    education_credit_df = (
        filtered_df.groupby("NAME_EDUCATION_TYPE")
        ["AMT_CREDIT"]
        .mean()
        .reset_index()
    )

    fig_education_credit = px.bar(
        education_credit_df,
        x="NAME_EDUCATION_TYPE",
        y="AMT_CREDIT",
        title="Credit by Education"
    )

    st.plotly_chart(
        fig_education_credit,
        use_container_width=True
    )

with col2:

    contract_credit_df = (
        filtered_df.groupby("NAME_CONTRACT_TYPE")
        ["AMT_CREDIT"]
        .mean()
        .reset_index()
    )

    fig_contract_credit = px.bar(
        contract_credit_df,
        x="NAME_CONTRACT_TYPE",
        y="AMT_CREDIT",
        title="Credit by Contract Type"
    )

    st.plotly_chart(
        fig_contract_credit,
        use_container_width=True
    )

st.divider()

# =====================================================
# DEFAULT RATE BY CREDIT RANGE
# =====================================================

credit_risk_df = (
    filtered_df.groupby(
        "CREDIT_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .reset_index()
)

credit_risk_df["Default Rate %"] = (
    credit_risk_df["TARGET"] * 100
)

fig_credit_risk = px.bar(
    credit_risk_df,
    x="CREDIT_GROUP",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Credit Range"
)

fig_credit_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_credit_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT INSIGHTS
# =====================================================

highest_risk_credit_group = (
    credit_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["CREDIT_GROUP"]
)

st.subheader("💡 Credit Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Total Credit: {total_credit:,.0f}

✅ Average Credit: {average_credit:,.0f}

✅ Median Credit: {median_credit:,.0f}

"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Credit Group: {highest_risk_credit_group}


⚠ Higher credit amounts may be associated with higher repayment risk.
"""
    )
