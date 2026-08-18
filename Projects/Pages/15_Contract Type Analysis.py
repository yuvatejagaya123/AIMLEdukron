import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Contract Type Analysis",
    page_icon="📄",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📄 Contract Type Analysis")

st.markdown("""
Analyze customer behavior and risk based on
loan contract type.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

df = clean_data(df)

df = create_features(df)

# =====================================================
# CREDIT INCOME RATIO
# =====================================================

df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"] /
    df["AMT_INCOME_TOTAL"]
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Contract Analysis Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["NAME_CONTRACT_TYPE"].dropna().unique(),
    default=df["NAME_CONTRACT_TYPE"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["NAME_CONTRACT_TYPE"].isin(contract_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

cash_loan_applications = (
    filtered_df["NAME_CONTRACT_TYPE"]
    == "Cash loans"
).sum()

revolving_loan_applications = (
    filtered_df["NAME_CONTRACT_TYPE"]
    == "Revolving loans"
).sum()

contract_risk_df = (
    filtered_df.groupby(
        "NAME_CONTRACT_TYPE"
    )["TARGET"]
    .mean()
    .reset_index()
)

contract_risk_df["Default Rate %"] = (
    contract_risk_df["TARGET"] * 100
)

cash_loan_default_rate = (
    contract_risk_df[
        contract_risk_df["NAME_CONTRACT_TYPE"]
        == "Cash loans"
    ]["Default Rate %"]
).iloc[0]

revolving_loan_default_rate = (
    contract_risk_df[
        contract_risk_df["NAME_CONTRACT_TYPE"]
        == "Revolving loans"
    ]["Default Rate %"]
).iloc[0]

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Cash Loan Applications",
    f"{cash_loan_applications:,}"
)

col2.metric(
    "Revolving Loan Applications",
    f"{revolving_loan_applications:,}"
)

col3.metric(
    "Cash Loan Default Rate",
    f"{cash_loan_default_rate:.2f}%"
)

col4.metric(
    "Revolving Loan Default Rate",
    f"{revolving_loan_default_rate:.2f}%"
)

st.divider()

# =====================================================
# APPLICATIONS BY CONTRACT TYPE
# =====================================================

contract_count_df = (
    filtered_df["NAME_CONTRACT_TYPE"]
    .value_counts()
    .reset_index()
)

contract_count_df.columns = [
    "Contract Type",
    "Applications"
]

fig_contract_count = px.bar(
    contract_count_df,
    x="Contract Type",
    y="Applications",
    text="Applications",
    title="Applications by Contract Type"
)

st.plotly_chart(
    fig_contract_count,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY CONTRACT TYPE
# =====================================================

fig_contract_risk = px.bar(
    contract_risk_df,
    x="NAME_CONTRACT_TYPE",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Contract Type"
)

fig_contract_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_contract_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# AVERAGE CREDIT BY CONTRACT TYPE
# =====================================================

contract_credit_df = (
    filtered_df.groupby(
        "NAME_CONTRACT_TYPE"
    )["AMT_CREDIT"]
    .mean()
    .reset_index()
)

fig_credit = px.bar(
    contract_credit_df,
    x="NAME_CONTRACT_TYPE",
    y="AMT_CREDIT",
    text="AMT_CREDIT",
    title="Average Credit by Contract Type"
)

fig_credit.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig_credit,
    use_container_width=True
)

st.divider()

# =====================================================
# AVERAGE INCOME BY CONTRACT TYPE
# =====================================================

contract_income_df = (
    filtered_df.groupby(
        "NAME_CONTRACT_TYPE"
    )["AMT_INCOME_TOTAL"]
    .mean()
    .reset_index()
)

fig_income = px.bar(
    contract_income_df,
    x="NAME_CONTRACT_TYPE",
    y="AMT_INCOME_TOTAL",
    text="AMT_INCOME_TOTAL",
    title="Average Income by Contract Type"
)

fig_income.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)

st.divider()

# =====================================================
# AVERAGE ANNUITY BY CONTRACT TYPE
# =====================================================

contract_annuity_df = (
    filtered_df.groupby(
        "NAME_CONTRACT_TYPE"
    )["AMT_ANNUITY"]
    .mean()
    .reset_index()
)

fig_annuity = px.bar(
    contract_annuity_df,
    x="NAME_CONTRACT_TYPE",
    y="AMT_ANNUITY",
    text="AMT_ANNUITY",
    title="Average Annuity by Contract Type"
)

fig_annuity.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig_annuity,
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT TO INCOME RATIO BY CONTRACT TYPE
# =====================================================

contract_ratio_df = (
    filtered_df.groupby(
        "NAME_CONTRACT_TYPE"
    )["CREDIT_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_ratio = px.bar(
    contract_ratio_df,
    x="NAME_CONTRACT_TYPE",
    y="CREDIT_INCOME_RATIO",
    text="CREDIT_INCOME_RATIO",
    title="Credit-To-Income Ratio by Contract Type"
)

fig_ratio.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

st.plotly_chart(
    fig_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# CONTRACT TYPE SUMMARY TABLE
# =====================================================

st.subheader("📋 Contract Type Summary")

contract_summary = (
    filtered_df.groupby(
        "NAME_CONTRACT_TYPE"
    )
    .agg(
        Applications=("SK_ID_CURR", "count"),
        Avg_Income=("AMT_INCOME_TOTAL", "mean"),
        Avg_Credit=("AMT_CREDIT", "mean"),
        Avg_Annuity=("AMT_ANNUITY", "mean"),
        Default_Rate=("TARGET", "mean")
    )
    .reset_index()
)

contract_summary

# =====================================================
# INSIGHTS
# =====================================================

highest_risk_contract = (
    contract_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_CONTRACT_TYPE"]
)

lowest_risk_contract = (
    contract_risk_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["NAME_CONTRACT_TYPE"]
)

highest_credit_contract = (
    contract_credit_df
    .sort_values(
        "AMT_CREDIT",
        ascending=False
    )
    .iloc[0]["NAME_CONTRACT_TYPE"]
)

highest_income_contract = (
    contract_income_df
    .sort_values(
        "AMT_INCOME_TOTAL",
        ascending=False
    )
    .iloc[0]["NAME_CONTRACT_TYPE"]
)

highest_ratio_contract = (
    contract_ratio_df
    .sort_values(
        "CREDIT_INCOME_RATIO",
        ascending=False
    )
    .iloc[0]["NAME_CONTRACT_TYPE"]
)

st.subheader("💡 Contract Type Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Cash Loan Applications: {cash_loan_applications:,}

✅ Revolving Loan Applications: {revolving_loan_applications:,}

✅ Highest Income Contract Type: {highest_income_contract}

✅ Highest Average Credit Contract Type: {highest_credit_contract}

✅ Lowest Risk Contract Type: {lowest_risk_contract}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Contract Type: {highest_risk_contract}

⚠ Highest Credit-To-Income Ratio Contract Type: {highest_ratio_contract}

⚠ Cash Loan Default Rate: {cash_loan_default_rate:.2f}%

⚠ Revolving Loan Default Rate: {revolving_loan_default_rate:.2f}%

⚠ Contract structure can significantly influence customer repayment behavior.
"""
    )