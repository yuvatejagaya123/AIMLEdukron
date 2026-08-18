import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.charts import (
    histogram,
    scatter_chart
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Income vs Credit Analysis",
    page_icon="⚖️",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("⚖️ Income vs Credit Analysis")

st.markdown("""
Determine whether customers are taking loans proportional
to their income using the Credit-to-Income Ratio.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# CREDIT TO INCOME RATIO
# =====================================================

df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"] /
    df["AMT_INCOME_TOTAL"]
)

# =====================================================
# RISK GROUPS
# =====================================================

df["RATIO_GROUP"] = pd.cut(
    df["CREDIT_INCOME_RATIO"],
    bins=[0, 2, 4, 6, float("inf")],
    labels=[
        "Low",
        "Moderate",
        "High",
        "Very High"
    ]
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Income vs Credit Filters")

ratio_filter = st.sidebar.multiselect(
    "Ratio Group",
    options=df["RATIO_GROUP"].dropna().unique(),
    default=df["RATIO_GROUP"].dropna().unique()
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
    (df["RATIO_GROUP"].isin(ratio_filter))
    &
    (df["CODE_GENDER"].isin(gender_filter))
    &
    (df["NAME_EDUCATION_TYPE"].isin(education_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

average_ratio = (
    filtered_df["CREDIT_INCOME_RATIO"]
    .mean()
)

highest_ratio = (
    filtered_df["CREDIT_INCOME_RATIO"]
    .max()
)

high_ratio_df = filtered_df[
    filtered_df["RATIO_GROUP"].isin(
        ["High", "Very High"]
    )
]

high_ratio_default_rate = (
    high_ratio_df["TARGET"]
    .mean()
) * 100

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Credit-To-Income Ratio",
    f"{average_ratio:.2f}"
)

col2.metric(
    "Highest Credit-To-Income Ratio",
    f"{highest_ratio:.2f}"
)

col3.metric(
    "Default Rate (High Ratio)",
    f"{high_ratio_default_rate:.2f}%"
)

st.divider()

# =====================================================
# SAMPLE DATA FOR SCATTER
# =====================================================

scatter_df = filtered_df.sample(
    min(10000, len(filtered_df)),
    random_state=42
)

# =====================================================
# INCOME VS CREDIT SCATTER
# =====================================================

st.subheader("📈 Income vs Credit Scatter Plot")

st.plotly_chart(
    scatter_chart(
        scatter_df,
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "TARGET",
        "Income vs Credit"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# RATIO DISTRIBUTION
# =====================================================

st.subheader("📊 Credit / Income Ratio Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "CREDIT_INCOME_RATIO",
        "Credit / Income Ratio Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE VS RATIO
# =====================================================

ratio_risk_df = (
    filtered_df.groupby(
        "RATIO_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .reset_index()
)

ratio_risk_df["Default Rate %"] = (
    ratio_risk_df["TARGET"] * 100
)

fig_ratio_risk = px.bar(
    ratio_risk_df,
    x="RATIO_GROUP",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate vs Credit / Income Ratio"
)

fig_ratio_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_ratio_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# GENDER WISE RATIO
# =====================================================

gender_ratio_df = (
    filtered_df.groupby("CODE_GENDER")
    ["CREDIT_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_gender_ratio = px.bar(
    gender_ratio_df,
    x="CODE_GENDER",
    y="CREDIT_INCOME_RATIO",
    title="Gender-wise Credit / Income Ratio"
)

st.plotly_chart(
    fig_gender_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# EDUCATION WISE RATIO
# =====================================================

education_ratio_df = (
    filtered_df.groupby("NAME_EDUCATION_TYPE")
    ["CREDIT_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_education_ratio = px.bar(
    education_ratio_df,
    x="NAME_EDUCATION_TYPE",
    y="CREDIT_INCOME_RATIO",
    title="Education-wise Credit / Income Ratio"
)

st.plotly_chart(
    fig_education_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# 


# =====================================================
# INSIGHTS
# =====================================================

highest_risk_ratio_group = (
    ratio_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["RATIO_GROUP"]
)

lowest_risk_ratio_group = (
    ratio_risk_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["RATIO_GROUP"]
)

most_common_ratio_group = (
    filtered_df["RATIO_GROUP"]
    .value_counts()
    .idxmax()
)

st.subheader("💡 Credit-to-Income Ratio Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Average Credit-to-Income Ratio: {average_ratio:.2f}

✅ Highest Ratio Observed: {highest_ratio:.2f}

✅ Most Common Ratio Group: {most_common_ratio_group}

✅ Lowest Risk Ratio Group: {lowest_risk_ratio_group}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Ratio Group: {highest_risk_ratio_group}

⚠ Default Rate (High Ratio Customers): {high_ratio_default_rate:.2f}%

⚠ Higher credit burden relative to income may indicate increased repayment risk.

⚠ Monitor customers in the High and Very High ratio categories.
"""
    )

# =====================================================
# SUMMARY TABLE
# =====================================================

st.subheader("📋 Ratio Group Summary")

ratio_summary = (
    filtered_df.groupby(
        "RATIO_GROUP",
        observed=True
    )
    .agg(
        Customers=("SK_ID_CURR", "count"),
        Default_Rate=("TARGET", "mean"),
        Avg_Income=("AMT_INCOME_TOTAL", "mean"),
        Avg_Credit=("AMT_CREDIT", "mean"),
        Avg_Ratio=("CREDIT_INCOME_RATIO", "mean")
    )
    .reset_index()
)

ratio_summary["Default_Rate"] = (
    ratio_summary["Default_Rate"] * 100
).round(2)

ratio_summary["Avg_Income"] = (
    ratio_summary["Avg_Income"]
).round(0)

ratio_summary["Avg_Credit"] = (
    ratio_summary["Avg_Credit"]
).round(0)

ratio_summary["Avg_Ratio"] = (
    ratio_summary["Avg_Ratio"]
).round(2)

st.dataframe(
    ratio_summary,
    use_container_width=True,
    hide_index=True
)