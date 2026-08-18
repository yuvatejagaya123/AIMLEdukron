import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features

from utils.charts import (
    histogram,
    bar_chart,
    scatter_chart
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Income Analysis",
    page_icon="💰",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("💰 Income Analysis")

st.markdown("""
Analyze customer income and understand its relationship
with credit risk, credit amount, and annuity burden.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

df = clean_data(df)

df = create_features(df)

# =====================================================
# INCOME GROUPS
# =====================================================

df["INCOME_GROUP"] = pd.cut(
    df["AMT_INCOME_TOTAL"],
    bins=[0, 50000, 100000, 150000, 200000, 300000, 500000, float("inf")],
    labels=[
        "Below 50K",
        "50K-100K",
        "100K-150K",
        "150K-200K",
        "200K-300K",
        "300K-500K",
        "Above 500K"
    ]
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Income Analysis Filters")

income_group_filter = st.sidebar.multiselect(
    "Income Group",
    options=df["INCOME_GROUP"].dropna().unique(),
    default=df["INCOME_GROUP"].dropna().unique()
)

education_filter = st.sidebar.multiselect(
    "Education",
    options=df["NAME_EDUCATION_TYPE"].dropna().unique(),
    default=df["NAME_EDUCATION_TYPE"].dropna().unique()
)

income_type_filter = st.sidebar.multiselect(
    "Income Type",
    options=df["NAME_INCOME_TYPE"].dropna().unique(),
    default=df["NAME_INCOME_TYPE"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["INCOME_GROUP"].isin(income_group_filter))
    & (df["NAME_EDUCATION_TYPE"].isin(education_filter))
    & (df["NAME_INCOME_TYPE"].isin(income_type_filter))
    & (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CARDS
# =====================================================

total_income = filtered_df["AMT_INCOME_TOTAL"].sum()

average_income = filtered_df["AMT_INCOME_TOTAL"].mean()

median_income = filtered_df["AMT_INCOME_TOTAL"].median()

maximum_income = filtered_df["AMT_INCOME_TOTAL"].max()

average_income_defaulters = (
    filtered_df[
        filtered_df["TARGET"] == 1
    ]["AMT_INCOME_TOTAL"]
    .mean()
)

st.subheader("📌 KPI Cards")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Income",
    f"{total_income:,.0f}"
)

col2.metric(
    "Average Income",
    f"{average_income:,.0f}"
)

col3.metric(
    "Median Income",
    f"{median_income:,.0f}"
)

col4.metric(
    "Maximum Income",
    f"{maximum_income:,.0f}"
)

col5.metric(
    "Avg Income of Defaulters",
    f"{average_income_defaulters:,.0f}"
)

st.divider()

# =====================================================
# INCOME DISTRIBUTION
# =====================================================

st.subheader("📊 Income Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "AMT_INCOME_TOTAL",
        "Income Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# CUSTOMERS BY INCOME GROUP
# =====================================================

income_group_count = (
    filtered_df["INCOME_GROUP"]
    .value_counts()
    .reset_index()
)

income_group_count.columns = [
    "Income Group",
    "Customers"
]

fig_income_group = px.bar(
    income_group_count,
    x="Income Group",
    y="Customers",
    text="Customers",
    title="Customers by Income Group"
)

st.plotly_chart(
    fig_income_group,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY INCOME GROUP
# =====================================================

income_risk_df = (
    filtered_df.groupby(
        "INCOME_GROUP",
        observed=True
    )["TARGET"]
    .mean()
    .reset_index()
)

income_risk_df["Default Rate %"] = (
    income_risk_df["TARGET"] * 100
)

fig_income_risk = px.bar(
    income_risk_df,
    x="INCOME_GROUP",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Income Group"
)

fig_income_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_income_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# INCOME VS CREDIT
# =====================================================


col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        scatter_chart(
            filtered_df,
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "TARGET",
            "Income vs Credit"
        ),
        use_container_width=True
    )

with col2:

    st.plotly_chart(
        scatter_chart(
            filtered_df,
            "AMT_INCOME_TOTAL",
            "AMT_ANNUITY",
            "TARGET",
            "Income vs Annuity"
        ),
        use_container_width=True
    )

st.divider()

# =====================================================
# INCOME BY EDUCATION
# =====================================================

education_income_df = (
    filtered_df.groupby(
        "NAME_EDUCATION_TYPE"
    )["AMT_INCOME_TOTAL"]
    .mean()
    .reset_index()
)

fig_education_income = px.bar(
    education_income_df,
    x="NAME_EDUCATION_TYPE",
    y="AMT_INCOME_TOTAL",
    title="Income by Education"
)

st.plotly_chart(
    fig_education_income,
    use_container_width=True
)

st.divider()

# =====================================================
# INCOME BY OCCUPATION
# =====================================================

occupation_income_df = (
    filtered_df.groupby(
        "OCCUPATION_TYPE"
    )["AMT_INCOME_TOTAL"]
    .mean()
    .reset_index()
)

occupation_income_df = (
    occupation_income_df
    .sort_values(
        "AMT_INCOME_TOTAL",
        ascending=False
    )
    .head(15)
)

fig_occupation_income = px.bar(
    occupation_income_df,
    x="OCCUPATION_TYPE",
    y="AMT_INCOME_TOTAL",
    title="Income by Occupation"
)

st.plotly_chart(
    fig_occupation_income,
    use_container_width=True
)

st.divider()

# =====================================================
# INCOME INSIGHTS
# =====================================================

highest_risk_income_group = (
    income_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["INCOME_GROUP"]
)

highest_income_group = (
    filtered_df["INCOME_GROUP"]
    .value_counts()
    .idxmax()
)

avg_income_non_default = (
    filtered_df[
        filtered_df["TARGET"] == 0
    ]["AMT_INCOME_TOTAL"]
    .mean()
)

st.subheader("💡 Income Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Most Common Income Group: {highest_income_group}

✅ Average Income of Defaulters: {average_income_defaulters:,.0f}

✅ Average Income of Non-Defaulters: {avg_income_non_default:,.0f}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Income Group: {highest_risk_income_group}

⚠ Income level has a significant relationship with default behavior.
"""
    )