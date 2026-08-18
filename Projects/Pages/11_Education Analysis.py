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
    page_title="Education Analysis",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🎓 Education Analysis")

st.markdown("""
Analyze customer behavior based on education level.
Understand income, credit, annuity, and default patterns
across education categories.
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
# FILTERS
# =====================================================

st.sidebar.header("Education Analysis Filters")

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
    (df["NAME_EDUCATION_TYPE"].isin(education_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

most_common_education = (
    filtered_df["NAME_EDUCATION_TYPE"]
    .mode()[0]
)

education_income_df = (
    filtered_df.groupby("NAME_EDUCATION_TYPE")
    ["AMT_INCOME_TOTAL"]
    .mean()
    .reset_index()
)

highest_income_education = (
    education_income_df
    .sort_values(
        "AMT_INCOME_TOTAL",
        ascending=False
    )
    .iloc[0]["NAME_EDUCATION_TYPE"]
)

education_default_df = (
    filtered_df.groupby("NAME_EDUCATION_TYPE")
    ["TARGET"]
    .mean()
    .reset_index()
)

education_default_df["Default Rate %"] = (
    education_default_df["TARGET"]
    * 100
)

lowest_default_education = (
    education_default_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["NAME_EDUCATION_TYPE"]
)

highest_default_education = (
    education_default_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_EDUCATION_TYPE"]
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Most Common Education",
    most_common_education
)

col2.metric(
    "Highest Income Education",
    highest_income_education
)

col3.metric(
    "Lowest Default Education",
    lowest_default_education
)

col4.metric(
    "Highest Default Education",
    highest_default_education
)

st.divider()

# =====================================================
# CUSTOMERS BY EDUCATION
# =====================================================

education_count_df = (
    filtered_df["NAME_EDUCATION_TYPE"]
    .value_counts()
    .reset_index()
)

education_count_df.columns = [
    "Education",
    "Customers"
]

fig_customers = px.bar(
    education_count_df,
    x="Education",
    y="Customers",
    text="Customers",
    title="Customers by Education"
)

st.plotly_chart(
    fig_customers,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY EDUCATION
# =====================================================

fig_default = px.bar(
    education_default_df,
    x="NAME_EDUCATION_TYPE",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Education"
)

fig_default.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_default,
    use_container_width=True
)

st.divider()

# =====================================================
# INCOME BY EDUCATION
# =====================================================

fig_income = px.bar(
    education_income_df,
    x="NAME_EDUCATION_TYPE",
    y="AMT_INCOME_TOTAL",
    title="Income by Education"
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT BY EDUCATION
# =====================================================

education_credit_df = (
    filtered_df.groupby("NAME_EDUCATION_TYPE")
    ["AMT_CREDIT"]
    .mean()
    .reset_index()
)

fig_credit = px.bar(
    education_credit_df,
    x="NAME_EDUCATION_TYPE",
    y="AMT_CREDIT",
    title="Credit by Education"
)

st.plotly_chart(
    fig_credit,
    use_container_width=True
)

st.divider()

# =====================================================
# ANNUITY BY EDUCATION
# =====================================================

education_annuity_df = (
    filtered_df.groupby("NAME_EDUCATION_TYPE")
    ["AMT_ANNUITY"]
    .mean()
    .reset_index()
)

fig_annuity = px.bar(
    education_annuity_df,
    x="NAME_EDUCATION_TYPE",
    y="AMT_ANNUITY",
    title="Annuity by Education"
)

st.plotly_chart(
    fig_annuity,
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT TO INCOME RATIO BY EDUCATION
# =====================================================

education_ratio_df = (
    filtered_df.groupby("NAME_EDUCATION_TYPE")
    ["CREDIT_INCOME_RATIO"]
    .mean()
    .reset_index()
)

fig_ratio = px.bar(
    education_ratio_df,
    x="NAME_EDUCATION_TYPE",
    y="CREDIT_INCOME_RATIO",
    title="Credit-To-Income Ratio by Education"
)

st.plotly_chart(
    fig_ratio,
    use_container_width=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

st.subheader("💡 Education Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Most Common Education: {most_common_education}

✅ Highest Income Education Group: {highest_income_education}

✅ Lowest Default Education Group: {lowest_default_education}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Default Education Group: {highest_default_education}

⚠ Education level shows a strong relationship with income and repayment performance.

⚠ Higher education categories generally have different credit behavior patterns.
"""
    )