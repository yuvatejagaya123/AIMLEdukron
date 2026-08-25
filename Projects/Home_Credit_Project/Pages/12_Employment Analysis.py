import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.charts import histogram

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Employment Analysis",
    page_icon="💼",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("💼 Employment Analysis")

st.markdown("""
Understand how employment history, income type,
occupation and organization type affect credit risk.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

df = clean_data(df)

df = create_features(df)

# =====================================================
# CLEAN DAYS_EMPLOYED
# =====================================================

df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
    365243,
    pd.NA
)

df["EMPLOYMENT_YEARS"] = (
    abs(df["DAYS_EMPLOYED"]) / 365
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Employment Analysis Filters")

income_type_filter = st.sidebar.multiselect(
    "Income Type",
    options=df["NAME_INCOME_TYPE"].dropna().unique(),
    default=df["NAME_INCOME_TYPE"].dropna().unique()
)

occupation_filter = st.sidebar.multiselect(
    "Occupation",
    options=df["OCCUPATION_TYPE"].dropna().unique(),
    default=df["OCCUPATION_TYPE"].dropna().unique()
)

organization_filter = st.sidebar.multiselect(
    "Organization Type",
    options=df["ORGANIZATION_TYPE"].dropna().unique(),
    default=df["ORGANIZATION_TYPE"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["NAME_INCOME_TYPE"].isin(income_type_filter))
    &
    (df["OCCUPATION_TYPE"].isin(occupation_filter))
    &
    (df["ORGANIZATION_TYPE"].isin(organization_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

avg_employment_years = (
    filtered_df["EMPLOYMENT_YEARS"]
    .mean()
)

most_common_occupation = (
    filtered_df["OCCUPATION_TYPE"]
    .mode()[0]
)

most_common_income_type = (
    filtered_df["NAME_INCOME_TYPE"]
    .mode()[0]
)

occupation_risk_df = (
    filtered_df.groupby(
        "OCCUPATION_TYPE"
    )["TARGET"]
    .mean()
    .reset_index()
)

occupation_risk_df["Default Rate %"] = (
    occupation_risk_df["TARGET"] * 100
)

highest_risk_occupation = (
    occupation_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["OCCUPATION_TYPE"]
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Employment Years",
    f"{avg_employment_years:.1f}"
)

col2.metric(
    "Most Common Occupation",
    most_common_occupation
)

col3.metric(
    "Most Common Income Type",
    most_common_income_type
)

col4.metric(
    "Highest Risk Occupation",
    highest_risk_occupation
)

st.divider()

# =====================================================
# EMPLOYMENT YEARS DISTRIBUTION
# =====================================================

st.subheader("📊 Employment Years Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "EMPLOYMENT_YEARS",
        "Employment Years Distribution"
    ),
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY EMPLOYMENT YEARS
# =====================================================

employment_default_df = (
    filtered_df.groupby(
        filtered_df["EMPLOYMENT_YEARS"].round()
    )["TARGET"]
    .mean()
    .reset_index()
)

employment_default_df["Default Rate %"] = (
    employment_default_df["TARGET"] * 100
)

fig_emp_years = px.line(
    employment_default_df,
    x="EMPLOYMENT_YEARS",
    y="Default Rate %",
    markers=True,
    title="Default Rate by Employment Years"
)

st.plotly_chart(
    fig_emp_years,
    use_container_width=True
)

st.divider()

# =====================================================
# APPLICATIONS BY INCOME TYPE
# =====================================================

income_count_df = (
    filtered_df["NAME_INCOME_TYPE"]
    .value_counts()
    .reset_index()
)

income_count_df.columns = [
    "Income Type",
    "Applications"
]

fig_income = px.bar(
    income_count_df,
    x="Income Type",
    y="Applications",
    text="Applications",
    title="Applications by Income Type"
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY INCOME TYPE
# =====================================================

income_risk_df = (
    filtered_df.groupby(
        "NAME_INCOME_TYPE"
    )["TARGET"]
    .mean()
    .reset_index()
)

income_risk_df["Default Rate %"] = (
    income_risk_df["TARGET"] * 100
)

fig_income_risk = px.bar(
    income_risk_df,
    x="NAME_INCOME_TYPE",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Income Type"
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
# APPLICATIONS BY OCCUPATION
# =====================================================

occupation_count_df = (
    filtered_df["OCCUPATION_TYPE"]
    .value_counts()
    .reset_index()
    .head(15)
)

occupation_count_df.columns = [
    "Occupation",
    "Applications"
]

fig_occ_apps = px.bar(
    occupation_count_df,
    x="Occupation",
    y="Applications",
    text="Applications",
    title="Applications by Occupation"
)

st.plotly_chart(
    fig_occ_apps,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY OCCUPATION
# =====================================================

occupation_risk_df = (
    occupation_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .head(15)
)

fig_occ_risk = px.bar(
    occupation_risk_df,
    x="OCCUPATION_TYPE",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Occupation"
)

fig_occ_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_occ_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY ORGANIZATION TYPE
# =====================================================

org_risk_df = (
    filtered_df.groupby(
        "ORGANIZATION_TYPE"
    )["TARGET"]
    .mean()
    .reset_index()
)

org_risk_df["Default Rate %"] = (
    org_risk_df["TARGET"] * 100
)

org_risk_df = (
    org_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .head(15)
)

fig_org_risk = px.bar(
    org_risk_df,
    x="ORGANIZATION_TYPE",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Organization Type"
)

fig_org_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_org_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# EMPLOYMENT SUMMARY TABLE
# =====================================================

st.subheader("📋 Employment Summary")

employment_summary = (
    filtered_df.groupby(
        "NAME_INCOME_TYPE"
    )
    .agg(
        Applications=("SK_ID_CURR", "count"),
        Avg_Income=("AMT_INCOME_TOTAL", "mean"),
        Avg_Credit=("AMT_CREDIT", "mean"),
        Default_Rate=("TARGET", "mean")
    )
    .reset_index()
)

employment_summary["Default_Rate"] = (
    employment_summary["Default_Rate"] * 100
).round(2)

employment_summary["Avg_Income"] = (
    employment_summary["Avg_Income"]
).round(0)

employment_summary["Avg_Credit"] = (
    employment_summary["Avg_Credit"]
).round(0)

st.dataframe(
    employment_summary,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

st.subheader("💡 Employment Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Average Employment Years: {avg_employment_years:.1f}

✅ Most Common Occupation: {most_common_occupation}

✅ Most Common Income Type: {most_common_income_type}

✅ Workers with longer employment histories generally show more stable applicant profiles.
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Occupation: {highest_risk_occupation}

⚠ Occupation and employment history have a strong relationship with repayment behavior.

⚠ Organization type may influence customer risk characteristics.
"""
    )