import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.charts import (
    pie_chart,
    donut_chart,
    bar_chart
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Gender Analysis",
    page_icon="👥",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("👥 Gender Analysis")

st.markdown("""
Compare applicant characteristics and credit risk across genders.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Gender Analysis Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["CODE_GENDER"].dropna().unique(),
    default=df["CODE_GENDER"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["CODE_GENDER"].isin(gender_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

male_applicants = (
    filtered_df["CODE_GENDER"] == "M"
).sum()

female_applicants = (
    filtered_df["CODE_GENDER"] == "F"
).sum()

male_default_rate = (
    filtered_df[
        filtered_df["CODE_GENDER"] == "M"
    ]["TARGET"].mean()
) * 100

female_default_rate = (
    filtered_df[
        filtered_df["CODE_GENDER"] == "F"
    ]["TARGET"].mean()
) * 100

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Male Applicants",
    f"{male_applicants:,}"
)

col2.metric(
    "Female Applicants",
    f"{female_applicants:,}"
)

col3.metric(
    "Male Default Rate",
    f"{male_default_rate:.2f}%"
)

col4.metric(
    "Female Default Rate",
    f"{female_default_rate:.2f}%"
)

st.divider()

# =====================================================
# APPLICANTS BY GENDER
# =====================================================

st.subheader("👥 Applicant Distribution")

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        pie_chart(
            filtered_df,
            "CODE_GENDER",
            "Applicants by Gender"
        ),
        use_container_width=True
    )

with col2:

    gender_count = (
        filtered_df["CODE_GENDER"]
        .value_counts()
        .reset_index()
    )

    gender_count.columns = [
        "Gender",
        "Applicants"
    ]

    fig = px.bar(
        gender_count,
        x="Gender",
        y="Applicants",
        text="Applicants",
        title="Applicants by Gender"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# =====================================================
# DEFAULT CUSTOMERS BY GENDER
# =====================================================

default_gender_df = (
    filtered_df.groupby("CODE_GENDER")["TARGET"]
    .sum()
    .reset_index()
)

default_gender_df.columns = [
    "Gender",
    "Defaults"
]

fig_defaults = px.bar(
    default_gender_df,
    x="Gender",
    y="Defaults",
    text="Defaults",
    title="Default Customers by Gender"
)

st.plotly_chart(
    fig_defaults,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY GENDER
# =====================================================

default_rate_df = (
    filtered_df.groupby("CODE_GENDER")["TARGET"]
    .mean()
    .reset_index()
)

default_rate_df["Default Rate %"] = (
    default_rate_df["TARGET"] * 100
)

fig_default_rate = px.bar(
    default_rate_df,
    x="CODE_GENDER",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Gender"
)

fig_default_rate.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_default_rate,
    use_container_width=True
)

st.divider()

# =====================================================
# AVG INCOME / CREDIT / ANNUITY
# =====================================================

gender_profile_df = (
    filtered_df.groupby("CODE_GENDER")
    .agg(
        Average_Income=("AMT_INCOME_TOTAL", "mean"),
        Average_Credit=("AMT_CREDIT", "mean"),
        Average_Annuity=("AMT_ANNUITY", "mean")
    )
    .reset_index()
)

col1, col2, col3 = st.columns(3)

with col1:

    fig_income = px.bar(
        gender_profile_df,
        x="CODE_GENDER",
        y="Average_Income",
        title="Average Income by Gender"
    )

    st.plotly_chart(
        fig_income,
        use_container_width=True
    )

with col2:

    fig_credit = px.bar(
        gender_profile_df,
                x="CODE_GENDER",
                y="Average_Credit",
                title="Average Credit by Gender"
    )

    st.plotly_chart(
        fig_credit,
        use_container_width=True
    )

with col3:

    fig_annuity = px.bar(
        gender_profile_df,
        x="CODE_GENDER",
        y="Average_Annuity",
        title="Average Annuity by Gender"
    )

    st.plotly_chart(
        fig_annuity,
        use_container_width=True
    )

st.divider()

# =====================================================
# COMPARISON TABLE
# =====================================================

st.subheader("📋 Gender Comparison Table")

comparison_table = (
    filtered_df.groupby("CODE_GENDER")
    .agg(
        Customers=("SK_ID_CURR", "count"),
        Defaults=("TARGET", "sum"),
        Default_Rate=("TARGET", "mean"),
        Avg_Income=("AMT_INCOME_TOTAL", "mean"),
        Avg_Credit=("AMT_CREDIT", "mean")
    )
    .reset_index()
)

comparison_table["Default_Rate"] = (
    comparison_table["Default_Rate"] * 100
).round(2)

comparison_table["Avg_Income"] = (
    comparison_table["Avg_Income"]
).round(0)

comparison_table["Avg_Credit"] = (
    comparison_table["Avg_Credit"]
).round(0)

comparison_table.rename(
    columns={
        "CODE_GENDER": "Gender"
    },
    inplace=True
)

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

highest_risk_gender = (
    default_rate_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["CODE_GENDER"]
)

st.subheader("💡 Gender Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Male Applicants: {male_applicants:,}

✅ Female Applicants: {female_applicants:,}

✅ Male Default Rate: {male_default_rate:.2f}%

✅ Female Default Rate: {female_default_rate:.2f}%
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Gender: {highest_risk_gender}

⚠ Gender with higher default tendency requires additional analysis.

⚠ Compare income and credit behavior between genders.
"""
    )