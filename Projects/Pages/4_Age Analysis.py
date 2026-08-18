import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.charts import histogram, bar_chart

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Age Analysis",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Age Analysis")

st.markdown("""
Analyze the relationship between applicant age and credit risk.
Age is calculated using:

**Age = abs(DAYS_BIRTH) / 365**
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# AGE GROUPS
# =====================================================

df["AGE_GROUP"] = pd.cut(
    df["AGE"],
    bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 100],
    labels=[
        "18-25",
        "26-30",
        "31-35",
        "36-40",
        "41-45",
        "46-50",
        "51-55",
        "56-60",
        "61+"
    ]
)

# =====================================================
# FILTERS
# =====================================================

st.sidebar.header("Age Analysis Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["CODE_GENDER"].dropna().unique(),
    default=df["CODE_GENDER"].dropna().unique()
)

age_group_filter = st.sidebar.multiselect(
    "Age Group",
    options=df["AGE_GROUP"].dropna().unique(),
    default=df["AGE_GROUP"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["CODE_GENDER"].isin(gender_filter))
    & (df["AGE_GROUP"].isin(age_group_filter))
    & (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

average_age = filtered_df["AGE"].mean()

youngest_customer = filtered_df["AGE"].min()

oldest_customer = filtered_df["AGE"].max()

age_risk_df = (
    filtered_df.groupby("AGE_GROUP", observed=True)["TARGET"]
    .mean()
    .reset_index()
)

age_risk_df["Default Rate %"] = (
    age_risk_df["TARGET"] * 100
)

highest_risk_age_group = (
    age_risk_df.sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["AGE_GROUP"]
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Age",
    f"{average_age:.0f}"
)

col2.metric(
    "Youngest Customer",
    f"{youngest_customer:.0f}"
)

col3.metric(
    "Oldest Customer",
    f"{oldest_customer:.0f}"
)

col4.metric(
    "Highest Risk Age Group",
    str(highest_risk_age_group)
)

st.divider()

# =====================================================
# AGE DISTRIBUTION
# =====================================================

st.subheader("📊 Age Distribution")

st.plotly_chart(
    histogram(
        filtered_df,
        "AGE",
        "Age Distribution Histogram"
    ),
    use_container_width=True
)

# =====================================================
# APPLICATIONS BY AGE GROUP
# =====================================================

age_group_count = (
    filtered_df["AGE_GROUP"]
    .value_counts()
    .reset_index()
)

age_group_count.columns = [
    "AGE_GROUP",
    "Applications"
]

fig_age_group = px.bar(
    age_group_count,
    x="AGE_GROUP",
    y="Applications",
    text="Applications",
    title="Applications by Age Group"
)

st.plotly_chart(
    fig_age_group,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY AGE
# =====================================================

age_default_df = (
    filtered_df.groupby(
        filtered_df["AGE"].round()
    )["TARGET"]
    .mean()
    .reset_index()
)

age_default_df["Default Rate %"] = (
    age_default_df["TARGET"] * 100
)

fig_default_age = px.line(
    age_default_df,
    x="AGE",
    y="Default Rate %",
    markers=True,
    title="Default Rate by Age"
)

st.plotly_chart(
    fig_default_age,
    use_container_width=True
)

# =====================================================
# DEFAULT RATE BY AGE GROUP
# =====================================================

fig_age_risk = px.bar(
    age_risk_df,
    x="AGE_GROUP",
    y="Default Rate %",
    title="Default Rate by Age Group",
    text="Default Rate %"
)

fig_age_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_age_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# CREDIT AND INCOME BY AGE
# =====================================================

age_financial_df = (
    filtered_df.groupby(
        "AGE_GROUP",
        observed=True
    )
    .agg(
        Average_Credit=("AMT_CREDIT", "mean"),
        Average_Income=("AMT_INCOME_TOTAL", "mean")
    )
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:

    fig_credit = px.bar(
        age_financial_df,
        x="AGE_GROUP",
        y="Average_Credit",
        title="Credit Amount by Age"
    )

    st.plotly_chart(
        fig_credit,
        use_container_width=True
    )

with col2:

    fig_income = px.bar(
        age_financial_df,
        x="AGE_GROUP",
        y="Average_Income",
        title="Income by Age"
    )

    st.plotly_chart(
        fig_income,
        use_container_width=True
    )

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

lowest_risk_age_group = (
    age_risk_df.sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["AGE_GROUP"]
)

most_common_age_group = (
    filtered_df["AGE_GROUP"]
    .mode()[0]
)

st.subheader("💡 Age Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Average Age: {average_age:.0f}

✅ Most Common Age Group: {most_common_age_group}

✅ Youngest Customer: {youngest_customer:.0f}

✅ Oldest Customer: {oldest_customer:.0f}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Age Group: {highest_risk_age_group}

⚠ Lowest Risk Age Group: {lowest_risk_age_group}
"""
    )