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
    page_title="Family & Children Analysis",
    page_icon="👨‍👩‍👧",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("👨‍👩‍👧 Family & Children Analysis")

st.markdown("""
Analyze family size, children count,
family status and their relationship with credit risk.
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

st.sidebar.header("Family Analysis Filters")

family_filter = st.sidebar.multiselect(
    "Family Status",
    options=df["NAME_FAMILY_STATUS"].dropna().unique(),
    default=df["NAME_FAMILY_STATUS"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["NAME_FAMILY_STATUS"].isin(family_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

avg_children = (
    filtered_df["CNT_CHILDREN"]
    .mean()
)

avg_family_members = (
    filtered_df["CNT_FAM_MEMBERS"]
    .mean()
)

customers_with_children = (
    filtered_df["CNT_CHILDREN"] > 0
).sum()

customers_without_children = (
    filtered_df["CNT_CHILDREN"] == 0
).sum()

family_risk_df = (
    filtered_df.groupby(
        "NAME_FAMILY_STATUS"
    )["TARGET"]
    .mean()
    .reset_index()
)

family_risk_df["Default Rate %"] = (
    family_risk_df["TARGET"] * 100
)

highest_risk_family = (
    family_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_FAMILY_STATUS"]
)

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Average Children",
    f"{avg_children:.1f}"
)

col2.metric(
    "Average Family Members",
    f"{avg_family_members:.1f}"
)

col3.metric(
    "Customers with Children",
    f"{customers_with_children:,}"
)

col4.metric(
    "Customers without Children",
    f"{customers_without_children:,}"
)

col5.metric(
    "Highest Risk Family Type",
    highest_risk_family
)

st.divider()

# =====================================================
# CUSTOMERS BY NUMBER OF CHILDREN
# =====================================================

children_count_df = (
    filtered_df["CNT_CHILDREN"]
    .value_counts()
    .reset_index()
)

children_count_df.columns = [
    "Children",
    "Customers"
]

fig_children = px.bar(
    children_count_df,
    x="Children",
    y="Customers",
    text="Customers",
    title="Customers by Number of Children"
)

st.plotly_chart(
    fig_children,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY NUMBER OF CHILDREN
# =====================================================

children_risk_df = (
    filtered_df.groupby(
        "CNT_CHILDREN"
    )["TARGET"]
    .mean()
    .reset_index()
)

children_risk_df["Default Rate %"] = (
    children_risk_df["TARGET"] * 100
)

fig_children_risk = px.bar(
    children_risk_df,
    x="CNT_CHILDREN",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Number of Children"
)

fig_children_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_children_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# CUSTOMERS BY FAMILY SIZE
# =====================================================

family_size_df = (
    filtered_df["CNT_FAM_MEMBERS"]
    .round()
    .value_counts()
    .reset_index()
)

family_size_df.columns = [
    "Family Size",
    "Customers"
]

fig_family_size = px.bar(
    family_size_df,
    x="Family Size",
    y="Customers",
    text="Customers",
    title="Customers by Family Size"
)

st.plotly_chart(
    fig_family_size,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY FAMILY SIZE
# =====================================================

family_size_risk_df = (
    filtered_df.groupby(
        filtered_df["CNT_FAM_MEMBERS"].round()
    )["TARGET"]
    .mean()
    .reset_index()
)

family_size_risk_df["Default Rate %"] = (
    family_size_risk_df["TARGET"] * 100
)

fig_family_risk = px.bar(
    family_size_risk_df,
    x="CNT_FAM_MEMBERS",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Family Size"
)

fig_family_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_family_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# APPLICATIONS BY FAMILY STATUS
# =====================================================

family_status_count = (
    filtered_df["NAME_FAMILY_STATUS"]
    .value_counts()
    .reset_index()
)

family_status_count.columns = [
    "Family Status",
    "Applications"
]

fig_status = px.bar(
    family_status_count,
    x="Family Status",
    y="Applications",
    text="Applications",
    title="Applications by Family Status"
)

st.plotly_chart(
    fig_status,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY FAMILY STATUS
# =====================================================

fig_family_status_risk = px.bar(
    family_risk_df,
    x="NAME_FAMILY_STATUS",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Family Status"
)

fig_family_status_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_family_status_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# INCOME VS FAMILY SIZE
# =====================================================

st.subheader("💰 Income vs Family Size")

income_family_df = (
    filtered_df.groupby(
        filtered_df["CNT_FAM_MEMBERS"].round()
    )["AMT_INCOME_TOTAL"]
    .mean()
    .reset_index()
)

income_family_df.columns = [
    "Family Size",
    "Average Income"
]

fig_income_family = px.bar(
    income_family_df,
    x="Family Size",
    y="Average Income",
    text="Average Income",
    title="Income vs Family Size"
)

fig_income_family.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

fig_income_family.update_layout(
    xaxis_title="Family Size",
    yaxis_title="Average Income"
)

st.plotly_chart(
    fig_income_family,
    use_container_width=True
)

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

most_common_family_status = (
    filtered_df["NAME_FAMILY_STATUS"]
    .mode()[0]
)

lowest_risk_family = (
    family_risk_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["NAME_FAMILY_STATUS"]
)

st.subheader("💡 Family Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Average Children: {avg_children:.1f}

✅ Average Family Members: {avg_family_members:.1f}

✅ Customers with Children: {customers_with_children:,}

✅ Customers without Children: {customers_without_children:,}

✅ Most Common Family Status: {most_common_family_status}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Highest Risk Family Type: {highest_risk_family}

⚠ Lowest Risk Family Type: {lowest_risk_family}

⚠ Family structure influences household expenses and repayment capability.

⚠ Customers with larger households may show different repayment behavior patterns.
"""
    )