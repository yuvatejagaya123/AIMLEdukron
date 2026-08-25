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
    page_title="Housing & Asset Analysis",
    page_icon="🏠",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🏠 Housing & Asset Analysis")

st.markdown("""
Analyze vehicle ownership, property ownership,
housing type and their relationship with credit risk.
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

st.sidebar.header("Housing Analysis Filters")

housing_filter = st.sidebar.multiselect(
    "Housing Type",
    options=df["NAME_HOUSING_TYPE"].dropna().unique(),
    default=df["NAME_HOUSING_TYPE"].dropna().unique()
)

target_filter = st.sidebar.multiselect(
    "Target",
    options=df["TARGET"].unique(),
    default=df["TARGET"].unique()
)

filtered_df = df[
    (df["NAME_HOUSING_TYPE"].isin(housing_filter))
    &
    (df["TARGET"].isin(target_filter))
]

# =====================================================
# KPI CALCULATIONS
# =====================================================

car_owners = (
    filtered_df["FLAG_OWN_CAR"] == "Y"
).sum()

property_owners = (
    filtered_df["FLAG_OWN_REALTY"] == "Y"
).sum()

both_owners = (
    (
        filtered_df["FLAG_OWN_CAR"] == "Y"
    )
    &
    (
        filtered_df["FLAG_OWN_REALTY"] == "Y"
    )
).sum()

property_owner_default_rate = (
    filtered_df[
        filtered_df["FLAG_OWN_REALTY"] == "Y"
    ]["TARGET"]
    .mean()
) * 100

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("📌 KPI Cards")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Car Owners",
    f"{car_owners:,}"
)

col2.metric(
    "Property Owners",
    f"{property_owners:,}"
)

col3.metric(
    "Own Both Assets",
    f"{both_owners:,}"
)

col4.metric(
    "Property Owner Default Rate",
    f"{property_owner_default_rate:.2f}%"
)

st.divider()

# =====================================================
# CAR OWNERSHIP DISTRIBUTION
# =====================================================

car_count_df = (
    filtered_df["FLAG_OWN_CAR"]
    .value_counts()
    .reset_index()
)

car_count_df.columns = [
    "Car Ownership",
    "Customers"
]

fig_car = px.pie(
    car_count_df,
    names="Car Ownership",
    values="Customers",
    title="Car Ownership Distribution"
)

st.plotly_chart(
    fig_car,
    use_container_width=True
)

st.divider()

# =====================================================
# PROPERTY OWNERSHIP DISTRIBUTION
# =====================================================

property_count_df = (
    filtered_df["FLAG_OWN_REALTY"]
    .value_counts()
    .reset_index()
)

property_count_df.columns = [
    "Property Ownership",
    "Customers"
]

fig_property = px.pie(
    property_count_df,
    names="Property Ownership",
    values="Customers",
    title="Property Ownership Distribution"
)

st.plotly_chart(
    fig_property,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY CAR OWNERSHIP
# =====================================================

car_risk_df = (
    filtered_df.groupby(
        "FLAG_OWN_CAR"
    )["TARGET"]
    .mean()
    .reset_index()
)

car_risk_df["Default Rate %"] = (
    car_risk_df["TARGET"] * 100
)

fig_car_risk = px.bar(
    car_risk_df,
    x="FLAG_OWN_CAR",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Car Ownership"
)

fig_car_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_car_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY PROPERTY OWNERSHIP
# =====================================================

property_risk_df = (
    filtered_df.groupby(
        "FLAG_OWN_REALTY"
    )["TARGET"]
    .mean()
    .reset_index()
)

property_risk_df["Default Rate %"] = (
    property_risk_df["TARGET"] * 100
)

fig_property_risk = px.bar(
    property_risk_df,
    x="FLAG_OWN_REALTY",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Property Ownership"
)

fig_property_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_property_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# APPLICANTS BY HOUSING TYPE
# =====================================================

housing_count_df = (
    filtered_df["NAME_HOUSING_TYPE"]
    .value_counts()
    .reset_index()
)

housing_count_df.columns = [
    "Housing Type",
    "Applicants"
]

fig_housing_count = px.bar(
    housing_count_df,
    x="Housing Type",
    y="Applicants",
    text="Applicants",
    title="Applicants by Housing Type"
)

st.plotly_chart(
    fig_housing_count,
    use_container_width=True
)

st.divider()

# =====================================================
# DEFAULT RATE BY HOUSING TYPE
# =====================================================

housing_risk_df = (
    filtered_df.groupby(
        "NAME_HOUSING_TYPE"
    )["TARGET"]
    .mean()
    .reset_index()
)

housing_risk_df["Default Rate %"] = (
    housing_risk_df["TARGET"] * 100
)

fig_housing_risk = px.bar(
    housing_risk_df,
    x="NAME_HOUSING_TYPE",
    y="Default Rate %",
    text="Default Rate %",
    title="Default Rate by Housing Type"
)

fig_housing_risk.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_housing_risk,
    use_container_width=True
)

st.divider()

# =====================================================
# AVERAGE CREDIT BY HOUSING TYPE
# =====================================================

housing_credit_df = (
    filtered_df.groupby(
        "NAME_HOUSING_TYPE"
    )["AMT_CREDIT"]
    .mean()
    .reset_index()
)

fig_housing_credit = px.bar(
    housing_credit_df,
    x="NAME_HOUSING_TYPE",
    y="AMT_CREDIT",
    text="AMT_CREDIT",
    title="Average Credit by Housing Type"
)

fig_housing_credit.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig_housing_credit,
    use_container_width=True
)

st.divider()

# =====================================================
# HOUSING & ASSET SUMMARY TABLE
# =====================================================

st.subheader("📋 Housing & Asset Summary")

housing_summary = (
    filtered_df.groupby(
        "NAME_HOUSING_TYPE"
    )
    .agg(
        Applicants=("SK_ID_CURR", "count"),
        Avg_Credit=("AMT_CREDIT", "mean"),
        Avg_Income=("AMT_INCOME_TOTAL", "mean"),
        Default_Rate=("TARGET", "mean")
    )
    .reset_index()
)

housing_summary["Default_Rate"] = (
    housing_summary["Default_Rate"] * 100
).round(2)

housing_summary["Avg_Credit"] = (
    housing_summary["Avg_Credit"]
).round(0)

housing_summary["Avg_Income"] = (
    housing_summary["Avg_Income"]
).round(0)

st.dataframe(
    housing_summary,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# INSIGHTS
# =====================================================

highest_risk_housing_type = (
    housing_risk_df
    .sort_values(
        "Default Rate %",
        ascending=False
    )
    .iloc[0]["NAME_HOUSING_TYPE"]
)

lowest_risk_housing_type = (
    housing_risk_df
    .sort_values(
        "Default Rate %",
        ascending=True
    )
    .iloc[0]["NAME_HOUSING_TYPE"]
)

most_common_housing_type = (
    filtered_df["NAME_HOUSING_TYPE"]
    .mode()[0]
)

st.subheader("💡 Housing & Asset Insights")

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
✅ Car Owners: {car_owners:,}

✅ Property Owners: {property_owners:,}

✅ Customers Owning Both Assets: {both_owners:,}

✅ Most Common Housing Type: {most_common_housing_type}

✅ Lowest Risk Housing Type: {lowest_risk_housing_type}
"""
    )

with col2:
    st.warning(
        f"""
⚠ Property Owner Default Rate: {property_owner_default_rate:.2f}%

⚠ Highest Risk Housing Type: {highest_risk_housing_type}

⚠ Housing type and asset ownership influence customer risk patterns.

⚠ Customers without major assets may show different repayment behavior.
"""
    )