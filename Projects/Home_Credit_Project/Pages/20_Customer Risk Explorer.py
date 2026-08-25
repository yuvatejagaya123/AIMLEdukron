import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.features import create_features
from utils.filters import sidebar_filters

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Risk Explorer",
    page_icon="🔍",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🔍 Customer Risk Explorer")

st.markdown("""
Explore individual customers and analyze
their risk profile using financial,
demographic and credit information.
""")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data("Data/application_train.csv")

df = clean_data(df)

df = create_features(df)

# =====================================================
# DERIVED FEATURES
# =====================================================

df["AGE"] = (
    abs(df["DAYS_BIRTH"]) / 365
).round(0)

df["DAYS_EMPLOYED"] = (
    df["DAYS_EMPLOYED"]
    .replace(365243, np.nan)
)

df["EMPLOYMENT_YEARS"] = (
    abs(df["DAYS_EMPLOYED"]) / 365
).round(1)

df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"]
    /
    df["AMT_INCOME_TOTAL"]
)

df["ANNUITY_INCOME_RATIO"] = (
    df["AMT_ANNUITY"]
    /
    df["AMT_INCOME_TOTAL"]
)

df["CREDIT_GOODS_RATIO"] = (
    df["AMT_CREDIT"]
    /
    df["AMT_GOODS_PRICE"]
)

df["AVG_EXT_SCORE"] = (
    df[
        [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        ]
    ]
    .mean(axis=1)
)

# =====================================================
# GLOBAL FILTERS
# =====================================================

filtered_df = sidebar_filters(df)

# =====================================================
# CUSTOMER SEARCH
# =====================================================

st.sidebar.header("Customer Search")

customer_id = st.sidebar.text_input(
    "Search SK_ID_CURR"
)

if customer_id:

    try:

        filtered_df = filtered_df[
            filtered_df["SK_ID_CURR"]
            ==
            int(customer_id)
        ]

    except:

        st.error(
            "Please enter a valid customer id"
        )

# =====================================================
# RESULTS
# =====================================================

st.subheader("📋 Matching Customers")

st.metric(
    "Customers Found",
    len(filtered_df)
)

display_columns = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "AGE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)

st.divider()
# =====================================================
# CUSTOMER PROFILE
# =====================================================

if len(filtered_df) == 1:

    customer = filtered_df.iloc[0]

    st.subheader("👤 Customer Risk Profile")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customer ID",
        int(customer["SK_ID_CURR"])
    )

    col2.metric(
        "TARGET",
        int(customer["TARGET"])
    )

    col3.metric(
        "Age",
        int(customer["AGE"])
    )

    col4.metric(
        "Gender",
        customer["CODE_GENDER"]
    )

    st.divider()

    # ==========================================
    # CUSTOMER DETAILS
    # ==========================================

    profile_df = pd.DataFrame({

        "Attribute":[
            "Income",
            "Credit Amount",
            "Annuity",
            "Education",
            "Occupation",
            "Family Status",
            "Children",
            "Housing Type"
        ],

        "Value":[
            customer["AMT_INCOME_TOTAL"],
            customer["AMT_CREDIT"],
            customer["AMT_ANNUITY"],
            customer["NAME_EDUCATION_TYPE"],
            customer["OCCUPATION_TYPE"],
            customer["NAME_FAMILY_STATUS"],
            customer["CNT_CHILDREN"],
            customer["NAME_HOUSING_TYPE"]
        ]
    })

    st.dataframe(
        profile_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================
    # EXTERNAL SCORES
    # ==========================================

    ext_df = pd.DataFrame({

        "Score":[
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "AVG_EXT_SCORE"
        ],

        "Value":[
            customer["EXT_SOURCE_1"],
            customer["EXT_SOURCE_2"],
            customer["EXT_SOURCE_3"],
            customer["AVG_EXT_SCORE"]
        ]
    })

    st.subheader("📊 External Scores")

    st.dataframe(
        ext_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================
    # RISK INDICATORS
    # ==========================================

    risk_df = pd.DataFrame({

        "Metric":[
            "Credit-To-Income Ratio",
            "Annuity-To-Income Ratio",
            "Credit-To-Goods Ratio",
            "Employment Years",
            "Average External Score"
        ],

        "Value":[
            round(customer["CREDIT_INCOME_RATIO"],2),
            round(customer["ANNUITY_INCOME_RATIO"],3),
            round(customer["CREDIT_GOODS_RATIO"],2),
            round(customer["EMPLOYMENT_YEARS"],1),
            round(customer["AVG_EXT_SCORE"],3)
        ]
    })

    st.subheader("⚠ Calculated Risk Indicators")

    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        """
        Search a single SK_ID_CURR
        to view detailed customer profile.
        """
    )

st.divider()

# =====================================================
# INSIGHTS
# =====================================================

st.subheader("💡 Explorer Insights")

if not filtered_df.empty:

    avg_credit_income_ratio = (
        filtered_df["CREDIT_INCOME_RATIO"]
        .mean()
    )

    avg_annuity_income_ratio = (
        filtered_df["ANNUITY_INCOME_RATIO"]
        .mean()
    )

    avg_external_score = (
        filtered_df["AVG_EXT_SCORE"]
        .mean()
    )

else:

    avg_credit_income_ratio = 0
    avg_annuity_income_ratio = 0
    avg_external_score = 0

col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
✅ Customers Found: {len(filtered_df):,}

✅ Average Credit-To-Income Ratio:
{avg_credit_income_ratio:.2f}

✅ Average External Score:
{avg_external_score:.3f}
"""
    )

with col2:

    st.warning(
        f"""
⚠ Average Annuity-To-Income Ratio:
{avg_annuity_income_ratio:.3f}

⚠ Use external score together with
income and credit ratios for risk analysis.
"""
    )