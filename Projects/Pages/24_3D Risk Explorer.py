import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="3D Risk Explorer",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = load_data(r"data/application_train.csv")

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🚀 3D Credit Risk Explorer")
st.markdown(
    """
    Explore customer risk using an interactive 3D visualization.
    Rotate, zoom, and inspect customer behaviour from different angles.
    """
)

# --------------------------------------------------
# CREATE AGE COLUMN
# --------------------------------------------------
df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)

# --------------------------------------------------
# CREATE RISK LEVEL
# --------------------------------------------------
df["RISK_LEVEL"] = df["TARGET"].map(
    {
        0: "Low Risk",
        1: "High Risk"
    }
)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("🎛 Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["RISK_LEVEL"].unique(),
    default=df["RISK_LEVEL"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["CODE_GENDER"].unique(),
    default=df["CODE_GENDER"].unique()
)

filtered_df = df[
    (df["RISK_LEVEL"].isin(risk_filter))
    &
    (df["CODE_GENDER"].isin(gender_filter))
]

# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------
if len(filtered_df) > 5000:
    chart_df = filtered_df.sample(
        5000,
        random_state=42
    )
else:
    chart_df = filtered_df

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{len(filtered_df):,}"
    )

with col2:
    st.metric(
        "Avg Income",
        f"₹{filtered_df['AMT_INCOME_TOTAL'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Avg Credit",
        f"₹{filtered_df['AMT_CREDIT'].mean():,.0f}"
    )

with col4:
    st.metric(
        "Default Rate",
        f"{filtered_df['TARGET'].mean()*100:.2f}%"
    )

st.markdown("---")

# --------------------------------------------------
# 3D CHART
# --------------------------------------------------
fig = px.scatter_3d(
    chart_df,
    x="AMT_INCOME_TOTAL",
    y="AMT_CREDIT",
    z="AMT_ANNUITY",
    color="RISK_LEVEL",
    size="CNT_FAM_MEMBERS",
    opacity=0.75,
    hover_data=[
        "AGE",
        "CODE_GENDER",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY"
    ],
    color_discrete_map={
        "Low Risk": "#22c55e",
        "High Risk": "#ef4444"
    }
)

fig.update_layout(
    template="plotly_dark",
    height=850,
    title="Income vs Credit vs Annuity (3D View)",
    scene=dict(
        xaxis_title="Income",
        yaxis_title="Credit Amount",
        zaxis_title="Annuity"
    ),
    margin=dict(
        l=0,
        r=0,
        t=50,
        b=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------
st.subheader("📌 Key Insights")

high_risk = filtered_df[
    filtered_df["TARGET"] == 1
]

low_risk = filtered_df[
    filtered_df["TARGET"] == 0
]

c1, c2 = st.columns(2)

with c1:

    st.success(
        f"""
        ✅ Average Income (Low Risk)

        ₹{low_risk['AMT_INCOME_TOTAL'].mean():,.0f}
        """
    )

    st.success(
        f"""
        ✅ Average Credit (Low Risk)

        ₹{low_risk['AMT_CREDIT'].mean():,.0f}
        """
    )

with c2:

    st.error(
        f"""
        ⚠ Average Income (High Risk)

        ₹{high_risk['AMT_INCOME_TOTAL'].mean():,.0f}
        """
    )

    st.error(
        f"""
        ⚠ Average Credit (High Risk)

        ₹{high_risk['AMT_CREDIT'].mean():,.0f}
        """
    )

# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------
st.markdown("---")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Filtered Data",
    data=csv,
    file_name="3d_risk_explorer_data.csv",
    mime="text/csv"
)