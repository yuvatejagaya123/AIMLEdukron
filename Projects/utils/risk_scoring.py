import pandas as pd


# =====================================================
# CALCULATE RISK SCORE
# =====================================================

def calculate_risk_score(row):

    score = 0

    # ===================================
    # External Score Risk
    # ===================================

    if pd.notna(row["AVG_EXT_SCORE"]):

        if row["AVG_EXT_SCORE"] < 0.30:
            score += 30

        elif row["AVG_EXT_SCORE"] < 0.50:
            score += 20

        elif row["AVG_EXT_SCORE"] < 0.70:
            score += 10

    # ===================================
    # Credit Income Ratio
    # ===================================

    if pd.notna(row["CREDIT_INCOME_RATIO"]):

        if row["CREDIT_INCOME_RATIO"] > 6:
            score += 25

        elif row["CREDIT_INCOME_RATIO"] > 4:
            score += 15

        elif row["CREDIT_INCOME_RATIO"] > 2:
            score += 8

    # ===================================
    # Annuity Income Ratio
    # ===================================

    if pd.notna(row["ANNUITY_INCOME_RATIO"]):

        if row["ANNUITY_INCOME_RATIO"] > 0.35:
            score += 20

        elif row["ANNUITY_INCOME_RATIO"] > 0.20:
            score += 10

        elif row["ANNUITY_INCOME_RATIO"] > 0.10:
            score += 5

    # ===================================
    # Employment Stability
    # ===================================

    if pd.notna(row["EMPLOYMENT_YEARS"]):

        if row["EMPLOYMENT_YEARS"] < 1:
            score += 10

        elif row["EMPLOYMENT_YEARS"] < 3:
            score += 5

    # ===================================
    # Age Risk
    # ===================================

    if pd.notna(row["AGE"]):

        if row["AGE"] < 30:
            score += 8

        elif row["AGE"] < 40:
            score += 4

    return min(score, 100)


# =====================================================
# RISK LEVEL
# =====================================================

def assign_risk_level(score):

    if score >= 75:
        return "Critical Risk"

    elif score >= 50:
        return "High Risk"

    elif score >= 25:
        return "Medium Risk"

    else:
        return "Low Risk"


# =====================================================
# ADD RISK COLUMNS
# =====================================================

def add_risk_columns(df):

    df = df.copy()

    df["RISK_SCORE"] = (
        df.apply(
            calculate_risk_score,
            axis=1
        )
    )

    df["RISK_LEVEL"] = (
        df["RISK_SCORE"]
        .apply(assign_risk_level)
    )

    return df