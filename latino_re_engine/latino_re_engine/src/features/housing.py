"""
Housing feature engineering — ownership, affordability, demand signals.
"""

import numpy as np
import pandas as pd

from src.features.demographic import _safe_div


def add_housing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    owner = df.get("owner_occupied", pd.Series(np.nan, index=df.index))
    renter = df.get("renter_occupied", pd.Series(np.nan, index=df.index))
    total_occ = df.get("occupied_housing_total", pd.Series(np.nan, index=df.index))

    home_value = df.get("median_home_value", pd.Series(np.nan, index=df.index))
    gross_rent = df.get("median_gross_rent", pd.Series(np.nan, index=df.index))
    income = df.get("median_household_income", pd.Series(np.nan, index=df.index))

    # Ownership / renter split
    df["homeownership_rate"] = _safe_div(owner, total_occ)
    df["renter_rate"] = _safe_div(renter, total_occ)

    # Affordability: standard rule-of-thumb thresholds
    # Price-to-Income: home value / annual income (lower = more affordable)
    df["price_to_income_ratio"] = _safe_div(home_value, income)

    # Rent-to-Income: monthly rent / monthly income
    monthly_income = income / 12
    df["rent_to_income_ratio"] = _safe_div(gross_rent, monthly_income)

    # Estimated monthly mortgage (30yr fixed, ~7% rate, 3.5% FHA down)
    # M = P * [r(1+r)^n] / [(1+r)^n - 1]
    loan_amount = home_value * 0.965  # 3.5% down
    monthly_rate = 0.07 / 12
    n = 360
    df["estimated_monthly_mortgage"] = loan_amount * (
        monthly_rate * (1 + monthly_rate) ** n
    ) / ((1 + monthly_rate) ** n - 1)

    df["mortgage_to_income_ratio"] = _safe_div(
        df["estimated_monthly_mortgage"], monthly_income
    )

    # Housing affordability ratio (inverse of price_to_income — higher = more affordable)
    # Capped at 1 to normalize direction: ratio = 1 / price_to_income (normalized later)
    df["housing_affordability_ratio"] = 1 / df["price_to_income_ratio"].replace(0, np.nan)

    # Demand proxy: high renter rate + high Hispanic % = latent buyer demand
    hispanic_pct = df.get("hispanic_pct", pd.Series(np.nan, index=df.index))
    df["latent_buyer_demand"] = df["renter_rate"].fillna(0) * hispanic_pct.fillna(0)

    # First-time buyer potential: renters who likely haven't bought yet
    df["first_time_buyer_potential"] = df["renter_rate"] * hispanic_pct

    # Rent burden: % of households paying >30% income on rent
    rent_income_pct = df.get("median_rent_income_pct", pd.Series(np.nan, index=df.index))
    df["rent_burden_flag"] = (rent_income_pct > 30).astype(float).where(rent_income_pct.notna())

    return df
