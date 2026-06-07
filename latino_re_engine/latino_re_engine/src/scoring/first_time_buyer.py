"""
First-Time Home Buyer Score (0–100).

Weights:
  30% Hispanic Population share
  20% Hispanic Renter share (latent buyers)
  15% Age 25–44 share
  10% Income Stability
  10% Employment Stability
  10% Population Growth
   5% Housing Affordability
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "hispanic_pct_norm":            0.30,
    "first_time_buyer_potential_norm": 0.20,
    "age_25_44_pct_norm":           0.15,
    "median_household_income_norm": 0.10,
    "employment_rate_norm":         0.10,
    "hispanic_growth_rate_norm":    0.10,
    "housing_affordability_ratio_norm": 0.05,
}


def compute_first_time_buyer_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    used_weight = 0.0

    for col, weight in WEIGHTS.items():
        if col in df.columns:
            score += df[col].fillna(0) * weight
            used_weight += weight

    # Re-scale if some components were missing
    if used_weight > 0 and used_weight < 1.0:
        score = score / used_weight

    return (score * 100).clip(0, 100).round(2)
