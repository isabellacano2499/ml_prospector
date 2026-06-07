"""
Real Estate Opportunity Score (0–100).

Weights:
  25% Migration inflow
  25% Population Growth
  20% Latent buyer demand (renter × Hispanic)
  15% Housing Affordability
  15% Income level
"""

import pandas as pd

WEIGHTS = {
    "migration_composite_norm":         0.25,
    "population_growth_rate_norm":      0.25,
    "latent_buyer_demand_norm":         0.20,
    "housing_affordability_ratio_norm": 0.15,
    "median_household_income_norm":     0.15,
}


def compute_real_estate_opportunity_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    used_weight = 0.0

    for col, weight in WEIGHTS.items():
        if col in df.columns:
            score += df[col].fillna(0) * weight
            used_weight += weight

    if used_weight > 0 and used_weight < 1.0:
        score = score / used_weight

    return (score * 100).clip(0, 100).round(2)
