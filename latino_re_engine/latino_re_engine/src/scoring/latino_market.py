"""
Latino Market Score (0–100).

Weights:
  25% Hispanic Population share
  20% Hispanic Growth rate
  15% Migration signal
  15% Housing Affordability
  15% Employment
  10% Spanish-language marketing opportunity
"""

import pandas as pd

WEIGHTS = {
    "hispanic_pct_norm":                    0.25,
    "hispanic_growth_rate_norm":            0.20,
    "migration_composite_norm":             0.15,
    "housing_affordability_ratio_norm":     0.15,
    "employment_rate_norm":                 0.15,
    "spanish_marketing_opportunity_norm":   0.10,
}


def compute_latino_market_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    used_weight = 0.0

    for col, weight in WEIGHTS.items():
        if col in df.columns:
            score += df[col].fillna(0) * weight
            used_weight += weight

    if used_weight > 0 and used_weight < 1.0:
        score = score / used_weight

    return (score * 100).clip(0, 100).round(2)
