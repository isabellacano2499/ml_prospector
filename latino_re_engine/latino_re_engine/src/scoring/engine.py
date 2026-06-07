"""
Scoring engine — applies all three scorers and adds growth_rate normalizations
for columns that only exist after the growth step.
"""

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.scoring.first_time_buyer import compute_first_time_buyer_score
from src.scoring.latino_market import compute_latino_market_score
from src.scoring.real_estate_opportunity import compute_real_estate_opportunity_score

# Growth/rate columns that need normalization before scoring
_GROWTH_COLS_TO_NORMALIZE = [
    "hispanic_growth_rate",
    "population_growth_rate",
    "household_growth_rate",
    "income_growth_rate",
    "migration_composite",
]


class ScoringEngine:
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Scoring: normalizing growth columns...")
        df = self._normalize_growth_cols(df)

        logger.info("Computing First-Time Home Buyer Score...")
        df["first_home_buyer_score"] = compute_first_time_buyer_score(df)

        logger.info("Computing Latino Market Score...")
        df["latino_market_score"] = compute_latino_market_score(df)

        logger.info("Computing Real Estate Opportunity Score...")
        df["real_estate_opportunity_score"] = compute_real_estate_opportunity_score(df)

        # Composite overall score
        df["overall_score"] = (
            df["first_home_buyer_score"] * 0.40
            + df["latino_market_score"] * 0.35
            + df["real_estate_opportunity_score"] * 0.25
        ).round(2)

        logger.info(
            f"Scoring complete. Score summary:\n"
            f"  FTHB:  min={df['first_home_buyer_score'].min():.1f} "
            f"max={df['first_home_buyer_score'].max():.1f} "
            f"mean={df['first_home_buyer_score'].mean():.1f}\n"
            f"  LMS:   min={df['latino_market_score'].min():.1f} "
            f"max={df['latino_market_score'].max():.1f} "
            f"mean={df['latino_market_score'].mean():.1f}\n"
            f"  REO:   min={df['real_estate_opportunity_score'].min():.1f} "
            f"max={df['real_estate_opportunity_score'].max():.1f} "
            f"mean={df['real_estate_opportunity_score'].mean():.1f}"
        )
        return df

    def _normalize_growth_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(0, 1))
        cols_present = [c for c in _GROWTH_COLS_TO_NORMALIZE if c in df.columns]
        if not cols_present:
            return df
        subset = df[cols_present].copy()
        median_fill = subset.median()
        scaled = scaler.fit_transform(subset.fillna(median_fill))
        for i, col in enumerate(cols_present):
            df[f"{col}_norm"] = scaled[:, i]
        return df
