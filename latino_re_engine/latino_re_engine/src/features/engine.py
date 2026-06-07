"""
Feature engine — applies all feature modules in sequence and normalizes scores.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.features.demographic import add_demographic_features
from src.features.economic import add_economic_features
from src.features.housing import add_housing_features
from src.features.migration import add_migration_features, add_growth_features
from src.features.language import add_language_features


# Columns to normalize to [0, 1] for use in scoring
NORMALIZE_COLS = [
    "hispanic_pct",
    "age_25_44_pct",
    "prime_home_buyer_pop",
    "family_formation_index",
    "married_pct",
    "median_household_income",
    "hispanic_median_income",
    "middle_income_pct",
    "employment_rate",
    "labor_force_participation_rate",
    "college_plus_pct",
    "homeownership_rate",
    "renter_rate",
    "median_home_value",
    "median_gross_rent",
    "housing_affordability_ratio",
    "latent_buyer_demand",
    "first_time_buyer_potential",
    "migration_inflow_rate",
    "interstate_inflow_rate",
    "migration_composite",
    "spanish_home_pct",
    "spanish_marketing_opportunity",
    "lep_spanish_pct",
]


class FeatureEngine:
    def __init__(self, df_previous: pd.DataFrame | None = None, year_prev: int | None = None):
        self.df_previous = df_previous
        self.year_prev = year_prev
        self._scaler = MinMaxScaler(feature_range=(0, 1))

    def transform(self, df: pd.DataFrame, year_current: int | None = None) -> pd.DataFrame:
        logger.info(f"Feature engineering: {len(df):,} records")

        df = add_demographic_features(df)
        df = add_economic_features(df)
        df = add_housing_features(df)
        df = add_migration_features(df)
        df = add_language_features(df)

        if self.df_previous is not None and year_current is not None:
            df = add_growth_features(df, self.df_previous, year_current, self.year_prev)

        df = self._normalize(df)

        logger.info("Feature engineering complete.")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_present = [c for c in NORMALIZE_COLS if c in df.columns]
        if not cols_present:
            return df

        subset = df[cols_present].copy()
        # Only scale rows with at least one non-NaN value per column
        scaled = self._scaler.fit_transform(subset.fillna(subset.median()))
        scaled_df = pd.DataFrame(scaled, columns=[f"{c}_norm" for c in cols_present], index=df.index)

        return pd.concat([df, scaled_df], axis=1)
