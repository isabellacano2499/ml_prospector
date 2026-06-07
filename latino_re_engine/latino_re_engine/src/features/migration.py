"""
Migration feature engineering.
ACS provides inflow proxies; true net migration requires IRS SOI data (future).
"""

import numpy as np
import pandas as pd

from src.features.demographic import _safe_div, _sum_cols


def add_migration_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    total_pop = df.get("total_population", pd.Series(np.nan, index=df.index))
    migration_universe = df.get("migration_universe", pd.Series(np.nan, index=df.index))

    # B07003 returns male (_m) and female (_f) separately — combine them
    df["same_house_1yr"] = _sum_cols(df, ["same_house_1yr_m", "same_house_1yr_f"])
    df["moved_within_county"] = _sum_cols(df, ["moved_within_county_m", "moved_within_county_f"])
    df["moved_diff_county_same_state"] = _sum_cols(df, ["moved_diff_county_same_state_m", "moved_diff_county_same_state_f"])
    df["moved_diff_state"] = _sum_cols(df, ["moved_diff_state_m", "moved_diff_state_f"])
    df["moved_from_abroad"] = _sum_cols(df, ["moved_from_abroad_m", "moved_from_abroad_f"])

    # Total in-movers (from outside the local county/ZIP)
    inflow_cols = ["moved_diff_county_same_state", "moved_diff_state", "moved_from_abroad"]
    df["migration_inflow"] = _sum_cols(df, inflow_cols)

    # Inflow as % of current population (migration intensity)
    df["migration_inflow_rate"] = _safe_div(df["migration_inflow"], total_pop)

    # Same-house stability (inverse of mobility)
    df["residential_stability_rate"] = _safe_div(df["same_house_1yr"], migration_universe)
    df["residential_mobility_rate"] = 1 - df["residential_stability_rate"]

    # Cross-state migration — strongest signal for emerging markets
    df["interstate_inflow_rate"] = _safe_div(df["moved_diff_state"], total_pop)
    df["international_inflow_rate"] = _safe_div(df["moved_from_abroad"], total_pop)

    # Composite migration opportunity: weights interstate > international > local
    df["migration_composite"] = (
        df["interstate_inflow_rate"].fillna(0) * 0.50
        + df["international_inflow_rate"].fillna(0) * 0.30
        + df["moved_diff_county_same_state"].fillna(0)
        / total_pop.replace(0, np.nan) * 0.20
    )

    return df


def add_growth_features(
    df_current: pd.DataFrame,
    df_previous: pd.DataFrame,
    year_current: int,
    year_previous: int,
) -> pd.DataFrame:
    """
    Merge two ACS snapshots and compute growth rates.
    Both DataFrames must share 'zip_code' (or county/state) as the join key.
    """
    id_col = _detect_geo_id(df_current)
    suffix_curr = f"_{year_current}"
    suffix_prev = f"_{year_previous}"

    merged = df_current.merge(
        df_previous,
        on=id_col,
        how="left",
        suffixes=(suffix_curr, suffix_prev),
    )

    def growth_rate(col: str) -> pd.Series:
        curr = merged.get(f"{col}{suffix_curr}", pd.Series(np.nan, index=merged.index))
        prev = merged.get(f"{col}{suffix_prev}", pd.Series(np.nan, index=merged.index))
        return _safe_div(curr - prev, prev.replace(0, np.nan))

    merged["population_growth_rate"] = growth_rate("total_population")
    merged["hispanic_growth_rate"] = growth_rate("hispanic_population")
    merged["household_growth_rate"] = growth_rate("occupied_housing_total")
    merged["income_growth_rate"] = growth_rate("median_household_income")

    # Restore current-year columns as primary (drop _year suffix for clean output)
    for col in df_current.columns:
        curr_col = f"{col}{suffix_curr}"
        if curr_col in merged.columns:
            merged[col] = merged[curr_col]
            merged.drop(columns=[curr_col], inplace=True)

    return merged


def _detect_geo_id(df: pd.DataFrame) -> str:
    for candidate in ("zip_code", "county_fips", "state_fips", "state"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Cannot detect geographic ID column in DataFrame.")
