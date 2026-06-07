"""
Demographic feature engineering.
Inputs: raw Census columns.
Outputs: derived demographic features added to the DataFrame.
"""

import numpy as np
import pandas as pd


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    total = df.get("total_population", pd.Series(np.nan, index=df.index))
    hispanic = df.get("hispanic_population", pd.Series(np.nan, index=df.index))

    # Hispanic share and density
    df["hispanic_pct"] = _safe_div(hispanic, total)

    # Age groups — collapse raw age buckets into meaningful ranges
    df["pop_18_24"] = _sum_cols(df, [
        "male_18_19", "male_20", "male_21", "male_22_24",
        "female_18_19", "female_20", "female_21", "female_22_24",
    ])
    df["pop_25_34"] = _sum_cols(df, [
        "male_25_29", "male_30_34",
        "female_25_29", "female_30_34",
    ])
    df["pop_35_44"] = _sum_cols(df, [
        "male_35_39", "male_40_44",
        "female_35_39", "female_40_44",
    ])
    df["pop_45_54"] = _sum_cols(df, [
        "male_45_49", "male_50_54",
        "female_45_49", "female_50_54",
    ])
    df["pop_25_44"] = df["pop_25_34"] + df["pop_35_44"]

    df["age_18_24_pct"] = _safe_div(df["pop_18_24"], total)
    df["age_25_34_pct"] = _safe_div(df["pop_25_34"], total)
    df["age_35_44_pct"] = _safe_div(df["pop_35_44"], total)
    df["age_25_44_pct"] = _safe_div(df["pop_25_44"], total)

    df["prime_home_buyer_pop"] = df["pop_25_44"]

    # Gender split
    male = df.get("male_total", pd.Series(np.nan, index=df.index))
    female = df.get("female_total", pd.Series(np.nan, index=df.index))
    df["male_pct"] = _safe_div(male, total)
    df["female_pct"] = _safe_div(female, total)

    # Hispanic by gender
    hisp_male = df.get("hispanic_male", pd.Series(np.nan, index=df.index))
    hisp_female = df.get("hispanic_female", pd.Series(np.nan, index=df.index))
    df["hispanic_male_pct"] = _safe_div(hisp_male, hispanic)
    df["hispanic_female_pct"] = _safe_div(hisp_female, hispanic)

    # Marital / family formation
    married_total = (
        df.get("male_married", pd.Series(0, index=df.index)).fillna(0)
        + df.get("female_married", pd.Series(0, index=df.index)).fillna(0)
    )
    marital_total = df.get("marital_total", pd.Series(np.nan, index=df.index))
    df["married_pct"] = _safe_div(married_total, marital_total)
    df["family_formation_index"] = df["married_pct"] * df["age_25_44_pct"]

    # Immigration
    nat = df.get("naturalized_citizens", pd.Series(np.nan, index=df.index))
    non_cit = df.get("non_citizens", pd.Series(np.nan, index=df.index))
    cit_total = df.get("citizenship_total", pd.Series(np.nan, index=df.index))
    df["foreign_born_pct"] = _safe_div(nat + non_cit.fillna(0), cit_total)
    df["naturalized_pct"] = _safe_div(nat, cit_total)

    return df


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Divide two series, returning NaN where denominator is 0 or NaN."""
    return num / denom.replace(0, np.nan)


def _sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum columns that may or may not exist, treating missing as 0."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].fillna(0).sum(axis=1)
