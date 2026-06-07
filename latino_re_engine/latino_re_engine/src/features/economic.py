"""
Economic feature engineering — income, employment, industry.
"""

import numpy as np
import pandas as pd

from src.features.demographic import _safe_div, _sum_cols


def add_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Income ---
    income = df.get("median_household_income", pd.Series(np.nan, index=df.index))
    hisp_income = df.get("hispanic_median_income", pd.Series(np.nan, index=df.index))

    df["income_gap_ratio"] = _safe_div(hisp_income, income)

    # Households earning above $50k (mortgage-eligible proxy)
    above_50k_cols = [
        "income_50_60k", "income_60_75k", "income_75_100k",
        "income_100_125k", "income_125_150k", "income_150_200k", "income_200k_plus",
    ]
    df["households_above_50k"] = _sum_cols(df, above_50k_cols)

    # Middle-income band: $30k–$100k (first-time buyer sweet spot)
    middle_cols = [
        "income_30_35k", "income_35_40k", "income_40_45k", "income_45_50k",
        "income_50_60k", "income_60_75k", "income_75_100k",
    ]
    all_income_cols = [
        "income_lt10k", "income_10_15k", "income_15_20k", "income_20_25k",
        "income_25_30k", "income_30_35k", "income_35_40k", "income_40_45k",
        "income_45_50k", "income_50_60k", "income_60_75k", "income_75_100k",
        "income_100_125k", "income_125_150k", "income_150_200k", "income_200k_plus",
    ]
    total_hh_income = _sum_cols(df, all_income_cols)
    df["middle_income_pct"] = _safe_div(_sum_cols(df, middle_cols), total_hh_income)

    # --- Employment ---
    employed = df.get("civilian_employed", pd.Series(np.nan, index=df.index))
    unemployed = df.get("civilian_unemployed", pd.Series(np.nan, index=df.index))
    labor_force = df.get("in_labor_force", pd.Series(np.nan, index=df.index))
    universe = df.get("employment_universe", pd.Series(np.nan, index=df.index))

    df["employment_rate"] = _safe_div(employed, labor_force)
    df["unemployment_rate"] = _safe_div(unemployed, labor_force)
    df["labor_force_participation_rate"] = _safe_div(labor_force, universe)

    # --- Industry sector shares ---
    ind_total = df.get("industry_total", pd.Series(np.nan, index=df.index))

    sector_pairs = [
        ("construction", ["industry_construction_m", "industry_construction_f"]),
        ("manufacturing", ["industry_manufacturing_m", "industry_manufacturing_f"]),
        ("transportation", ["industry_transportation_m", "industry_transportation_f"]),
        ("finance_re", ["industry_finance_re_m", "industry_finance_re_f"]),
        ("professional", ["industry_professional_m", "industry_professional_f"]),
        ("healthcare_edu", ["industry_healthcare_edu_m", "industry_healthcare_edu_f"]),
        ("hospitality", ["industry_hospitality_m", "industry_hospitality_f"]),
    ]

    for sector_name, sector_cols in sector_pairs:
        total_sector = _sum_cols(df, sector_cols)
        df[f"sector_{sector_name}_pct"] = _safe_div(total_sector, ind_total)

    # Education score: share of adults with at least some college
    edu_total = df.get("education_total", pd.Series(np.nan, index=df.index))
    college_plus_cols = [
        "some_college_lt1yr", "some_college_gt1yr", "associates_degree",
        "bachelors_degree", "masters_degree", "professional_degree", "doctorate_degree",
    ]
    df["college_plus_pct"] = _safe_div(_sum_cols(df, college_plus_cols), edu_total)

    bachelors_plus_cols = ["bachelors_degree", "masters_degree", "professional_degree", "doctorate_degree"]
    df["bachelors_plus_pct"] = _safe_div(_sum_cols(df, bachelors_plus_cols), edu_total)

    return df
