"""
Export pipeline output to Parquet (partitioned by state) and CSV.
"""

from pathlib import Path
import pandas as pd
from loguru import logger

from src.config.settings import settings

# Canonical output schema — columns guaranteed in the final dataset
OUTPUT_COLUMNS = [
    # Geography
    "zip_code", "state", "state_fips", "county",
    # Raw demographics
    "total_population", "hispanic_population", "hispanic_pct",
    "age_25_44_pct", "prime_home_buyer_pop",
    "married_pct", "family_formation_index",
    "foreign_born_pct", "naturalized_pct",
    # Economic
    "median_household_income", "hispanic_median_income",
    "middle_income_pct", "income_gap_ratio",
    "employment_rate", "unemployment_rate", "labor_force_participation_rate",
    "college_plus_pct", "bachelors_plus_pct",
    # Industry
    "sector_construction_pct", "sector_healthcare_edu_pct",
    "sector_hospitality_pct", "sector_professional_pct",
    # Housing
    "homeownership_rate", "renter_rate",
    "median_home_value", "median_gross_rent",
    "price_to_income_ratio", "rent_to_income_ratio",
    "estimated_monthly_mortgage", "mortgage_to_income_ratio",
    "housing_affordability_ratio", "rent_burden_flag",
    "first_time_buyer_potential",
    # Language
    "spanish_home_pct", "lep_spanish_pct",
    "bilingual_spanish_pct", "spanish_marketing_opportunity",
    # Migration
    "migration_inflow", "migration_inflow_rate",
    "interstate_inflow_rate", "international_inflow_rate",
    "residential_mobility_rate", "migration_composite",
    # Growth (present only when dual-year mode is enabled)
    "population_growth_rate", "hispanic_growth_rate",
    "household_growth_rate", "income_growth_rate",
    # Scores
    "first_home_buyer_score",
    "latino_market_score",
    "real_estate_opportunity_score",
    "overall_score",
]


class Exporter:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or settings.output_dir

    def export(self, df: pd.DataFrame, label: str = "latino_market") -> dict[str, Path]:
        df_out = self._align_schema(df)
        paths: dict[str, Path] = {}

        # Parquet — partitioned by state (efficient for downstream queries)
        parquet_path = self.output_dir / f"{label}.parquet"
        if "state" in df_out.columns:
            df_out.to_parquet(
                parquet_path,
                engine="pyarrow",
                compression="snappy",
                partition_cols=["state"],
                index=False,
            )
        else:
            df_out.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
        paths["parquet"] = parquet_path
        logger.info(f"Parquet written → {parquet_path}")

        # CSV — flat file for spreadsheet / BI tools
        csv_path = self.output_dir / f"{label}.csv"
        df_out.to_csv(csv_path, index=False)
        paths["csv"] = csv_path
        logger.info(f"CSV written → {csv_path}  ({len(df_out):,} rows)")

        self._print_summary(df_out)
        return paths

    def _align_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only OUTPUT_COLUMNS that exist; add missing ones as NaN."""
        present = [c for c in OUTPUT_COLUMNS if c in df.columns]
        missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
        if missing:
            logger.debug(f"Columns not available in this run: {missing}")
        result = df[present].copy()
        for col in missing:
            result[col] = float("nan")
        return result[OUTPUT_COLUMNS]

    def _print_summary(self, df: pd.DataFrame) -> None:
        score_cols = ["first_home_buyer_score", "latino_market_score",
                      "real_estate_opportunity_score", "overall_score"]
        present_scores = [c for c in score_cols if c in df.columns]
        if not present_scores:
            return
        logger.info("\n=== TOP 10 MARKETS BY OVERALL SCORE ===")
        top = df.nlargest(10, "overall_score")[
            ["zip_code", "state", "hispanic_pct"] + present_scores
        ].to_string(index=False)
        logger.info(f"\n{top}")
