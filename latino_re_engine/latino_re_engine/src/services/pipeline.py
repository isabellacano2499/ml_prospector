"""
Main pipeline orchestrator.
Coordinates: fetch → validate → features → score → export.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.config.settings import settings
from src.clients.census.client import CensusClient
from src.features.engine import FeatureEngine
from src.scoring.engine import ScoringEngine
from src.exports.exporter import Exporter


class Pipeline:
    def __init__(
        self,
        client: Optional[CensusClient] = None,
        feature_engine: Optional[FeatureEngine] = None,
        scoring_engine: Optional[ScoringEngine] = None,
        exporter: Optional[Exporter] = None,
    ) -> None:
        self.client = client or CensusClient()
        self.feature_engine = feature_engine
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.exporter = exporter or Exporter()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, dual_year: bool = True) -> dict[str, Path]:
        """
        Full pipeline: fetch current year, optionally compare to previous year,
        run features + scoring, export.
        """
        logger.info(
            f"Pipeline start — level={settings.geographic_level} "
            f"year={settings.acs_year}"
        )

        # 1. Fetch current year
        df_current = self._fetch_year(settings.acs_year)
        if df_current.empty:
            logger.error("No data returned from Census API. Aborting.")
            sys.exit(1)
        logger.info(f"Fetched {len(df_current):,} records for {settings.acs_year}.")

        # 2. Optionally fetch previous year for growth features
        df_previous: Optional[pd.DataFrame] = None
        if dual_year:
            logger.info(f"Fetching previous year ({settings.acs_year_prev}) for growth comparison...")
            df_previous = self._fetch_year(settings.acs_year_prev)
            if df_previous.empty:
                logger.warning(
                    f"No data for {settings.acs_year_prev}. Growth features will be skipped."
                )
                df_previous = None

        # 3. Feature engineering
        feature_engine = self.feature_engine or FeatureEngine(
            df_previous=df_previous,
            year_prev=settings.acs_year_prev if df_previous is not None else None,
        )
        df = feature_engine.transform(df_current, year_current=settings.acs_year)

        # 4. Scoring
        df = self.scoring_engine.score(df)

        # 5. Export
        label = f"latino_market_{settings.geographic_level}_{settings.acs_year}"
        paths = self.exporter.export(df, label=label)

        logger.info(f"Pipeline complete. Output: {paths}")
        return paths

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_year(self, year: int) -> pd.DataFrame:
        """Temporarily override the year setting, fetch, then restore."""
        original_year = settings.acs_year
        settings.__dict__["acs_year"] = year
        self.client.year = year
        self.client.base_url = self.client.base_url.replace(
            str(original_year), str(year)
        )
        try:
            df = self.client.fetch_all()
        finally:
            settings.__dict__["acs_year"] = original_year
            self.client.year = original_year
        return df
