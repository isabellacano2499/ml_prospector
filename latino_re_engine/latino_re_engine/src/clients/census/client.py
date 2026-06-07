"""
Census ACS API client with batched variable requests, retry, and disk cache.
"""

import json
import hashlib
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging
from tqdm import tqdm

from src.config.settings import settings
from src.config.census_variables import (
    CENSUS_SENTINEL_VALUES,
    get_variable_groups_for_level,
)
from src.clients.census.endpoints import (
    build_base_url,
    build_zip_params,
    build_county_params,
    build_state_params,
    get_geo_id_columns,
    STATE_FIPS,
    STATE_FIPS_NAMES,
)


class CensusAPIError(Exception):
    pass


class CensusClient:
    def __init__(self) -> None:
        self.api_key = settings.census_api_key
        self.year = settings.acs_year
        self.dataset = settings.acs_dataset
        self.geo_level = settings.geographic_level
        self.raw_dir = settings.raw_data_dir
        self.var_batch_size = settings.variable_batch_size
        self.timeout = settings.request_timeout
        self.base_url = build_base_url(self.year, self.dataset)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> pd.DataFrame:
        """Fetch all variable groups for the configured geographic level."""
        groups = get_variable_groups_for_level(self.geo_level)
        all_codes = {}
        for group_vars in groups.values():
            all_codes.update(group_vars)

        variable_codes = list(all_codes.keys())
        logger.info(
            f"Fetching {len(variable_codes)} variables at {self.geo_level} level "
            f"(ACS {self.dataset} {self.year})"
        )

        if self.geo_level == "zip":
            return self._fetch_zip_all_states(variable_codes, all_codes)
        elif self.geo_level == "county":
            return self._fetch_geography(
                variable_codes, all_codes, build_county_params, scope="national"
            )
        elif self.geo_level == "state":
            return self._fetch_geography(
                variable_codes, all_codes, build_state_params, scope="national"
            )
        else:
            raise ValueError(f"Unsupported geographic level: {self.geo_level}")

    def _fetch_zip_all_states(
        self, variable_codes: list[str], code_map: dict[str, str]
    ) -> pd.DataFrame:
        """Fetch ZCTA data with a single national query.

        ZCTAs are not nested within states in the Census hierarchy, so
        we fetch all ZCTAs at once instead of iterating state-by-state.
        """
        logger.info("Fetching all ZCTAs nationally (single query)...")
        geo_id_cols = get_geo_id_columns("zip")

        def build_params(codes: list[str]) -> dict:
            return build_zip_params(codes, "", self.api_key)

        batch_frames = self._fetch_variables_resilient(variable_codes, build_params, geo_id_cols)
        if not batch_frames:
            return pd.DataFrame()

        df = self._merge_batches(batch_frames, geo_id_cols)
        df = self._rename_and_clean(df, code_map)
        return df

    def _fetch_zip_state(
        self,
        variable_codes: list[str],
        code_map: dict[str, str],
        state_fips: str,
        state_abbr: str,
    ) -> pd.DataFrame | None:
        geo_id_cols = get_geo_id_columns("zip")

        def build_params(codes: list[str]) -> dict:
            return build_zip_params(codes, state_fips, self.api_key)

        batch_frames = self._fetch_variables_resilient(variable_codes, build_params, geo_id_cols)
        if not batch_frames:
            return None

        df = self._merge_batches(batch_frames, geo_id_cols)
        df = self._rename_and_clean(df, code_map)
        df["state_fips"] = state_fips
        df["state"] = state_abbr
        return df

    # ------------------------------------------------------------------
    # County / State: single national call
    # ------------------------------------------------------------------

    def _fetch_geography(
        self,
        variable_codes: list[str],
        code_map: dict[str, str],
        params_fn,
        scope: str = "",  # noqa: kept for call-site clarity
    ) -> pd.DataFrame:
        geo_id_cols = get_geo_id_columns(self.geo_level)

        def build_params(codes: list[str]) -> dict:
            return params_fn(codes, self.api_key)

        batch_frames = self._fetch_variables_resilient(variable_codes, build_params, geo_id_cols)
        if not batch_frames:
            return pd.DataFrame()

        df = self._merge_batches(batch_frames, geo_id_cols)
        return self._rename_and_clean(df, code_map)

    # ------------------------------------------------------------------
    # HTTP + retry
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request(self, url: str, params: dict) -> list[list] | None:
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 204:
                return None
            if resp.status_code == 429:
                raise requests.ConnectionError("Rate limited (429)")
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or len(data) < 2:
                logger.warning(f"Empty response from Census API: {url}")
                return None
            return data
        except requests.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}: {e}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from Census API: {url}")
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: list[list], geo_id_cols: list[str]
    ) -> pd.DataFrame | None:
        if not raw or len(raw) < 2:
            return None
        headers = raw[0]
        rows = raw[1:]
        df = pd.DataFrame(rows, columns=headers)
        return df

    def _merge_batches(
        self, frames: list[pd.DataFrame], geo_id_cols: list[str]
    ) -> pd.DataFrame:
        if len(frames) == 1:
            return frames[0]
        merge_on = ["NAME"] + geo_id_cols
        result = frames[0]
        for frame in frames[1:]:
            # keep only columns that don't already exist (avoid duplicates)
            new_cols = [c for c in frame.columns if c not in result.columns or c in merge_on]
            result = result.merge(frame[new_cols], on=merge_on, how="outer")
        return result

    def _rename_and_clean(
        self, df: pd.DataFrame, code_map: dict[str, str]
    ) -> pd.DataFrame:
        df = df.rename(columns=code_map)
        df = df.rename(columns={"zip code tabulation area": "zip_code"})

        numeric_cols = [c for c in df.columns if c not in ("NAME", "zip_code", "state", "state_fips", "county")]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Replace Census sentinel values with NaN
            df[col] = df[col].where(~df[col].isin(CENSUS_SENTINEL_VALUES))

        return df

    # ------------------------------------------------------------------
    # Resilient variable fetching (binary-search on 400 errors)
    # ------------------------------------------------------------------

    def _fetch_variables_resilient(
        self,
        codes: list[str],
        build_params: Callable[[list[str]], dict],
        geo_id_cols: list[str],
    ) -> list[pd.DataFrame]:
        """Split codes into batches; on 400, subdivide to isolate bad variables."""
        frames: list[pd.DataFrame] = []
        for batch in self._split_batches(codes):
            df = self._fetch_batch_resilient(batch, build_params, geo_id_cols)
            if df is not None:
                frames.append(df)
        return frames

    def _fetch_batch_resilient(
        self,
        codes: list[str],
        build_params: Callable[[list[str]], dict],
        geo_id_cols: list[str],
    ) -> pd.DataFrame | None:
        """Try a batch; on failure split in half and retry each side recursively."""
        if not codes:
            return None

        params = build_params(codes)
        cache_key = self._cache_key(params)
        raw = self._load_cache(cache_key)
        if raw is None:
            raw = self._request(self.base_url, params)
            if raw is not None:
                self._save_cache(cache_key, raw)

        if raw is not None:
            return self._parse_response(raw, geo_id_cols)

        # Single variable and it failed — skip it permanently
        if len(codes) == 1:
            logger.debug(f"Variable {codes[0]} unavailable at {self.geo_level} level — skipping.")
            return None

        # Binary split: find which half has the bad variable
        mid = len(codes) // 2
        left = self._fetch_batch_resilient(codes[:mid], build_params, geo_id_cols)
        right = self._fetch_batch_resilient(codes[mid:], build_params, geo_id_cols)

        if left is None:
            return right
        if right is None:
            return left

        merge_on = ["NAME"] + geo_id_cols
        new_cols = [c for c in right.columns if c not in left.columns or c in merge_on]
        return left.merge(right[new_cols], on=merge_on, how="outer")

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _split_batches(self, codes: list[str]) -> list[list[str]]:
        size = self.var_batch_size
        return [codes[i: i + size] for i in range(0, len(codes), size)]

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _cache_key(self, params: dict) -> str:
        payload = json.dumps({k: v for k, v in sorted(params.items()) if k != "key"})
        return hashlib.md5(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.raw_dir / f"{key}.json"

    def _load_cache(self, key: str) -> list[list] | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                path.unlink(missing_ok=True)
        return None

    def _save_cache(self, key: str, data: list[list]) -> None:
        path = self._cache_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
