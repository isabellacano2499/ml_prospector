"""
Processes downloaded state license files (CSV/Excel) from TREC (TX) and DBPR (FL).
Filters to agents in our target cities/counties and returns a clean DataFrame.
"""
import pandas as pd
from pathlib import Path
from loguru import logger
from state_licenses.zip_to_geo import zip_list_to_cities

DATA_DIR = Path(__file__).parent.parent / "data"

# Possible column name variants across TREC and DBPR exports
NAME_COLS       = ["Name", "Full Name", "FullName", "Licensee Name", "NAME", "FULL_NAME", "QUALIFIER_NAME"]
FIRST_COLS      = ["First Name", "FIRST_NAME", "FirstName", "FNAME"]
LAST_COLS       = ["Last Name", "LAST_NAME", "LastName", "LNAME"]
LICENSE_COLS    = ["License Number", "LICENSE_NUMBER", "LicNbr", "LIC_NBR", "LICENSE_NBR", "License #"]
TYPE_COLS       = ["License Type", "LICENSE_TYPE", "LicType", "LIC_TYPE", "TYPE"]
STATUS_COLS     = ["Status", "LICENSE_STATUS", "LicStatus", "LIC_STATUS", "CURRENT_STATUS"]
CITY_COLS       = ["City", "CITY", "Mailing City", "MAIL_CITY", "BUS_CITY"]
COUNTY_COLS     = ["County", "COUNTY", "Mailing County", "MAIL_COUNTY"]
AGENCY_COLS     = ["Sponsor", "Sponsoring Broker", "SPONSOR", "Company", "COMPANY", "Business Name",
                   "BUSINESS_NAME", "Broker Name", "BROKER_NAME"]


def load_and_filter(filepath: str | Path, target_cities: set[str] = None,
                    target_counties: set[str] = None, state: str = "") -> pd.DataFrame:
    """Load a license file and filter to target cities/counties."""
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()

    logger.info(f"Loading {path.name}...")
    df = _read_file(path)
    if df.empty:
        logger.warning(f"Empty or unreadable file: {path}")
        return pd.DataFrame()

    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.debug(f"  Columns: {list(df.columns)}")

    df = _normalize_columns(df, state)
    df = _filter_active(df)
    df = _filter_real_estate(df)

    if target_cities or target_counties:
        df = _filter_geography(df, target_cities or set(), target_counties or set())

    df = _clean_names(df)
    logger.info(f"  After filtering: {len(df):,} agents")
    return df


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path, dtype=str)
        elif suffix == ".csv":
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
                except UnicodeDecodeError:
                    continue
        elif suffix == ".txt":
            for sep in ("\t", "|", ","):
                try:
                    df = pd.read_csv(path, dtype=str, sep=sep, encoding="latin-1", low_memory=False)
                    if len(df.columns) > 2:
                        return df
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Read error: {e}")
    return pd.DataFrame()


def _normalize_columns(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Rename varied column names to a standard schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def pick(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # Case-insensitive fallback
        for c in candidates:
            matches = [col for col in df.columns if col.lower() == c.lower()]
            if matches:
                return matches[0]
        return None

    name_col = pick(NAME_COLS)
    first_col = pick(FIRST_COLS)
    last_col = pick(LAST_COLS)

    if name_col:
        df["name"] = df[name_col].fillna("").str.strip()
    elif first_col and last_col:
        df["name"] = (df[first_col].fillna("") + " " + df[last_col].fillna("")).str.strip()
    else:
        df["name"] = ""

    col_map = {
        "license_number": pick(LICENSE_COLS),
        "license_type":   pick(TYPE_COLS),
        "license_status": pick(STATUS_COLS),
        "city":           pick(CITY_COLS),
        "county":         pick(COUNTY_COLS),
        "agency":         pick(AGENCY_COLS),
    }
    for new_col, old_col in col_map.items():
        if old_col:
            df[new_col] = df[old_col].fillna("").str.strip()
        else:
            df[new_col] = ""

    df["state"] = state
    return df


def _filter_active(df: pd.DataFrame) -> pd.DataFrame:
    if "license_status" not in df.columns or df["license_status"].eq("").all():
        return df
    mask = df["license_status"].str.lower().str.contains("active|current|a$", na=False)
    return df[mask]


def _filter_real_estate(df: pd.DataFrame) -> pd.DataFrame:
    if "license_type" not in df.columns or df["license_type"].eq("").all():
        return df
    mask = df["license_type"].str.lower().str.contains(
        "sales|broker|agent|sl$|bk$|re-sl|re-bk", na=False
    )
    # If nothing matches, return all (might be a pre-filtered file)
    return df[mask] if mask.any() else df


def _filter_geography(df: pd.DataFrame, cities: set[str], counties: set[str]) -> pd.DataFrame:
    cities_lower  = {c.lower() for c in cities}
    counties_lower = {c.lower() for c in counties}

    city_mask   = df.get("city", pd.Series(dtype=str)).str.lower().isin(cities_lower)
    county_mask = df.get("county", pd.Series(dtype=str)).str.lower().str.contains(
        "|".join(counties_lower), na=False
    ) if counties_lower else pd.Series(False, index=df.index)

    combined = city_mask | county_mask
    if combined.any():
        return df[combined]
    logger.warning("Geography filter matched 0 rows — returning all agents (check city/county names)")
    return df


def _clean_names(df: pd.DataFrame) -> pd.DataFrame:
    def fmt(name: str) -> str:
        name = name.strip()
        if "," in name:
            parts = name.split(",", 1)
            return f"{parts[1].strip().title()} {parts[0].strip().title()}"
        return name.title()
    df["name"] = df["name"].apply(fmt)
    return df


def build_target_geo(zip_list: list[str]) -> tuple[set[str], set[str]]:
    """Return (cities, counties) sets for a list of target ZIPs."""
    geo = zip_list_to_cities(zip_list)
    cities  = {city   for city, county in geo.values() if city}
    counties = {county for city, county in geo.values() if county}
    return cities, counties
