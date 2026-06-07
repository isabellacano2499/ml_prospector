"""
Census ACS API endpoint builders by geographic level.
Reference: https://api.census.gov/data/{year}/acs/acs5/examples.html
"""

BASE_URL = "https://api.census.gov/data/{year}/acs/{dataset}"

# FIPS state codes for iterating ZCTAs state-by-state
# (ZCTAs queried nationally can timeout; per-state is more reliable)
STATE_FIPS: list[str] = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56", "72",
]

STATE_FIPS_NAMES: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR",
}


def build_base_url(year: int, dataset: str = "acs5") -> str:
    return BASE_URL.format(year=year, dataset=dataset)


def build_zip_params(variables: list[str], state_fips: str, api_key: str) -> dict:
    """Build params for ZCTA queries.

    ZCTAs are NOT nested within states in the Census API hierarchy —
    they cross state boundaries — so the ``in=state:`` parameter is
    invalid and triggers a 400 error.  We query all ZCTAs nationally.
    The *state_fips* argument is kept for call-site compatibility but
    is not sent to the API.
    """
    return {
        "get": ",".join(["NAME"] + variables),
        "for": "zip code tabulation area:*",
        "key": api_key,
    }


def build_county_params(variables: list[str], api_key: str) -> dict:
    return {
        "get": ",".join(["NAME"] + variables),
        "for": "county:*",
        "in": "state:*",
        "key": api_key,
    }


def build_state_params(variables: list[str], api_key: str) -> dict:
    return {
        "get": ",".join(["NAME"] + variables),
        "for": "state:*",
        "key": api_key,
    }


def get_geo_id_columns(geographic_level: str) -> list[str]:
    """Return the geography identifier columns returned by the Census API."""
    if geographic_level == "zip":
        return ["zip code tabulation area"]
    elif geographic_level == "county":
        return ["state", "county"]
    elif geographic_level == "state":
        return ["state"]
    return []
