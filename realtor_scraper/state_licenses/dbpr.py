"""
Florida DBPR (Dept. of Business & Professional Regulation) license search.
Public government database — no bot protection.
URL: https://www.myfloridalicense.com/wl11.asp
License types: SL = Sales Associate, BK = Broker
"""
import time
import random
import requests
from bs4 import BeautifulSoup
from loguru import logger

SEARCH_URL = "https://www.myfloridalicense.com/wl11.asp"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
}

# FL county FIPS → county name mapping for target counties
FL_COUNTIES = {
    "Miami-Dade": "DADE",
    "Broward": "BROWARD",
    "Palm Beach": "PALM BEACH",
    "Hillsborough": "HILLSBOROUGH",
    "Orange": "ORANGE",
    "Osceola": "OSCEOLA",
    "Duval": "DUVAL",
    "Lee": "LEE",
    "Polk": "POLK",
    "Collier": "COLLIER",
}

LICENSE_TYPES = [
    ("SL", "Sales Associate"),
    ("BK", "Broker"),
]


def search_dbpr_by_county(county: str, session: requests.Session) -> list[dict]:
    """Search DBPR for active real estate licensees in a Florida county."""
    # Normalize county name to DBPR code
    dbpr_county = _normalize_county(county)
    if not dbpr_county:
        logger.debug(f"DBPR: unknown county '{county}', skipping")
        return []

    agents = []
    for lic_code, lic_name in LICENSE_TYPES:
        results = _search(dbpr_county, lic_code, lic_name, session)
        agents.extend(results)
        time.sleep(random.uniform(1.0, 2.5))

    # Deduplicate by license number
    seen = set()
    unique = []
    for a in agents:
        key = a.get("license_number") or a.get("name")
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def _search(county: str, lic_code: str, lic_name: str, session: requests.Session) -> list[dict]:
    # DBPR uses a POST form
    data = {
        "SearchTerm": "",
        "LicenseType": f"RE-{lic_code}",      # RE-SL or RE-BK
        "County": county,
        "City": "",
        "LicenseStatus": "Current,Active",
        "Search": "Search",
    }
    # Also try GET-based search
    params = {
        "mode": "0",
        "SID": "",
        "bsds": "",
        "county": county,
        "LicenseType": f"RE-{lic_code}",
        "status": "A",
    }

    try:
        # Try POST first
        resp = session.post(SEARCH_URL, data=data, timeout=30)
        if resp.status_code != 200 or "no records" in resp.text.lower():
            resp = session.get(SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"DBPR request failed for county={county}: {e}")
        return []

    return _parse_dbpr(resp.text, county, lic_name)


def _parse_dbpr(html: str, county: str, license_type: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    agents = []

    table = (
        soup.find("table", {"id": "searchResultsTable"})
        or soup.find("table", {"class": "resultsTable"})
        or soup.find("table")
    )
    if not table:
        logger.debug(f"DBPR: no table found for county={county}")
        return []

    rows = table.find_all("tr")
    headers = []

    for row in rows:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(separator=" ", strip=True) for c in cells]

        if not texts:
            continue

        if any(h in texts[0].lower() for h in ("name", "license", "type", "status", "qualifier")):
            headers = [t.lower().replace(" ", "_").replace("/", "_") for t in texts]
            continue

        if not headers or len(texts) < 2:
            continue

        row_data = dict(zip(headers, texts))
        name = row_data.get("name") or row_data.get("qualifier_name") or texts[0]
        lic_no = row_data.get("license_#") or row_data.get("license") or row_data.get("lic_#") or ""
        status = row_data.get("status") or row_data.get("license_status") or ""
        business = row_data.get("business_name") or row_data.get("dba") or ""
        address = row_data.get("mailing_address") or row_data.get("address") or ""

        if not name or name.lower() in ("name", "qualifier name", ""):
            continue

        agents.append({
            "name": _format_name(name),
            "license_number": lic_no,
            "license_type": license_type,
            "license_status": status,
            "agency": business,
            "address": address,
            "county": county,
            "state": "FL",
        })

    logger.debug(f"DBPR county={county} type={license_type}: {len(agents)} agents parsed")
    return agents


def _normalize_county(county: str) -> str:
    """Map uszipcode county name to DBPR county code."""
    c = county.upper().replace(" COUNTY", "").strip()
    # Handle Miami-Dade specifically
    if "DADE" in c or "MIAMI" in c:
        return "DADE"
    for name, code in FL_COUNTIES.items():
        if name.upper() in c or c in name.upper():
            return code
    return c  # Pass through as-is for unknown counties


def _format_name(raw: str) -> str:
    raw = raw.strip()
    if "," in raw:
        parts = raw.split(",", 1)
        last = parts[0].strip().title()
        first = parts[1].strip().title()
        return f"{first} {last}"
    return raw.title()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s
