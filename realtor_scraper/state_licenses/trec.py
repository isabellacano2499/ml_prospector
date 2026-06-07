"""
Texas Real Estate Commission (TREC) license holder search.
Public government database — no bot protection.
URL: https://www.trec.texas.gov/apps/license-holder-search/
"""
import time
import random
import requests
from bs4 import BeautifulSoup
from loguru import logger

SEARCH_URL = "https://www.trec.texas.gov/apps/license-holder-search/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

LICENSE_TYPES = ["Sales Agent", "Broker"]


def search_trec_by_city(city: str, session: requests.Session) -> list[dict]:
    """Search TREC for all active real estate agents/brokers in a city."""
    agents = []
    for lic_type in LICENSE_TYPES:
        results = _search(city=city, license_type=lic_type, session=session)
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


def _search(city: str, license_type: str, session: requests.Session) -> list[dict]:
    params = {
        "real_fname": "",
        "real_lname": "",
        "real_lic_no": "",
        "real_city": city,
        "real_county": "",
        "real_lic_type": license_type,
        "real_status": "Active",
        "Submit": "Search",
    }
    try:
        resp = session.get(SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"TREC request failed for city={city}: {e}")
        return []

    return _parse_trec(resp.text, city, license_type)


def _parse_trec(html: str, city: str, license_type: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    agents = []

    # TREC results are in an HTML table
    table = soup.find("table", {"id": "license-holder-table"}) or soup.find("table")
    if not table:
        logger.debug(f"TREC: no table found for city={city}")
        return []

    rows = table.find_all("tr")
    headers = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]

        if not texts:
            continue

        # Detect header row
        if any(h in texts[0].lower() for h in ("name", "license", "type", "status")):
            headers = [t.lower().replace(" ", "_") for t in texts]
            continue

        if not headers or len(texts) < 2:
            continue

        row_data = dict(zip(headers, texts))

        name = row_data.get("name") or texts[0]
        lic_no = row_data.get("license_#") or row_data.get("license_number") or row_data.get("lic_#") or ""
        status = row_data.get("status") or ""
        exp_date = row_data.get("expiration") or row_data.get("expiration_date") or ""
        sponsor = row_data.get("sponsoring_broker") or ""
        address_city = row_data.get("city") or city

        if not name or name.lower() in ("name", ""):
            continue

        # Only active licenses
        if status and "active" not in status.lower():
            continue

        agents.append({
            "name": _format_name(name),
            "license_number": lic_no,
            "license_type": license_type,
            "license_status": status,
            "license_expiration": exp_date,
            "sponsoring_broker": sponsor,
            "city": address_city,
            "state": "TX",
        })

    logger.debug(f"TREC city={city} type={license_type}: {len(agents)} agents parsed")
    return agents


def _format_name(raw: str) -> str:
    """Convert 'GARCIA, JUAN CARLOS' → 'Juan Carlos Garcia'"""
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
