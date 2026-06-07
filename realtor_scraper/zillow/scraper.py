import time
import random
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import HEADERS, DELAY_MIN, DELAY_MAX, MAX_AGENTS_PER_ZIP
from zillow.parser import parse_agents_from_html

SEARCH_URL = "https://www.zillow.com/professionals/real-estate-agent-reviews/"


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _get(session: requests.Session, url: str, params: dict) -> requests.Response:
    resp = session.get(url, params=params, timeout=20)
    if resp.status_code == 429:
        wait = random.uniform(30, 60)
        logger.warning(f"Rate limited (429). Waiting {wait:.0f}s...")
        time.sleep(wait)
        resp.raise_for_status()
    resp.raise_for_status()
    return resp


def scrape_zip(zip_code: str, session: requests.Session | None = None) -> list[dict]:
    """Return list of agent dicts for a given ZIP code."""
    if session is None:
        session = _make_session()

    agents = []
    page = 1

    while len(agents) < MAX_AGENTS_PER_ZIP:
        params = {
            "zip": zip_code,
            "sort": "rating",
            "page": page,
        }

        try:
            resp = _get(session, SEARCH_URL, params)
        except requests.HTTPError as e:
            logger.warning(f"ZIP {zip_code} page {page}: HTTP {e.response.status_code}, stopping")
            break
        except requests.RequestException as e:
            logger.warning(f"ZIP {zip_code} page {page}: {e}, stopping")
            break

        page_agents = parse_agents_from_html(resp.text, zip_code)

        if not page_agents:
            if page == 1:
                logger.warning(f"ZIP {zip_code}: no agents parsed (may need Selenium)")
            break

        agents.extend(page_agents)
        logger.debug(f"ZIP {zip_code} page {page}: {len(page_agents)} agents found")

        if len(page_agents) < 10:
            break

        page += 1
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    return agents[:MAX_AGENTS_PER_ZIP]


def scrape_zips(zip_list: list[str], on_progress=None) -> list[dict]:
    """Scrape all ZIPs. Calls on_progress(zip_code, agents) after each ZIP."""
    session = _make_session()
    all_agents = []

    for i, zip_code in enumerate(zip_list):
        logger.info(f"[{i+1}/{len(zip_list)}] Scraping ZIP {zip_code}...")
        agents = scrape_zip(zip_code, session)
        all_agents.extend(agents)

        if on_progress:
            on_progress(zip_code, agents)

        logger.info(f"  → {len(agents)} agents found for {zip_code}")

        if i < len(zip_list) - 1:
            delay = random.uniform(DELAY_MIN, DELAY_MAX)
            time.sleep(delay)

    return all_agents
