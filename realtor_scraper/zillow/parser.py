import json
import re
from bs4 import BeautifulSoup
from loguru import logger


def _extract_next_data(html: str) -> dict | None:
    """Pull pre-rendered JSON from Next.js __NEXT_DATA__ script tag."""
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if tag:
        try:
            return json.loads(tag.string)
        except Exception:
            pass
    return None


def _extract_apollo_state(html: str) -> dict | None:
    """Pull Apollo cache from inline script (older Zillow pages)."""
    match = re.search(r'window\.__APOLLO_STATE__\s*=\s*(\{.*?\});', html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


def parse_agents_from_html(html: str, zip_code: str) -> list[dict]:
    agents = []

    data = _extract_next_data(html)
    if data:
        agents = _parse_next_data(data, zip_code)
        if agents:
            return agents

    data = _extract_apollo_state(html)
    if data:
        agents = _parse_apollo_state(data, zip_code)
        if agents:
            return agents

    # Fallback: parse visible HTML cards
    agents = _parse_html_cards(html, zip_code)
    return agents


def _parse_next_data(data: dict, zip_code: str) -> list[dict]:
    agents = []
    try:
        # Navigate common Next.js Zillow structures
        props = data.get("props", {}).get("pageProps", {})
        agent_list = (
            props.get("agents")
            or props.get("agentSearchResults", {}).get("agents")
            or props.get("searchResults", {}).get("agents")
            or []
        )
        for a in agent_list:
            agent = _normalize_agent(a, zip_code)
            if agent:
                agents.append(agent)
    except Exception as e:
        logger.debug(f"__NEXT_DATA__ parse error for {zip_code}: {e}")
    return agents


def _parse_apollo_state(data: dict, zip_code: str) -> list[dict]:
    agents = []
    try:
        for key, val in data.items():
            if isinstance(val, dict) and val.get("__typename") in ("Agent", "ProProfile"):
                agent = _normalize_agent(val, zip_code)
                if agent:
                    agents.append(agent)
    except Exception as e:
        logger.debug(f"Apollo state parse error for {zip_code}: {e}")
    return agents


def _parse_html_cards(html: str, zip_code: str) -> list[dict]:
    """Last-resort HTML card extraction."""
    agents = []
    soup = BeautifulSoup(html, "lxml")

    cards = (
        soup.select("article[data-test='agent-card']")
        or soup.select("div.agent-card")
        or soup.select("[class*='AgentCard']")
        or soup.select("[class*='agent-card']")
    )

    for card in cards:
        try:
            name = _text(card, ["[data-test='agent-name']", "h2", "h3", ".agent-name", "[class*='agentName']"])
            agency = _text(card, ["[data-test='agent-business-name']", ".agent-business", "[class*='businessName']"])
            rating_raw = _text(card, ["[data-test='agent-rating']", ".agent-rating", "[class*='rating']"])
            reviews_raw = _text(card, ["[data-test='agent-reviews']", ".agent-reviews", "[class*='reviews']"])
            link_tag = card.find("a", href=re.compile(r"/profile/"))
            profile_url = f"https://www.zillow.com{link_tag['href']}" if link_tag else None

            full_text = card.get_text(" ", strip=True).lower()
            speaks_spanish = any(w in full_text for w in ("spanish", "español", "habla español", "se habla"))

            rating = _parse_float(rating_raw)
            reviews = _parse_int(reviews_raw)

            if not name:
                continue

            agents.append({
                "zip_code": zip_code,
                "name": name,
                "agency": agency,
                "rating": rating,
                "review_count": reviews,
                "speaks_spanish": speaks_spanish,
                "years_experience": None,
                "recent_sales": None,
                "profile_url": profile_url,
                "phone": None,
            })
        except Exception:
            continue

    return agents


def _normalize_agent(raw: dict, zip_code: str) -> dict | None:
    try:
        name = raw.get("fullName") or raw.get("name") or raw.get("displayName")
        if not name:
            return None

        agency = (
            raw.get("businessName")
            or raw.get("brokerageName")
            or raw.get("companyName")
        )

        rating = _parse_float(raw.get("rating") or raw.get("averageRating") or raw.get("reviewAvgRating"))
        reviews = _parse_int(raw.get("reviewCount") or raw.get("totalReviews") or raw.get("numReviews"))
        years = _parse_int(raw.get("yearsExperience") or raw.get("experienceYears"))
        sales = _parse_int(raw.get("recentSales") or raw.get("pastYearSales") or raw.get("soldCount"))

        langs = raw.get("languages") or raw.get("spokenLanguages") or []
        if isinstance(langs, list):
            lang_text = " ".join(str(l).lower() for l in langs)
        else:
            lang_text = str(langs).lower()
        speaks_spanish = "spanish" in lang_text or "español" in lang_text

        slug = raw.get("profileUrl") or raw.get("pageUrl") or raw.get("url") or ""
        profile_url = slug if slug.startswith("http") else f"https://www.zillow.com{slug}" if slug else None

        phone = raw.get("phone") or raw.get("phoneNumber") or raw.get("mobilePhone")

        return {
            "zip_code": zip_code,
            "name": name,
            "agency": agency,
            "rating": rating,
            "review_count": reviews,
            "speaks_spanish": speaks_spanish,
            "years_experience": years,
            "recent_sales": sales,
            "profile_url": profile_url,
            "phone": phone,
        }
    except Exception:
        return None


def _text(tag, selectors: list) -> str | None:
    for sel in selectors:
        el = tag.select_one(sel)
        if el:
            return el.get_text(strip=True)
    return None


def _parse_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except Exception:
        return None


def _parse_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(re.sub(r"[^\d]", "", str(val)))
    except Exception:
        return None
