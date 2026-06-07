"""
Step 1: Scrape Realtor.com agent search by ZIP to get agent names + basic info.
Realtor.com is much less aggressively protected than Zillow.
URL: https://www.realtor.com/realestateagents/{zip}/
"""
import re
import time
import random
import json
from loguru import logger
from playwright.sync_api import Page, TimeoutError as PwTimeout

from config import MAX_AGENTS_PER_ZIP, OUTPUT_DIR

DEBUG_DIR = OUTPUT_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

SEARCH_URL = "https://www.realtor.com/realestateagents/{zip}/"


def scrape_zip_realtor(zip_code: str, page: Page, spanish_only: bool = False) -> list[dict]:
    """Get agent list from Realtor.com for a ZIP code."""
    url = SEARCH_URL.format(zip=zip_code)

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    except PwTimeout:
        logger.warning(f"Realtor.com timeout for ZIP {zip_code}")
        return []

    time.sleep(random.uniform(2.0, 4.0))

    content = (page.content() or "").lower()
    if "captcha" in content or "blocked" in content or "robot" in content:
        logger.warning(f"Bot wall on Realtor.com ZIP {zip_code}")
        page.screenshot(path=str(DEBUG_DIR / f"realtor_blocked_{zip_code}.png"))
        return []

    # Wait for agent cards
    try:
        page.wait_for_selector(
            "[class*='agent-list-card'], [data-testid='agent-card'], [class*='AgentCard'], li[class*='agent']",
            timeout=15_000,
        )
    except PwTimeout:
        logger.warning(f"Realtor.com ZIP {zip_code}: no cards loaded — saving screenshot")
        page.screenshot(path=str(DEBUG_DIR / f"realtor_empty_{zip_code}.png"))

    # Try to get data from embedded JSON first
    agents = _from_next_data(page, zip_code)
    if not agents:
        agents = _from_dom(page, zip_code)

    if spanish_only:
        agents = [a for a in agents if a.get("speaks_spanish")]

    logger.debug(f"Realtor.com ZIP {zip_code}: {len(agents)} agents")
    return agents[:MAX_AGENTS_PER_ZIP]


def _from_next_data(page: Page, zip_code: str) -> list[dict]:
    try:
        raw = page.evaluate("""() => {
            const el = document.getElementById('__NEXT_DATA__');
            return el ? el.textContent : null;
        }""")
        if not raw:
            return []
        data = json.loads(raw)
        # Search common locations for agent list
        props = data.get("props", {}).get("pageProps", {})
        agent_list = (
            props.get("agents")
            or props.get("agentList")
            or props.get("agentResults", {}).get("agents")
            or props.get("results")
            or []
        )
        return [a for a in (_parse_realtor_agent(x, zip_code) for x in agent_list) if a]
    except Exception as e:
        logger.debug(f"Realtor.com __NEXT_DATA__ failed: {e}")
        return []


def _from_dom(page: Page, zip_code: str) -> list[dict]:
    agents = []
    card_selectors = [
        "[data-testid='agent-card']",
        "[class*='agent-list-card']",
        "[class*='AgentCard']",
        "li[class*='agent']",
        "article[class*='agent']",
    ]

    cards = []
    for sel in card_selectors:
        try:
            cards = page.query_selector_all(sel)
            if cards:
                break
        except Exception:
            continue

    for card in cards:
        try:
            def get(selectors):
                for s in selectors:
                    try:
                        el = card.query_selector(s)
                        if el:
                            t = (el.inner_text() or "").strip()
                            if t:
                                return t
                    except Exception:
                        continue
                return None

            name = get([
                "[data-testid='agent-name']", "h2", "h3",
                "[class*='agentName']", "[class*='agent-name']",
            ])
            if not name:
                continue

            agency = get([
                "[data-testid='agent-office']", "[class*='office']",
                "[class*='brokerage']", "[class*='company']",
            ])
            phone = get([
                "[data-testid='agent-phone']", "a[href^='tel:']",
                "[class*='phone']",
            ])
            rating_raw = get(["[class*='rating']", "[aria-label*='rating']"])
            reviews_raw = get(["[class*='review']", "[class*='Review']"])

            full_text = (card.inner_text() or "").lower()
            speaks_spanish = any(w in full_text for w in (
                "spanish", "español", "habla español",
            ))

            # Realtor.com profile link
            link = card.query_selector("a[href*='/realestateagents/']")
            profile_url = None
            if link:
                href = link.get_attribute("href") or ""
                profile_url = href if href.startswith("http") else f"https://www.realtor.com{href}"

            agents.append({
                "zip_code": zip_code,
                "name": name.strip(),
                "agency": agency,
                "phone": _clean_phone(phone),
                "rating": _parse_float(rating_raw),
                "review_count": _parse_int(reviews_raw),
                "speaks_spanish": speaks_spanish,
                "realtor_profile_url": profile_url,
            })
        except Exception as e:
            logger.debug(f"Card parse error: {e}")

    return agents


def _parse_realtor_agent(raw: dict, zip_code: str) -> dict | None:
    try:
        name = (
            raw.get("full_name") or raw.get("fullName")
            or raw.get("name") or raw.get("agent_name")
        )
        if not name:
            return None

        langs = raw.get("languages") or raw.get("spoken_languages") or []
        if isinstance(langs, list):
            lang_text = " ".join(str(l).lower() for l in langs)
        else:
            lang_text = str(langs).lower()
        speaks_spanish = "spanish" in lang_text or "español" in lang_text

        slug = raw.get("permalink") or raw.get("profile_url") or raw.get("href") or ""
        profile_url = slug if slug.startswith("http") else f"https://www.realtor.com{slug}" if slug else None

        phone = (
            raw.get("phones", [{}])[0].get("number")
            if isinstance(raw.get("phones"), list) and raw.get("phones")
            else raw.get("phone") or raw.get("office_phone")
        )

        return {
            "zip_code": zip_code,
            "name": name,
            "agency": raw.get("broker_name") or raw.get("office_name") or raw.get("brokerage"),
            "phone": _clean_phone(phone),
            "rating": _parse_float(raw.get("ratings", {}).get("average") if isinstance(raw.get("ratings"), dict) else raw.get("rating")),
            "review_count": _parse_int(raw.get("ratings", {}).get("count") if isinstance(raw.get("ratings"), dict) else raw.get("review_count")),
            "speaks_spanish": speaks_spanish,
            "realtor_profile_url": profile_url,
        }
    except Exception:
        return None


def _clean_phone(val) -> str | None:
    if not val:
        return None
    digits = re.sub(r"[^\d]", "", str(val))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits[0] == "1":
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return str(val).strip() or None


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
