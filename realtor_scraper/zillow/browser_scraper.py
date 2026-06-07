import time
import random
import json
import re
from pathlib import Path
from loguru import logger
from playwright.sync_api import sync_playwright, Page, TimeoutError as PwTimeout

from config import MAX_AGENTS_PER_ZIP, DELAY_MIN, DELAY_MAX, OUTPUT_DIR

SEARCH_URL = "https://www.zillow.com/professionals/real-estate-agent-reviews/"

DEBUG_DIR = OUTPUT_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)


def _make_browser_context(playwright, headless: bool = True):
    browser = playwright.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    )
    context = browser.new_context(
        viewport={"width": 1366, "height": 768},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        java_script_enabled=True,
    )
    # Hide automation fingerprint
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en', 'es'] });
    """)
    return browser, context


def _extract_agents_from_page(page: Page, zip_code: str) -> list[dict]:
    """Try multiple extraction strategies."""

    # Strategy 1: intercept __NEXT_DATA__ JSON
    agents = _extract_from_next_data(page, zip_code)
    if agents:
        return agents

    # Strategy 2: parse visible DOM cards
    agents = _extract_from_dom(page, zip_code)
    if agents:
        return agents

    # Strategy 3: look for JSON in script tags
    agents = _extract_from_scripts(page, zip_code)
    return agents


def _extract_from_next_data(page: Page, zip_code: str) -> list[dict]:
    try:
        raw = page.evaluate("""() => {
            const el = document.getElementById('__NEXT_DATA__');
            return el ? el.textContent : null;
        }""")
        if not raw:
            return []
        data = json.loads(raw)
        props = data.get("props", {}).get("pageProps", {})
        agent_list = (
            props.get("agents")
            or props.get("agentSearchResults", {}).get("agents")
            or []
        )
        return [_normalize(a, zip_code) for a in agent_list if _normalize(a, zip_code)]
    except Exception as e:
        logger.debug(f"__NEXT_DATA__ extraction failed: {e}")
        return []


def _extract_from_dom(page: Page, zip_code: str) -> list[dict]:
    agents = []
    selectors = [
        "article[data-test='agent-card']",
        "div[data-test='agent-card']",
        "[class*='AgentCard']",
        "[class*='agent-card']",
        "li[class*='agent']",
    ]
    cards = []
    for sel in selectors:
        try:
            cards = page.query_selector_all(sel)
            if cards:
                logger.debug(f"ZIP {zip_code}: found {len(cards)} cards with selector '{sel}'")
                break
        except Exception:
            continue

    for card in cards:
        try:
            name = _card_text(card, [
                "[data-test='agent-name']", "h2", "h3", "h4",
                "[class*='agentName']", "[class*='agent-name']",
            ])
            if not name:
                continue

            agency = _card_text(card, [
                "[data-test='agent-business-name']",
                "[class*='businessName']", "[class*='brokerage']",
            ])
            rating_raw = _card_text(card, [
                "[data-test='agent-rating']", "[class*='rating']", "[aria-label*='rating']",
            ])
            reviews_raw = _card_text(card, [
                "[data-test='agent-reviews-count']", "[class*='reviews']",
                "[class*='reviewCount']",
            ])
            sales_raw = _card_text(card, [
                "[data-test='agent-sales']", "[class*='sales']", "[class*='sold']",
            ])
            experience_raw = _card_text(card, [
                "[class*='experience']", "[class*='years']",
            ])

            full_text = (card.inner_text() or "").lower()
            speaks_spanish = any(w in full_text for w in (
                "spanish", "español", "habla español", "se habla", "hablan español",
            ))

            link = card.query_selector("a[href*='/profile/']") or card.query_selector("a[href*='realtor']")
            profile_url = None
            if link:
                href = link.get_attribute("href") or ""
                profile_url = href if href.startswith("http") else f"https://www.zillow.com{href}"

            agents.append({
                "zip_code": zip_code,
                "name": name.strip(),
                "agency": agency,
                "rating": _parse_float(rating_raw),
                "review_count": _parse_int(reviews_raw),
                "speaks_spanish": speaks_spanish,
                "years_experience": _parse_int(experience_raw),
                "recent_sales": _parse_int(sales_raw),
                "profile_url": profile_url,
                "phone": None,
            })
        except Exception as e:
            logger.debug(f"Card parse error: {e}")
            continue

    return agents


def _extract_from_scripts(page: Page, zip_code: str) -> list[dict]:
    try:
        scripts = page.evaluate("""() => {
            return Array.from(document.scripts)
                .map(s => s.textContent)
                .filter(t => t && t.includes('agent') && t.includes('rating'));
        }""")
        for script in (scripts or []):
            match = re.search(r'"agents"\s*:\s*(\[.*?\])', script, re.DOTALL)
            if match:
                agent_list = json.loads(match.group(1))
                result = [_normalize(a, zip_code) for a in agent_list if _normalize(a, zip_code)]
                if result:
                    return result
    except Exception as e:
        logger.debug(f"Script extraction failed: {e}")
    return []


def scrape_zip_browser(zip_code: str, page: Page) -> list[dict]:
    url = f"{SEARCH_URL}?zip={zip_code}&sort=rating"

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    except PwTimeout:
        logger.warning(f"ZIP {zip_code}: page load timeout")
        return []

    # Random human-like pause after load
    time.sleep(random.uniform(2.0, 4.0))

    # Check for bot wall / CAPTCHA
    title = page.title().lower()
    content = (page.content() or "").lower()
    if "captcha" in content or "robot" in content or "blocked" in title:
        logger.warning(f"ZIP {zip_code}: bot detection triggered — saving screenshot")
        page.screenshot(path=str(DEBUG_DIR / f"blocked_{zip_code}.png"))
        return []

    # Wait for agent content
    try:
        page.wait_for_selector(
            "article, [class*='AgentCard'], [class*='agent-card'], [data-test='agent-card']",
            timeout=15_000,
        )
    except PwTimeout:
        logger.warning(f"ZIP {zip_code}: no agent cards appeared — saving screenshot")
        page.screenshot(path=str(DEBUG_DIR / f"empty_{zip_code}.png"))

    agents = _extract_agents_from_page(page, zip_code)
    logger.debug(f"ZIP {zip_code}: extracted {len(agents)} agents from page")
    return agents[:MAX_AGENTS_PER_ZIP]


def scrape_zips_browser(zip_list: list[str], headless: bool = True, on_progress=None) -> list[dict]:
    all_agents = []

    with sync_playwright() as p:
        browser, context = _make_browser_context(p, headless=headless)
        page = context.new_page()

        # Warm up: visit Zillow homepage first (looks more natural)
        logger.info("Warming up browser on Zillow homepage...")
        try:
            page.goto("https://www.zillow.com", wait_until="domcontentloaded", timeout=20_000)
            time.sleep(random.uniform(2.0, 4.0))
        except Exception:
            pass

        for i, zip_code in enumerate(zip_list):
            logger.info(f"[{i+1}/{len(zip_list)}] Scraping ZIP {zip_code}...")
            agents = scrape_zip_browser(zip_code, page)
            all_agents.extend(agents)
            logger.info(f"  → {len(agents)} agents for ZIP {zip_code}")

            if on_progress:
                on_progress(zip_code, agents)

            if i < len(zip_list) - 1:
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        browser.close()

    return all_agents


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize(raw: dict, zip_code: str) -> dict | None:
    try:
        name = raw.get("fullName") or raw.get("name") or raw.get("displayName")
        if not name:
            return None
        langs = raw.get("languages") or raw.get("spokenLanguages") or []
        if isinstance(langs, list):
            lang_text = " ".join(str(l).lower() for l in langs)
        else:
            lang_text = str(langs).lower()
        speaks_spanish = "spanish" in lang_text or "español" in lang_text
        slug = raw.get("profileUrl") or raw.get("pageUrl") or raw.get("url") or ""
        profile_url = slug if slug.startswith("http") else f"https://www.zillow.com{slug}" if slug else None
        return {
            "zip_code": zip_code,
            "name": name,
            "agency": raw.get("businessName") or raw.get("brokerageName"),
            "rating": _parse_float(raw.get("rating") or raw.get("averageRating")),
            "review_count": _parse_int(raw.get("reviewCount") or raw.get("totalReviews")),
            "speaks_spanish": speaks_spanish,
            "years_experience": _parse_int(raw.get("yearsExperience")),
            "recent_sales": _parse_int(raw.get("recentSales") or raw.get("pastYearSales")),
            "profile_url": profile_url,
            "phone": raw.get("phone") or raw.get("phoneNumber"),
        }
    except Exception:
        return None


def _card_text(card, selectors: list) -> str | None:
    for sel in selectors:
        try:
            el = card.query_selector(sel)
            if el:
                t = el.inner_text()
                if t:
                    return t.strip()
        except Exception:
            continue
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
