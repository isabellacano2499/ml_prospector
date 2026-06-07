"""
Scrapes individual Zillow agent profile pages.
URL format: https://www.zillow.com/profile/{AgentSlug}
Uses patchright (patched Playwright) to bypass Cloudflare bot detection.
"""
import re
import sys
import time
import random
import json
from pathlib import Path
from loguru import logger

# Use patchright (drop-in replacement for playwright) to evade bot detection
try:
    from patchright.sync_api import Page, TimeoutError as PwTimeout, sync_playwright
    USING_PATCHRIGHT = True
except ImportError:
    from playwright.sync_api import Page, TimeoutError as PwTimeout, sync_playwright
    USING_PATCHRIGHT = False
    logger.warning("patchright not found — falling back to playwright (may be blocked by Zillow)")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

DEBUG_DIR = OUTPUT_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

# Persistent profile dir so cookies accumulate over time
PROFILE_DIR = OUTPUT_DIR / "browser_profile"
PROFILE_DIR.mkdir(exist_ok=True)

PROFILE_BASE = "https://www.zillow.com/profile/"
ZILLOW_HOME  = "https://www.zillow.com/"


def build_profile_url(name: str) -> str:
    """Convert 'Juan Garcia' → 'https://www.zillow.com/profile/JuanGarcia'"""
    slug = re.sub(r"[^a-zA-Z0-9]", "", name.title().replace(" ", ""))
    return f"{PROFILE_BASE}{slug}"


def make_browser_context(playwright_instance, headless: bool = True):
    """
    Create a persistent browser context that saves cookies between runs.
    patchright patches Chrome to evade Cloudflare's fingerprinting.
    """
    context = playwright_instance.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-infobars",
        ],
        viewport={"width": 1366, "height": 768},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/Chicago",
    )
    return context


def warmup_session(page: Page):
    """
    Visit Zillow homepage first to build up a real browsing session.
    Mimics a user who landed on Zillow before searching for an agent.
    """
    try:
        logger.info("Warming up Zillow session...")
        page.goto(ZILLOW_HOME, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(random.uniform(2.5, 4.5))

        # Simulate scrolling like a real user
        page.mouse.move(400 + random.randint(-50, 50), 300 + random.randint(-30, 30))
        page.evaluate("window.scrollTo(0, 300)")
        time.sleep(random.uniform(0.8, 1.5))
        page.evaluate("window.scrollTo(0, 600)")
        time.sleep(random.uniform(1.0, 2.0))
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(random.uniform(0.5, 1.0))

        logger.info("Session warmup complete")
    except Exception as e:
        logger.debug(f"Warmup failed (non-fatal): {e}")


def reset_session(page: Page):
    """
    Navigate back to Zillow homepage between profile visits to break
    the consecutive-profile pattern that triggers bot detection.
    """
    try:
        # Pick a random Zillow page to make the navigation look organic
        pages = [
            ZILLOW_HOME,
            "https://www.zillow.com/homes/for_sale/",
            "https://www.zillow.com/mortgage-rates/",
        ]
        target = random.choice(pages)
        page.goto(target, wait_until="domcontentloaded", timeout=20_000)
        time.sleep(random.uniform(3.0, 6.0))
        page.evaluate(f"window.scrollTo(0, {random.randint(200, 500)})")
        time.sleep(random.uniform(1.0, 2.5))
    except Exception as e:
        logger.debug(f"Session reset failed (non-fatal): {e}")


def scrape_profile(url: str, page: Page) -> dict | None:
    """Visit a Zillow profile page and extract all contact/performance data."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    except PwTimeout:
        logger.debug(f"Timeout loading {url}")
        return None

    # Wait a bit + human-like micro-scroll
    time.sleep(random.uniform(2.0, 3.5))
    page.evaluate(f"window.scrollTo(0, {random.randint(100, 300)})")
    time.sleep(random.uniform(0.5, 1.2))

    # Check for 404 / not found
    title = (page.title() or "").lower()
    if "not found" in title or "404" in title or "error" in title:
        logger.debug(f"Profile not found: {url}")
        return None

    # Check for bot wall (Cloudflare "Press & Hold" or CAPTCHA)
    content = page.content() or ""
    content_lower = content.lower()
    if "press & hold" in content_lower or "captcha" in content_lower or "are you a human" in content_lower:
        logger.warning(f"Bot wall hit on profile: {url}")
        safe_name = re.sub(r"[^a-zA-Z0-9]", "_", url.split("/")[-1])
        page.screenshot(path=str(DEBUG_DIR / f"blocked_{safe_name}_{int(time.time())}.png"))
        return None

    # Try Next.js embedded JSON first (fastest, most data)
    result = _from_next_data(page, url)
    if result == "DIRECTORY":
        # Zillow returned a directory/search page instead of the profile.
        # The agent either has no Zillow profile or uses a different slug.
        logger.debug(f"Directory page hit for {url} — agent may not be on Zillow")
        return None
    if isinstance(result, tuple) and result[0] == "REDIRECT":
        # A matching agent was found in the directory results; follow their real URL.
        real_url = result[1]
        logger.debug(f"Following real profile URL: {real_url}")
        try:
            page.goto(real_url, wait_until="domcontentloaded", timeout=30_000)
        except PwTimeout:
            return None
        time.sleep(random.uniform(2.0, 3.5))
        content_check = (page.content() or "").lower()
        if "press & hold" in content_check or "captcha" in content_check:
            logger.warning(f"Bot wall on redirect: {real_url}")
            return None
        inner = _from_next_data(page, real_url)
        if inner and inner not in ("DIRECTORY",) and not isinstance(inner, tuple):
            return inner
        return _from_dom(page, real_url)
    if result:
        return result

    # Fallback: parse DOM directly
    return _from_dom(page, url)


def _from_next_data(page: Page, url: str):
    """
    Parse Zillow's __NEXT_DATA__ JSON.
    Returns:
      - dict                   : individual profile data
      - "DIRECTORY"            : directory/search page, agent not on Zillow
      - ("REDIRECT", real_url) : directory page with a matching agent card
      - None                   : no usable data
    """
    try:
        raw = page.evaluate("""() => {
            const el = document.getElementById('__NEXT_DATA__');
            return el ? el.textContent : null;
        }""")
        if not raw:
            return None
        data = json.loads(raw)
        props = data.get("props", {}).get("pageProps", {})

        # ── Detect directory/search page ─────────────────────────────────
        display = props.get("displayData", {})
        if "agentDirectoryFinderDisplay" in display or props.get("profileType") == 2:
            return _handle_directory_page(props, url)

        # ── Individual profile: Zillow stores data in displayUser ─────────
        if "displayUser" in props:
            return _normalize_display_user(props, url)

        return None
    except Exception as e:
        logger.debug(f"__NEXT_DATA__ parse failed for {url}: {e}")
        return None


def _normalize_display_user(props: dict, url: str) -> dict | None:
    """
    Extract agent data from the real Zillow profile page structure.
    Key paths:
      - displayUser → name, email, phoneNumbers, businessName
      - professionalInformation → list of {term, description/lines} (languages, member since)
      - graphQLData.agentListingSalesSection.content.sold.headerText → "Sold (N)"
      - reviewsData → rating for agents with reviews
    """
    from datetime import datetime

    user = props.get("displayUser") or {}
    name = user.get("name")
    if not name:
        return None

    # Phone
    phones = user.get("phoneNumbers") or {}
    phone  = phones.get("cell") or phones.get("business")

    # Email
    email = user.get("email")

    # Agency
    agency = user.get("businessName")

    # ── professionalInformation: list of {term, description, lines} ───────
    prof_info = props.get("professionalInformation") or []
    speaks_spanish = False
    years_exp      = None

    for item in (prof_info if isinstance(prof_info, list) else []):
        term = (item.get("term") or "").lower()
        desc = item.get("description") or ""
        lines_text = " ".join(item.get("lines") or [])
        combined   = f"{desc} {lines_text}".lower()

        if "language" in term:
            speaks_spanish = "spanish" in combined or "español" in combined

        if "member since" in term and desc:
            try:
                # desc format: "11/09/2007" or "2007"
                year = int(desc.strip()[-4:])
                years_exp = datetime.now().year - year
            except Exception:
                pass

    # Also check full-text fields that sometimes list languages
    du_str = json.dumps(user).lower()
    if not speaks_spanish:
        speaks_spanish = "spanish" in du_str or "español" in du_str

    # ── Total sales from graphQLData ──────────────────────────────────────
    total_sales = None
    try:
        sold_header = (
            props.get("graphQLData", {})
                 .get("agentListingSalesSection", {})
                 .get("content", {})
                 .get("sold", {})
                 .get("headerText") or ""
        )
        m = re.search(r'\((\d+)\)', sold_header)
        if m:
            total_sales = int(m.group(1))
    except Exception:
        pass

    # ── Rating / review count ─────────────────────────────────────────────
    rating       = None
    review_count = None
    try:
        rd = props.get("reviewsData") or {}
        # Some profiles surface averageRating directly
        rating = _parse_float(rd.get("averageRating") or rd.get("rating"))
        # Count from filters list (first filter = "All reviews" with count)
        filters = rd.get("filters") or []
        if filters:
            review_count = _parse_int(filters[0].get("count"))
    except Exception:
        pass

    return {
        "profile_url":      url,
        "name":             name,
        "agency":           agency,
        "phone":            phone,
        "email":            email,
        "rating":           rating,
        "review_count":     review_count,
        "speaks_spanish":   speaks_spanish,
        "years_experience": years_exp,
        "sales_last_12m":   None,   # requires date-filtering sold list — omitted for now
        "total_sales":      total_sales,
    }


def _handle_directory_page(props: dict, original_url: str):
    """
    When Zillow serves a directory page instead of a profile, check if any
    result card's name matches the slug we were looking for.
    Returns ("REDIRECT", real_url) on match, else "DIRECTORY".
    """
    slug = original_url.rstrip("/").split("/")[-1].lower()  # e.g. "dianasifuentes"

    try:
        display = props.get("displayData", {})
        finder  = display.get("agentDirectoryFinderDisplay", {})
        results = finder.get("searchResults", {}).get("results", {})
        cards   = results.get("resultsCards", [])

        for card in cards:
            card_name = (card.get("cardTitle") or "").lower().replace(" ", "")
            link      = card.get("cardActionLink", "")
            if card_name and slug.startswith(card_name[:6]):
                logger.debug(f"Directory match: '{card.get('cardTitle')}' → {link}")
                return ("REDIRECT", link)
    except Exception as e:
        logger.debug(f"Directory page parse error: {e}")

    return "DIRECTORY"


def _from_dom(page: Page, url: str) -> dict | None:
    try:
        def get(selectors: list) -> str | None:
            for sel in selectors:
                try:
                    el = page.query_selector(sel)
                    if el:
                        t = (el.inner_text() or "").strip()
                        if t:
                            return t
                except Exception:
                    continue
            return None

        name = get([
            "h1[data-test='agent-name']", "h1", "[class*='agentName']",
            "[class*='profileName']", "h1[class*='name']",
        ])
        if not name:
            return None

        agency = get([
            "[data-test='agent-business-name']", "[class*='businessName']",
            "[class*='brokerage']", "[class*='company']",
        ])

        # Phone — tel: links first
        phone = None
        phone_link = page.query_selector("a[href^='tel:']")
        if phone_link:
            href = phone_link.get_attribute("href") or ""
            phone = href.replace("tel:", "").strip()
        if not phone:
            phone = get(["[data-test='agent-phone']", "[class*='phone']", "[class*='Phone']"])

        # Email — mailto: links first
        email = None
        email_link = page.query_selector("a[href^='mailto:']")
        if email_link:
            href = email_link.get_attribute("href") or ""
            email = href.replace("mailto:", "").strip()
        if not email:
            email = get(["[data-test='agent-email']", "[class*='email']", "[class*='Email']"])

        rating_raw  = get(["[data-test='rating']", "[class*='rating']", "[aria-label*='rating']"])
        reviews_raw = get(["[data-test='review-count']", "[class*='reviewCount']", "[class*='reviews']"])

        full_text   = page.inner_text("body") or ""
        sales_12m   = _extract_number(full_text, r"(\d+)\s*sales?\s*last\s*12\s*months?")
        total_sales = _extract_number(full_text, r"(\d[\d,]*)\s*total\s*sales?")
        years_exp   = _extract_number(full_text, r"(\d+)\s*years?\s*(?:of\s*)?experience")

        speaks_spanish = any(w in full_text.lower() for w in (
            "spanish", "español", "habla español", "se habla",
        ))

        return {
            "profile_url":     url,
            "name":            name,
            "agency":          agency,
            "phone":           phone,
            "email":           email,
            "rating":          _parse_float(rating_raw),
            "review_count":    _parse_int(reviews_raw),
            "speaks_spanish":  speaks_spanish,
            "years_experience": years_exp,
            "sales_last_12m":  sales_12m,
            "total_sales":     total_sales,
        }
    except Exception as e:
        logger.debug(f"DOM profile parse failed for {url}: {e}")
        return None


def _normalize_profile(agent: dict, url: str) -> dict | None:
    try:
        name = (
            agent.get("fullName") or agent.get("name")
            or agent.get("displayName") or agent.get("agentName")
        )
        if not name:
            return None

        langs = agent.get("languages") or agent.get("spokenLanguages") or []
        if isinstance(langs, list):
            lang_text = " ".join(str(l).lower() for l in langs)
        else:
            lang_text = str(langs).lower()
        speaks_spanish = "spanish" in lang_text or "español" in lang_text

        phone = (
            agent.get("phone") or agent.get("phoneNumber")
            or agent.get("mobilePhone") or agent.get("businessPhone")
        )
        email = (
            agent.get("email") or agent.get("emailAddress")
            or agent.get("businessEmail")
        )

        return {
            "profile_url":     url,
            "name":            name,
            "agency":          (agent.get("businessName") or agent.get("brokerageName")
                                or agent.get("companyName")),
            "phone":           phone,
            "email":           email,
            "rating":          _parse_float(agent.get("rating") or agent.get("averageRating")
                                            or agent.get("reviewAvgRating")),
            "review_count":    _parse_int(agent.get("reviewCount") or agent.get("totalReviews")),
            "speaks_spanish":  speaks_spanish,
            "years_experience": _parse_int(agent.get("yearsExperience")
                                           or agent.get("experienceYears")),
            "sales_last_12m":  _parse_int(agent.get("recentSales") or agent.get("salesLast12Months")
                                          or agent.get("pastYearSales")),
            "total_sales":     _parse_int(agent.get("totalSales") or agent.get("soldCount")),
        }
    except Exception:
        return None


def _extract_number(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return _parse_int(m.group(1))
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
