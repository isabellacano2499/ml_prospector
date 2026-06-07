"""
Instagram profile finder using Playwright browser.
Flow:
  1. Search DuckDuckGo: site:instagram.com "Name" realtor STATE
  2. Extract first valid Instagram handle from results
  3. Navigate to instagram.com/handle/ and read rendered page + post alt texts
"""
import re
import time
import random
import unicodedata
from loguru import logger

try:
    from patchright.sync_api import Page, TimeoutError as PwTimeout
except ImportError:
    from playwright.sync_api import Page, TimeoutError as PwTimeout

_SKIP_HANDLES = {
    "p", "reel", "explore", "accounts", "stories", "tv",
    "reels", "about", "privacy", "legal", "help", "tags", "directory",
}

DDG_SEARCH = "https://duckduckgo.com/"
IG_BASE    = "https://www.instagram.com/"

# Distinctive Spanish words unlikely to appear in English text
_SPANISH_INDICATORS = [
    "casa", "comprar", "vender", "hogar", "precio", "familia",
    "comunidad", "hispano", "espanol", "habla", "hablo", "gracias",
    "bienvenidos", "vendido", "vendida", "hogares", "propiedades",
    "bienes raices", "vivienda", "primera", "somos", "estamos",
    "ayudo", "compradores", "vendedores", "agente", "nuevo hogar",
    "tu casa", "mi casa", "renta", "casas", "bilingue",
]

_LATINO_INDICATORS = [
    "nahrep", "latino", "latina", "hispano", "hispanic", "latinx",
    "hispanic heritage", "mercado latino", "comunidad latina",
]

_COMMUNITY_TYPES = {
    "first_time_buyer": [
        "first-time", "first time", "primera casa", "fha",
        "down payment", "dpa", "primer hogar", "first home", "buyers",
    ],
    "veterans": ["veteran", "military", "va loan", "va home", "vets"],
    "luxury": ["luxury", "million dollar", "high-end", "premium listing", "exclusive listing"],
    "investor": ["investor", "investment", "flip", "rental", "portfolio", "airbnb"],
    "family_community": [
        "familia", "family", "community", "comunidad", "families",
        "neighborhood", "vecindad", "barrio",
    ],
    "relocation": [
        "relocation", "relocating", "relo", "moving to", "mudanza", "relocate",
    ],
}

_US_STATES_LOWER = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
]

# State abbreviations → full name for display
_STATE_ABBREV = {
    "al":"Alabama","ak":"Alaska","az":"Arizona","ar":"Arkansas","ca":"California",
    "co":"Colorado","ct":"Connecticut","de":"Delaware","fl":"Florida","ga":"Georgia",
    "hi":"Hawaii","id":"Idaho","il":"Illinois","in":"Indiana","ia":"Iowa",
    "ks":"Kansas","ky":"Kentucky","la":"Louisiana","me":"Maine","md":"Maryland",
    "ma":"Massachusetts","mi":"Michigan","mn":"Minnesota","ms":"Mississippi",
    "mo":"Missouri","mt":"Montana","ne":"Nebraska","nv":"Nevada","nh":"New Hampshire",
    "nj":"New Jersey","nm":"New Mexico","ny":"New York","nc":"North Carolina",
    "nd":"North Dakota","oh":"Ohio","ok":"Oklahoma","or":"Oregon","pa":"Pennsylvania",
    "ri":"Rhode Island","sc":"South Carolina","sd":"South Dakota","tn":"Tennessee",
    "tx":"Texas","ut":"Utah","vt":"Vermont","va":"Virginia","wa":"Washington",
    "wv":"West Virginia","wi":"Wisconsin","wy":"Wyoming",
}


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")


def find_instagram(first: str, last: str, state: str, page: Page) -> dict:
    handle = _ddg_search_handle(first, last, state, page)
    if not handle:
        return {}

    time.sleep(random.uniform(2.0, 4.0))
    profile = _fetch_ig_profile(handle, page)
    if not profile:
        return {}

    profile["ig_handle"] = handle
    profile["ig_url"]    = IG_BASE + handle + "/"
    return profile


def _ddg_search_handle(first: str, last: str, state: str, page: Page) -> str | None:
    name    = (first + " " + last).strip()
    queries = [
        "site:instagram.com " + '"' + name + '"' + " realtor " + state,
        "site:instagram.com " + name + " realtor " + state,
    ]

    for query in queries:
        try:
            import urllib.parse
            search_url = DDG_SEARCH + "?q=" + urllib.parse.quote_plus(query) + "&kl=us-en"
            page.goto(search_url, wait_until="domcontentloaded", timeout=20_000)
            time.sleep(random.uniform(2.0, 3.5))

            content = page.content()
            if not content or len(content) < 1000:
                continue

            handles = re.findall(r"instagram\.com/([A-Za-z0-9_.]{3,30})", content)
            for h in handles:
                if h.lower() not in _SKIP_HANDLES and not h.startswith("_"):
                    logger.debug("DDG found IG handle for " + name + ": @" + h)
                    return h

        except (PwTimeout, Exception) as e:
            logger.debug("DDG search failed for " + name + ": " + str(e))

        time.sleep(random.uniform(1.5, 3.0))

    return None


def _fetch_ig_profile(handle: str, page: Page) -> dict | None:
    url = IG_BASE + handle + "/"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        time.sleep(random.uniform(2.0, 3.5))

        content = page.content()
        if not content:
            return None

        title = (page.title() or "").lower()
        if "page not found" in title or "instagram" not in title:
            return None

        body_text = page.inner_text("body") or ""

        # meta[name=description] format (most reliable):
        # "X Followers, Y Following, Z Posts - See Instagram photos from @handle"
        followers, posts_n = _counts_from_meta(page)

        # Fall back to body text regex if meta didn't have them
        if followers is None:
            followers = _parse_ig_count(body_text, r"([\d,\.]+[KkMm]?)\s*[Ff]ollowers")
        if posts_n is None:
            posts_n = _parse_ig_count(body_text, r"([\d,\.]+[KkMm]?)\s*[Pp]osts?")

        bio = _extract_bio(page)
        bio_lower = _strip_accents(bio or "").lower()

        is_private = "This account is private" in body_text

        # Read up to 12 visible post alt texts for content analysis
        post_texts = _extract_post_alt_texts(page)
        all_content = bio_lower + " " + " ".join(
            _strip_accents(t).lower() for t in post_texts
        )

        # Basic bio signals — accent-normalized so "Español" matches "espanol"
        ig_spanish_signals = any(w in bio_lower for w in (
            "espanol", "spanish", "hispano", "latino", "latina",
            "bilingue", "bilingual", "habla", "hablo", "se habla",
        ))
        ig_realtor_signals = any(w in bio_lower for w in (
            "realtor", "real estate", "realty", "homes", "casa",
            "broker", "agent", "properties",
        ))

        # Posts in Spanish: count distinctive Spanish words across bio + posts
        spanish_word_count = sum(1 for w in _SPANISH_INDICATORS if w in all_content)
        ig_posts_spanish = spanish_word_count >= 2

        # Explicit latino market mention (bio or posts)
        ig_mentions_latino = any(w in all_content for w in _LATINO_INDICATORS)

        # NAHREP membership: very strong signal
        ig_nahrep = "nahrep" in all_content

        # Community type served (can be multiple, comma-separated)
        detected = []
        for ctype, keywords in _COMMUNITY_TYPES.items():
            if any(kw in all_content for kw in keywords):
                detected.append(ctype)
        ig_community_type = ",".join(detected) if detected else None

        # Collaborates with other accounts (@mentions in posts)
        at_mentions = re.findall(r"@[A-Za-z0-9_.]{3,30}", " ".join(post_texts))
        ig_collaborates = len(at_mentions) > 0

        # States or metro areas mentioned in bio + posts
        ig_area_mentions = _detect_state_mentions(all_content)

        # Overall content language based on Spanish word density
        ig_content_language = _classify_language(spanish_word_count)

        # Engagement proxy: followers per post (when both available)
        ig_engagement_proxy = None
        if followers and posts_n and posts_n > 0:
            ig_engagement_proxy = round(followers / posts_n, 1)

        return {
            "ig_followers":         followers,
            "ig_posts":             posts_n,
            "ig_bio":               (bio or "")[:200] or None,
            "ig_is_private":        is_private,
            "ig_spanish_signals":   ig_spanish_signals,
            "ig_realtor_signals":   ig_realtor_signals,
            "ig_posts_spanish":     ig_posts_spanish,
            "ig_mentions_latino":   ig_mentions_latino,
            "ig_nahrep":            ig_nahrep,
            "ig_community_type":    ig_community_type,
            "ig_collaborates":      ig_collaborates,
            "ig_area_mentions":     ig_area_mentions,
            "ig_content_language":  ig_content_language,
            "ig_engagement_proxy":  ig_engagement_proxy,
        }

    except (PwTimeout, Exception) as e:
        logger.debug("IG profile fetch failed for @" + handle + ": " + str(e))
        return None


def _counts_from_meta(page: Page) -> tuple:
    # meta[name=description] typically contains:
    # "X Followers, Y Following, Z Posts - See Instagram photos from @handle"
    try:
        meta = page.query_selector("meta[name=description]")
        if meta:
            content = meta.get_attribute("content") or ""
            followers = _parse_ig_count(content, r"([\d,\.]+[KkMm]?)\s*[Ff]ollowers")
            posts_n   = _parse_ig_count(content, r"([\d,\.]+[KkMm]?)\s*[Pp]osts?")
            return followers, posts_n
    except Exception:
        pass
    return None, None


def _extract_post_alt_texts(page: Page) -> list:
    try:
        imgs = page.query_selector_all("article img[alt]")
        texts = []
        for img in imgs[:12]:
            alt = img.get_attribute("alt") or ""
            if alt and len(alt) > 20:
                texts.append(alt)
        return texts
    except Exception:
        return []


def _classify_language(spanish_word_count: int) -> str:
    if spanish_word_count >= 3:
        return "spanish"
    if spanish_word_count >= 1:
        return "mixed"
    return "english"


def _detect_state_mentions(text: str) -> str | None:
    found = []
    seen = set()
    # Full names first
    for state in _US_STATES_LOWER:
        if re.search(r"\b" + re.escape(state) + r"\b", text):
            title = state.title()
            if title not in seen:
                found.append(title)
                seen.add(title)
    # Then abbreviations (e.g. CA, TX, FL — uppercase 2-letter word boundary)
    for abbr, full in _STATE_ABBREV.items():
        if full not in seen and re.search(r"\b" + abbr.upper() + r"\b", text.upper()):
            found.append(full)
            seen.add(full)
    return ", ".join(found[:4]) if found else None


def _extract_bio(page: Page) -> str:
    try:
        meta = page.query_selector("meta[name=description]")
        if meta:
            content = meta.get_attribute("content") or ""
            idx = content.find("on Instagram:")
            if idx != -1:
                bio = content[idx + 13:].strip().strip('"').strip("'").strip()
                if bio and "See Instagram" not in bio:
                    return bio

        og = page.query_selector("meta[property=og\\:description]")
        if og:
            content = og.get_attribute("content") or ""
            m = re.search(r"from (.+?)$", content)
            if m:
                part = m.group(1).strip()
                if part and "See Instagram" not in part:
                    return part

        return ""
    except Exception:
        return ""


def _parse_ig_count(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    if not m:
        return None
    raw = m.group(1).replace(",", "").strip()
    try:
        if raw.upper().endswith("K"):
            return int(float(raw[:-1]) * 1_000)
        if raw.upper().endswith("M"):
            return int(float(raw[:-1]) * 1_000_000)
        return int(float(raw))
    except ValueError:
        return None


# Kept for backward compatibility
import requests

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s
