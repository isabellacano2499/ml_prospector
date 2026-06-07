"""
Realtor Scraper — Latino Markets (TX + FL)

Flow:
  1. Load target ZIPs from Census output (score >= 40, pop >= 1000)
  2. Read downloaded state license files:
       TX → data/tx_licenses.csv   (from TREC: trec.texas.gov)
       FL → data/fl_licenses.csv   (from DBPR: myfloridalicense.com)
  3. Filter agents to our target cities / counties
  4. Enrich each agent via Zillow individual profile → phone, email, sales
  5. Save to CSV (checkpoint after every agent)

How to get the license files:
  TX: https://www.trec.texas.gov/agency-information/open-records
      Download "License Holder" data → save as data/tx_licenses.csv
  FL: https://www.myfloridalicense.com  → "clicking here" (free download)
      Select Real Estate → save as data/fl_licenses.csv
"""
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger
from tqdm import tqdm
from playwright.sync_api import sync_playwright

from config import OUTPUT_DIR
from zip_loader import load_target_zips
from state_licenses.zip_to_geo import zip_list_to_cities
from state_licenses.file_processor import load_and_filter, build_target_geo
from zillow.profile_scraper import build_profile_url, scrape_profile

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level="INFO")
logger.add(OUTPUT_DIR / "scraper.log", rotation="10 MB", level="DEBUG")

DATA_DIR = Path(__file__).parent / "data"
TX_FILE  = DATA_DIR / "tx_licenses.csv"
FL_FILE  = DATA_DIR / "fl_licenses.csv"


def main():
    parser = argparse.ArgumentParser(description="Realtor Scraper — Latino Markets")
    parser.add_argument("--states",      nargs="+", default=["TX", "FL"])
    parser.add_argument("--limit",       type=int,  default=None, help="Limit ZIPs (testing)")
    parser.add_argument("--no-headless", action="store_true",     help="Show Zillow browser")
    parser.add_argument("--skip-zillow", action="store_true",     help="Skip Zillow enrichment")
    args = parser.parse_args()

    # ── Load target ZIPs ──────────────────────────────────────────────────────
    zips_df = load_target_zips(args.states)
    if args.limit:
        zips_df = zips_df.head(args.limit)

    logger.info(f"Target ZIPs: {len(zips_df)}")
    logger.info(f"\n{zips_df['state_derived'].value_counts().to_string()}")

    zip_list = list(zips_df["zip_code"])
    zip_market = zips_df.set_index("zip_code").to_dict("index")

    tx_zips = [z for z in zip_list if zip_market[z]["state_derived"] == "TX"]
    fl_zips = [z for z in zip_list if zip_market[z]["state_derived"] == "FL"]

    logger.info("Resolving ZIP → city/county...")
    geo_map = zip_list_to_cities(zip_list)

    tx_cities, tx_counties   = build_target_geo(tx_zips) if tx_zips else (set(), set())
    fl_cities, fl_counties   = build_target_geo(fl_zips) if fl_zips else (set(), set())

    logger.info(f"TX target cities : {len(tx_cities)} | counties: {len(tx_counties)}")
    logger.info(f"FL target cities : {len(fl_cities)} | counties: {len(fl_counties)}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    states_str  = "_".join(args.states)
    output_path = OUTPUT_DIR / f"realtors_{states_str}_{timestamp}.csv"

    all_agents: list[dict] = []

    # ── STEP 1: Load state license files ────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1 — Loading state license files")
    logger.info("=" * 55)

    if "TX" in args.states:
        if TX_FILE.exists():
            tx_df = load_and_filter(TX_FILE, target_cities=tx_cities,
                                    target_counties=tx_counties, state="TX")
            tx_agents = _df_to_agents(tx_df, zip_list=tx_zips,
                                      geo_map=geo_map, zip_market=zip_market)
            all_agents.extend(tx_agents)
            logger.info(f"TX agents loaded: {len(tx_agents)}")
        else:
            logger.warning(f"TX license file not found: {TX_FILE}")
            logger.warning("Download from: https://www.trec.texas.gov/agency-information/open-records")

    if "FL" in args.states:
        if FL_FILE.exists():
            fl_df = load_and_filter(FL_FILE, target_cities=fl_cities,
                                    target_counties=fl_counties, state="FL")
            fl_agents = _df_to_agents(fl_df, zip_list=fl_zips,
                                      geo_map=geo_map, zip_market=zip_market)
            all_agents.extend(fl_agents)
            logger.info(f"FL agents loaded: {len(fl_agents)}")
        else:
            logger.warning(f"FL license file not found: {FL_FILE}")
            logger.warning("Download from: https://www.myfloridalicense.com → 'clicking here'")

    if not all_agents:
        logger.error("No agents loaded. Please download the license files first.")
        logger.error(f"Expected locations:\n  TX: {TX_FILE}\n  FL: {FL_FILE}")
        return

    logger.info(f"Total agents from license files: {len(all_agents)}")
    _save(all_agents, output_path)

    # ── STEP 2: Zillow enrichment ─────────────────────────────────────────────
    if not args.skip_zillow:
        logger.info("=" * 55)
        logger.info("STEP 2 — Zillow profiles: phone, email, sales")
        logger.info("=" * 55)
        _enrich_zillow(all_agents, output_path, headless=not args.no_headless)

    _save(all_agents, output_path)
    _print_summary(all_agents, output_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _df_to_agents(df: pd.DataFrame, zip_list: list, geo_map: dict,
                  zip_market: dict) -> list[dict]:
    """Convert filtered license DataFrame to agent dicts with market data attached."""
    if df.empty:
        return []

    agents = []
    # Map city → best ZIP by market score
    city_to_zip: dict[str, str] = {}
    for z in zip_list:
        city, _ = geo_map.get(z, ("", ""))
        if city:
            existing = city_to_zip.get(city)
            if not existing or (zip_market.get(z, {}).get("overall_score", 0) >
                                zip_market.get(existing, {}).get("overall_score", 0)):
                city_to_zip[city] = z

    for _, row in df.iterrows():
        agent_city = str(row.get("city", "")).strip()
        best_zip   = city_to_zip.get(agent_city) or (zip_list[0] if zip_list else "")
        market     = zip_market.get(best_zip, {})

        agents.append({
            "state":            row.get("state", ""),
            "zip_code":         best_zip,
            "city":             agent_city,
            "county":           row.get("county", ""),
            "name":             row.get("name", ""),
            "agency":           row.get("agency", ""),
            "license_number":   row.get("license_number", ""),
            "license_type":     row.get("license_type", ""),
            "license_status":   row.get("license_status", ""),
            "license_expiration": row.get("license_expiration", ""),
            "sponsoring_broker": row.get("sponsoring_broker", ""),
            "market_score":     market.get("overall_score"),
            "hispanic_pct":     round((market.get("hispanic_pct") or 0) * 100, 1),
            "spanish_home_pct": round((market.get("spanish_home_pct") or 0) * 100, 1),
            "total_population": market.get("total_population"),
            "median_income":    market.get("median_household_income"),
        })
    return agents


def _enrich_zillow(agents: list[dict], output_path: Path, headless: bool = True):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--no-first-run"],
        )
        context = browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
        page = context.new_page()

        bar = tqdm(agents, desc="Zillow profiles", unit="agent")
        for agent in bar:
            name = agent.get("name", "")
            if not name:
                continue

            url     = build_profile_url(name)
            profile = scrape_profile(url, page)

            if profile:
                agent["email"]            = profile.get("email")
                agent["phone"]            = agent.get("phone") or profile.get("phone")
                agent["zillow_profile_url"] = profile.get("profile_url")
                agent["sales_last_12m"]   = profile.get("sales_last_12m")
                agent["total_sales"]      = profile.get("total_sales")
                agent["years_experience"] = profile.get("years_experience")
                agent["speaks_spanish"]   = profile.get("speaks_spanish", False)
                agent["rating"]           = profile.get("rating")
                agent["review_count"]     = profile.get("review_count")
                if not agent.get("agency") and profile.get("agency"):
                    agent["agency"]       = profile.get("agency")

            emails = sum(1 for a in agents if a.get("email"))
            bar.set_postfix({"emails": emails})
            _save(agents, output_path)
            time.sleep(random.uniform(2.0, 4.5))

        browser.close()


def _save(agents: list[dict], path: Path):
    if not agents:
        return
    df = pd.DataFrame(agents)
    df = df.drop_duplicates(subset=["name", "state"], keep="last")
    col_order = [
        "state", "zip_code", "city", "county", "market_score",
        "hispanic_pct", "spanish_home_pct",
        "name", "agency", "phone", "email",
        "license_number", "license_type", "license_status", "license_expiration",
        "sponsoring_broker",
        "rating", "review_count", "speaks_spanish",
        "years_experience", "sales_last_12m", "total_sales",
        "zillow_profile_url",
        "total_population", "median_income",
    ]
    existing = [c for c in col_order if c in df.columns]
    df[existing].to_csv(path, index=False)


def _print_summary(agents: list[dict], path: Path):
    df = pd.DataFrame(agents)
    total = len(df)
    if total == 0:
        logger.warning("No agents found.")
        return
    with_email = int(df["email"].notna().sum()) if "email" in df.columns else 0
    with_phone = int(df["phone"].notna().sum()) if "phone" in df.columns else 0
    spanish    = int(df["speaks_spanish"].sum()) if "speaks_spanish" in df.columns else 0
    logger.info("=" * 55)
    logger.info(f"Total agents      : {total:,}")
    logger.info(f"Con telefono      : {with_phone:,} ({with_phone/total*100:.0f}%)")
    logger.info(f"Con email         : {with_email:,} ({with_email/total*100:.0f}%)")
    logger.info(f"Spanish speakers  : {spanish:,} ({spanish/total*100:.0f}%)")
    if "state" in df.columns:
        logger.info(f"Por estado:\n{df.groupby('state').size().to_string()}")
    logger.info(f"Guardado en: {path}")


if __name__ == "__main__":
    main()
