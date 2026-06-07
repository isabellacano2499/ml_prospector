"""
MMI Data Enricher
=================
Reads MMI Data.xlsx (name, company, email, phone, state, units sold, loan volume),
then enriches each realtor with:
  - Zillow profile: rating, reviews, Spanish speaker, years exp, sales 12m, total sales,
                    price range, profile URL, verified email/phone
  - Instagram:      handle, followers, posts, bio, Spanish/Latino market signals

Output: realtor_scraper/output/mmi_enriched_YYYYMMDD_HHMM.csv
        (saves checkpoint after every realtor)

Usage:
  python mmi_enricher.py
  python mmi_enricher.py --limit 20          # test with first 20
  python mmi_enricher.py --state Texas       # only one state
  python mmi_enricher.py --no-headless       # show browser window
  python mmi_enricher.py --skip-instagram    # skip Instagram lookup
  python mmi_enricher.py --resume output/mmi_enriched_20240101_1200.csv
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

try:
    from patchright.sync_api import sync_playwright
except ImportError:
    from playwright.sync_api import sync_playwright

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MMI_FILE    = ROOT.parent / "latino_re_engine" / "MMI Data.xlsx"
OUTPUT_DIR  = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from zillow.profile_scraper import build_profile_url, scrape_profile, make_browser_context, warmup_session, reset_session
from instagram.finder import find_instagram

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:5}</level> | {message}",
           level="INFO")
logger.add(OUTPUT_DIR / "mmi_enricher.log", rotation="10 MB", level="DEBUG")

DELAY_ZILLOW    = (8.0, 15.0)
DELAY_INSTAGRAM = (3.0, 7.0)
DELAY_GOOGLE    = (4.0, 9.0)


def main():
    parser = argparse.ArgumentParser(description="Enrich MMI realtors with Zillow + Instagram")
    parser.add_argument("--limit",          type=int,   default=None)
    parser.add_argument("--state",          type=str,   default=None, help="Filter by state name")
    parser.add_argument("--no-headless",    action="store_true")
    parser.add_argument("--skip-zillow",    action="store_true")
    parser.add_argument("--skip-instagram", action="store_true")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to existing output CSV to resume from")
    args = parser.parse_args()

    # ── Load MMI data ─────────────────────────────────────────────────────────
    logger.info(f"Reading {MMI_FILE.name}...")
    df = pd.read_excel(MMI_FILE, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names
    df = df.rename(columns={
        "First Name":           "first_name",
        "Last Name":            "last_name",
        "Company / Account":   "company",
        "Email":                "email_mmi",
        "Phone":                "phone_mmi",
        "State":                "state",
        "BS Sold # Units":      "units_sold",
        "Loan Volume 14 months":"loan_volume",
    })
    df["full_name"] = (df["first_name"].str.strip() + " " + df["last_name"].str.strip()).str.strip()

    if args.state:
        df = df[df["state"].str.lower() == args.state.lower()]
        logger.info(f"Filtered to state '{args.state}': {len(df)} realtors")

    if args.limit:
        df = df.head(args.limit)

    logger.info(f"Total realtors to process: {len(df)}")

    # ── Resume from existing output ───────────────────────────────────────────
    already_done: set[str] = set()
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            done_df = pd.read_csv(resume_path, dtype=str)
            already_done = set(done_df["full_name"].dropna().str.strip())
            logger.info(f"Resuming — {len(already_done)} realtors already processed")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = resume_path or (OUTPUT_DIR / f"mmi_enriched_{timestamp}.csv")

    # ── Convert DataFrame to list of dicts ────────────────────────────────────
    records = df.to_dict("records")
    enriched: list[dict] = []

    # Load already-done records if resuming
    if resume_path and resume_path.exists():
        existing = pd.read_csv(resume_path, dtype=str)
        enriched = existing.to_dict("records")

    with sync_playwright() as p:
        # Persistent context + patchright patches Cloudflare fingerprinting
        context = make_browser_context(p, headless=not args.no_headless)
        page    = context.new_page()   # tab 1: Zillow profiles
        ig_page = context.new_page()   # tab 2: DuckDuckGo + Instagram

        # Warm up session on Zillow homepage before scraping profiles
        if not args.skip_zillow:
            warmup_session(page)

        bar = tqdm(records, desc="Enriching", unit="realtor")
        zillow_count = 0  # tracks profile visits for reset/pause cadence
        for rec in bar:
            name = rec.get("full_name", "").strip()
            if not name:
                continue
            if name in already_done:
                bar.set_postfix({"skip": name[:20]})
                continue

            row = _base_record(rec)

            # ── Zillow enrichment ─────────────────────────────────────────────
            if not args.skip_zillow:
                # Every 2 profiles: visit a random Zillow page to break the
                # consecutive /profile/ pattern that triggers bot detection.
                if zillow_count > 0 and zillow_count % 2 == 0:
                    reset_session(page)

                # Every 8 profiles: longer pause so Zillow's session rate
                # limiter resets — prevents bot wall from building up.
                if zillow_count > 0 and zillow_count % 8 == 0:
                    logger.info(f"Deep pause after {zillow_count} profiles...")
                    time.sleep(random.uniform(75, 100))
                    warmup_session(page)

                zurl    = build_profile_url(name)
                profile = scrape_profile(zurl, page)
                zillow_count += 1
                if profile:
                    row.update({
                        "zillow_url":           profile.get("profile_url"),
                        "zillow_rating":        profile.get("rating"),
                        "zillow_reviews":       profile.get("review_count"),
                        "zillow_speaks_spanish": profile.get("speaks_spanish", False),
                        "zillow_years_exp":     profile.get("years_experience"),
                        "zillow_sales_12m":     profile.get("sales_last_12m"),
                        "zillow_total_sales":   profile.get("total_sales"),
                        "zillow_agency":        profile.get("agency"),
                        "email_zillow":         profile.get("email"),
                        "phone_zillow":         profile.get("phone"),
                    })
                time.sleep(random.uniform(*DELAY_ZILLOW))

            # ── Instagram lookup ──────────────────────────────────────────────
            if not args.skip_instagram:
                ig = find_instagram(
                    rec.get("first_name", ""),
                    rec.get("last_name", ""),
                    rec.get("state", ""),
                    ig_page,
                )
                if ig:
                    row.update({
                        "ig_handle":          ig.get("ig_handle"),
                        "ig_url":             ig.get("ig_url"),
                        "ig_followers":       ig.get("ig_followers"),
                        "ig_posts":           ig.get("ig_posts"),
                        "ig_bio":             ig.get("ig_bio"),
                        "ig_is_private":      ig.get("ig_is_private"),
                        "ig_spanish_signals": ig.get("ig_spanish_signals"),
                        "ig_realtor_signals": ig.get("ig_realtor_signals"),
                    })
                time.sleep(random.uniform(*DELAY_GOOGLE))

            enriched.append(row)
            already_done.add(name)
            _save(enriched, output_path)

            ig_found    = sum(1 for r in enriched if r.get("ig_handle"))
            email_found = sum(1 for r in enriched if r.get("email_zillow"))
            bar.set_postfix({
                "ig": ig_found,
                "emails": email_found,
                "last": name[:18],
            })

        context.close()

    _save(enriched, output_path)
    _summary(enriched, output_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_record(rec: dict) -> dict:
    return {
        "full_name":   rec.get("full_name", "").strip(),
        "first_name":  rec.get("first_name", "").strip().title(),
        "last_name":   rec.get("last_name", "").strip().title(),
        "company":     rec.get("company", "").strip(),
        "state":       rec.get("state", "").strip(),
        "email_mmi":   rec.get("email_mmi", "").strip() or None,
        "phone_mmi":   rec.get("phone_mmi", "").strip() or None,
        "units_sold":  rec.get("units_sold", "").strip() or None,
        "loan_volume": rec.get("loan_volume", "").strip() or None,
    }


def _save(records: list[dict], path: Path):
    if not records:
        return
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["full_name"], keep="last")
    col_order = [
        # MMI original
        "full_name", "first_name", "last_name", "company", "state",
        "email_mmi", "phone_mmi", "units_sold", "loan_volume",
        # Zillow enriched
        "zillow_rating", "zillow_reviews", "zillow_speaks_spanish",
        "zillow_years_exp", "zillow_sales_12m", "zillow_total_sales",
        "zillow_agency", "email_zillow", "phone_zillow", "zillow_url",
        # Instagram
        "ig_handle", "ig_url", "ig_followers", "ig_posts",
        "ig_bio", "ig_is_private", "ig_spanish_signals", "ig_realtor_signals",
    ]
    existing = [c for c in col_order if c in df.columns]
    df[existing].to_csv(path, index=False)


def _summary(records: list[dict], path: Path):
    df = pd.DataFrame(records)
    total = len(df)
    if total == 0:
        logger.warning("No records processed.")
        return

    def pct(n): return f"{n} ({n/total*100:.0f}%)"

    zil_found   = int(df["zillow_url"].notna().sum())            if "zillow_url"    in df.columns else 0
    spanish     = int(df["zillow_speaks_spanish"].eq(True).sum()) if "zillow_speaks_spanish" in df.columns else 0
    ig_found    = int(df["ig_handle"].notna().sum())             if "ig_handle"     in df.columns else 0
    email_zil   = int(df["email_zillow"].notna().sum())          if "email_zillow"  in df.columns else 0
    email_mmi   = int(df["email_mmi"].notna().sum())             if "email_mmi"     in df.columns else 0

    logger.info("=" * 55)
    logger.info(f"Total realtors procesados : {total:,}")
    logger.info(f"Perfil Zillow encontrado  : {pct(zil_found)}")
    logger.info(f"Spanish speaker (Zillow)  : {pct(spanish)}")
    logger.info(f"Instagram encontrado      : {pct(ig_found)}")
    logger.info(f"Email MMI                 : {pct(email_mmi)}")
    logger.info(f"Email Zillow (adicional)  : {pct(email_zil)}")
    logger.info(f"Guardado en: {path}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
