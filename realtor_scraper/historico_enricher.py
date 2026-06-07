"""
Historico Realtors Enricher
============================
Reads Historico Realtors.xlsx, filters to Qualified + Discarded (labeled records),
enriches each with Instagram data, and saves a CSV ready for XGBoost training.

Label: 1 = Qualified (les intereso), 0 = Discarded (no les intereso)

Usage:
  python historico_enricher.py --no-headless
  python historico_enricher.py --no-headless --limit 10
  python historico_enricher.py --no-headless --state Texas
  python historico_enricher.py --no-headless --resume output/historico_enriched_20240101_1200.csv
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

ROOT       = Path(__file__).parent
HIST_FILE  = ROOT.parent / "latino_re_engine" / "Historico Realtors.xlsx"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from instagram.finder import find_instagram
from zillow.profile_scraper import make_browser_context, warmup_session

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:5}</level> | {message}",
           level="INFO")
logger.add(OUTPUT_DIR / "historico_enricher.log", rotation="10 MB", level="DEBUG")

DELAY_IG = (5.0, 10.0)

# Column order for output CSV
COL_ORDER = [
    # Original fields
    "full_name", "first_name", "last_name", "company", "state",
    "email", "phone", "units_sold", "loan_volume",
    "lead_status", "label",
    # Instagram enriched
    "ig_handle", "ig_url", "ig_followers", "ig_posts",
    "ig_bio", "ig_is_private",
    "ig_spanish_signals", "ig_realtor_signals",
    "ig_posts_spanish", "ig_mentions_latino", "ig_nahrep",
    "ig_community_type", "ig_collaborates",
    "ig_area_mentions", "ig_content_language", "ig_engagement_proxy",
]


def main():
    parser = argparse.ArgumentParser(description="Enrich Historico Realtors with Instagram")
    parser.add_argument("--limit",        type=int,  default=None,
                        help="Process only first N records (for testing)")
    parser.add_argument("--state",        type=str,  default=None,
                        help="Filter to one state (e.g. Texas)")
    parser.add_argument("--skip-states",  nargs="+", default=None,
                        help="Skip these states — process everything else (including empty state)")
    parser.add_argument("--no-headless",  action="store_true",
                        help="Show browser window (required to bypass bot detection)")
    parser.add_argument("--resume",       type=str,  default=None,
                        help="Path to existing output CSV to resume from")
    args = parser.parse_args()

    # ── Load and filter historico data ────────────────────────────────────────
    logger.info("Reading " + HIST_FILE.name + "...")
    df = pd.read_excel(HIST_FILE, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "First Name":            "first_name",
        "Last Name":             "last_name",
        "Company / Account":     "company",
        "Email":                 "email",
        "Phone":                 "phone",
        "State":                 "state",
        "Lead Status":           "lead_status",
        "BS Sold # Units":       "units_sold",
        "Loan Volume 14 months": "loan_volume",
        "Converted":             "converted",
    })
    df["full_name"] = (
        df["first_name"].str.strip() + " " + df["last_name"].str.strip()
    ).str.strip()

    # Keep only labeled records
    status_lower = df["lead_status"].str.strip().str.lower()
    df = df[status_lower.isin(["qualified", "discarded"])].copy()
    df["label"] = (df["lead_status"].str.strip().str.lower() == "qualified").astype(int)

    logger.info(
        "Labels  ->  Qualified: " + str((df["label"] == 1).sum()) +
        "  |  Discarded: " + str((df["label"] == 0).sum())
    )

    if args.state:
        df = df[df["state"].str.lower() == args.state.lower()].copy()
        logger.info("Filtered to " + args.state + ": " + str(len(df)) + " records")

    if args.skip_states:
        skip_lower = [s.lower() for s in args.skip_states]
        df = df[~df["state"].str.lower().isin(skip_lower)].copy()
        logger.info("Skipping " + str(len(args.skip_states)) + " states | Remaining: " + str(len(df)))

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limit applied: " + str(len(df)) + " records")

    logger.info("Total to process: " + str(len(df)))

    # ── Resume logic ──────────────────────────────────────────────────────────
    already_done: set = set()
    resume_path = None
    enriched: list = []

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            done_df = pd.read_csv(resume_path, dtype=str)
            already_done = set(done_df["full_name"].dropna().str.strip())
            enriched = done_df.to_dict("records")
            logger.info("Resuming: " + str(len(already_done)) + " already done")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = resume_path or (OUTPUT_DIR / ("historico_enriched_" + timestamp + ".csv"))
    logger.info("Output: " + str(output_path))

    records = df.to_dict("records")

    # ── Browser + scraping loop ───────────────────────────────────────────────
    with sync_playwright() as p:
        context = make_browser_context(p, headless=not args.no_headless)
        ig_page = context.new_page()

        # Brief warmup: navigate to a neutral page so the browser fingerprint
        # is established before hitting DuckDuckGo/Instagram
        logger.info("Warming up browser...")
        try:
            ig_page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15_000)
            time.sleep(random.uniform(2.0, 3.5))
        except Exception:
            pass

        bar = tqdm(records, desc="Historico IG", unit="realtor")
        for rec in bar:
            name = rec.get("full_name", "").strip()
            if not name or name in already_done:
                continue

            row = _base_row(rec)

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
                    "ig_posts_spanish":   ig.get("ig_posts_spanish"),
                    "ig_mentions_latino": ig.get("ig_mentions_latino"),
                    "ig_nahrep":          ig.get("ig_nahrep"),
                    "ig_community_type":  ig.get("ig_community_type"),
                    "ig_collaborates":    ig.get("ig_collaborates"),
                    "ig_area_mentions":   ig.get("ig_area_mentions"),
                    "ig_content_language":ig.get("ig_content_language"),
                    "ig_engagement_proxy":ig.get("ig_engagement_proxy"),
                })

            enriched.append(row)
            already_done.add(name)
            _save(enriched, output_path)

            time.sleep(random.uniform(*DELAY_IG))

            ig_found = sum(1 for r in enriched if r.get("ig_handle"))
            bar.set_postfix({"ig": ig_found, "last": name[:20]})

        context.close()

    _save(enriched, output_path)
    _summary(enriched, output_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_row(rec: dict) -> dict:
    return {
        "full_name":   rec.get("full_name", "").strip(),
        "first_name":  rec.get("first_name", "").strip().title(),
        "last_name":   rec.get("last_name", "").strip().title(),
        "company":     rec.get("company", "").strip(),
        "state":       rec.get("state", "").strip(),
        "email":       rec.get("email", "").strip() or None,
        "phone":       rec.get("phone", "").strip() or None,
        "units_sold":  rec.get("units_sold", "").strip() or None,
        "loan_volume": rec.get("loan_volume", "").strip() or None,
        "lead_status": rec.get("lead_status", "").strip(),
        "label":       rec.get("label"),
    }


def _save(records: list, path: Path):
    if not records:
        return
    out = pd.DataFrame(records)
    out = out.drop_duplicates(subset=["full_name"], keep="last")
    cols = [c for c in COL_ORDER if c in out.columns]
    out[cols].to_csv(path, index=False, encoding="utf-8")


def _summary(records: list, path: Path):
    df = pd.DataFrame(records)
    total = len(df)
    if total == 0:
        logger.warning("No records processed.")
        return

    def pct(n):
        return str(n) + " (" + str(round(n / total * 100)) + "%)"

    qual   = int((df["label"].astype(str) == "1").sum())
    disc   = int((df["label"].astype(str) == "0").sum())
    ig     = int(df["ig_handle"].notna().sum())             if "ig_handle"          in df.columns else 0
    nahrep = int(df["ig_nahrep"].astype(str).eq("True").sum()) if "ig_nahrep"       in df.columns else 0
    latino = int(df["ig_mentions_latino"].astype(str).eq("True").sum()) if "ig_mentions_latino" in df.columns else 0
    span_p = int(df["ig_posts_spanish"].astype(str).eq("True").sum()) if "ig_posts_spanish"    in df.columns else 0
    span_s = int(df["ig_spanish_signals"].astype(str).eq("True").sum()) if "ig_spanish_signals" in df.columns else 0

    logger.info("=" * 55)
    logger.info("Total procesados       : " + str(total))
    logger.info("Qualified  (label=1)   : " + str(qual))
    logger.info("Discarded  (label=0)   : " + str(disc))
    logger.info("IG encontrado          : " + pct(ig))
    logger.info("Spanish signals (bio)  : " + pct(span_s))
    logger.info("Posts en espanol       : " + pct(span_p))
    logger.info("Menciona latino/NAHREP : " + pct(latino))
    logger.info("NAHREP especifico      : " + str(nahrep))
    logger.info("Output: " + str(path))
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
