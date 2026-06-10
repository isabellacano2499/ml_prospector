"""
Build Training Dataset — Paso 2 & 3
=====================================
1. Carga los CSVs del historico enriquecidos (todas las PCs)
2. Agrega Census ZIP data a nivel estado (promedio ponderado por poblacion)
3. Une Census state metrics a cada realtor por estado
4. Extrae features de nombre de empresa (spanish_company, latino_brokerage)
5. Guarda: realtor_scraper/output/historico_training_ready.csv

Usage:
  python build_training_dataset.py
"""
import sys
import re
import pandas as pd
import pgeocode
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT       = Path(__file__).parent
CENSUS_CSV = ROOT.parent / "latino_re_engine" / "data" / "output" / "latino_market_zip_2024.csv"
OUTPUT_DIR = ROOT / "output"
OUT_FILE   = OUTPUT_DIR / "historico_training_ready.csv"

STATE_ABBREV = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
    "PR":"Puerto Rico",
}
STATE_NAME_TO_ABBREV = {v: k for k, v in STATE_ABBREV.items()}

# Companies strongly associated with Latino/Hispanic agent market
LATINO_BROKERAGES = {
    "la rosa", "casa buena", "movil realty", "vive realty", "agent trust",
    "naim real estate", "home prime", "casa", "hogar", "latin", "hispano",
    "hispanic", "latino", "habla", "bilingual", "multicultural",
}

# Spanish words that might appear in company names
SPANISH_NAME_WORDS = {
    "casa", "hogar", "vive", "movil", "buena", "bueno", "real", "realty",
    "casas", "hogares", "vida", "sol", "estrella", "luna", "tierra",
    "familia", "unidos", "prime", "latina", "latino",
}


# ── Step 1: Load and merge all historico CSVs ─────────────────────────────────

print("Loading historico CSVs...")
hist_files = sorted(OUTPUT_DIR.glob("historico_enriched_*.csv"))
print(f"  Files found: {len(hist_files)}")

dfs = []
for f in hist_files:
    dfs.append(pd.read_csv(f, dtype=str))

historico = pd.concat(dfs, ignore_index=True)
historico = historico.drop_duplicates(subset=["full_name"], keep="last")
historico["label"] = pd.to_numeric(historico["label"], errors="coerce")
historico = historico[historico["label"].notna()].copy()
historico["label"] = historico["label"].astype(int)

print(f"  Total realtors: {len(historico)}")
print(f"  Qualified (1): {(historico['label']==1).sum()}")
print(f"  Discarded (0): {(historico['label']==0).sum()}")
print(f"  Instagram hit: {historico['ig_handle'].notna().sum()}")


# ── Step 2: Build Census state-level aggregates ───────────────────────────────

print("\nBuilding Census state aggregates...")
census = pd.read_csv(CENSUS_CSV)

# Map ZIP codes to state abbreviations using pgeocode
print("  Mapping ZIP codes to states (this takes ~1 min)...")
nomi      = pgeocode.Nominatim("us")
zip_str   = census["zip_code"].astype(str).str.zfill(5).tolist()

# Batch in chunks of 5000 to avoid memory issues
chunk     = 5000
geo_parts = []
for i in range(0, len(zip_str), chunk):
    geo_parts.append(nomi.query_postal_code(zip_str[i : i + chunk]))
geo = pd.concat(geo_parts, ignore_index=True)

census["state_abbr"] = geo["state_code"].values
census = census[census["state_abbr"].notna()].copy()

# Weighted average by total_population for each state
CENSUS_METRICS = [
    "hispanic_pct",
    "overall_score",
    "median_household_income",
    "first_home_buyer_score",
    "spanish_marketing_opportunity",
    "lep_spanish_pct",
    "bilingual_spanish_pct",
    "homeownership_rate",
    "latino_market_score",
    "first_time_buyer_potential",
]
census[CENSUS_METRICS] = census[CENSUS_METRICS].apply(pd.to_numeric, errors="coerce")
census["total_population"] = pd.to_numeric(census["total_population"], errors="coerce").fillna(0)

def weighted_avg(group):
    pop = group["total_population"]
    total = pop.sum()
    result = {}
    for col in CENSUS_METRICS:
        vals = group[col].fillna(0)
        result["census_" + col] = (vals * pop).sum() / total if total > 0 else None
    return pd.Series(result)

state_census = census.groupby("state_abbr").apply(weighted_avg).reset_index()
state_census["state"] = state_census["state_abbr"].map(STATE_ABBREV)

print(f"  States with Census data: {len(state_census)}")
print("  Census metrics computed: " + ", ".join("census_" + m for m in CENSUS_METRICS))


# ── Step 3: Join Census to historico ─────────────────────────────────────────

print("\nJoining Census to historico...")

# Normalize state names — fix "FL" abbreviations in raw data
historico["state_clean"] = historico["state"].str.strip()
historico.loc[historico["state_clean"].str.len() == 2, "state_clean"] = (
    historico.loc[historico["state_clean"].str.len() == 2, "state_clean"]
    .str.upper()
    .map(STATE_ABBREV)
)

state_census_lookup = state_census.set_index("state")[
    ["state_abbr"] + ["census_" + m for m in CENSUS_METRICS]
]
historico = historico.merge(
    state_census_lookup,
    left_on="state_clean",
    right_index=True,
    how="left",
)

census_joined = historico["census_hispanic_pct"].notna().sum()
print(f"  Realtors with Census data joined: {census_joined} ({census_joined/len(historico)*100:.0f}%)")


# ── Step 4: Company name features ────────────────────────────────────────────

print("\nExtracting company features...")

def has_spanish_company(name: str) -> bool:
    if not name:
        return False
    n = name.lower()
    return any(w in n for w in SPANISH_NAME_WORDS)

def is_latino_brokerage(name: str) -> bool:
    if not name:
        return False
    n = name.lower()
    return any(w in n for w in LATINO_BROKERAGES)

historico["company_has_spanish_name"] = historico["company"].fillna("").apply(has_spanish_company)
historico["company_is_latino_brokerage"] = historico["company"].fillna("").apply(is_latino_brokerage)

print(f"  Spanish company name: {historico['company_has_spanish_name'].sum()}")
print(f"  Latino brokerage: {historico['company_is_latino_brokerage'].sum()}")


# ── Step 5: Numeric casting of ML features ────────────────────────────────────

print("\nCasting feature types...")

NUMERIC_COLS = [
    "units_sold", "ig_followers", "ig_posts", "ig_engagement_proxy",
] + ["census_" + m for m in CENSUS_METRICS]

BOOL_COLS = [
    "ig_spanish_signals", "ig_realtor_signals", "ig_posts_spanish",
    "ig_mentions_latino", "ig_nahrep", "ig_collaborates", "ig_is_private",
    "company_has_spanish_name", "company_is_latino_brokerage",
]

for col in NUMERIC_COLS:
    if col in historico.columns:
        historico[col] = pd.to_numeric(historico[col], errors="coerce")

for col in BOOL_COLS:
    if col in historico.columns:
        historico[col] = historico[col].map(
            {"True": True, "False": False, True: True, False: False}
        ).astype("boolean")

# Clip units_sold outliers (max realistic: 200 units/year)
if "units_sold" in historico.columns:
    historico["units_sold"] = historico["units_sold"].clip(upper=200)


# ── Step 6: Save ─────────────────────────────────────────────────────────────

COL_ORDER = [
    # Identity
    "full_name", "first_name", "last_name", "company", "state", "state_clean", "state_abbr",
    # Label
    "label",
    # Original features
    "units_sold",
    # Company features
    "company_has_spanish_name", "company_is_latino_brokerage",
    # Instagram features
    "ig_handle", "ig_followers", "ig_posts", "ig_engagement_proxy",
    "ig_spanish_signals", "ig_realtor_signals",
    "ig_posts_spanish", "ig_mentions_latino", "ig_nahrep",
    "ig_community_type", "ig_collaborates",
    "ig_area_mentions", "ig_content_language", "ig_is_private",
    "ig_bio",
    # Census features
    "census_hispanic_pct", "census_overall_score",
    "census_median_household_income", "census_first_home_buyer_score",
    "census_spanish_marketing_opportunity", "census_lep_spanish_pct",
    "census_bilingual_spanish_pct", "census_homeownership_rate",
    "census_latino_market_score", "census_first_time_buyer_potential",
]

cols = [c for c in COL_ORDER if c in historico.columns]
historico[cols].to_csv(OUT_FILE, index=False, encoding="utf-8")

print(f"\nGuardado: {OUT_FILE}")
print(f"Total filas: {len(historico)}")
print(f"Total columnas: {len(cols)}")

print("\n=== RESUMEN DE FEATURES ===")
feature_cols = [c for c in cols if c not in (
    "full_name","first_name","last_name","company","state","state_clean",
    "state_abbr","label","ig_handle","ig_bio","ig_area_mentions","ig_community_type",
    "ig_content_language"
)]
print(f"Features numericas/booleanas para ML: {len(feature_cols)}")
for c in feature_cols:
    non_null = historico[c].notna().sum()
    pct = non_null / len(historico) * 100
    print(f"  {c}: {non_null:,}/{len(historico):,} ({pct:.0f}%)")
