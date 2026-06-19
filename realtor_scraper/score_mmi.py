"""
Score MMI Realtors — aplica el modelo XGBoost entrenado
=========================================================
1. Carga y combina todos los mmi_enriched_*.csv
2. Aplica el mismo feature engineering del modelo
3. Une Census por estado
4. Pone call features en 0 (nunca han sido llamados)
5. Genera score 0-100 por realtor
6. Guarda: output/mmi_scored_YYYYMMDD.csv  (ordenado por score desc)

Usage:
  python score_mmi.py
"""
import sys, json, joblib, re
import numpy as np
import pandas as pd
import pgeocode
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "output" / "model"
OUT_DIR   = ROOT / "output"
CENSUS_CSV = ROOT.parent / "latino_re_engine" / "data" / "output" / "latino_market_zip_2024.csv"

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

LATINO_BROKERAGES = {
    "la rosa","casa buena","movil realty","vive realty","agent trust",
    "naim real estate","home prime","casa","hogar","latin","hispano",
    "hispanic","latino","habla","bilingual","multicultural",
}
SPANISH_NAME_WORDS = {
    "casa","hogar","vive","movil","buena","bueno",
    "casas","hogares","vida","sol","estrella","luna","tierra",
    "familia","unidos","latina","latino",
}
CENSUS_METRICS = [
    "hispanic_pct","overall_score","median_household_income",
    "first_home_buyer_score","spanish_marketing_opportunity",
    "lep_spanish_pct","bilingual_spanish_pct","homeownership_rate",
    "latino_market_score","first_time_buyer_potential",
]

print("=" * 55)
print("Scoring MMI Realtors con XGBoost")
print("=" * 55)

# ── Load model artifacts ──────────────────────────────────────────────────────
print("\nCargando modelo...")
model       = joblib.load(MODEL_DIR / "xgboost_model.pkl")
features    = json.load(open(MODEL_DIR / "features.json"))
medians     = json.load(open(MODEL_DIR / "medians.json"))
state_rates = json.load(open(MODEL_DIR / "state_rates.json"))
global_rate = state_rates.pop("__global__", 0.57)
print(f"  Features esperadas: {len(features)}")

# ── Load and merge all MMI enriched CSVs ─────────────────────────────────────
print("\nCargando MMI enriched CSVs...")
files = sorted(OUT_DIR.glob("mmi_enriched_*.csv"), key=lambda p: p.stat().st_mtime)
print(f"  Archivos: {len(files)}")
dfs = []
for f in files:
    d = pd.read_csv(f, dtype=str)
    if "batch_name" not in d.columns:
        d["batch_name"] = f.stem.replace("mmi_enriched_", "carga_")
    dfs.append(d)
df  = pd.concat(dfs, ignore_index=True)
df  = df.drop_duplicates(subset=["full_name"], keep="last")
print(f"  Realtors unicos: {len(df):,}")
print(f"  Instagram hit: {df['ig_handle'].notna().sum():,} ({df['ig_handle'].notna().mean()*100:.0f}%)")

# ── Census state aggregates ───────────────────────────────────────────────────
print("\nComputando Census por estado...")
census = pd.read_csv(CENSUS_CSV)
nomi   = pgeocode.Nominatim("us")
zip_str = census["zip_code"].astype(str).str.zfill(5).tolist()
chunk   = 5000
parts   = []
for i in range(0, len(zip_str), chunk):
    parts.append(nomi.query_postal_code(zip_str[i:i+chunk]))
geo = pd.concat(parts, ignore_index=True)
census["state_abbr"] = geo["state_code"].values
census = census[census["state_abbr"].notna()].copy()
census[CENSUS_METRICS] = census[CENSUS_METRICS].apply(pd.to_numeric, errors="coerce")
census["total_population"] = pd.to_numeric(census["total_population"], errors="coerce").fillna(0)

def weighted_avg(group):
    pop = group["total_population"]
    total = pop.sum()
    return pd.Series({
        "census_" + m: (group[m].fillna(0) * pop).sum() / total if total > 0 else None
        for m in CENSUS_METRICS
    })

state_census = census.groupby("state_abbr").apply(weighted_avg).reset_index()
state_census["state"] = state_census["state_abbr"].map(STATE_ABBREV)
census_lookup = state_census.set_index("state")[
    ["state_abbr"] + ["census_" + m for m in CENSUS_METRICS]
]
print(f"  Estados con Census: {len(state_census)}")

# Save state-level Hispanic population totals for the dashboard map
_hisp_series = (
    census["hispanic_population"] if "hispanic_population" in census.columns
    else pd.Series(0.0, index=census.index)
)
census["_hisp_abs"] = pd.to_numeric(_hisp_series, errors="coerce").fillna(0)
state_pop = census.groupby("state_abbr").agg(
    total_population = ("total_population", "sum"),
    hispanic_pop     = ("_hisp_abs",        "sum"),
).reset_index()
state_pop["hispanic_pct"] = (
    state_pop["hispanic_pop"] / state_pop["total_population"].replace(0, np.nan)
).fillna(0)
state_pop["state"] = state_pop["state_abbr"].map(STATE_ABBREV)
state_pop = state_pop.dropna(subset=["state"])
state_pop.to_csv(OUT_DIR / "state_census_summary.csv", index=False, encoding="utf-8")
print(f"  Guardado: state_census_summary.csv ({len(state_pop)} estados)")

# ── Feature engineering ───────────────────────────────────────────────────────
print("\nEngineering features...")

# Normalize state
df["state_clean"] = df["state"].str.strip()
df.loc[df["state_clean"].str.len() == 2, "state_clean"] = (
    df.loc[df["state_clean"].str.len() == 2, "state_clean"]
    .str.upper().map(STATE_ABBREV)
)

# Join Census
df = df.merge(census_lookup, left_on="state_clean", right_index=True, how="left")

# Company features — word-boundary matching so "latin" no matchea "platinum"
def _keyword_match(text: str, keywords: set) -> bool:
    n = text.lower()
    for w in keywords:
        if " " in w:          # frases multi-palabra: substring exacto es suficiente
            if w in n:
                return True
        else:                 # palabra sola: requiere limite de palabra
            if re.search(r"\b" + re.escape(w) + r"\b", n):
                return True
    return False

df["company_has_spanish_name"]    = df["company"].fillna("").apply(lambda n: _keyword_match(n, SPANISH_NAME_WORDS))
df["company_is_latino_brokerage"] = df["company"].fillna("").apply(lambda n: _keyword_match(n, LATINO_BROKERAGES))

# Log transforms
for col in ["ig_followers", "ig_posts", "ig_engagement_proxy"]:
    src = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df[col + "_log"] = np.log1p(src)

# has_instagram — solo cuenta si hay datos reales scrapeados (bio, followers o posts)
ig_has_data = (
    df["ig_bio"].notna() |
    df["ig_followers"].notna() |
    df["ig_posts"].notna()
)
df["has_instagram"] = (df["ig_handle"].notna() & ig_has_data).astype(int)

# content_lang_score
df["content_lang_score"] = (
    df["ig_content_language"].map({"spanish": 2, "mixed": 1, "english": 0}).fillna(0)
)

# Community type flags
for ctype in ["first_time_buyer", "family_community", "relocation",
              "veterans", "luxury", "investor"]:
    df["comm_" + ctype] = (
        df["ig_community_type"].fillna("").str.contains(ctype).astype(int)
    )

# State conversion rate from training history
df["state_conversion_rate"] = df["state_abbr"].map(state_rates).fillna(global_rate)

# Boolean cols
BOOL_COLS = [
    "has_instagram","company_has_spanish_name","company_is_latino_brokerage",
    "ig_spanish_signals","ig_realtor_signals","ig_posts_spanish",
    "ig_mentions_latino","ig_nahrep","ig_collaborates","ig_is_private",
]
for col in BOOL_COLS:
    if col in df.columns:
        df[col] = (
            df[col].map({True:1, False:0, "True":1, "False":0})
            .fillna(0).astype(int)
        )

# Build feature matrix in exact model order
X = pd.DataFrame(index=df.index)
for feat in features:
    if feat in df.columns:
        X[feat] = pd.to_numeric(df[feat], errors="coerce")
    else:
        X[feat] = 0.0
    X[feat] = X[feat].fillna(medians.get(feat, 0.0))

print(f"  Feature matrix: {X.shape[0]:,} x {X.shape[1]}")
print(f"  NaN residual: {X.isna().sum().sum()}")

# ── Score ─────────────────────────────────────────────────────────────────────
print("\nGenerando scores...")
proba  = model.predict_proba(X)[:, 1]
scores = (proba * 100).round(1)
df["propensity_score"] = scores

# ── Priority tier ─────────────────────────────────────────────────────────────
def tier(s):
    if s >= 80: return "A — Prioridad Alta"
    if s >= 65: return "B — Prioridad Media"
    if s >= 50: return "C — Seguimiento"
    return "D — Baja prioridad"

df["priority_tier"] = df["propensity_score"].apply(tier)

# ── Save output ───────────────────────────────────────────────────────────────
out_cols = [
    "propensity_score", "priority_tier",
    "full_name", "company", "state", "batch_name",
    "email_mmi", "phone_mmi", "units_sold",
    "ig_handle", "ig_url", "ig_followers", "ig_bio",
    "ig_spanish_signals", "ig_posts_spanish", "ig_mentions_latino",
    "ig_nahrep", "ig_community_type", "ig_content_language",
    "company_has_spanish_name", "company_is_latino_brokerage",
    "census_hispanic_pct", "census_overall_score",
    "state_conversion_rate",
]
out_cols = [c for c in out_cols if c in df.columns]
result   = df[out_cols].sort_values("propensity_score", ascending=False).reset_index(drop=True)

ts       = datetime.now().strftime("%Y%m%d_%H%M")
out_path = OUT_DIR / f"mmi_scored_{ts}.csv"
result.to_csv(out_path, index=False, encoding="utf-8")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"RESULTADO FINAL: {len(result):,} realtors puntuados")
print(f"{'='*55}")
print(f"\nDistribucion de scores:")
print(f"  Media:   {scores.mean():.1f}")
print(f"  Mediana: {float(np.median(scores)):.1f}")
print(f"  Max:     {scores.max():.1f}")
print(f"  Min:     {scores.min():.1f}")

print(f"\nTiers de prioridad:")
tier_counts = result["priority_tier"].value_counts().sort_index()
for t, n in tier_counts.items():
    print(f"  {t}: {n:,} realtors")

print(f"\nTop 20 realtors para llamar HOY:")
top = result.head(20)[["propensity_score","full_name","company","state",
                         "phone_mmi","ig_handle","ig_followers"]]
print(top.to_string(index=False))

print(f"\nGuardado en: {out_path}")
