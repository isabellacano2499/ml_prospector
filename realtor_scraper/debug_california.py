import pandas as pd, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR   = Path(__file__).parent / "output" / "model"
OUTPUT_DIR  = Path(__file__).parent / "output"

mmi         = pd.read_csv(sorted(OUTPUT_DIR.glob("mmi_scored_*.csv"))[-1])
hist        = pd.read_csv(MODEL_DIR / "historico_scored.csv")
state_rates = json.load(open(MODEL_DIR / "state_rates.json"))
importances = pd.read_csv(MODEL_DIR / "feature_importances.csv", index_col=0)
importances.columns = ["imp"]

# ── California en MMI
ca = mmi[mmi["state"].str.lower() == "california"].copy()
print(f"California en MMI: {len(ca)} realtors")
print(f"  Score: mean={ca['propensity_score'].mean():.1f}  max={ca['propensity_score'].max():.1f}  min={ca['propensity_score'].min():.1f}")
print(f"  Tiers: {ca['priority_tier'].value_counts().to_dict()}")
ca_rate = ca["state_conversion_rate"].iloc[0] if len(ca) else 0
print(f"  state_conversion_rate: {ca_rate:.4f}  ({ca_rate*100:.1f}%)")
hisp = ca["census_hispanic_pct"].iloc[0] if len(ca) else 0
print(f"  census_hispanic_pct:   {hisp:.4f}")
print(f"  Con ig_content_language=spanish: {(ca['ig_content_language']=='spanish').sum()}")
print(f"  Con ig_content_language=mixed:   {(ca['ig_content_language']=='mixed').sum()}")

print()
# ── California en historico (training)
state_col = "state_clean" if "state_clean" in hist.columns else "state"
ca_h = hist[hist[state_col].str.lower().str.strip() == "california"]
print(f"California en historico (training): {len(ca_h)} realtors")
if len(ca_h):
    print(f"  Qualified (1): {(ca_h['label']==1).sum()} ({(ca_h['label']==1).mean()*100:.1f}%)")
    print(f"  Discarded (0): {(ca_h['label']==0).sum()} ({(ca_h['label']==0).mean()*100:.1f}%)")
else:
    print("  *** NO HAY REGISTROS DE CALIFORNIA EN EL HISTORICO ***")
    print("  -> state_conversion_rate = global fallback")

print()
# ── state_rates.json — que valor tiene CA
ca_key = None
for k in state_rates:
    if "ca" in k.lower() or "california" in k.lower():
        ca_key = k
        break
print(f"California en state_rates.json: key='{ca_key}' valor={state_rates.get(ca_key, 'NO ENCONTRADO')}")
print(f"Global rate fallback: {state_rates.get('__global__', 'no guardado')}")

print()
# ── Comparar conversion rate de todos los estados en MMI
print("state_conversion_rate por estado (MMI):")
state_agg = mmi.groupby("state").agg(
    n=("full_name","count"),
    conv_rate=("state_conversion_rate","mean"),
    avg_score=("propensity_score","mean"),
    n_tier_a=("priority_tier", lambda x: x.str.startswith("A").sum()),
).sort_values("conv_rate")
for st, row in state_agg.iterrows():
    flag = " <<< CALIFORNIA" if st.lower() == "california" else ""
    print(f"  {st:<25} conv={row['conv_rate']:.3f} ({row['conv_rate']*100:.1f}%)  avg_score={row['avg_score']:.1f}  tier_A={int(row['n_tier_a'])}{flag}")

print()
# ── Top CA realtors con español — por que no suben
print("Top 10 California con ig_content_language=spanish o mixed:")
ca_esp = ca[ca["ig_content_language"].isin(["spanish","mixed"])].sort_values("propensity_score", ascending=False)
if len(ca_esp):
    print(ca_esp[["propensity_score","priority_tier","full_name","ig_content_language",
                   "ig_mentions_latino","company","state_conversion_rate","census_hispanic_pct"]].head(10).to_string())
else:
    print("  Ninguno con español/mixto en CA")

print()
# ── Importancia de state_conversion_rate
conv_imp = importances.loc["state_conversion_rate","imp"] if "state_conversion_rate" in importances.index else 0
total    = importances["imp"].sum()
print(f"state_conversion_rate importancia: {conv_imp:.4f} ({conv_imp/total*100:.1f}% del modelo)")
if conv_imp == 0:
    print("-> ELIMINADA del modelo correctamente")

# ── California Census profile vs top states
print()
print("California Census vs estados con mejor score:")
mmi_csv = sorted(OUTPUT_DIR.glob("mmi_scored_*.csv"))[-1]
full = pd.read_csv(mmi_csv)
census_cols = ["census_hispanic_pct","census_bilingual_spanish_pct",
               "census_lep_spanish_pct","census_overall_score",
               "census_first_time_buyer_potential"]
compare_states = ["California","Nevada","Michigan","Minnesota","Missouri","Ohio","Texas"]
for st in compare_states:
    rows = full[full["state"]==st]
    if len(rows) == 0:
        continue
    vals = {c: pd.to_numeric(rows[c], errors="coerce").mean() for c in census_cols if c in rows.columns}
    avg_sc = rows["propensity_score"].mean()
    print(f"\n  {st} (n={len(rows)}, avg_score={avg_sc:.1f}):")
    for k, v in vals.items():
        col_short = k.replace("census_","").replace("_"," ")
        print(f"    {col_short:<35} {v:.4f}")
