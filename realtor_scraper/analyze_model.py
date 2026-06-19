"""Honest model quality analysis."""
import sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "output" / "model"

features    = json.load(open(MODEL_DIR / "features.json"))
importances = pd.read_csv(MODEL_DIR / "feature_importances.csv", index_col=0)
hist        = pd.read_csv(MODEL_DIR / "historico_scored.csv")
mmi         = pd.read_csv(sorted((ROOT / "output").glob("mmi_scored_*.csv"))[-1])

importances.columns = ["imp"]
importances = importances.sort_values("imp", ascending=False)
total_imp   = importances["imp"].sum()

q = hist[hist["label"] == 1]["propensity_score"]
d = hist[hist["label"] == 0]["propensity_score"]

train_auc = roc_auc_score(hist["label"], hist["propensity_score"] / 100)
train_ap  = average_precision_score(hist["label"], hist["propensity_score"] / 100)
CV_AUC    = 0.702
CV_STD    = 0.014

# Confusion matrix at threshold 50
y_true = hist["label"].values
y_pred = (hist["propensity_score"] >= 50).astype(int).values
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("=" * 65)
print("ANALISIS HONESTO DEL MODELO — XGBoost Realtor Propensity")
print("=" * 65)

print("\n--- 1. OVERFITTING / UNDERFITTING ---")
print(f"  AUC en TRAINING SET:      {train_auc:.4f}")
print(f"  AUC en CV (5-fold):       {CV_AUC:.4f} +/- {CV_STD:.4f}")
print(f"  GAP (train - cv):         {train_auc - CV_AUC:+.4f}")
print(f"  -> {'LIGERO overfitting (normal en XGB)' if train_auc - CV_AUC < 0.05 else 'OVERFITTING SIGNIFICATIVO'}")

print("\n--- 2. CONFUSION MATRIX (threshold = 50) ---")
print(f"  Verdaderos Positivos (TP): {tp}  — Qualified bien predichos")
print(f"  Falsos Negativos  (FN):    {fn}  — Qualified perdidos ({fn/(tp+fn)*100:.1f}%)")
print(f"  Verdaderos Negativos (TN): {tn}  — Discarded bien predichos")
print(f"  Falsos Positivos  (FP):    {fp}  — Discarded mal predichos ({fp/(tn+fp)*100:.1f}%)")
print(f"  Precision:   {tp/(tp+fp)*100:.1f}%  (de los que el modelo dice 'si', cuantos son qualified)")
print(f"  Recall:      {tp/(tp+fn)*100:.1f}%  (de los qualified, cuantos detecta)")
print(f"  F1:          {2*tp/(2*tp+fp+fn):.3f}")

print("\n--- 3. SEPARACION REAL ENTRE CLASES ---")
print(f"  Qualified — mean={q.mean():.1f}  median={q.median():.1f}  p25={q.quantile(.25):.1f}  p75={q.quantile(.75):.1f}")
print(f"  Discarded — mean={d.mean():.1f}  median={d.median():.1f}  p25={d.quantile(.25):.1f}  p75={d.quantile(.75):.1f}")
overlap = ((q < 50).sum() + (d > 50).sum())
total_poss = len(q) + len(d)
print(f"  Overlap a threshold 50: {overlap} registros errados ({overlap/total_poss*100:.1f}% del set)")

print("\n--- 4. TOP 15 FEATURES (por importancia ganada) ---")
for i, (feat, row) in enumerate(importances.head(15).iterrows()):
    bar = "#" * int(row["imp"] * 200)
    pct = row["imp"] / total_imp * 100
    print(f"  {i+1:2}. {feat:<42} {row['imp']:.4f}  ({pct:.1f}%)  {bar}")

print(f"\n  Concentracion: top-1 = {importances.iloc[0,0]/total_imp*100:.1f}%  |  top-3 = {importances.head(3)['imp'].sum()/total_imp*100:.1f}%  |  top-5 = {importances.head(5)['imp'].sum()/total_imp*100:.1f}%")

print("\n--- 5. CALL FEATURES: PROBLEMA ESTRUCTURAL ---")
call_feats = ["was_called", "call_duration_min", "total_calls_clip", "answer_rate"]
call_imp   = importances.loc[importances.index.isin(call_feats)]
call_total = call_imp["imp"].sum()
for feat, row in call_imp.iterrows():
    vals    = pd.to_numeric(hist.get(feat, pd.Series(0, index=hist.index)), errors="coerce")
    nonzero = (vals > 0).sum()
    print(f"  {feat:<30} imp={row['imp']:.4f} ({row['imp']/total_imp*100:.1f}%)  nonzero en train={nonzero}/{len(hist)} ({nonzero/len(hist)*100:.1f}%)")
print(f"  -> Call features acumulan {call_total/total_imp*100:.1f}% del modelo")
print(f"     En MMI estas 4 features son TODAS 0 (nunca llamados)")
print(f"     El modelo fue entrenado con {(pd.to_numeric(hist.get('was_called',0),errors='coerce')>0).sum()/len(hist)*100:.1f}% del dataset con calls=0 tambien")

print("\n--- 6. DISTRIBUCION MMI SCORING ---")
mmi_s = mmi["propensity_score"]
print(f"  MMI:      mean={mmi_s.mean():.1f}  median={mmi_s.median():.1f}  std={mmi_s.std():.1f}  max={mmi_s.max():.1f}  min={mmi_s.min():.1f}")
print(f"  Training: mean={(q.mean()+d.mean())/2:.1f}  (Q={q.mean():.1f}, D={d.mean():.1f})")
print(f"  -> El MMI se parece mas a la clase Discarded? {'SI' if abs(mmi_s.mean()-d.mean()) < abs(mmi_s.mean()-q.mean()) else 'NO'} (MMI media={mmi_s.mean():.1f}, D media={d.mean():.1f}, Q media={q.mean():.1f})")
print(f"\n  MMI por buckets:")
for lo, hi in [(0,20),(20,40),(40,60),(60,80),(80,100)]:
    n = int(((mmi_s >= lo) & (mmi_s < hi)).sum())
    pct = n / len(mmi_s) * 100
    bar = "#" * int(pct / 2)
    print(f"    {lo:3}-{hi}: {n:5} ({pct:5.1f}%)  {bar}")

print("\n--- 7. VARIABLES CON MUCHOS NULOS EN ENTRENAMIENTO ---")
key_features = ["units_sold","ig_followers","ig_posts","ig_engagement_proxy",
                "ig_handle","ig_bio","ig_spanish_signals","ig_mentions_latino"]
for col in key_features:
    if col in hist.columns:
        null_pct = hist[col].isna().mean() * 100
        if col == "ig_handle":
            null_pct = hist[col].isna().mean() * 100
        print(f"  {col:<40} {null_pct:.1f}% nulos en training")

print("\n--- 8. VEREDICTO FINAL ---")
print(f"  AUC CV = {CV_AUC:.3f} -> En contexto: random=0.5, perfecto=1.0, util_en_negocio=0.7+")
print(f"  Este modelo es UTIL pero no PRECISO en el sentido literal.")
