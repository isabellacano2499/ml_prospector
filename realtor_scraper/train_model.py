"""
XGBoost Realtor Propensity Model
==================================
Entrena el modelo que predice si un realtor es Qualified (1) o Discarded (0).
Guarda el modelo y todos los artefactos necesarios para puntuar nuevos realtors.

Usage:
  python train_model.py
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).parent
DATA_FILE = ROOT / "output" / "historico_training_ready.csv"
MODEL_DIR = ROOT / "output" / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Feature definitions ───────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "units_sold",
    "ig_followers_log",
    "ig_posts_log",
    "ig_engagement_proxy_log",
    "content_lang_score",
]

BOOLEAN_FEATURES = [
    "has_instagram",
    "company_has_spanish_name",
    "company_is_latino_brokerage",
    "ig_spanish_signals",
    "ig_realtor_signals",
    "ig_posts_spanish",
    "ig_mentions_latino",
    "ig_nahrep",
    "ig_collaborates",
    "ig_is_private",
]

COMMUNITY_FLAGS = [
    "comm_first_time_buyer",
    "comm_family_community",
    "comm_relocation",
    "comm_veterans",
    "comm_luxury",
    "comm_investor",
]

ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + COMMUNITY_FLAGS


# ── Load data ─────────────────────────────────────────────────────────────────

print("=" * 55)
print("PASO 4: XGBoost Realtor Propensity Model")
print("=" * 55)
print(f"\nCargando: {DATA_FILE.name}")
df = pd.read_csv(DATA_FILE)
print(f"  Filas: {len(df):,}  |  Columnas: {len(df.columns)}")
print(f"  Qualified (1): {(df['label']==1).sum():,}")
print(f"  Discarded (0): {(df['label']==0).sum():,}")


# ── Feature engineering ───────────────────────────────────────────────────────

print("\nEngineering features...")

# Log-transform skewed counts (Instagram followers/posts can be 0 to 116K)
for col in ["ig_followers", "ig_posts", "ig_engagement_proxy"]:
    src = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df[col + "_log"] = np.log1p(src)

# Has Instagram at all (binary)
df["has_instagram"] = df["ig_handle"].notna().astype(int)

# Content language → ordinal score: spanish=2, mixed=1, english=0
df["content_lang_score"] = (
    df["ig_content_language"]
    .map({"spanish": 2, "mixed": 1, "english": 0})
    .fillna(0)
)

# Community type flags from comma-separated string
for ctype in ["first_time_buyer", "family_community", "relocation",
              "veterans", "luxury", "investor"]:
    df["comm_" + ctype] = (
        df["ig_community_type"].fillna("").str.contains(ctype).astype(int)
    )

# Target-encode state with Bayesian smoothing.
# Raw rates on small states are unreliable: Nevada had n=22 → 100%, Rhode Island n=1 → 100%.
# Smoothing formula: (state_positives + m * global_rate) / (state_n + m)
# m=30 means: a state needs ≥30 samples before its rate is trusted over the global average.
global_rate   = df["label"].mean()
m             = 30
state_counts  = df.groupby("state_abbr")["label"].count()
state_sums    = df.groupby("state_abbr")["label"].sum()
state_rates   = ((state_sums + m * global_rate) / (state_counts + m)).rename("state_conversion_rate")
df["state_conversion_rate"] = df["state_abbr"].map(state_rates).fillna(global_rate)

print("\nstate_conversion_rate (smoothed, top estados en MMI):")
for st, raw, smooth in sorted(
    [(s, state_sums[s]/state_counts[s], state_rates[s])
     for s in ["CA","NV","TX","FL","RI","MO","MI","OH"] if s in state_rates.index],
    key=lambda x: x[2]
):
    n = int(state_counts[st])
    print(f"  {st}  n={n:4}  raw={raw:.3f}  smoothed={smooth:.3f}")

# Boolean columns: True/False → 1/0
for col in BOOLEAN_FEATURES:
    if col in df.columns:
        df[col] = (
            df[col].map({True: 1, False: 0, "True": 1, "False": 0})
            .fillna(0).astype(int)
        )

# Build feature matrix
X = df[ALL_FEATURES].copy()

# Cast numerics and impute NaN with median
medians = {}
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    med = X[col].median()
    medians[col] = float(med) if not np.isnan(med) else 0.0
    X[col] = X[col].fillna(medians[col])

y = df["label"].astype(int)

print(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
print(f"  NaN residual: {X.isna().sum().sum()} (should be 0)")


# ── Sample weights from call history ─────────────────────────────────────────
# Call data is NOT a feature — it informs how confident we are in each label.
# Qualified + long call  → we're very sure this is a good lead (weight 2.0)
# Qualified + medium call → confident positive (weight 1.5)
# Qualified + short call  → answered but maybe lukewarm (weight 1.0)
# Discarded + was called  → confirmed rejection, trust this label more (weight 1.5)
# Discarded + never called → uncertain; might have answered if called (weight 0.8)

print("\nCalculando sample weights por calidad de llamada...")
_zero        = pd.Series(0.0, index=df.index)
_call_raw    = df["was_called"].fillna(0).astype(float)        if "was_called"        in df.columns else _zero
_dur_raw     = df["call_duration_min"].fillna(0).astype(float) if "call_duration_min" in df.columns else _zero
was_called   = _call_raw > 0
duration_min = _dur_raw
qualified    = y == 1

sample_weights = pd.Series(1.0, index=df.index)
sample_weights[qualified  & was_called & (duration_min >= 2)]               = 2.0
sample_weights[qualified  & was_called & (duration_min >= 1) & (duration_min < 2)] = 1.5
sample_weights[qualified  & was_called & (duration_min < 1)]                = 1.0
sample_weights[qualified  & ~was_called]                                    = 1.0
sample_weights[~qualified & was_called]                                     = 1.5
sample_weights[~qualified & ~was_called]                                    = 0.8

w = sample_weights.values
print(f"  Qualified con llamada larga (>=2 min, w=2.0): {(qualified & was_called & (duration_min>=2)).sum()}")
print(f"  Qualified con llamada media (1-2 min, w=1.5): {(qualified & was_called & (duration_min>=1) & (duration_min<2)).sum()}")
print(f"  Discarded con llamada confirmada (w=1.5):     {(~qualified & was_called).sum()}")
print(f"  Discarded sin llamada (incertidumbre, w=0.8): {(~qualified & ~was_called).sum()}")


# ── Train XGBoost ─────────────────────────────────────────────────────────────

print("\nEntrenando XGBoost...")

# Class ratio (57/43 — almost balanced, minor adjustment)
scale_pw = (y == 0).sum() / (y == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pw,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

# 5-fold stratified cross-validation — manual loop to pass sample_weight per fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
    y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]
    w_tr          = w[tr_idx]
    m_fold = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=scale_pw,
        eval_metric="auc", random_state=42, n_jobs=-1,
    )
    m_fold.fit(X_tr, y_tr, sample_weight=w_tr)
    fold_scores.append(roc_auc_score(y_val, m_fold.predict_proba(X_val)[:, 1]))
    print(f"    Fold {fold}: AUC = {fold_scores[-1]:.4f}")

cv_auc = np.array(fold_scores)
print(f"\n  AUC cross-val (5-fold): {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
print(f"  Folds: {[round(s, 3) for s in cv_auc]}")

# Train final model on all data with call-quality weights
model.fit(X, y, sample_weight=w)

# Training-set metrics (optimistic but useful for threshold calibration)
y_prob  = model.predict_proba(X)[:, 1]
y_pred  = (y_prob >= 0.5).astype(int)
train_auc = roc_auc_score(y, y_prob)

print(f"\n  AUC en training set: {train_auc:.3f}")
print("\n  Classification report (training):")
print(classification_report(y, y_pred, target_names=["Discarded", "Qualified"]))


# ── Feature importances ───────────────────────────────────────────────────────

importances = (
    pd.Series(model.feature_importances_, index=ALL_FEATURES)
    .sort_values(ascending=False)
)

print("\n=== TOP 20 FEATURES MAS IMPORTANTES ===")
for feat, imp in importances.head(20).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<40} {imp:.4f}  {bar}")


# ── Confusion matrix plot ─────────────────────────────────────────────────────

cm = confusion_matrix(y, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay(cm, display_labels=["Discarded", "Qualified"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues"
)
axes[0].set_title("Confusion Matrix (training set)")

importances.head(20).sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title("Top 20 Feature Importances")
axes[1].set_xlabel("Importance (gain)")

plt.tight_layout()
fig.savefig(MODEL_DIR / "model_report.png", dpi=150)
plt.close()
print(f"\n  Plot guardado: {MODEL_DIR / 'model_report.png'}")


# ── Score distribution check ──────────────────────────────────────────────────

scores = (y_prob * 100).round(1)
df["propensity_score"] = scores

print("\n=== DISTRIBUCION DE SCORES 0-100 ===")
for label, name in [(1, "Qualified"), (0, "Discarded")]:
    s = df.loc[df["label"] == label, "propensity_score"]
    print(f"  {name}: mean={s.mean():.1f}  median={s.median():.1f}  "
          f"p25={s.quantile(.25):.1f}  p75={s.quantile(.75):.1f}")

# Threshold analysis
print("\n=== CUANTOS REALTORS QUEDAN A CADA THRESHOLD ===")
for thr in [50, 60, 70, 80]:
    above = (scores >= thr).sum()
    qual_above = ((scores >= thr) & (y == 1)).sum()
    prec = qual_above / above * 100 if above > 0 else 0
    print(f"  Score >= {thr}: {above:,} realtors  |  precision={prec:.0f}% son Qualified")


# ── Save artifacts ────────────────────────────────────────────────────────────

joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")

with open(MODEL_DIR / "features.json", "w") as f:
    json.dump(ALL_FEATURES, f, indent=2)

with open(MODEL_DIR / "medians.json", "w") as f:
    json.dump(medians, f, indent=2)

state_rates_dict = state_rates.to_dict()
state_rates_dict["__global__"] = float(global_rate)
with open(MODEL_DIR / "state_rates.json", "w") as f:
    json.dump(state_rates_dict, f, indent=2)

importances.to_csv(MODEL_DIR / "feature_importances.csv", header=True)

# Save scored training set
df.to_csv(MODEL_DIR / "historico_scored.csv", index=False, encoding="utf-8")

print(f"\n=== ARCHIVOS GUARDADOS EN {MODEL_DIR} ===")
for p in sorted(MODEL_DIR.iterdir()):
    print(f"  {p.name}")

print("\n✓ Modelo listo. Siguiente paso: score_mmi.py para puntuar los 3,643 del MMI Data.")
