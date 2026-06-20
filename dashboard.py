"""
HomeSi · ML Prospector — Dashboard
"""
import re as _re
import io
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "realtor_scraper" / "output"
MODEL_DIR   = ROOT / "realtor_scraper" / "output" / "model"
UPLOADS_DIR = ROOT / "realtor_scraper" / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

SPANISH_NAME_WORDS = {
    "casa","hogar","vive","movil","buena","bueno",
    "casas","hogares","vida","sol","estrella","luna","tierra",
    "familia","unidos","latina","latino",
}
LATINO_BROKERAGES = {
    "la rosa","casa buena","movil realty","vive realty","agent trust",
    "naim real estate","home prime","casa","hogar","latin","hispano",
    "hispanic","latino","habla","bilingual","multicultural",
}

def _keyword_match(text: str, keywords: set) -> bool:
    n = text.lower()
    for w in keywords:
        if " " in w:
            if w in n:
                return True
        else:
            if _re.search(r"\b" + _re.escape(w) + r"\b", n):
                return True
    return False

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
}
NAME_TO_ABBR = {v: k for k, v in STATE_ABBREV.items()}

# ── Session state ──────────────────────────────────────────────────────────────
if "selected_state" not in st.session_state:
    st.session_state.selected_state = None
if "n_show" not in st.session_state:
    st.session_state.n_show = 25
if "uploaded_batches" not in st.session_state:
    st.session_state.uploaded_batches = []   # list of DataFrames scored in-browser

# ── Data loading ───────────────────────────────────────────────────────────────

def _normalize_realtors(df: pd.DataFrame) -> pd.DataFrame:
    df["propensity_score"] = pd.to_numeric(df.get("propensity_score", 0), errors="coerce").fillna(0)
    if "ig_followers" in df.columns:
        df["ig_followers"] = pd.to_numeric(df["ig_followers"], errors="coerce")
    else:
        df["ig_followers"] = np.nan
    for col in [
        "ig_spanish_signals","ig_posts_spanish","ig_mentions_latino","ig_nahrep",
        "company_is_latino_brokerage","company_has_spanish_name",
        "ig_collaborates","ig_realtor_signals","ig_is_private",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    def _ig_url(handle):
        if pd.isna(handle) or str(handle).strip() in ("", "nan"):
            return None
        return f"https://www.instagram.com/{str(handle).strip().lstrip('@').rstrip('/')}/"

    df["ig_link"] = df["ig_handle"].apply(_ig_url) if "ig_handle" in df.columns else None
    if "batch_name" not in df.columns:
        df["batch_name"] = "carga_original"
    df["batch_name"] = df["batch_name"].fillna("carga_original")
    return df


@st.cache_data
def load_realtors() -> pd.DataFrame:
    # En Streamlit Cloud lee desde data/; localmente acepta también output/
    candidates = [
        DATA_DIR / "realtors.csv",
        *sorted(OUTPUT_DIR.glob("mmi_scored_*.csv")),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        st.error("No se encontró ningún archivo de realtors.")
        st.stop()
    return _normalize_realtors(pd.read_csv(path))


@st.cache_data
def load_census() -> pd.DataFrame:
    for p in [DATA_DIR / "state_census_summary.csv",
              OUTPUT_DIR / "state_census_summary.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            for col in ["total_population", "hispanic_pop", "hispanic_pct"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return df
    return pd.DataFrame(columns=["state","state_abbr","total_population","hispanic_pop","hispanic_pct"])


@st.cache_data
def load_model_artifacts():
    model     = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    features  = json.load(open(MODEL_DIR / "features.json"))
    medians   = json.load(open(MODEL_DIR / "medians.json"))
    sr        = json.load(open(MODEL_DIR / "state_rates.json"))
    global_r  = sr.pop("__global__", 0.57)
    return model, features, medians, sr, global_r


def score_excel_in_browser(file_bytes: bytes, batch_label: str) -> pd.DataFrame:
    """Score an uploaded Excel without Instagram scraping — runs entirely in-browser."""
    model, features, medians, state_rates, global_rate = load_model_artifacts()

    df = pd.read_excel(io.BytesIO(file_bytes), dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "First Name":           "first_name",
        "Last Name":            "last_name",
        "Company / Account":    "company",
        "Email":                "email_mmi",
        "Phone":                "phone_mmi",
        "State":                "state",
        "BS Sold # Units":      "units_sold",
    })
    if "first_name" in df.columns and "last_name" in df.columns:
        df["full_name"] = (df["first_name"].str.strip() + " " + df["last_name"].str.strip()).str.strip()
    if "full_name" not in df.columns:
        return None

    # Normalize state to full name
    df["state_clean"] = df["state"].str.strip()
    df.loc[df["state_clean"].str.len() == 2, "state_clean"] = (
        df.loc[df["state_clean"].str.len() == 2, "state_clean"]
        .str.upper().map(STATE_ABBREV)
    )
    state_abbr_map = {v: k for k, v in STATE_ABBREV.items()}
    df["state_abbr"] = df["state_clean"].map(state_abbr_map)

    # Company signals
    df["company_has_spanish_name"]    = df["company"].fillna("").apply(lambda n: _keyword_match(n, SPANISH_NAME_WORDS))
    df["company_is_latino_brokerage"] = df["company"].fillna("").apply(lambda n: _keyword_match(n, LATINO_BROKERAGES))

    # Instagram features — all zero (not yet scraped)
    for col in ["ig_followers_log","ig_posts_log","ig_engagement_proxy_log","content_lang_score"]:
        df[col] = 0.0
    df["has_instagram"] = 0
    for b in ["ig_spanish_signals","ig_realtor_signals","ig_posts_spanish",
              "ig_mentions_latino","ig_nahrep","ig_collaborates","ig_is_private"]:
        df[b] = 0
    for ct in ["first_time_buyer","family_community","relocation","veterans","luxury","investor"]:
        df["comm_" + ct] = 0

    # State conversion rate
    df["state_conversion_rate"] = df["state_abbr"].map(state_rates).fillna(global_rate)

    # Feature matrix
    X = pd.DataFrame(index=df.index)
    for feat in features:
        X[feat] = pd.to_numeric(df.get(feat, 0), errors="coerce").fillna(medians.get(feat, 0.0))

    proba = model.predict_proba(X)[:, 1]
    df["propensity_score"] = (proba * 100).round(1)
    df["priority_tier"]    = df["propensity_score"].apply(
        lambda s: "A" if s >= 80 else "B" if s >= 65 else "C" if s >= 50 else "D"
    )
    df["batch_name"] = batch_label
    df["ig_link"]    = None
    df["state"]      = df["state_clean"]
    return _normalize_realtors(df)


def build_signal_description(row) -> str:
    parts = []
    handle = row.get("ig_handle", "")
    has_ig = pd.notna(handle) and str(handle).strip() not in ("", "nan")
    lang   = str(row.get("ig_content_language", "")).lower()

    if has_ig:
        raw_fol = row.get("ig_followers", None)
        fol_str = f"{int(raw_fol):,} seguidores" if pd.notna(raw_fol) and raw_fol > 0 else "sin conteo de seguidores"
        if lang == "spanish":
            parts.append(f"Publica exclusivamente en espanol en Instagram ({fol_str})")
        elif lang == "mixed":
            parts.append(f"Publica contenido bilingue —espanol e ingles— en Instagram ({fol_str})")
        elif lang == "english":
            parts.append(f"Instagram activo, publica en ingles ({fol_str})")
        else:
            parts.append(f"Tiene Instagram ({fol_str})")
    else:
        parts.append("No se detecto Instagram")

    co = str(row.get("company", "")).strip()
    if row.get("company_is_latino_brokerage", 0):
        parts.append(f'Trabaja en "{co}", empresa especializada en el mercado latino o hispano')
    elif row.get("company_has_spanish_name", 0):
        parts.append(f'Su empresa tiene nombre en espanol ("{co}")')

    sigs = []
    if row.get("ig_mentions_latino", 0):
        sigs.append("menciona explicitamente la comunidad latina en su perfil")
    if row.get("ig_nahrep", 0):
        sigs.append("es miembro de NAHREP o lo referencia en su contenido")
    if row.get("ig_posts_spanish", 0) and lang not in ("spanish", "mixed"):
        sigs.append("incluye posts en espanol aunque su idioma principal sea otro")
    if row.get("ig_spanish_signals", 0) and lang == "english":
        sigs.append("usa palabras o hashtags en espanol en bio o publicaciones")
    if row.get("ig_collaborates", 0):
        sigs.append("hace colaboraciones frecuentes en Instagram")
    if sigs:
        parts.append("Ademas: " + " y ".join(sigs))

    community = str(row.get("ig_community_type", "")).strip()
    if community and community.lower() not in ("nan", "none", ""):
        comm_map = {
            "first_time_buyer": "compradores de primera vivienda",
            "family_community": "familias y comunidad",
            "relocation":       "relocation / mudanzas",
            "veterans":         "veteranos",
            "luxury":           "propiedades de lujo",
            "investor":         "inversionistas",
        }
        labels = [v for k, v in comm_map.items() if k in community.lower()]
        if labels:
            parts.append(f"Audiencia: {', '.join(labels)}")

    try:
        u = int(float(str(row.get("units_sold", "")).strip()))
        if u > 0:
            parts.append(f"{u} unidades vendidas registradas")
    except Exception:
        pass

    return ". ".join(parts) + "." if parts else "Sin senales especificas detectadas."


def tier_info(score: float):
    if score >= 80:
        return "#1a7a4a", "#d4efdf", "Prioridad Alta"
    if score >= 65:
        return "#1565c0", "#dce8fb", "Prioridad Media"
    if score >= 50:
        return "#c07a00", "#fef3cd", "Seguimiento"
    return "#777", "#f2f3f4", "Baja prioridad"


def render_card(row) -> str:
    score  = float(row.get("propensity_score", 0))
    sc, bg, lbl = tier_info(score)

    name    = str(row.get("full_name", "—")).title()
    company = str(row.get("company", "")).strip()
    state   = str(row.get("state", "")).strip()
    phone   = str(row.get("phone_mmi", "")).strip()
    ig_link = row.get("ig_link", None)
    handle  = str(row.get("ig_handle", "")).strip().lstrip("@")
    signal  = str(row.get("_signal", ""))

    phone_html = (
        f'<span style="color:#555;">&#x260E; {phone}</span>'
        if phone and phone != "nan" else ""
    )
    ig_html = (
        f'<a href="{ig_link}" target="_blank" '
        f'style="color:#1B4F72;font-weight:600;text-decoration:none;">'
        f'@{handle} &#8599;</a>'
        if ig_link else
        '<span style="color:#bbb;">sin Instagram</span>'
    )
    co_html = f"<b>{company}</b>" if company and company != "nan" else ""

    score_bar = f'<div style="height:4px;background:#e0e0e0;border-radius:2px;margin-top:6px;">' \
                f'<div style="height:4px;width:{score:.0f}%;background:{sc};border-radius:2px;"></div></div>'

    return (
        f'<div style="background:#fff;border:1px solid #e8ecf0;border-radius:14px;'
        f'padding:18px 20px 14px 20px;margin-bottom:10px;border-left:5px solid {sc};'
        f'box-shadow:0 1px 5px rgba(0,0,0,0.06);font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">'
        f'<div style="display:flex;align-items:flex-start;gap:16px;">'
        f'<div style="min-width:62px;text-align:center;background:{bg};border-radius:10px;padding:8px 6px;">'
        f'<div style="font-size:1.65rem;font-weight:800;color:{sc};line-height:1;">{score:.0f}</div>'
        f'<div style="font-size:0.65rem;color:{sc};font-weight:600;margin-top:2px;letter-spacing:.3px;">{lbl.upper()}</div>'
        f'{score_bar}'
        f'</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:1.05rem;font-weight:700;color:#1B4F72;margin-bottom:4px;">{name}</div>'
        f'<div style="font-size:0.85rem;color:#444;margin-bottom:6px;display:flex;flex-wrap:wrap;gap:6px 14px;">'
        f'{co_html}<span style="color:#888;">{state}</span>{phone_html}{ig_html}'
        f'</div>'
        f'<div style="font-size:0.82rem;color:#333;line-height:1.55;background:#f7f9fc;'
        f'border-radius:8px;padding:9px 13px;border-left:3px solid #aec6e8;">'
        f'{signal}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


# ── App layout ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HomeSi Prospector", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Fuente global */
html, body, [class*="css"], [data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar base */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a3f5c 0%, #1B4F72 60%, #1a5276 100%) !important;
}
[data-testid="stSidebar"] * { color: white !important; }

/* Titulo del sidebar */
[data-testid="stSidebar"] h2 {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px !important;
    color: white !important;
    margin-bottom: 2px !important;
}

/* Labels de filtros */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.7) !important;
}

/* Texto de caption */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.74rem !important;
    line-height: 1.5 !important;
}

/* Inputs y selectbox */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}

/* Checkbox */
[data-testid="stSidebar"] .stCheckbox label span { color: white !important; font-size: 0.86rem !important; }

/* Boton */
[data-testid="stSidebar"] .stButton button {
    background: #F39C12; color: #000 !important; border: none;
    border-radius: 8px; width: 100%; font-weight: 700;
    padding: 9px 0; font-size: 0.88rem; letter-spacing: 0.2px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover { background: #e67e22; }

/* Icono de ayuda */
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { fill:#F39C12 !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"]:hover svg { fill:#f8c471 !important; }

/* Ocultar texto del boton de colapsar sidebar */
[data-testid="stSidebar"] button[kind="header"] span { display: none !important; }
[data-testid="collapsedControl"] { font-size: 0 !important; }
[data-testid="stSidebarCollapseButton"] span { font-size: 0 !important; color: transparent !important; }
button[data-testid="stBaseButton-headerNoPadding"] span { display: none !important; }

/* Divider */
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; margin: 12px 0 !important; }

/* Main */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.metric-card { background: #f0f5fb; border-radius: 12px; padding: 14px 18px; text-align: center; }
.metric-num  { font-size: 1.8rem; font-weight: 800; color: #1B4F72; line-height: 1; }
.metric-lbl  { font-size: .76rem; color: #666; margin-top: 5px; font-weight: 500; }
h3 { color: #1B4F72 !important; }
</style>
""", unsafe_allow_html=True)

df_all  = load_realtors()
if st.session_state.uploaded_batches:
    df_all = pd.concat([df_all] + st.session_state.uploaded_batches, ignore_index=True)
census  = load_census()
states_available = sorted(df_all["state"].dropna().unique().tolist())
all_states_opts  = ["Todos los estados"] + states_available

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style='padding:4px 0 18px 0;'>
  <div style='font-size:1.45rem;font-weight:800;letter-spacing:-0.3px;line-height:1.2;'>HomeSi Prospector</div>
  <div style='font-size:0.8rem;color:rgba(255,255,255,0.6);margin-top:6px;font-weight:400;'>
    Selecciona un estado en el <b style="color:white;">mapa</b> o filtra aqui abajo
  </div>
</div>
""", unsafe_allow_html=True)

    # State search/select — synced with map click
    current = st.session_state.selected_state or "Todos los estados"
    idx     = all_states_opts.index(current) if current in all_states_opts else 0
    sidebar_state = st.selectbox("Estado", all_states_opts, index=idx)
    if sidebar_state != current:
        st.session_state.selected_state = None if sidebar_state == "Todos los estados" else sidebar_state
        st.session_state.n_show = 25
        st.rerun()

    st.markdown("---")
    min_score = st.slider("Score minimo", 0, 100, 0, step=5)

    lang_opts = st.multiselect(
        "Contenido Instagram",
        ["spanish", "mixed", "english"],
        default=[],
        format_func=lambda x: {
            "spanish": "Espanol (exclusivo)",
            "mixed":   "Bilingue (mixto)",
            "english": "Solo ingles",
        }.get(x, x),
    )
    only_latino_co = st.checkbox("Solo empresas latinas / hispanas", value=False)
    only_ig        = st.checkbox("Solo con Instagram detectado",     value=False)

    st.markdown("---")
    all_batches = sorted(df_all["batch_name"].unique().tolist())
    batch_filter = st.multiselect(
        "Filtrar por subida",
        all_batches,
        default=[],
        help="Muestra solo realtors de una carga especifica. Vacio = todas las cargas.",
    )

    st.markdown("---")
    st.caption(
        "El score refleja el perfil individual: actividad e idioma en Instagram, "
        "tipo de empresa, menciones a la comunidad latina y unidades vendidas."
    )

# ── Map ────────────────────────────────────────────────────────────────────────

realtor_stats = df_all.groupby("state").agg(
    n_realtors = ("full_name",        "count"),
    avg_score  = ("propensity_score", "mean"),
    n_high     = ("propensity_score", lambda x: (x >= 65).sum()),
).reset_index()

map_df = census.merge(realtor_stats, on="state", how="outer")
map_df["state_abbr"]       = map_df["state"].map(NAME_TO_ABBR)
map_df["hispanic_pct_pct"] = (map_df["hispanic_pct"] * 100).round(1)
map_df["avg_score"]        = map_df["avg_score"].fillna(0).round(1)
map_df["n_realtors"]       = map_df["n_realtors"].fillna(0).astype(int)
map_df["n_high"]           = map_df["n_high"].fillna(0).astype(int)
map_df["hispanic_pop"]     = map_df["hispanic_pop"].fillna(0)
map_df = map_df[map_df["state_abbr"].notna()].copy()

sel = st.session_state.selected_state
fig = go.Figure()

fig.add_trace(go.Choropleth(
    locations    = map_df["state_abbr"],
    z            = map_df["hispanic_pct_pct"],
    locationmode = "USA-states",
    colorscale   = [[0,"#e8f4f8"],[0.15,"#aed6f1"],[0.4,"#2e86c1"],[1,"#1b2a6b"]],
    zmin=0, zmax=50,
    colorbar     = dict(title="% Hispano", thickness=12, len=0.42, x=1.01, xanchor="left", y=0.82, yanchor="top", ticksuffix="%"),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Hispanohablantes: <b>%{customdata[0]:,.0f}</b><br>"
        "Porcentaje: <b>%{z:.1f}%</b><extra></extra>"
    ),
    text       = map_df["state"],
    customdata = map_df[["hispanic_pop"]].values,
    marker     = dict(line=dict(color="white", width=0.8)),
))

if sel:
    sel_row = map_df[map_df["state"] == sel]
    if len(sel_row):
        fig.add_trace(go.Choropleth(
            locations=sel_row["state_abbr"], z=[1],
            locationmode="USA-states",
            colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(0,0,0,0)"]],
            showscale=False,
            marker=dict(line=dict(color="#F39C12", width=3.5)),
            hoverinfo="skip",
        ))

fig.add_trace(go.Scattergeo(
    locations    = map_df["state_abbr"],
    locationmode = "USA-states",
    mode         = "markers",
    marker=dict(
        size       = np.sqrt(map_df["n_realtors"].clip(1)) * 3.2,
        color      = map_df["avg_score"],
        cmin=25, cmax=75,
        colorscale = [[0,"#f9ebea"],[0.45,"#f39c12"],[1,"#1a7a4a"]],
        colorbar   = dict(title="Score<br>promedio", thickness=12, len=0.42, x=1.01, xanchor="left", y=0.3, yanchor="top"),
        line=dict(width=1.5, color="white"),
        opacity=0.88,
    ),
    text       = map_df["state"],
    customdata = map_df[["n_realtors","avg_score","n_high","hispanic_pop","hispanic_pct_pct"]].values,
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Hispanohablantes: <b>%{customdata[3]:,.0f}</b> (%{customdata[4]:.1f}%)<br>"
        "Realtors MMI: <b>%{customdata[0]}</b> · Score promedio: %{customdata[1]:.1f}<br>"
        "Con score alto (>=65): %{customdata[2]}<br>"
        "<i>Click para filtrar lista</i><extra></extra>"
    ),
    showlegend=False,
))

fig.update_layout(
    geo=dict(
        scope="usa", showlakes=False, showland=True,
        landcolor="#f0f0eb", bgcolor="rgba(0,0,0,0)",
        projection_type="albers usa",
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=345,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
)

st.markdown("### Hispanohablantes por Estado")
st.caption(
    "Tono: % de poblacion hispana (Census ACS 2024)  ·  "
    "Burbujas: realtors en MMI (tamano) y score promedio (color)  ·  "
    "**Haz click en una burbuja** o usa el selector del panel izquierdo"
)

chart_event = st.plotly_chart(
    fig, use_container_width=True, key="map_chart",
    on_select="rerun", config={"displayModeBar": False},
)

if chart_event and chart_event.selection and chart_event.selection.points:
    for pt in chart_event.selection.points:
        clicked = pt.get("text", "")
        if clicked and clicked in states_available:
            if clicked != st.session_state.selected_state:
                st.session_state.selected_state = clicked
                st.session_state.n_show = 25
                st.rerun()
            break

# ── Filter ─────────────────────────────────────────────────────────────────────

df = df_all.copy()
if st.session_state.selected_state:
    df = df[df["state"] == st.session_state.selected_state]
if min_score > 0:
    df = df[df["propensity_score"] >= min_score]
if lang_opts:
    df = df[df["ig_content_language"].str.lower().isin(lang_opts)]
if only_latino_co:
    df = df[df["company_is_latino_brokerage"] == 1]
if only_ig:
    df = df[df["ig_handle"].notna()]
if batch_filter:
    df = df[df["batch_name"].isin(batch_filter)]

df = df.sort_values("propensity_score", ascending=False).reset_index(drop=True)
df["_signal"] = df.apply(build_signal_description, axis=1)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("---")

if st.session_state.selected_state:
    c_row = census[census["state"] == st.session_state.selected_state]
    if len(c_row):
        hp   = int(c_row["hispanic_pop"].values[0])
        hpct = c_row["hispanic_pct"].values[0] * 100
        tot  = int(c_row["total_population"].values[0])
        st.markdown(f"### {st.session_state.selected_state}")
        st.caption(
            f"**{hp:,} hispanohablantes** · {hpct:.1f}% de {tot:,} habitantes  ·  "
            f"**{len(df):,} realtors** con filtros actuales"
        )
    else:
        st.markdown(f"### {st.session_state.selected_state} — {len(df):,} realtors")
else:
    total_hisp = int(census["hispanic_pop"].sum()) if len(census) else 0
    st.markdown("### Todos los estados")
    st.caption(
        f"**{total_hisp:,} hispanohablantes** en EE.UU. (Census ACS 2024)  ·  "
        f"**{len(df):,} realtors** con filtros actuales"
    )

# ── KPIs ───────────────────────────────────────────────────────────────────────

n_high = (df["propensity_score"] >= 65).sum()
n_esp  = df["ig_content_language"].str.lower().isin(["spanish","mixed"]).sum()
n_lat  = df["company_is_latino_brokerage"].sum()
n_ig   = df["ig_link"].notna().sum()

cols = st.columns(4)
for col, num, lbl in [
    (cols[0], f"{len(df):,}",  "realtors"),
    (cols[1], f"{n_high:,}",   "score alto (>=65)"),
    (cols[2], f"{n_esp:,}",    "IG espanol / bilingue"),
    (cols[3], f"{n_lat:,}",    "empresa latina"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-num">{num}</div>'
        f'<div class="metric-lbl">{lbl}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ── Realtor table ──────────────────────────────────────────────────────────────

df["Senales"]    = df["_signal"]
df["Seguidores"] = df["ig_followers"].apply(
    lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else ""
)

display = df.rename(columns={
    "propensity_score": "Score",
    "full_name":        "Nombre",
    "company":          "Empresa",
    "state":            "Estado",
    "batch_name":       "Carga",
    "phone_mmi":        "Telefono",
    "ig_link":          "Instagram",
})[["Score","Nombre","Empresa","Estado","Carga","Telefono","Instagram","Senales"]].head(500)

st.dataframe(
    display,
    use_container_width=True,
    height=560,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.0f", width="small",
        ),
        "Instagram": st.column_config.LinkColumn(
            "Instagram",
            display_text=r"instagram\.com/([^/]+)",
            width="medium",
        ),
        "Nombre":  st.column_config.TextColumn("Nombre",         width="medium"),
        "Empresa": st.column_config.TextColumn("Empresa",        width="medium"),
        "Estado":  st.column_config.TextColumn("Estado",         width="small"),
        "Carga":   st.column_config.TextColumn("Carga",          width="small"),
        "Telefono":st.column_config.TextColumn("Telefono",       width="medium"),
        "Senales": st.column_config.TextColumn("Por que llamar", width="large"),
    },
    hide_index=True,
)

if len(df) > 500:
    st.caption(f"Primeros 500 de {len(df):,}. Descarga el CSV para ver todos.")

# ── Downloads ──────────────────────────────────────────────────────────────────

st.markdown("---")
_SKIP_COLS   = {"_signal", "ig_link", "state_abbr", "_hisp_abs", "Senales", "Seguidores"}
export_cols  = [c for c in df_all.columns if c not in _SKIP_COLS]

col_dl1, col_dl2, _ = st.columns([1, 1, 2])

with col_dl1:
    _df_export = df[[c for c in export_cols if c in df.columns]]
    st.download_button(
        "⬇ Descargar con filtros activos (CSV)",
        _df_export.to_csv(index=False).encode("utf-8"),
        f"realtors_{(st.session_state.selected_state or 'todos').replace(' ','_').lower()}.csv",
        "text/csv",
        help="Exporta exactamente lo que ves: estado, score minimo, idioma y carga seleccionados",
    )
with col_dl2:
    _all_cols = [c for c in df_all.columns if c not in _SKIP_COLS]
    st.download_button(
        "⬇ Todos los realtors — sin filtros (CSV)",
        df_all[_all_cols].to_csv(index=False).encode("utf-8"),
        "realtors_todos.csv",
        "text/csv",
        help="Lista completa sin aplicar ningun filtro, con toda la informacion de Instagram y demas",
    )

# ── Nueva Carga ────────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("Subir nueva lista de realtors", expanded=False):
    st.markdown(
        "Sube un Excel con las columnas: "
        "**First Name, Last Name, Company / Account, Email, Phone, State, BS Sold # Units**  \n"
        "Los realtors se puntuan al instante. El score es parcial hasta que se busque su Instagram."
    )

    uploaded = st.file_uploader(
        "Selecciona el archivo Excel",
        type=["xlsx", "xls"],
        key="new_batch",
    )
    batch_label_up = st.text_input(
        "Nombre para esta carga",
        placeholder="julio_2026",
        key="batch_label_up",
    )

    if uploaded:
        file_stem = uploaded.name.rsplit(".", 1)[0]
        label     = batch_label_up.strip() or _re.sub(r"[^a-zA-Z0-9_-]", "_", file_stem)

        already_loaded = any(
            b["batch_name"].iloc[0] == label
            for b in st.session_state.uploaded_batches
            if len(b)
        )

        if not already_loaded:
            with st.spinner(f"Puntuando {label}…"):
                scored = score_excel_in_browser(uploaded.getvalue(), label)
            if scored is not None and len(scored):
                st.session_state.uploaded_batches.append(scored)
                st.success(
                    f"{len(scored):,} realtors puntuados y añadidos con etiqueta **{label}**. "
                    "Puedes filtrarlos en el sidebar con 'Filtrar por subida'."
                )
                st.info(
                    "Score parcial — sin datos de Instagram. "
                    "Para el score completo con Instagram, corre el pipeline localmente y actualiza el repo."
                )
                st.rerun()
            else:
                st.error("No se pudo procesar el archivo. Verifica que tenga las columnas correctas.")
        else:
            st.info(f"La carga **{label}** ya está en la sesión actual.")

st.markdown("---")
with st.expander("Como obtener el score completo con Instagram (paso a paso)", expanded=False):
    st.markdown("""
### El score que ves al subir el Excel es parcial
Sin datos de Instagram, el modelo no puede evaluar si el realtor publica en español,
cuántos seguidores tiene ni si menciona la comunidad latina.
Para el score completo hay que buscar su Instagram primero.

---

### Paso a paso para enriquecer con Instagram

**1. Abre Claude Code en tu computadora**
En la terminal de VS Code o en la app de Claude Code, escríbele esto:

> *"Tengo un nuevo Excel de realtors en `realtor_scraper/uploads/NOMBRE_DEL_ARCHIVO.xlsx`.
> Corre el pipeline completo: primero `mmi_enricher.py` con ese archivo y el batch name
> que quieras, y después `score_mmi.py`. Cuando termine, haz commit y push a GitHub."*

Claude Code va a ejecutar los comandos, monitorear el progreso y subir los resultados.

---

**2. Lo que hace el pipeline automáticamente**

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `mmi_enricher.py` | Busca el Instagram de cada realtor en Google, entra al perfil, lee la bio, seguidores, idioma del contenido y señales latinas |
| 2 | `score_mmi.py` | Aplica el modelo XGBoost con todos los datos y genera el score final |
| 3 | `git push` | Sube el nuevo CSV a GitHub → el dashboard se actualiza solo |

---

**3. Cuánto tarda**

El scraping de Instagram tarda entre **3 y 8 segundos por realtor** para no activar
el bot-detector. Para 100 realtors, espera unos 15-20 minutos.
Para listas grandes puedes pedirle a Claude Code que corra solo un estado:

> *"Corre el enricher solo para Texas con `--state Texas`"*

---

**4. Cuando termine**

Recarga esta página — los realtors nuevos aparecerán en la lista con sus señales
de Instagram y un score actualizado. Filtralos por nombre de carga en el sidebar.
""")
