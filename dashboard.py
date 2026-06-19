"""
HomeSi · ML Prospector — Dashboard
"""
import re as _re
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT        = Path(__file__).parent
OUTPUT_DIR  = ROOT / "realtor_scraper" / "output"
UPLOADS_DIR = ROOT / "realtor_scraper" / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

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

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_realtors() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("mmi_scored_*.csv"))
    if not files:
        st.error("No se encontro mmi_scored_*.csv en realtor_scraper/output/")
        st.stop()
    df = pd.read_csv(files[-1])
    df["propensity_score"] = pd.to_numeric(df["propensity_score"], errors="coerce").fillna(0)
    df["ig_followers"]     = pd.to_numeric(df["ig_followers"],     errors="coerce")
    for col in [
        "ig_spanish_signals","ig_posts_spanish","ig_mentions_latino","ig_nahrep",
        "company_is_latino_brokerage","company_has_spanish_name",
        "ig_collaborates","ig_realtor_signals","ig_is_private",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    def build_ig_url(handle):
        if pd.isna(handle) or str(handle).strip() in ("", "nan"):
            return None
        h = str(handle).strip().lstrip("@").rstrip("/")
        return f"https://www.instagram.com/{h}/"

    df["ig_link"] = df["ig_handle"].apply(build_ig_url)
    if "batch_name" not in df.columns:
        df["batch_name"] = "carga_original"
    df["batch_name"] = df["batch_name"].fillna("carga_original")
    return df


@st.cache_data
def load_census() -> pd.DataFrame:
    path = OUTPUT_DIR / "state_census_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["state","state_abbr","total_population","hispanic_pop","hispanic_pct"])
    df = pd.read_csv(path)
    for col in ["total_population","hispanic_pop","hispanic_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


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
/* Sidebar */
[data-testid="stSidebar"] { background-color:#1B4F72 !important; }
[data-testid="stSidebar"] * { color:white !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg { fill:#F39C12 !important; color:#F39C12 !important; }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"]:hover svg { fill:#f8c471 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background:#1e5f88 !important; border-color:#2980b9 !important; }
[data-testid="stSidebar"] .stCheckbox label span { color:white !important; }
[data-testid="stSidebar"] .stButton button {
    background:#F39C12; color:#000 !important; border:none;
    border-radius:8px; width:100%; font-weight:700; padding:8px 0;
}
[data-testid="stSidebar"] .stButton button:hover { background:#e67e22; }
/* Main */
.main .block-container { padding-top:1.5rem; padding-bottom:2rem; }
.metric-card { background:#f0f5fb; border-radius:12px; padding:14px 18px; text-align:center; }
.metric-num  { font-size:1.8rem; font-weight:800; color:#1B4F72; line-height:1; }
.metric-lbl  { font-size:.76rem; color:#666; margin-top:5px; font-weight:500; }
h3 { color:#1B4F72 !important; }
</style>
""", unsafe_allow_html=True)

df_all  = load_realtors()
census  = load_census()
states_available = sorted(df_all["state"].dropna().unique().tolist())
all_states_opts  = ["Todos los estados"] + states_available

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## HomeSi Prospector")
    st.markdown("Selecciona un estado en el **mapa** o busca aqui abajo.")
    st.markdown("")

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
with st.expander("Procesar nueva lista de realtors", expanded=False):
    st.markdown(
        "El archivo debe ser Excel (.xlsx) o CSV con las columnas: "
        "**First Name, Last Name, Company / Account, Email, Phone, State, BS Sold # Units**"
    )

    tab_path, tab_upload = st.tabs(["Pegar ruta del archivo", "Subir desde el navegador"])

    # ── Opcion A: pegar ruta del explorador ──────────────────────────────────
    with tab_path:
        st.caption(
            "En el Explorador de Windows: click derecho sobre el archivo → "
            "**Copiar como ruta** → pega aqui abajo"
        )
        raw_path = st.text_input(
            "Ruta del archivo",
            placeholder=r'C:\Users\Isabella\Desktop\Nueva_Lista.xlsx',
            key="batch_path",
        )
        batch_label_path = st.text_input(
            "Nombre para esta carga (aparece en la columna Carga)",
            placeholder="junio_2026",
            key="batch_label_path",
        )
        if raw_path.strip():
            clean_path  = raw_path.strip().strip('"').strip("'")
            file_stem   = Path(clean_path).stem
            label       = batch_label_path.strip() or _re.sub(r"[^a-zA-Z0-9_-]", "_", file_stem)
            venv        = r"latino_re_engine\.venv\Scripts\python.exe"
            enricher    = r"realtor_scraper\mmi_enricher.py"
            scorer      = r"realtor_scraper\score_mmi.py"
            st.markdown("**Copia y pega esto en la terminal (en la carpeta del proyecto):**")
            st.code(
                f'{venv} {enricher} --input "{clean_path}" --batch-name "{label}"\n'
                f'{venv} {scorer}',
                language="bash",
            )
            st.info(
                f"Cuando termine, recarga el dashboard — los nuevos realtors apareceran "
                f"con la etiqueta **{label}** en la columna Carga."
            )

    # ── Opcion B: subir via navegador ─────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Selecciona el archivo",
            type=["xlsx", "xls", "csv"],
            key="new_batch",
        )
        batch_label_up = st.text_input(
            "Nombre para esta carga",
            placeholder="junio_2026",
            key="batch_label_up",
        )
        if uploaded:
            save_path  = UPLOADS_DIR / uploaded.name
            save_path.write_bytes(uploaded.getvalue())
            file_stem  = uploaded.name.rsplit(".", 1)[0]
            label      = batch_label_up.strip() or _re.sub(r"[^a-zA-Z0-9_-]", "_", file_stem)
            saved_rel  = f"realtor_scraper/uploads/{uploaded.name}"
            venv       = r"latino_re_engine\.venv\Scripts\python.exe"
            enricher   = r"realtor_scraper\mmi_enricher.py"
            scorer     = r"realtor_scraper\score_mmi.py"
            st.success(f"Guardado en: `{saved_rel}`")
            st.markdown("**Copia y pega esto en la terminal:**")
            st.code(
                f'{venv} {enricher} --input "{saved_rel}" --batch-name "{label}"\n'
                f'{venv} {scorer}',
                language="bash",
            )
            st.info(
                f"Cuando termine, recarga el dashboard — los nuevos realtors apareceran "
                f"con la etiqueta **{label}** en la columna Carga."
            )
