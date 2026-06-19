# ML Prospector — HomeSí B2B Prospecting Engine

Sistema de machine learning para identificar y priorizar agentes inmobiliarios (realtors) con alto potencial de conversión en el mercado hispano de EE.UU.

## Descripción

HomeSí es una plataforma enfocada en conectar familias latinas con agentes inmobiliarios que hablan español y conocen su comunidad. Este pipeline automatiza la prospección B2B: dado un listado de realtors del MMI Data, el sistema encuentra su Instagram, analiza su contenido, les asigna un score de 0 a 100 y los prioriza para el equipo de ventas.

## Cómo funciona

```
MMI Data.xlsx
     │
     ▼
mmi_enricher.py       ← busca Instagram y Zillow por cada realtor (web scraping)
     │
     ▼
mmi_enriched_*.csv    ← datos enriquecidos con señales de Instagram
     │
     ▼
score_mmi.py          ← aplica el modelo XGBoost, genera score 0-100
     │
     ▼
mmi_scored_*.csv      ← lista final ordenada por score
     │
     ▼
dashboard.py          ← visualización interactiva en Streamlit
```

## Señales que usa el modelo

| Señal | Descripción |
|---|---|
| `ig_content_language` | Idioma del contenido en Instagram (español / bilingüe / inglés) |
| `ig_spanish_signals` | Palabras o hashtags en español en bio o publicaciones |
| `ig_mentions_latino` | Menciona explícitamente la comunidad latina |
| `ig_nahrep` | Miembro o referencia a NAHREP |
| `company_is_latino_brokerage` | Trabaja en una empresa especializada en el mercado latino |
| `company_has_spanish_name` | Nombre de la empresa en español |
| `ig_followers_log` | Seguidores en Instagram (escala logarítmica) |
| `ig_posts_log` | Número de publicaciones |
| `units_sold` | Unidades vendidas registradas en MMI |
| `comm_*` | Tipo de comunidad: primera vivienda, familia, veteranos, etc. |

## Tiers de prioridad

| Score | Tier |
|---|---|
| ≥ 80 | **A — Prioridad Alta** |
| 65 – 79 | **B — Prioridad Media** |
| 50 – 64 | **C — Seguimiento** |
| < 50 | **D — Baja prioridad** |

## Estructura del proyecto

```
ML Prospector/
├── dashboard.py                  # Dashboard Streamlit interactivo
├── realtor_scraper/
│   ├── mmi_enricher.py           # Scraping Instagram + Zillow
│   ├── score_mmi.py              # Scoring con XGBoost
│   ├── train_model.py            # Entrenamiento del modelo
│   ├── analyze_model.py          # Diagnóstico y análisis del modelo
│   └── output/
│       └── model/                # Artefactos del modelo entrenado
│           ├── xgboost_model.pkl
│           ├── features.json
│           ├── medians.json
│           └── state_rates.json
└── latino_re_engine/             # Pipeline Census ACS + datos de mercado
```

## Instalación y uso

```bash
# Activar el entorno virtual
latino_re_engine\.venv\Scripts\activate

# 1. Enriquecer realtors con Instagram y Zillow
python realtor_scraper/mmi_enricher.py

# 2. Puntuar y generar lista priorizada
python realtor_scraper/score_mmi.py

# 3. Lanzar el dashboard
latino_re_engine\.venv\Scripts\streamlit.exe run dashboard.py
```

## Subir una nueva lista de realtors

1. Abre el dashboard → sección **"Procesar nueva lista de realtors"**
2. Pega la ruta del Excel o súbelo desde el navegador
3. Copia el comando que aparece y ejecútalo en la terminal
4. Cuando termine, recarga el dashboard — los nuevos realtors aparecen con su etiqueta en la columna **Carga**

## Modelo

- **Algoritmo:** XGBoost Classifier
- **Features:** 21 (5 numéricas + 10 booleanas + 6 flags de comunidad)
- **AUC cross-validation (5-fold):** 0.637 ± 0.017
- **AUC training set:** 0.822
- **Precisión en score ≥ 80:** 97% son Qualified

El modelo usa pesos por calidad de llamada en el entrenamiento: llamadas largas confirmadas pesan más que leads sin contacto previo. Las métricas del Census (% hispano por estado) fueron excluidas del modelo para evitar que el modelo memorizara estados en lugar de evaluar el perfil individual del realtor.
