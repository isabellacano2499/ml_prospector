# Latino Real Estate Market Intelligence Engine 

Este motor de inteligencia de mercado inmobiliario está diseñado para identificar, analizar y calificar el potencial del mercado hispano en los Estados Unidos a nivel de códigos postales (ZIP Codes/ZCTAs), condados o estados.

El sistema descarga datos socioeconómicos detallados directamente de la API del Censo de EE.UU. (ACS 5-Year), calcula variables derivadas complejas y genera calificaciones de oportunidad (`0` a `100`) para guiar decisiones de inversión, préstamos hipotecarios y estrategias de marketing.

---

##  1. Fuente de Datos y Temporalidad
El análisis principal utiliza los datos oficiales de la encuesta **ACS 5-Year (American Community Survey)** del **Census Bureau**. 

* **Año de análisis principal:** `2022` (configurable vía `.env` o CLI).
* **Año de comparación histórica:** `2019` (usado para calcular tasas de crecimiento y dinámicas de cambio a 3 años en el modo de crecimiento activo).

---

##  2. Variables del Censo Utilizadas (Raw Census Variables)
El motor consulta un total de **106 variables del Censo**, organizadas en los siguientes grupos:

### A. Demografía Básica y Edad
* `B01003_001E`: Población total.
* `B03003_003E`: Población de origen hispano o latino.
* `B01001_002E` y `B01001_026E`: Población total masculina y femenina.
* `B01001I_002E` y `B01001I_017E`: Población hispana masculina y femenina.
* **Rangos de Edad (Masculino y Femenino):** Variables desde los 18 hasta los 54 años (`B01001_007E` a `_016E` y `B01001_031E` a `_040E`) para identificar la concentración de jóvenes adultos.

### B. Ciudadanía e Inmigración
* `B05001_001E`: Universo total de ciudadanía.
* `B05001_005E`: Ciudadanos por naturalización.
* `B05001_006E`: No ciudadanos.

### C. Estado Civil y Estructura Familiar
* `B12001_001E`: Universo de estado civil (mayores de 15 años).
* `B12001_003E` a `_013E`: Distribución de solteros, casados, separados y divorciados por género.

### D. Educación
* `B15003_001E`: Población de 25 años o más (universo de educación).
* `B15003_017E` a `_025E`: Niveles educativos alcanzados (diploma de secundaria, GED, algunos créditos universitarios, títulos asociados, licenciaturas, maestrías, doctorados).

### E. Ingresos del Hogar (Household Income)
* `B19013_001E`: Ingreso medio anual del hogar (población general).
* `B19013I_001E`: Ingreso medio anual del hogar (específico de hogares hispanos).
* `B19001_002E` a `_017E`: Distribución de hogares por 16 rangos de ingresos (desde menos de $10,000 hasta más de $200,000 anuales).

### F. Empleo e Industrias
* `B23025_001E` a `_007E`: Fuerza laboral activa, empleados civiles, desempleados y personas fuera de la fuerza laboral.
* `C24030_001E` a `_031E`: Empleados distribuidos en sectores industriales clave (construcción, manufactura, transporte, finanzas/bienes raíces, servicios profesionales, educación/salud, hotelería/turismo) segmentados por género.

### G. Vivienda (Housing)
* `B25003_001E`: Total de viviendas ocupadas.
* `B25003_002E`: Viviendas ocupadas por sus propietarios (Homeowners).
* `B25003_003E`: Viviendas rentadas (Renters).
* `B25077_001E`: Valor medio de la vivienda.
* `B25064_001E`: Renta bruta mensual media.
* `B25071_001E`: Porcentaje medio de ingresos destinado al pago de renta (medida de carga financiera).

### H. Idioma en el Hogar
* `B16002_001E`: Total de hogares clasificados por idioma.
* `B16002_003E`: Hogares de habla hispana que hablan inglés "muy bien" (Bilingües).
* `B16002_004E`: Hogares de habla hispana con dominio limitado del inglés (LEP - Limited English Proficiency).

### I. Movilidad y Migración (Migration)
* `B07003_001E` a `_013E`: Movilidad geográfica en el último año por género (personas que viven en la misma casa, que se mudaron dentro del mismo condado, desde otro condado del mismo estado, desde otro estado o desde el extranjero).

---

##  3. Ingeniería de Variables (Features Calculadas)
El motor toma los datos crudos anteriores y calcula **indicadores avanzados** para evaluar el mercado de manera más inteligente:

* **Concentración Hispana (`hispanic_pct`):** Porcentaje de la población que se identifica como latina.
* **Población en Edad Clave de Compra (`age_25_44_pct`):** Porcentaje de la población de entre 25 y 44 años (edad típica del comprador de primera vivienda).
* **Índice de Formación Familiar (`family_formation_index`):** Multiplica la tasa de personas casadas por el porcentaje de población en edad clave de compra. Identifica zonas con núcleos familiares en crecimiento.
* **Rango de Ingresos Medios (`middle_income_pct`):** Porcentaje de hogares con ingresos anuales entre **$30,000 y $100,000** (el mercado meta principal para primeros compradores).
* **Brecha de Ingresos Hispana (`income_gap_ratio`):** Proporción entre el ingreso medio de los hogares hispanos y el ingreso medio general de la zona.
* **Tasa de Asequibilidad de Vivienda (`housing_affordability_ratio`):** Relación inversa entre el valor medio de la vivienda y el ingreso familiar (a menor relación precio-ingreso, mayor asequibilidad).
* **Demanda Latente de Compradores (`latent_buyer_demand`):** Multiplicación de la tasa de inquilinos (renters) por el porcentaje de hispanos en la zona.
* **Potencial de Compradores de Primera Vez (`first_time_buyer_potential`):** Estima el porcentaje de inquilinos hispanos activos en el área.
* **Índice de Oportunidad de Marketing en Español (`spanish_marketing_opportunity`):** Pondera la necesidad de comunicación bilingüe dando un **65%** de peso a hogares con dominio limitado del inglés (LEP) y un **35%** a hogares bilingües.
* **Indicador Compuesto de Migración (`migration_composite`):** Mide la atracción de población ponderando la procedencia: **50%** migración interestatal, **30%** internacional y **20%** intercondado.
* **Tasas de Crecimiento Histórico (2019 vs 2022):**
  * Crecimiento de la población general (`population_growth_rate`).
  * Crecimiento de la población hispana (`hispanic_growth_rate`).
  * Crecimiento de hogares ocupados (`household_growth_rate`).
  * Crecimiento de ingresos del hogar (`income_growth_rate`).

---

##  4. Modelos de Calificación y Algoritmo de Scoring
El sistema asigna una calificación de **0 a 100** a cada zona utilizando tres modelos especializados, normalizando previamente todas las variables de entrada a un rango de `0` a `1`.

### A. Calificación de Comprador de Primera Vivienda (FTHB Score)
* **Objetivo:** Identificar códigos postales ideales para campañas de crédito y adquisición de vivienda enfocadas en familias hispanas que actualmente rentan.
* **Pesos y Variables:**
  * **30%**: Porcentaje de población hispana (`hispanic_pct`).
  * **20%**: Inquilinos hispanos / demanda latente (`first_time_buyer_potential`).
  * **15%**: Población en edad de compra de 25 a 44 años (`age_25_44_pct`).
  * **10%**: Estabilidad de ingresos del hogar (`median_household_income`).
  * **10%**: Tasa de empleo (`employment_rate`).
  * **10%**: Crecimiento de la población hispana (`hispanic_growth_rate`).
  * **5%**: Asequibilidad de la vivienda (`housing_affordability_ratio`).

### B. Calificación de Fuerza del Mercado Latino (LMS Score)
* **Objetivo:** Medir la madurez, tamaño, estabilidad y potencial comercial general de la comunidad hispana local.
* **Pesos y Variables:**
  * **25%**: Porcentaje de población hispana (`hispanic_pct`).
  * **20%**: Crecimiento de la población hispana (`hispanic_growth_rate`).
  * **15%**: Índice compuesto de migración/atracción de población (`migration_composite`).
  * **15%**: Asequibilidad de la vivienda (`housing_affordability_ratio`).
  * **15%**: Tasa de empleo (`employment_rate`).
  * **10%**: Oportunidad de marketing y comunicación en español (`spanish_marketing_opportunity`).

### C. Calificación de Oportunidad Inmobiliaria (REO Score)
* **Objetivo:** Encontrar áreas con dinámicas de mercado atractivas para la inversión de capital, construcción de vivienda y desarrollo inmobiliario.
* **Pesos y Variables:**
  * **25%**: Índice compuesto de migración/atracción de población (`migration_composite`).
  * **25%**: Tasa de crecimiento poblacional general (`population_growth_rate`).
  * **20%**: Demanda latente de compradores (`latent_buyer_demand`).
  * **15%**: Asequibilidad de la vivienda (`housing_affordability_ratio`).
  * **15%**: Nivel de ingresos de la zona (`median_household_income`).

### D. Calificación General Compuesta (Overall Score)
El resultado global que consolida el potencial de cada código postal se calcula mediante una suma ponderada de los tres modelos anteriores:
$$\text{Overall Score} = (\text{FTHB Score} \times 0.40) + (\text{LMS Score} \times 0.35) + (\text{REO Score} \times 0.25)$$

---

##  5. Instrucciones de Ejecución
Para volver a ejecutar el motor y actualizar los datos, utiliza el entorno virtual (`.venv`):

```powershell
# Ejecución estándar (ZIP codes nacionales, 2022 vs 2019, incluye crecimiento)
& "C:\Users\lalai\OneDrive\Desktop\7. AI\.venv\Scripts\python.exe" main.py

# Ejecución rápida (sólo año 2022, omitiendo la comparación de crecimiento)
& "C:\Users\lalai\OneDrive\Desktop\7. AI\.venv\Scripts\python.exe" main.py --no-growth

# Ejecución por condados (County)
& "C:\Users\lalai\OneDrive\Desktop\7. AI\.venv\Scripts\python.exe" main.py --level county

# Ejecución por estados (State)
& "C:\Users\lalai\OneDrive\Desktop\7. AI\.venv\Scripts\python.exe" main.py --level state
```

### Resultados generados:
Los resultados calificados y procesados se guardan automáticamente en:
* 📄 `data/output/latino_market_{nivel}_{año}.csv` (ideal para abrir con Excel).
* 📦 `data/output/latino_market_{nivel}_{año}.parquet` (ideal para análisis avanzado con Python/Pandas).
