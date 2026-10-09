# GCI Credit Default Risk 2026 — Project Spec

> Documento maestro del proyecto. Fuente de verdad para stack, preprocesamiento,
> features, validación y estrategia de experimentos.
> Última actualización: 2026-10-09.

## 1. Contexto y reglas

- **Competencia**: GCI Global 2026 — Home Credit Default Risk (subset de 1 tabla).
- **Tarea**: predecir probabilidad de impago (`TARGET` ∈ {0,1}).
- **Métrica**: ROC-AUC.
- **Submission**: CSV `SK_ID_CURR,TARGET` (probabilidad), formato `sample_submission.csv`. Formato incorrecto = score -1.
- **Cierre**: 2026-11-20 01:00 UTC. Submission final planeada: **2026-11-18**.
- ⚠️ **Solo cuenta la ÚLTIMA submission** para el ranking final (no la mejor del public LB).
- Public LB = subset del test; Private LB (final) = test completo. No sobreajustar al public LB.
- **Prohibido**: datos externos, hand-labeling (incl. ajustes manuales por reglas del EDA).
- **Requerido**: reproducibilidad (seeds fijas, código que regenera el CSV final) y **citas** de todo código/idea ajena en `refs.md`.

## 2. Datos

Origen oficial: carpeta Drive de la competencia → `input/`. Copia local: `files/data/` (sandbox) y `data/raw/` (repo, no versionado).

| Archivo | Filas | Columnas |
|---|---|---|
| `train.csv` | 171,202 | 34 (32 features + `SK_ID_CURR` + `TARGET`) |
| `test.csv` | 61,500 | 33 |
| `sample_submission.csv` | 61,500 | 2 |

Grupos de variables (32 features):

| Grupo | Columnas | Señal esperada |
|---|---|---|
| Scores externos | `EXT_SOURCE_1/2/3` | Muy alta (top predictores) |
| Montos | `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `AMT_INCOME_TOTAL` | Alta (vía ratios) |
| Tiempo/estabilidad | `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`, `OWN_CAR_AGE` | Alta |
| Consultas buró | `AMT_REQ_CREDIT_BUREAU_HOUR/MON/QRT/YEAR` | Media-alta |
| Categóricas | `NAME_CONTRACT_TYPE`, `CODE_GENDER`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE` | Media |
| Región | `REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT` | Baja-media |
| Flags contacto | `FLAG_MOBIL`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE` | Baja |
| Hogar | `CNT_CHILDREN`, `CNT_FAM_MEMBERS` | Baja |

## 3. Stack

- **Lenguaje**: Python 3.x, `requirements.txt` con versiones fijadas (`==`).
- **Modelado**: LightGBM (base), XGBoost, CatBoost (categóricas nativas), scikit-learn (CV, métricas, meta-modelo).
- **Tuning (M4)**: Optuna, presupuesto acotado (50–100 trials).
- **Experimental (M4)**: TabPFN como miembro extra del ensemble (si corre en CPU).
- **Repo**: `github.com/mariomarroquin3/gci_home_credit` — `src/{config,data,features,train}.py`, `notebooks/`, `submissions/`, `refs.md`.
- **Gestión**: Linear workspace GCI World, proyecto "GCI Credit Default Risk 2026", 5 sprints por label (Sprint 1–5), hitos M1–M5.

## 4. Preprocesamiento

1. **Sin imputación de `EXT_SOURCE_*`**: los NaN son informativos → LightGBM los maneja nativamente + flags binarios `*_ISNA`.
2. **Anomalía `DAYS_EMPLOYED == 365243`** (código de desempleado/pensionado): flag binario + reemplazar valor por NaN.
3. **Flags de nulos** para columnas con >30% NaN.
4. **`CNT_FAM_MEMBERS`**: coherencia con `CNT_CHILDREN` (feature derivada, ver §5).
5. **Categóricas**:
   - Baja cardinalidad → one-hot (LGBM/XGB) o nativas (CatBoost).
   - Alta cardinalidad (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`) → **target encoding OOF con smoothing** (nunca encoding global: leakage).
6. **Numéricas**: sin escalado (modelos de árboles). Días negativos → convertir a años/positivos solo en features derivadas.
7. **Nada de hand-labeling ni reglas manuales** — todo criterio sale del CV (regla de la competencia).

## 5. Feature engineering (variables a crear)

### Ratios financieros (dominio)
| Feature | Fórmula |
|---|---|
| `CREDIT_ANNUITY_RATIO` | `AMT_CREDIT / AMT_ANNUITY` |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` (carga de pago) |
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` (apalancamiento) |
| `CREDIT_GOODS_RATIO` | `AMT_CREDIT / AMT_GOODS_PRICE` |
| `DOWNPAYMENT` | `AMT_GOODS_PRICE - AMT_CREDIT` |
| `INCOME_PER_FAMILY` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` |

### Tiempo
| Feature | Fórmula |
|---|---|
| `AGE_YEARS` | `-DAYS_BIRTH / 365.25` |
| `EMPLOYED_YEARS` | `-DAYS_EMPLOYED / 365.25` (NaN si anomalía) |
| `EMPLOYED_BIRTH_PCT` | `DAYS_EMPLOYED / DAYS_BIRTH` (% vida laboral) |
| `REGISTRATION_YEARS` | `-DAYS_REGISTRATION / 365.25` |
| `ID_PUBLISH_YEARS` | `-DAYS_ID_PUBLISH / 365.25` |
| `PHONE_CHANGE_YEARS` | `-DAYS_LAST_PHONE_CHANGE / 365.25` |

### EXT_SOURCE (la señal dominante)
- Polinómicas grado 2–3: `EXT1²`, `EXT2²`, `EXT3²`, `EXT1*EXT2`, `EXT1*EXT3`, `EXT2*EXT3`, `EXT1*EXT2*EXT3`.
- Estadísticos: `EXT_MEAN`, `EXT_STD`, `EXT_MIN`, `EXT_MAX`, conteo de EXT nulos.

### Buró / contacto / hogar
- `BUREAU_REQ_TOTAL` = suma HOUR+MON+QRT+YEAR; ratios YEAR/MON (frecuencia reciente).
- `PHONE_FLAGS_SUM` = FLAG_MOBIL+FLAG_EMP_PHONE+FLAG_WORK_PHONE.
- `CHILDREN_RATIO` = `CNT_CHILDREN / CNT_FAM_MEMBERS`.

## 6. Validación

- **Desarrollo**: StratifiedKFold 5-fold, seed fija (en `src/config.py`). Métrica de control: AUC OOF.
- **Selección de modelo final**: **Repeated StratifiedKFold 3×5** (seeds 42/43/44), comparar mean ± std. Decisión de Mario (2026-10-09): **nunca elegir por public LB**.
- **Brecha CV↔LB**: si |CV − LB| > 0.02 → **parar y arreglar validación antes de seguir** (domain shift, leakage).
- **Adversarial validation (M4)**: LGBM train-vs-test; AUC ≥ 0.6 → investigar features que separan.
- Early stopping por fold (200 rondas, métrica AUC).

## 7. Estrategia de experimentos (decisión de Mario #4)

- **Un branch por experimento**: `feat/ratios`, `feat/target-enc`, `model/catboost`, `model/xgboost`, `feat/stacking`, `tuning/lgbm`… `main` siempre contiene el mejor pipeline verificado.
- **Cada modelo se guarda** en `outputs/` (objeto + OOF + params + seed).
- Criterio de adopción a `main`: mejora de CV (RSKF cuando aplique) ≥ umbral acordado (0.003 Sprint 2, 0.002 tuning).
- Si el blend no supera al mejor individual → se queda el individual (simplicidad = reproducibilidad).
- Todo experimento se registra en el **submission log** del README: fecha | modelo | branch | CV AUC | LB AUC | archivo.

## 8. Sprints (Linear, labels Sprint 1–5)

| Sprint | Fechas | Hito | Contenido (issues) |
|---|---|---|---|
| Sprint 1 · Baseline | oct 9–16 | M1 | GCI-5 ✅ datos · GCI-6 repo público · GCI-7 EDA · GCI-8 baseline submission · GCI-9 submission log |
| Sprint 2 · Ensemble | oct 17–23 | M2 | GCI-10 FE ratios/polinómicas · GCI-11 encodings + Cat/XGB · GCI-12 blend + submission #2 |
| Sprint 3 · Requisitos mínimos | oct 24–nov 6 | M3 | GCI-13 reproducibilidad+citas · GCI-14 RSKF · GCI-15 congelar mínimo (tag v1.0) · GCI-16 README público |
| Sprint 4 · Métodos extra | nov 7–15 | M4 | GCI-17 Optuna · GCI-18 stacking+TabPFN · GCI-19 adversarial validation |
| Sprint 5 · Cierre | nov 16–18 | M5 | GCI-20 elegir final (RSKF) · GCI-21 submission final · GCI-22 repo reproducible (tag v2.0) |

**Definición de M3 cumplido** (acordada): pipeline end-to-end con `python -m src.train` + seeds fijas + `refs.md` completo + submission con CV ≥ 0.7681 (baseline 0.7581 + 0.01).

## 9. Referencias base (citar en `refs.md`)

- Kaggle Home Credit Default Risk (2018) — writeups de soluciones (p. ej. 1st place "Home Aloan").
- Kernel de Olivier (FE de referencia) y kernels de Yang (stacking).
- `deepseek_markdown_20261009_5cd6d6.md` (research propio, en `files/`).
- Hollmann et al., TabPFN (Nature, 2025) si se usa en M4.

## 10. Estado actual (2026-10-09)

- Datos descargados y verificados (GCI-5 ✅).
- Baseline LightGBM 5-fold: **CV AUC 0.7581** (commit inicial 2026-10-08), sin submission LB aún.
- Repo privado; publicación la gestiona Mario (GCI-6).
- Próximo: GCI-7 EDA (vence oct 13).
