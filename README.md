# GCI Global 2026 · Home Credit Default Risk

Predicción de probabilidad de impago (`TARGET`). Métrica: **ROC-AUC**. Entrega: CSV `SK_ID_CURR,TARGET` en Omnicampus.
Cierre: **2026-11-20 01:00 UTC** (19 nov, 19:00 El Salvador; entrega interna: 18 nov). Objetivo: Outstanding (código reproducible del CSV final).

## Datos (versión de la competencia)

Una sola tabla a nivel solicitud — **sin** las 7 tablas auxiliares del Kaggle original.

| Archivo | Dimensiones | Notas |
|---|---|---|
| `train.csv` | 171,202 × 34 | `TARGET` desbalanceado: **8.1%** positivos |
| `test.csv` | 61,500 × 33 | Sin `TARGET` |
| `sample_submission.csv` | 61,500 × 2 | Formato exacto obligatorio (error = score −1) |

Hallazgos del EDA inicial (oct 2026):
- `EXT_SOURCE_1/2/3` dominan la señal (corr. −0.15 a −0.18 con TARGET); **sus nulos predicen** (EXT3 faltante: 8.9% vs 7.7% default) → no imputar, flags `*_ISNA`.
- `DAYS_EMPLOYED == 365243` es código especial en el **18%** de las filas → flag binario + NaN.
- Alta cardinalidad: `ORGANIZATION_TYPE`, `OCCUPATION_TYPE` → target encoding **out-of-fold** con smoothing.

## Estado actual

- **Baseline (commit inicial, 2026-10-08):** LightGBM, CV 5-fold **AUC 0.7581**.
  Params: `lr=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.8, bagging=0.8, lambda_l2=1.0`, early stopping 200, seeds en `src/config.py`.
- Meta M3 (red de seguridad): pipeline end-to-end con CV ≥ 0.7681 (baseline + 0.01).

## Metodología

1. **Sprint 1 — Baseline** (oct 9–16): EDA, features de ratios/anomalías, primer submission y lectura del LB.
2. **Sprint 2 — Ensemble base** (oct 17–23): features EXT_SOURCE polinómicas + target encoding OOF; XGBoost y CatBoost; blend ponderado.
3. **Sprint 3 — Requisitos mínimos** (oct 24–nov 6): reproducibilidad total, refs.md completo, submission con CV ≥ 0.7681.
4. **Sprint 4 — Métodos extra** (nov 7–15): tuning Optuna, stacking, TabPFN (si suma ≥ 0.002), adversarial validation.
5. **Sprint 5 — Cierre** (nov 16–18): modelo final por **Repeated StratifiedKFold 3×5** (seeds 42/43/44), nunca por public LB.

Reglas de decisión (acordadas 2026-10-09):
- Si `|CV − LB| > 0.02` → parar y arreglar validación antes de seguir.
- Un branch por experimento; `main` = mejor verificado por CV. Todo modelo guarda OOF, params y seed.
- Umbrales de adopción a `main`: +0.003 features · +0.002 tuning · +0.001 blend/stacking.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
Copia `train.csv`, `test.csv` y `sample_submission.csv` a `data/raw/` (no se versionan).

## Uso
```bash
python -m src.train            # CV 5-fold LightGBM -> outputs/ + submissions/
```
EDA en `notebooks/01_eda.ipynb`.

## Estructura
```
src/config.py     rutas, SEED, seed_everything
src/data.py       carga de datos
src/features.py   limpieza y features
src/train.py      CV + predicción + submission
notebooks/        EDA y experimentos
submissions/      CSVs (solo final_submission.csv se versiona)
refs.md           fuentes de código citadas
project.md        spec técnica completa del proyecto
```

## Reglas de la competencia
- Sin datos externos; sin hand-labeling (todo criterio sale del modelo/CV).
- Seeds fijas (`src/config.py`); 2 corridas = CSV idéntico. Citar código ajeno en `refs.md`.
- Cuenta la **última** submission — el archivo final se elige por CV repetido, no por el leaderboard público.

## Seguimiento
- Plan operativo: proyecto Linear **GCI Credit Default Risk 2026** (workspace GCI World), issues GCI-5…GCI-22 con milestones M1–M5.
- Issues públicos: pestaña Issues de este repo (espejo de los hitos).
