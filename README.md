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

## Estado actual (2026-10-21)

**Pipeline de producción (`python -m src.train`, main tras PR #14/#15/#17): blend de 5 miembros —
LGBM+CatBoost+XGBoost tuneados (Optuna) + variantes descorrelacionadas `lgbm_init` y `lgbm_linear`
(src/variants.py, v30) — seed-averaging 5 seeds × 10 folds, restricciones monótonas en EXT_SOURCE_1/2/3
(LGBM/XGB), pseudo-labeling APAGADO (v27) y pesos por hill-climbing anidado.**
**Blend anidado RSKF 3×5: 0.76439 ± ruido de partición ~0.0004** (src/validate.py, PR #15).

Scoreboard (CV OOF honesto; LB = leaderboard público):

| Pipeline | CV | LB |
|---|---|---|
| Baseline LightGBM (commit inicial) | 0.7581 | — |
| + features v1 | 0.7583 | — |
| LGBM tuneado Optuna (v4/v6) | 0.7615 | — |
| Blend 3 GBMs (v7) | 0.7637 | **0.7666** (pos. 169/462) |
| + seed-averaging 5 seeds (v12) | 0.7646 | — |
| ~~+ pseudo-labeling (v20)~~ | ~~0.7649~~ inflado → corregido (v27) | — |
| Blend 3 GBMs, RSKF anidado (PR #15) | 0.76426 | — |
| **+ variantes v30 (`lgbm_init`+`lgbm_linear`, producción)** | **0.76439** | — |

Correcciones metodológicas (2026-10-13, exp/pl-leak-v27, exp/monotone-v28, exp/monotone-blend-v29):
- El +0.0003..+0.0005 atribuido al pseudo-labeling era **fuga de selección** (+0.00096 medida; con
  labeler anidado por fold el efecto honesto es ~0). PL fuera de producción; `--pseudo` queda solo
  para reproducción histórica.
- Una ganancia pareada y sistemática también puede ser una *fuga sistemática*: toda selección de
  datos auxiliares debe anidarse por fold.
- Monotonías EXT_SOURCE: efecto real en componente LGBM (+0.00025, 5/5 seeds) pero se diluye en el
  blend (+0.00002, n.s.); se conservan como regularización con prior de dominio.
- Ruido de partición RSKF ~0.0004 → deltas < ~0.0008 en un solo split no son concluyentes (PR #15);
  el sesgo de ajustar pesos de blend sobre el mismo OOF es pequeño (~0.0001).

Experimentos v1–v30: una rama `exp/*` por experimento con README y resultados (incluye los negativos
documentados: features de 2º nivel, KNN-500, TE-OOF, stacking, NN-MLP, PL ronda-2, VAE, TabPFN,
frecuency-encoding, selección por permutación, variantes GBDT no adoptadas).
Decisión de submission final: **Repeated StratifiedKFold**, nunca public LB (ruido ±0.003 del subset;
el reglamento confirma leaderboard privado sobre el test completo al cierre).
- Meta M3 (red de seguridad): pipeline end-to-end reproducible con seeds fijas y refs completo.

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
python -m src.train                        # pipeline completa v20 (~4 h CPU) -> submissions/
python -m src.train --seeds 42 --folds 2   # smoke test
python -m src.train --no-pseudo            # solo pass 1 (equivale a v12)
```
Cada corrida guarda OOF/test preds (`.npy`), modelos, pesos del blend y `metrics.json` en
`outputs/run_<timestamp>/`; 2 corridas con mismas seeds = mismo CSV. EDA en `notebooks/01_eda.ipynb`.

Selección de candidatos (RSKF pareado, nunca por public LB):
```bash
python -m src.train --seeds 42,43,44 --folds 5 --tag A     # RSKF 3x5 por candidato
python -m src.validate outputs/run_<A>                      # AUC media ± std + blend con pesos anidados
python -m src.validate outputs/run_<A> outputs/run_<B>      # Δ pareado por seed (B - A)
python -m src.variants outputs/run_<A> --only lgbm_init      # OOF de una variante GBDT con los mismos folds
python -m src.validate outputs/run_<A> --add lgbm_init,a+b   # ¿el miembro extra mejora el blend anidado en cada seed?
```

## Estructura
```
src/config.py     rutas, seeds, hiperparámetros tuneados (Optuna), umbral pseudo-label
src/data.py       carga de datos
src/validate.py   RSKF pareado entre corridas + AUC del blend con pesos anidados (honesto)
src/features.py   limpieza y features v1
src/models.py     LGBM / CatBoost / XGBoost por seed con early stopping (+ pseudo-rows)
src/variants.py   variantes GBDT descorrelacionadas (DART, extra_trees, linear_tree, XGB d2, init_score)
src/blend.py      hill-climbing de pesos sobre OOF
src/train.py      pipeline end-to-end: pass1 -> pseudo-labels -> pass2 -> blend -> submission
experiments/      (en ramas exp/*) un experimento por rama con README y resultados
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
