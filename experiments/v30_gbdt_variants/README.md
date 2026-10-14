# Experimento v30 — variantes GBDT descorrelacionadas como miembros del blend

Idea (research/Métodos nuevos para ganar GCI.md, opción 1): sumar al blend LGBM+CB+XGB modelos
con otro sesgo inductivo. Código: `src/variants.py`; evaluación: `python -m src.validate <run> --add`.

## Protocolo
- Base: `python -m src.train --seeds 42,43,44 --folds 5 --extra "" --tag base` (RSKF 3x5, sin pseudo).
  Reproduce PR #15: blend in-sample 0.76434, anidado seed-avg **0.76426**.
- Cada variante usa exactamente los mismos folds (StratifiedKFold por seed) -> comparación pareada.
- Métrica: AUC del blend con pesos anidados (5-fold sobre filas), por seed, base vs base + candidato.
- Criterio de aceptación: gana en las 3 seeds.

## Resultados (blend anidado base por seed: 0.76388 / 0.76347 / 0.76302)

| candidato | AUC solo (media 3 seeds) | Δ blend por seed | gana | Δ seed-avg | peso |
|---|---|---|---|---|---|
| xgb_d2 (max_depth 2) | 0.76184 | -0.00009 / -0.00002 / -0.00003 | 0/3 | -0.00001 | 0.01 |
| lgbm_et (extra_trees, 127 hojas) | 0.75488 | 0 / 0 / 0 | 0/3 | 0 | 0.00 |
| lgbm_et2 (extra_trees, params Optuna) | 0.75500 | 0 / 0 / 0 | 0/3 | 0 | 0.00 |
| lgbm_dart (1000 rondas fijas) | 0.76006 | 0 / +0.00001 / 0 | 2/3 | 0 | 0.00 |
| **lgbm_linear** (linear_tree) | 0.76126 | +0.00006 / +0.00004 / +0.00003 | **3/3** | +0.00005 | 0.23 |
| **lgbm_init** (init_score = logit OOF LogReg) | 0.75999 | +0.00015 / +0.00010 / +0.00009 | **3/3** | +0.00008 | 0.15 |
| **lgbm_init + lgbm_linear** | — | +0.00021 / +0.00013 / +0.00012 | **3/3** | **+0.00013** (0.76439) | 0.22 |

Adoptado: `EXTRA_MODELS = ["lgbm_init", "lgbm_linear"]` en `src/config.py`.

## Lecturas
- La ganancia es consistente (3/3 seeds, pareada) pero chica: +0.00013, por debajo del ruido de
  partición (~0.0004) y del umbral de blend de project.md (+0.001). Pasa el criterio pareado, no el umbral.
- Correlación con XGB: 0.996 (xgb_d2), 0.975 (linear), 0.971 (dart), 0.963 (init), 0.958 (extra_trees).
  extra_trees es el más descorrelacionado pero pierde 0.007 de AUC solo y el blend le da peso 0.
- Lo que aporta es otro sesgo funcional (lineal en hoja; boosting desde un modelo lineal), no más azar.
- DART: rondas elegidas en un split 80/20 aparte (seed 999), mejor ronda 964 -> 1000
  (`dart_rounds.py`).
- lgbm_init: LogReg (mediana + indicador de nulos, QuantileTransformer normal, one-hot, C=0.1);
  logit OOF interno de 5 folds para las filas de train de cada fold externo, modelo completo
  para val/test -> sin fuga.
