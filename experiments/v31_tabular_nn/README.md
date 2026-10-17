# Experimento v31 — TabM y RealMLP como miembros del blend (negativo)

Idea (research/Métodos nuevos para ganar GCI.md, candidato recomendado): redes tabulares modernas
como fuente de diversidad frente a los GBDT. Código: `src/nn.py` (pytabkit 1.7.3, torch 2.14.1 CPU,
defaults de los autores sin HPO); evaluación: `screen.py`.

## Protocolo
- Base: `python -m src.train --seeds 42,43,44 --folds 5 --tag base` (v30 con lgbm_init + lgbm_linear),
  blend in-sample 0.76446.
- `python -m src.nn outputs/run_X --only tabm,realmlp --seeds 42`: mismos folds, mejor época por AUC
  en el fold de validación (igual que el early stopping de los GBDT). Checkpoint por fold en `run_X/nn_ckpt`.
- Screening en seed 42 (5 folds): ~1 h por fold y modelo en 4 cores -> 3x5 completo serían ~30 h.
- Blend anidado en 2 etapas (5 GBDT -> blend v30; luego pesos {v30, redes}), pesos solo con filas de train.
- Preprocesado NN: inf -> NaN, mediana + indicador de nulo, categóricas con 'MISSING'.

## Resultados (seed 42, blend anidado base v30 = 0.76403)

| candidato | AUC solo | corr con XGB | blend | Δ | peso |
|---|---|---|---|---|---|
| TabM-D (k=32) | 0.75038 | 0.924 | 0.76403 | 0.00000 | 0.00 |
| RealMLP-TD | 0.75049 | 0.917 | 0.76389 | -0.00014 | 0.04 |
| TabM + RealMLP | — | — | 0.76389 | -0.00014 | 0 / 0.04 |

Rechazado sin correr seeds 43-44: ningún candidato mejora la seed 42.

## Lecturas
- Las redes quedan 0.013 de AUC por debajo de los GBDT (0.763); con 32 columnas tabulares
  simples los árboles dominan y la descorrelación (0.92) no compensa la pérdida.
- RealMLP recibe peso 0.04 en train pero empeora fuera de muestra: sobreajuste del peso.
