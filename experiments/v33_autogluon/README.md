# Experimento v33 — AutoGluon best_quality como miembro del blend

## Protocolo
- `python experiments/v33_autogluon/dump_features.py` (features de src.features -> CSV).
- Entorno aparte (AutoGluon 1.6.3 requiere pandas 2): `python -m venv agenv && agenv/bin/pip install "autogluon.tabular[all]==1.6.3"`.
- `agenv/bin/python experiments/v33_autogluon/run_ag.py 42 5 3600`: best_quality, 1 h por fold externo,
  mismos folds que src.models (SKF shuffle seed 42), dynamic_stacking off, 4 CPU. Reanudable por fold.
- `python experiments/v33_autogluon/screen.py outputs/run_<...>_v32base 42`: blend anidado en 2 etapas
  (pesos v30 y luego v30-blend vs AG, ambos dentro de cada fold anidado).

## Resultados (seed 42)

| | AUC |
|---|---|
| AutoGluon solo (folds 0.7603 / 0.7629 / 0.7667 / 0.7583 / 0.7631) | 0.76222 |
| XGB solo (referencia) | ≈0.7630 |
| Blend anidado v30 | 0.76409 |
| v30 + AutoGluon | 0.76409 (Δ +0.00000, peso 0.00) |

Correlación con XGB 0.9825. **Rechazado**; seeds 43–44 no se corren. AutoGluon arma en esencia otro
ensemble de GBDTs (LightGBM/CatBoost/XGB/RF) y no aporta señal nueva. Costo: ≈5.5 h de CPU por seed.
