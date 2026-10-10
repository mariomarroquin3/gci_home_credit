# Exp v9 — Optuna CatBoost

- Espacio: lr [0.01,0.1] log, depth 4-8, l2 [1,30] log, random_strength [0.1,10] log, bagging_temperature [0,2], min_data_in_leaf [20,300].
- 26/45 trials (sandbox se reinicio; meseta evidente en ultimos 13 trials -> no se relanzo).
- **Mejor trial 12: CV AUC 0.76243** (vs 0.76143 sin tunear, +0.001).
- Re-entrenado completo (`train_v9_final.py`): **CV 0.76243** — reproduce exacto.
- Params en `best_params.json`. Patron: depth 4 + lr bajo + regularizacion fuerte (igual que LGBM).
- OOF/test npy no versionados (quedan en sandbox); blend v10 los consume.

Decision: CatBoost tuneado pasa a ser componente del blend v10 (reemplaza CB default).
