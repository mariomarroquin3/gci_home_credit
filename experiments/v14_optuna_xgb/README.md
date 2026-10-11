# Exp v14 — Optuna XGBoost (45 trials, 5-fold seed 42)

Best: trial 42, **CV 0.76297** (vs XGB default v5: 0.76186 → **+0.00111**).
Bajo el umbral de tuning para main (+0.002) pero mejora el componente del blend.

Best params: `{"learning_rate": 0.0067, "max_depth": 3, "min_child_weight": 14,
"subsample": 0.917, "colsample_bytree": 0.504, "reg_alpha": 8.78, "reg_lambda": 3.01, "gamma": 0.257}`
(depth 3 + regularizacion fuerte: mismo patron convergente que LGBM y CB).

Guardado en `best_params.json`. Consumido por exp_v15 (seed-avg + re-blend).
