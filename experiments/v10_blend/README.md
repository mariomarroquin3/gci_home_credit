# Exp v10 — Blend LGBM-tuned + CB-tuned + XGBoost (hill climbing)

- Pesos: simplex grueso 0.05 -> hill climbing 0.01 sobre OOF (5-fold, seed 42).
- Componentes: LGBM-tuned 0.76151, CB-tuned (v9) 0.76243, XGBoost 0.76186.
- **CV blend: 0.76414** (pesos lgbm .26 / cb .45 / xgb .29).
- Delta vs blend v7 (0.76366): **+0.00048** -> debajo del umbral de blend (+0.001).
- Promedio simple 1/3 daria 0.76406 (los pesos solo afinan +0.0001).

Decision: NO adoptado (bajo umbral). Se documenta; la palanca grande es v11 (features Tier-1).
Si se desea submission intermedia, `submission_v10_blend.csv` esta validado (61500 filas, rango ok).
