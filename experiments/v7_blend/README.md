# Experimento v7 — Blend 3 modelos (pesos en simplex, grid 0.05)
- Individuales: LGBM 0.76151, CB 0.76143, XGB 0.76186
- Pesos: {'lgbm': 0.3, 'catboost': 0.35, 'xgboost': 0.35}
- **CV blend: 0.76366** (+0.0018 vs mejor individual) ✔ supera umbral de blend (+0.001)
- Total vs baseline 0.75809: **+0.00557**