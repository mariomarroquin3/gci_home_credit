# Experimento v5 — Diversificadores (CatBoost + XGBoost)
Features v1, CV 5-fold seed 42.
- **CatBoost: 0.76143** (categóricas nativas, depth 6, lr 0.03)
- **XGBoost: 0.76186** (one-hot, depth 5, lr 0.03)
Ambos en la liga del LGBM tuneado (0.76151) -> diversidad real para el blend.