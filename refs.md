# Referencias de código externo

Regla GCI: citar toda fuente de código usada. Una fila por referencia.

| Fecha | Fuente (URL) | Qué se tomó | Dónde se usa (archivo) |
|---|---|---|---|
| 2026-10 | `tutorial.ipynb` (material GCI) | Estructura del baseline | `src/train.py` |
| 2026-10 | Ke et al. 2017, *LightGBM* (NeurIPS) | Algoritmo GBDT principal | `src/models.py` (`run_lgbm`) |
| 2026-10 | Chen & Guestrin 2016, *XGBoost* (KDD) | Diversificador GBDT | `src/models.py` (`run_xgb`) |
| 2026-10 | Prokhorenkova et al. 2018, *CatBoost* (NeurIPS) | GBDT con categóricas nativas | `src/models.py` (`run_cb`) |
| 2026-10 | Akiba et al. 2019, *Optuna* (KDD) | Tuning de hiperparámetros | `experiments/v4_optuna`, `v9_optuna_cb`, `v14_optuna_xgb` |
| 2026-10 | Lee 2013, *Pseudo-Label* (ICML workshop) | Semi-supervisado: filas de test p<0.03 como negativos extra (1 ronda) | `src/train.py` (pass 2), `experiments/v19`, `v20` |
| 2026-10 | Caruana et al. 2004, *Ensemble Selection* (ICML) | Hill-climbing de pesos del blend sobre OOF | `src/blend.py` |
| 2026-10 | Kaggle 2018, *Home Credit Default Risk* | Dataset original del que deriva el subset GCI | `data/raw/` |
