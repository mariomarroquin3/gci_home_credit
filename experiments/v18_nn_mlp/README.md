# Exp v18 — NN-MLP con embeddings (3 seeds x 5 folds) — NO APORTA

- NN standalone seed-avg OOF: **0.74906** (vs GBMs ~0.763)
- Blend 4 modelos: **0.76459** con peso NN = **0.0** (el hill-climbing lo descarta)

Conclusion: el NN no captura nada que los GBMs no tengan; diversidad aparente, valor real cero.
Familias probadas: LGBM, CB, XGB (adoptados), NN/stacking/KNN/TE (todos neutros o negativos).
Techo confirmado del stack GBM sobre features v1: ~0.7646.
