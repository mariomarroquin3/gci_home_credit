# Exp v26 — VAE augmentacion minoritaria — NEGATIVO FUERTE

- CV 0.74966 vs re-check 0.76106 (**-0.0114**) — muy por debajo del umbral +0.003.
- Protocolo honesto: VAE (z=16, 15 epochs) entrenado por fold SOLO en defaulters del fold-train,
  10k sinteticos label=1, LGBM-tuneado. Proxy estandar (v6/v19/v23-25).
- Interpretacion: los sinteticos del VAE borran la frontera de decision (oversampling borroso);
  AUC es ranking y los GBMs ya manejan el 8% de desbalance. Consistente con literatura tabular:
  modelos generativos profundos < GBDT. Ref: estilo notebooks Kaggle Home Credit (citado en refs).
- LINEA MODELOS GENERATIVOS CERRADA. Con esto quedan cerradas TODAS las lineas:
  modelos (7 familias), features (0/9), stacking (p=0.006), generativos (VAE), semi-supervisado
  (PL ronda-2), diversificadores (NN/KNN/TabPFN).
