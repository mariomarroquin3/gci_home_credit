# Exp v19 — pseudo-labeling (LGBM-tuned) — POSITIVO MENOR

Test rows con p<0.03 del blend v12 como negativos extra: 15,156 filas (0 positivos con p>0.97).
OOF honesto (solo sobre train): **CV 0.76215 (+0.00064 vs v6)** — mayor ganancia de un componente desde el tuning.
Bajo umbral de features (+0.003) para main, pero suficiente para probar pipeline completa -> v20.
