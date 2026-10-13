# Exp v19 — pseudo-labeling (LGBM-tuned) — POSITIVO MENOR

Test rows con p<0.03 del blend v12 como negativos extra: 15,156 filas (0 positivos con p>0.97).
OOF honesto (solo sobre train): **CV 0.76215 (+0.00064 vs v6)** — mayor ganancia de un componente desde el tuning.
Bajo umbral de features (+0.003) para main, pero suficiente para probar pipeline completa -> v20.

---

> **CORRECCION 2026-10-13 (exp/pl-leak-v27):** la ganancia atribuida al pseudo-labeling en este
> experimento fue medida con seleccion GLOBAL de pseudo-filas (labeler entrenado sobre el 100%
> de train, incluidos los folds de validacion) -> fuga de seleccion de **+0.00096**. Con labeler
> anidado por fold (protocolo honesto), el efecto del PL es **-0.00017 ~ 0**. El pipeline de
> produccion queda SIN pseudo-labels (= v12). Este README se conserva como registro del error
> metodologico y su correccion.
