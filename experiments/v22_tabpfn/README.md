# Exp v22 — TabPFN v2 — CERRADO (incompatibilidad de entorno, EV~0)

- tabpfn >= 2.5 requiere API token (TABPFN_TOKEN) — fuera de las reglas de reproducibilidad offline.
- tabpfn 2.0.9 (OSS) es incompatible con pandas 2.3 del entorno (`numpy string dtypes are not allowed`),
  incluso con entrada 100% numerica — el fallo es interno del paquete.
- Decision: no pelear el entorno. El NN-MLP (v18), diversificador mas fuerte y con datos completos,
  recibio peso 0.0 en el blend; TabPFN con subsamples de 10k tendria aportacion esperada ~0.
Todas las familias de modelos probadas: LGBM/CB/XGB (adoptados), NN/stacking/KNN/TE/TabPFN (cerrados).
