# Experimento v4 — Optuna (60 trials)

- **CV AUC (OOF, 5-fold, seed 42): 0.76145**
- Baseline (main): 0.75809 -> delta **+0.00336** ✔ SUPERA el umbral de tuning (+0.002)
- Leccion clave: lr muy bajo (0.008) + min_child_samples alto (197) + feature_fraction bajo (0.40) + min_split_gain 0.23 -> regularizacion fuerte, arboles mas pequeños pero muchos mas.

## Mejor config
```json
{
  "learning_rate": 0.008135825913706748,
  "num_leaves": 37,
  "min_child_samples": 197,
  "feature_fraction": 0.40362837123334017,
  "bagging_fraction": 0.9525103687282102,
  "lambda_l1": 0.00011657904410727821,
  "lambda_l2": 3.0181881636817223,
  "min_split_gain": 0.22728540197274544,
  "max_depth": 9
}
```

## Top 5 trials
| trial | AUC |
|---|---|
| 15 | 0.76145 |
| 25 | 0.76144 |
| 44 | 0.76142 |
| 56 | 0.76136 |
| 28 | 0.76133 |
