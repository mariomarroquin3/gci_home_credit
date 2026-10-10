# Experimento v3_tuning

- CV AUC (OOF, 5-fold StratifiedKFold, seed 42): **0.75888**
- Baseline (main, 2026-10-08): 0.75810 -> delta **+0.00078**
- Umbral de adopcion a main: +0.003 (features) / +0.002 (tuning) -> **NO adoptado**
- lr 0.02, num_leaves 63, min_child 100, feature_fraction 0.7, l1 0.1, l2 5.0, min_split_gain 0.01. Mejor delta hasta ahora: el tuning mueve mas que las features. Direccion para Optuna en Sprint 4.
- Reproducir: `python experiments/v3_tuning/train.py` (datos en `data/raw/`)
- Resultados completos: `results.json`, importancias en `importance_*.csv`
