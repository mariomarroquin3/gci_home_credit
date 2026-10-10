# Experimento v2_ext_polys

- CV AUC (OOF, 5-fold StratifiedKFold, seed 42): **0.75804**
- Baseline (main, 2026-10-08): 0.75810 -> delta **-0.00006**
- Umbral de adopcion a main: +0.003 (features) / +0.002 (tuning) -> **NO adoptado**
- Polinomicas grado 2-3 + stats EXT + target encoding OOF (ORGANIZATION_TYPE, OCCUPATION_TYPE, m=100) + flag BUREAU_YEAR_ISNA. EXT_MEAN domina la ganancia pero no mueve el CV: la señal EXT ya estaba capturada.
- Reproducir: `python experiments/v2_ext_polys/train.py` (datos en `data/raw/`)
- Resultados completos: `results.json`, importancias en `importance_*.csv`
