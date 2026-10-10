# Experimento v1_ratios

- CV AUC (OOF, 5-fold StratifiedKFold, seed 42): **0.75827**
- Baseline (main, 2026-10-08): 0.75810 -> delta **+0.00017**
- Umbral de adopcion a main: +0.003 (features) / +0.002 (tuning) -> **NO adoptado**
- Features nuevas entran al top-15 por ganancia (CREDIT_ANNUITY, CREDIT_GOODS, DOWNPAYMENT) pero el efecto neto es ruido (fold_std 0.0032). src/features.py integrado.
- Reproducir: `python experiments/v1_ratios/train.py` (datos en `data/raw/`)
- Resultados completos: `results.json`, importancias en `importance_*.csv`
