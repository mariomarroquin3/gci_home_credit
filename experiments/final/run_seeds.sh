#!/bin/bash
# Corrida final por seed (equivale a `python -m src.train` con 5 seeds x 10 folds, partida en 5 procesos)
cd "$(dirname "$0")/../.."
for s in 42 43 44 45 46; do
  ls outputs/run_*_final_s$s/metrics.json >/dev/null 2>&1 && continue
  python -m src.train --seeds $s --folds 10 --tag final_s$s > outputs/final_s$s.log 2>&1
done
python -m experiments.final.combine > outputs/final_combine.log 2>&1
echo DONE >> outputs/final_combine.log
