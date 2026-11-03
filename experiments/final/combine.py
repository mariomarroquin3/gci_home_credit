"""Une las corridas por seed de run_seeds.sh: media por modelo entre seeds + hill-climbing.

Es el mismo cálculo que src.train con todas las seeds en un proceso (OOF y test se promedian
por seed, luego se ajustan los pesos sobre el OOF promedio).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.blend import apply_weights, fast_auc, hill_climb_weights
from src.config import ID_COL, OUTPUT_DIR, SUBMISSION_DIR, TARGET
from src.data import load_data

runs = sorted(OUTPUT_DIR.glob(sys.argv[1] if len(sys.argv) > 1 else "run_*_final_s*"))
models = [p.name[4:].replace("_perseed_pass1.npy", "") for p in sorted(runs[0].glob("oof_*_perseed_pass1.npy"))]
train, test, sample = load_data()
y = train[TARGET].values
oof = {m: np.mean([np.load(r / f"oof_{m}.npy") for r in runs], axis=0) for m in models}
tp = {m: np.mean([np.load(r / f"test_{m}.npy") for r in runs], axis=0) for m in models}
for m in models:
    print(f"{m}: {fast_auc(y, oof[m]):.5f}")
w, _ = hill_climb_weights(oof, y, score_fn=fast_auc)
cv = fast_auc(y, apply_weights(oof, w))
sub = sample[[ID_COL]].merge(pd.DataFrame({ID_COL: test[ID_COL], TARGET: apply_weights(tp, w)}), on=ID_COL, how="left")
assert sub[TARGET].notna().all() and len(sub) == len(sample)
tag = "final" if len(sys.argv) == 1 else "combo"
path = SUBMISSION_DIR / f"sub_{tag}_{len(runs)}seeds_cv{cv:.5f}.csv"
sub.to_csv(path, index=False)
print("runs", [r.name for r in runs], "weights", w)
print(f"CV final: {cv:.5f} -> {path}")
(OUTPUT_DIR / "final_combine.json").write_text(json.dumps({"runs": [r.name for r in runs], "cv": cv, "weights": w}))
