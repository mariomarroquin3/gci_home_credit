"""Blend anidado v30 (5 miembros) vs v30 + AutoGluon, por seed (pesos fuera de la fila evaluada).

  python experiments/v33_autogluon/screen.py outputs/run_<...>_v32base 42[,43,44]
La corrida base debe tener las seeds en el mismo orden que metrics.json["seeds"].
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import OUTPUT_DIR, TARGET  # noqa: E402
from src.data import load_data  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from src.blend import fast_auc, hill_climb_weights  # noqa: E402
from src.validate import NESTED_FOLDS, NESTED_SEED  # noqa: E402


def nested_two_stage(mats, y, extra=None):
    """Pesos v30 por hill-climbing y luego (blend v30, extra) en 2 miembros; ambos dentro del fold anidado.
    La grilla de 6 miembros de hill_climb_weights es inviable en tiempo."""
    base, new, w_extra = np.zeros(len(y)), np.zeros(len(y)), []
    skf = StratifiedKFold(n_splits=NESTED_FOLDS, shuffle=True, random_state=NESTED_SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        w, _ = hill_climb_weights({m: mats[m][tr] for m in V30}, y[tr], score_fn=fast_auc)
        b = sum(w[m] * mats[m] for m in V30)
        base[va] = b[va]
        if extra is not None:
            w2, _ = hill_climb_weights({"v30": b[tr], "x": extra[tr]}, y[tr], grid=100, score_fn=fast_auc)
            new[va] = w2["v30"] * b[va] + w2["x"] * extra[va]
            w_extra.append(w2["x"])
    return roc_auc_score(y, base), roc_auc_score(y, new), float(np.mean(w_extra))

V30 = ["lgbm", "cb", "xgb", "lgbm_init", "lgbm_linear"]

run, seeds = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(",")]
all_seeds = json.loads((run / "metrics.json").read_text())["seeds"]
y = load_data()[0][TARGET].values
oof = {m: np.load(run / f"oof_{m}_perseed_pass1.npy") for m in V30}
for seed in seeds:
    si = all_seeds.index(seed)
    ag = np.load(OUTPUT_DIR / "v33_ag" / f"oof_ag_s{seed}.npy")
    o = {m: oof[m][si] for m in V30}
    base, new, w_ag = nested_two_stage(o, y, ag)
    print(f"seed {seed}: AG solo {roc_auc_score(y, ag):.5f} corr_xgb {np.corrcoef(ag, o['xgb'])[0, 1]:.4f} | "
          f"v30 {base:.5f} -> +ag {new:.5f} (Δ {new - base:+.5f}) peso ag {w_ag:.2f}")
