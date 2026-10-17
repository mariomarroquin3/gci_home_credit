"""Screening de una seed: blend anidado v30 (lgbm,cb,xgb,lgbm_init,lgbm_linear) vs + red(es).

Lee los checkpoints por fold de src.nn (run/nn_ckpt/{nombre}_s{seed}_f{fold}.npz).

  python experiments/v31_tabular_nn/screen.py outputs/run_X --seed 42 --add tabm,realmlp,tabm+realmlp
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import TARGET  # noqa: E402
from src.data import load_data  # noqa: E402
from src.models import _folds  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.blend import fast_auc, hill_climb_weights  # noqa: E402
from src.validate import NESTED_FOLDS, NESTED_SEED, blend  # noqa: E402

BASE = ["lgbm", "cb", "xgb", "lgbm_init", "lgbm_linear"]


def load_nn(run, name, seed, y, n_folds):
    o = np.full(len(y), np.nan)
    for fi, (_, va) in enumerate(_folds(y, n_folds, seed)):
        o[va] = np.load(run / "nn_ckpt" / f"{name}_s{seed}_f{fi}.npz")["va"]
    return o


def nested_two_stage(mats, y, extra):
    """Blend anidado en 2 etapas (el grid de hill_climb con 6+ miembros es inviable):
    pesos v30 sobre los 5 GBDT y luego pesos {blend_v30, redes}; ambos solo con filas de train."""
    pred_base, pred_new, ws = np.zeros(len(y)), np.zeros(len(y)), []
    skf = StratifiedKFold(n_splits=NESTED_FOLDS, shuffle=True, random_state=NESTED_SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        w5, _ = hill_climb_weights({m: mats[m][tr] for m in BASE}, y[tr], score_fn=fast_auc)
        b = blend({m: mats[m] for m in BASE}, w5)
        pred_base[va] = b[va]
        if extra:
            st = {"v30": b, **{c: mats[c] for c in extra}}
            w, _ = hill_climb_weights({k: v[tr] for k, v in st.items()}, y[tr], score_fn=fast_auc)
            pred_new[va] = blend({k: v[va] for k, v in st.items()}, w)
            ws.append(w)
    return roc_auc_score(y, pred_base), (roc_auc_score(y, pred_new) if extra else None), ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add", type=str, default="tabm,realmlp,tabm+realmlp")
    args = ap.parse_args()
    met = json.loads((args.run / "metrics.json").read_text())
    si = met["seeds"].index(args.seed)
    train, _, _ = load_data()
    y = train[TARGET].values
    oof = {m: np.load(args.run / f"oof_{m}_perseed_pass1.npy")[si] for m in BASE}
    base, _, _ = nested_two_stage(oof, y, [])
    print(f"seed {args.seed} base v30 blend anidado {base:.5f}")
    for group in args.add.split(","):
        extra = group.split("+")
        try:
            for c in extra:
                oof[c] = load_nn(args.run, c, args.seed, y, met["n_folds"])
        except FileNotFoundError:
            print(f"  +{group}: faltan folds")
            continue
        solo = {c: round(roc_auc_score(y, oof[c]), 5) for c in extra}
        corr = {c: round(np.corrcoef(oof[c], oof["xgb"])[0, 1], 4) for c in extra}
        _, new, ws = nested_two_stage(oof, y, extra)
        w = {c: round(np.mean([x[c] for x in ws]), 3) for c in extra}
        print(f"  +{group:14s} solo {solo} corr_xgb {corr} -> {new:.5f} (Δ {new - base:+.5f}) peso {w}")


if __name__ == "__main__":
    main()
