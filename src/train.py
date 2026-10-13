"""Pipeline de producción (exp/pseudo-pipeline-v20, CV OOF 0.76488):

  pass 1: LGBM-t + CB-t + XGB-t, seed-averaging 5 seeds x 5 folds (OOF honesto)
  pseudo: filas de test con prob(blend pass1) < 0.03 como negativos extra (15k aprox)
  pass 2: mismos 3 modelos re-entrenados con pseudo-filas en cada fold de train
  blend:  pesos por hill-climbing sobre OOF pass 2 -> submission

Uso:
  python -m src.train                      # pipeline completa (~4 h en CPU)
  python -m src.train --seeds 42 --folds 2 # smoke test rápido
  python -m src.train --no-pseudo          # solo pass 1 (equivale a exp/seedavg-v12)
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.blend import apply_weights, hill_climb_weights
from src.config import (ID_COL, N_FOLDS, OUTPUT_DIR, PSEUDO_MAX_PROB, SEEDS,
                        SUBMISSION_DIR, TARGET, seed_everything)
from src.data import load_data
from src.features import build
from src.models import run_cb, run_lgbm, run_xgb

MODELS = ["lgbm", "cb", "xgb"]


def run_pass(X, y, X_test, cat_cols, seeds, n_folds, Xp=None, yp=None, save_dir=None):
    oof, tpred = {}, {}
    oof["lgbm"], tpred["lgbm"] = run_lgbm(X, y, X_test, seeds, n_folds, Xp, yp, save_dir)
    oof["cb"], tpred["cb"] = run_cb(X, y, X_test, cat_cols, seeds, n_folds, Xp, yp, save_dir)
    oof["xgb"], tpred["xgb"] = run_xgb(X, y, X_test, cat_cols, seeds, n_folds, Xp, yp, save_dir)
    # seed-averaging: media de OOF por seed (cada seed = SKF propio -> sigue siendo OOF honesto)
    return {m: o.mean(axis=0) for m, o in oof.items()}, \
           {m: t.mean(axis=0) for m, t in tpred.items()}, oof, tpred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--no-pseudo", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    seed_everything(seeds[0])

    train, test, sample = load_data()
    train, test, cat_cols = build(train, test)
    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X, y = train[features], train[TARGET].values
    X_test = test[features]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = OUTPUT_DIR / f"run_{stamp}{tag}"
    (out / "models").mkdir(parents=True, exist_ok=True)

    print("== pass 1 (sin pseudo) ==")
    oof1, tp1, oof1_raw, _ = run_pass(X, y, X_test, cat_cols, seeds, args.folds)
    for m in MODELS:  # per-seed OOF: insumo del RSKF pareado de seleccion final (nov)
        np.save(out / f"oof_{m}_perseed_pass1.npy", oof1_raw[m])
    for m in MODELS:
        print(f"  {m}: OOF seed-avg {roc_auc_score(y, oof1[m]):.5f}")
    w1, cv1 = hill_climb_weights(oof1, y)
    print(f"  blend pass1: {cv1:.5f} {w1}")
    p1_test = apply_weights(tp1, w1)

    if args.no_pseudo:
        oof2, tp2, w2, cv2, n_pseudo = oof1, tp1, w1, cv1, 0
    else:
        mask = p1_test < PSEUDO_MAX_PROB
        Xp = X_test[mask]  # conserva indice original: run_xgb lo usa para mapear filas one-hot
        yp = np.zeros(int(mask.sum()), dtype=int)
        n_pseudo = int(mask.sum())
        print(f"== pass 2 (pseudo-labels: {n_pseudo} filas negativas) ==")
        oof2, tp2, oof2_raw, _ = run_pass(X, y, X_test, cat_cols, seeds, args.folds, Xp, yp, out / "models")
        for m in MODELS:
            print(f"  {m}: OOF PL seed-avg {roc_auc_score(y, oof2[m]):.5f}")
            np.save(out / f"oof_{m}_perseed.npy", oof2_raw[m])
        w2, cv2 = hill_climb_weights(oof2, y)
        print(f"  blend pass2: {cv2:.5f} {w2}")

    pred = apply_weights(tp2, w2)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    sub = sample[[ID_COL]].merge(pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred}), on=ID_COL, how="left")
    assert sub[TARGET].notna().all() and len(sub) == len(sample)
    sub_path = SUBMISSION_DIR / f"sub_{stamp}{tag}_cv{cv2:.5f}.csv"
    sub.to_csv(sub_path, index=False)
    for m in MODELS:
        np.save(out / f"oof_{m}.npy", oof2[m])
        np.save(out / f"test_{m}.npy", tp2[m])
    metrics = {"time": stamp, "seeds": seeds, "n_folds": args.folds, "pseudo_rows": n_pseudo,
               "cv_auc": round(cv2, 5), "weights": w2, "pass1": {"cv_auc": round(cv1, 5), "weights": w1},
               "submission": sub_path.name}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    with open(OUTPUT_DIR / "experiments.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")
    print(f"CV final: {cv2:.5f} -> {sub_path}")


if __name__ == "__main__":
    main()
