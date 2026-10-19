"""AutoGluon best_quality por fold externo (mismos folds que src.models: SKF shuffle, seed).

  agenv/bin/python experiments/v33_autogluon/run_ag.py <seed> <n_folds> <time_limit_s>
Guarda outputs/v33_ag/oof_ag_s<seed>.npy y test_ag_s<seed>.npy (test = media de folds).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "v33_ag"
TARGET = "TARGET"

seed, n_folds, tl = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
train = pd.read_csv(OUT / "train_feat.csv")
test = pd.read_csv(OUT / "test_feat.csv")
y = train[TARGET].values
oof_p, test_p = OUT / f"oof_ag_s{seed}.npy", OUT / f"test_ag_s{seed}.npy"
oof = np.load(oof_p) if oof_p.exists() else np.full(len(train), np.nan)
tpred = np.load(test_p) if test_p.exists() else np.zeros(len(test))
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
    if not np.isnan(oof[va]).any():
        continue  # fold ya hecho (reanudable)
    pred = TabularPredictor(label=TARGET, eval_metric="roc_auc", path=str(OUT / f"ag_s{seed}_f{fi}"),
                            verbosity=1).fit(
        train.iloc[tr], presets="best_quality", time_limit=tl, num_cpus=4,
        dynamic_stacking=False, ag_args_fit={"random_seed": seed})
    oof[va] = pred.predict_proba(train.iloc[va].drop(columns=[TARGET]))[1].values
    tpred += pred.predict_proba(test)[1].values / n_folds
    np.save(oof_p, oof)
    np.save(test_p, tpred)
    print(f"seed {seed} fold {fi}: AUC {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
    print(pred.leaderboard(silent=True)[["model", "score_val"]].head(8).to_string(), flush=True)
print(f"seed {seed} OOF AUC {roc_auc_score(y, oof):.5f}")
