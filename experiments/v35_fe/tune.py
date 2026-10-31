"""v36: re-tuneo de LGBM para el set de features actual (v35 + agregados EXT). Optuna, lr 0.03.

Tunea con folds seed 42 y valida el mejor contra los params actuales con folds seed 43.
Uso: python -m experiments.v35_fe.tune --trials 40
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.blend import fast_auc
from src.config import ID_COL, LGBM_PARAMS, TARGET
from src.data import load_data
from src.features import build
from src.structure import FoldFeatures

EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def ext_agg(X):
    E = X[EXT].to_numpy(float)
    P = X[[f"{c}_PRED" for c in EXT]].to_numpy(float)
    F = np.where(np.isnan(E), P, E)
    f = pd.DataFrame(index=X.index)
    f["EXT_MEAN"] = np.nanmean(E, 1)
    f["EXT_STD"] = np.nanstd(E, 1)
    f["EXT_PROD_F"] = F.prod(1)
    f["EXT_MEAN_F"] = F.mean(1)
    f["EXT_MIN_F"] = F.min(1)
    f["EXT_WMEAN_F"] = (2 * F[:, 0] + 3 * F[:, 1] + 4 * F[:, 2]) / 9
    for i, c in enumerate(["E1", "E2", "E3"]):
        f[f"{c}_FILL"] = F[:, i]
    f["EXT_F_x_CA"] = f["EXT_MEAN_F"] * X["CREDIT_ANNUITY"].values
    f["EXT_F_x_AGE"] = f["EXT_MEAN_F"] * X["AGE_YEARS"].values
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    train, test, _ = load_data()
    n = len(train)
    trb, teb, _ = build(train, test)
    feats = [c for c in trb.columns if c not in (ID_COL, TARGET)]
    X0 = pd.concat([trb[feats], teb[feats]], ignore_index=True)
    X0 = pd.concat([X0, ext_agg(X0)], axis=1)
    y = train[TARGET].values
    ff = FoldFeatures(train, test, y)
    folds = {}
    for seed in (42, 43):
        folds[seed] = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y):
            F_tr, F_te = ff(tr, seed)
            X = pd.concat([X0, pd.concat([F_tr, F_te], ignore_index=True)], axis=1)
            folds[seed].append((lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=False),
                                X.iloc[va], va))
    print(f"prep {time.time()-t0:.0f}s", flush=True)

    def cv(params, seed):
        oof = np.zeros(n)
        for dtr, Xva, va in folds[seed]:
            m = lgb.train(dict(params, seed=seed), dtr, 20000, valid_sets=[lgb.Dataset(Xva, y[va])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        return fast_auc(y, oof)

    base = dict(LGBM_PARAMS, learning_rate=0.03)

    def objective(trial):
        p = dict(base,
                 num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 20, 600, log=True),
                 feature_fraction=trial.suggest_float("feature_fraction", 0.15, 0.8),
                 bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
                 lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
                 min_split_gain=trial.suggest_float("min_split_gain", 0, 1),
                 max_depth=trial.suggest_int("max_depth", 3, 12))
        s = cv(p, 42)
        print(f"trial {trial.number} {s:.5f} {time.time()-t0:.0f}s", flush=True)
        return s

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial({k: LGBM_PARAMS[k] for k in ["num_leaves", "min_child_samples", "feature_fraction",
                                                  "bagging_fraction", "lambda_l2", "min_split_gain", "max_depth"]})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(objective, n_trials=args.trials)
    best = dict(base, **st.best_params)
    print("best s42", f"{st.best_value:.5f}", "current s42", f"{st.trials[0].value:.5f}", json.dumps(st.best_params))
    print("holdout s43: current", f"{cv(base, 43):.5f}", "best", f"{cv(best, 43):.5f}", flush=True)


if __name__ == "__main__":
    main()
