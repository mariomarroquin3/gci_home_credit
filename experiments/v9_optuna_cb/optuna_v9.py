"""Exp v9: Optuna CatBoost (features v1, CV 5-fold seed 42). 45 trials.
Log incremental en trials.csv. Ref: CB default-tuned 0.76143."""
import json
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v9_optuna_cb")
OUT.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build(df):
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        df[f"{c}_ISNA"] = df[c].isna().astype(int)
    df["CREDIT_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["ANNUITY_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["INCOME_PER_FAM"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_BIRTH_PCT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    df["PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25
    bur = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    df["BUREAU_REQ_TOTAL"] = df[bur].sum(axis=1, min_count=1)
    df["PHONE_FLAGS_SUM"] = df[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE"]].sum(axis=1)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
    return df


train = pd.read_csv("data/raw/train.csv")
y = train["TARGET"].values
X = build(train).drop(columns=["SK_ID_CURR", "TARGET"]).reset_index(drop=True)
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("str")
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
LOG = OUT / "trials.csv"
LOG.write_text("")


def objective(trial):
    params = dict(iterations=8000,
                  learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                  depth=trial.suggest_int("depth", 4, 8),
                  l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                  random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
                  bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
                  min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 300),
                  od_wait=300, od_type="Iter", eval_metric="AUC", random_seed=SEED, verbose=False)
    oof = np.zeros(len(train))
    for tr, va in splits:
        m = CatBoostClassifier(**params)
        m.fit(X.iloc[tr], y[tr], cat_features=cat_idx, eval_set=(X.iloc[va], y[va]),
              use_best_model=True)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    score = roc_auc_score(y, oof)
    with open(LOG, "a") as f:
        f.write(json.dumps({"trial": trial.number, "auc": round(score, 5),
                            **{k: v for k, v in params.items() if not isinstance(v, bool)}}) + "\n")
    return score


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.enqueue_trial({"learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0,
                     "random_strength": 1.0, "bagging_temperature": 0.5, "min_data_in_leaf": 50})
study.optimize(objective, n_trials=45, show_progress_bar=False)
best = {"best_auc": round(study.best_value, 5), "best_params": study.best_params,
        "delta_vs_v5_cb": round(study.best_value - 0.76143, 5)}
(OUT / "best_params.json").write_text(json.dumps(best, indent=2))
print(json.dumps(best, indent=2))
