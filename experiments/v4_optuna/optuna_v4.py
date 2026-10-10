"""Exp v4: Optuna sobre features v1 (branch exp/optuna-v4).
Objetivo: maximizar CV AUC (OOF 5-fold, seed 42). Log incremental en optuna_trials.csv."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v4_optuna")
OUT.mkdir(parents=True, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_STATIC = dict(objective="binary", metric="auc", seed=SEED, deterministic=True,
                     force_row_wise=True, verbose=-1, bagging_freq=1)


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
    for c in df.select_dtypes(include=["object", "str"]).columns:
        df[c] = df[c].astype("category")
    return df


train = pd.read_csv("data/raw/train.csv")
y = train["TARGET"].values
X = build(train).drop(columns=["SK_ID_CURR", "TARGET"]).reset_index(drop=True)
cat_feats = [c for c in X.columns if str(X[c].dtype) == "category"]
print(f"Features: {X.shape[1]}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(skf.split(X, y))

LOG = OUT / "optuna_trials.csv"


def cv_auc(params):
    oof = np.zeros(len(train))
    for tr, va in splits:
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_feats)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_feats, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=8000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    return roc_auc_score(y, oof)


def objective(trial):
    params = dict(PARAMS_STATIC,
                  learning_rate=trial.suggest_float("learning_rate", 0.005, 0.06, log=True),
                  num_leaves=trial.suggest_int("num_leaves", 16, 256, log=True),
                  min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
                  feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
                  bagging_fraction=trial.suggest_float("bagging_fraction", 0.4, 1.0),
                  lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
                  lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 20.0, log=True),
                  min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.3),
                  max_depth=trial.suggest_int("max_depth", 3, 14))
    score = cv_auc(params)
    with open(LOG, "a") as f:
        f.write(json.dumps({"trial": trial.number, "auc": round(score, 5),
                            **{k: params[k] for k in params if k not in PARAMS_STATIC}}) + "\n")
    return score


if not LOG.exists():
    LOG.write_text("")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
# sembrar baseline y v3 como puntos de partida
study.enqueue_trial({"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
                     "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l1": 0.001,
                     "lambda_l2": 1.0, "min_split_gain": 0.0, "max_depth": 8})
study.enqueue_trial({"learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 100,
                     "feature_fraction": 0.7, "bagging_fraction": 0.8, "lambda_l1": 0.1,
                     "lambda_l2": 5.0, "min_split_gain": 0.01, "max_depth": 10})
study.optimize(objective, n_trials=60, show_progress_bar=False)

best = {"best_auc": round(study.best_value, 5), "best_params": study.best_params,
        "baseline": 0.75809, "delta": round(study.best_value - 0.75809, 5)}
(OUT / "best_params.json").write_text(json.dumps(best, indent=2))
print(json.dumps(best, indent=2))
