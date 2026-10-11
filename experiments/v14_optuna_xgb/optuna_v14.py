"""Exp v14: Optuna XGBoost (45 trials, 5-fold seed 42, features v1 + one-hot).
Ref: XGB default (v5) 0.76186; v5 en blend."""
import json
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v14_optuna_xgb"); OUT.mkdir(parents=True, exist_ok=True)

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

train = pd.read_csv("data/raw/train.csv"); y = train["TARGET"].values
X = build(train.drop(columns=["TARGET"])).drop(columns=["SK_ID_CURR"])
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

def objective(trial):
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", nthread=-1, seed=42,
             learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
             max_depth=trial.suggest_int("max_depth", 3, 8),
             min_child_weight=trial.suggest_int("min_child_weight", 5, 400, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
             gamma=trial.suggest_float("gamma", 0.0, 1.0))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = xgb.train(p, xgb.DMatrix(X.iloc[tr], y[tr]), num_boost_round=20000,
                      evals=[(xgb.DMatrix(X.iloc[va], y[va]), "va")], early_stopping_rounds=300, verbose_eval=False)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va]), iteration_range=(0, m.best_iteration + 1))
    return roc_auc_score(y, oof)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=45, callbacks=[
    lambda s, t: print(f"trial {t.number} value {t.value:.5f} best {s.best_value:.5f}", flush=True)])
(OUT / "best_params.json").write_text(json.dumps(study.best_params, indent=1))
print("BEST", study.best_value, json.dumps(study.best_params))
