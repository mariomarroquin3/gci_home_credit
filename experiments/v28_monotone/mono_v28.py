"""Exp v28: restricciones monotonas EXT_SOURCE_1/2/3 (revision PR #14).
LGBM-tuned, 3 seeds (42-44) x 5 folds, con y sin monotone -> t pareada.
Adoptar solo si mejora sistematica (media >0 y consistente en las 3 seeds)."""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
OUT = Path("files/exp_v28"); OUT.mkdir(parents=True, exist_ok=True)

BASE = dict(json.loads((Path("files/exp_v4/best_params.json")).read_text())["best_params"],
            objective="binary", metric="auc", deterministic=True,
            force_row_wise=True, verbose=-1)

train = pd.read_csv("files/data/train.csv")
y = train["TARGET"].values

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

X = build(train.drop(columns=["TARGET"])).drop(columns=["SK_ID_CURR"])
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")

MONO = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
aucs = {"off": [0.76106, 0.761, 0.76032], "on": [0.76117, 0.76133, 0.76075]}  # seeds 42-44 medidas
for seed in [45, 46]:
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oofs = {k: np.zeros(len(X)) for k in aucs}
    for tr, va in skf.split(X, y):
        for k, mono in (("off", None), ("on", [-1 if c in MONO else 0 for c in X.columns])):
            p = dict(BASE, seed=seed)
            if mono is not None:
                p.update(monotone_constraints=mono, monotone_constraints_method="advanced")
            m = lgb.train(p, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=20000,
                          valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            oofs[k][va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    for k in aucs:
        aucs[k].append(roc_auc_score(y, oofs[k]))
    print(f"seed {seed}: off {aucs['off'][-1]:.5f} | on {aucs['on'][-1]:.5f} "
          f"({aucs['on'][-1]-aucs['off'][-1]:+.5f})", flush=True)

d = np.array(aucs["on"]) - np.array(aucs["off"])
t, p = stats.ttest_rel(aucs["on"], aucs["off"])
print(f"media delta: {d.mean():+.5f} (sd {d.std():.5f}, t={t:.2f}, p={p:.4f})", flush=True)
(OUT / "results.json").write_text(json.dumps({
    "off": [round(a, 5) for a in aucs["off"]], "on": [round(a, 5) for a in aucs["on"]],
    "delta_mean": round(float(d.mean()), 5), "delta_sd": round(float(d.std()), 5),
    "t": round(float(t), 2), "p": round(float(p), 4)}, indent=1))
print("DONE_V28", flush=True)
