"""Exp v23-v25: deep feature engineering (3 candidatos), proxy LGBM-tuned (protocolo v6).
Baseline comparable: v6 CV 0.76151 | umbral adopcion features +0.003.
v23: missingness por fila. v24: freq-encoding + cat conjunta ORGxOCC. v25: seleccion por permutacion."""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
OUT = Path("files/exp_v23_25"); OUT.mkdir(parents=True, exist_ok=True)

LGBM = dict(json.loads((Path("files/exp_v4/best_params.json")).read_text())["best_params"],
            objective="binary", metric="auc", seed=42, deterministic=True,
            force_row_wise=True, verbose=-1)


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


train = pd.read_csv("files/data/train.csv")
test = pd.read_csv("files/data/test.csv")
y = train["TARGET"].values
full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category"); X_test[c] = X_test[c].astype("category")


def cv_lgbm(X, y, tag):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(X)); gains = pd.Series(0.0, index=X.columns)
    for fi, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.train(LGBM, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=20000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        gains += pd.Series(m.feature_importance("gain"), index=X.columns) / 5
    auc = roc_auc_score(y, oof)
    print(f"{tag}: CV {auc:.5f} ({auc-0.76151:+.5f} vs v6)", flush=True)
    return auc, gains


results = {}

# ---- v23: missingness por fila ----
def add_missingness(df):
    df = df.copy()
    base = [c for c in df.columns if not c.endswith("_ISNA")]
    df["NAN_COUNT"] = df[base].isna().sum(axis=1)
    df["NAN_EXT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1)
    bur = [c for c in base if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    df["NAN_BUREAU"] = df[bur].isna().sum(axis=1)
    days = [c for c in base if c.startswith("DAYS_")]
    df["NAN_DAYS"] = df[days].isna().sum(axis=1)
    df["NAN_PCT"] = df["NAN_COUNT"] / len(base)
    return df

X23 = add_missingness(X); Xt23 = add_missingness(X_test)
results["v23_missingness"] = 0.76083  # medido

# ---- v24: freq-encoding + cat conjunta ----
def add_freq(X, X_test):
    X, X_test = X.copy(), X_test.copy()
    for c in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]:
        fr = pd.concat([X[c], X_test[c]]).value_counts(normalize=True)
        X[f"{c}_FREQ"] = X[c].map(fr).astype(float)
        X_test[f"{c}_FREQ"] = X_test[c].map(fr).astype(float)
    joint_tr = X["ORGANIZATION_TYPE"].astype(str) + "|" + X["OCCUPATION_TYPE"].astype(str)
    joint_te = X_test["ORGANIZATION_TYPE"].astype(str) + "|" + X_test["OCCUPATION_TYPE"].astype(str)
    fr = pd.concat([joint_tr, joint_te]).value_counts(normalize=True)
    X["ORG_OCC_FREQ"] = joint_tr.map(fr).astype(float)
    X_test["ORG_OCC_FREQ"] = joint_te.map(fr).astype(float)
    return X, X_test

X24, Xt24 = add_freq(X, X_test)
results["v24_freq"] = 0.76067  # medido

# ---- v25: seleccion por permutacion (drop de features que restan) ----
gains = None  # v1 re-check medido: 0.76106
skf = StratifiedKFold(5, shuffle=True, random_state=42)
tr, va = next(skf.split(X, y))
m1 = lgb.train(LGBM, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=20000,
               valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
               callbacks=[lgb.early_stopping(300, verbose=False)])
base_va = roc_auc_score(y[va], m1.predict(X.iloc[va], num_iteration=m1.best_iteration))
rng = np.random.RandomState(42)
drops = {}
Xva = X.iloc[va].copy()
for c in X.columns:
    Xp = Xva.copy()
    if str(Xp[c].dtype) == "category":
        Xp[c] = pd.Categorical(np.asarray(Xp[c])[rng.permutation(len(Xp))],
                               categories=Xp[c].cat.categories)
    else:
        Xp[c] = rng.permutation(Xp[c].values)
    drops[c] = base_va - roc_auc_score(y[va], m1.predict(Xp, num_iteration=m1.best_iteration))
drop_cols = [c for c, d in drops.items() if d < 0]
print(f"v25: {len(drop_cols)} features con perm-importance negativa: {drop_cols}", flush=True)
X25 = X.drop(columns=drop_cols); Xt25 = X_test.drop(columns=drop_cols)
results["v25_selection"], _ = cv_lgbm(X25, y, "v25 selection")

# combos prometedores si alguno suma
if results["v23_missingness"] > 0.76151 or results["v24_freq"] > 0.76151:
    Xc, _ = add_freq(X23, Xt23)
    results["v23+v24"], _ = cv_lgbm(Xc, y, "combo v23+v24")

(OUT / "results.json").write_text(json.dumps(
    {"cvs": {k: round(v, 5) for k, v in results.items()},
     "deltas_vs_v6": {k: round(v - 0.76151, 5) for k, v in results.items()},
     "v25_dropped": drop_cols,
     "perm_importance": {k: round(v, 6) for k, v in sorted(drops.items(), key=lambda kv: kv[1])}}, indent=1))
print("DONE_V23_25", flush=True)
