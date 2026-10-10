"""Exp v9-final: CatBoost con params Optuna trial 12 (CV esperado 0.76243).
Features v1 canonicas. Guarda oof_cb_tuned.npy + test_cb_tuned.npy para blend v10."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v9_optuna_cb")
OUT.mkdir(parents=True, exist_ok=True)
BEST = json.loads((OUT / "best_params.json").read_text())


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
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values
full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
id_test = test["SK_ID_CURR"].values
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

from catboost import CatBoostClassifier
Xc, Xtc = X.copy(), X_test.copy()
for c in cat_cols:
    Xc[c] = Xc[c].astype("str"); Xtc[c] = Xtc[c].astype("str")
cat_idx = [Xc.columns.get_loc(c) for c in cat_cols]

params = dict(iterations=8000, od_wait=BEST["od_wait"], od_type=BEST["od_type"],
              eval_metric="AUC", random_seed=SEED, verbose=False,
              learning_rate=BEST["learning_rate"], depth=BEST["depth"],
              l2_leaf_reg=BEST["l2_leaf_reg"], random_strength=BEST["random_strength"],
              bagging_temperature=BEST["bagging_temperature"],
              min_data_in_leaf=BEST["min_data_in_leaf"])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train)); pred = np.zeros(len(test))
folds = []
for fold, (tr, va) in enumerate(skf.split(Xc, y)):
    m = CatBoostClassifier(**params)
    m.fit(Xc.iloc[tr], y[tr], cat_features=cat_idx,
          eval_set=(Xc.iloc[va], y[va]), use_best_model=True)
    oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
    pred += m.predict_proba(Xtc)[:, 1] / 5
    fa = roc_auc_score(y[va], oof[va]); folds.append(round(fa, 5))
    print(f"fold {fold}: {fa:.5f} (iter {m.best_iteration_})", flush=True)

cv = roc_auc_score(y, oof)
print(f"CatBoost-tuned CV: {cv:.5f}")
np.save(OUT / "oof_cb_tuned.npy", oof)
np.save(OUT / "test_cb_tuned.npy", pred)
(OUT / "results.json").write_text(json.dumps({
    "cv_auc": round(cv, 5), "folds": folds, "params": {k: v for k, v in params.items() if k != "verbose"},
    "optuna_trial": 12, "note": "Optuna 26/45 trials (sandbox restart), meseta evidente"}, indent=1))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v9_cb.csv", index=False)
print("OK guardado")
