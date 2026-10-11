"""Exp v19: pseudo-labeling. Test rows con p>0.97 (label 1) o p<0.03 (label 0) del blend v12
se agregan como entrenamiento extra en cada fold; OOF se mantiene SOLO sobre train -> honesto.
Ref v6 LGBM-tuned: 0.76151. Si suma, se extiende a CB/XGB."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v19_pseudo_label"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
LGBM_P = json.loads(Path("experiments/v4_optuna/best_params.json").read_text())["best_params"]

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
Xtr = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
Xte = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
id_test = test["SK_ID_CURR"].values
cat_cols = Xtr.select_dtypes(include=["object", "str"]).columns.tolist()
for c in cat_cols:
    Xtr[c] = Xtr[c].astype("category"); Xte[c] = Xte[c].astype("category")

p_blend = pd.read_csv("experiments/v12_seedavg/submission_v12.csv")["TARGET"].values
hi, lo = 0.97, 0.03
mask = (p_blend > hi) | (p_blend < lo)
Xp = Xte[mask]; yp = (p_blend[mask] > hi).astype(int)
print(f"pseudo rows: {mask.sum()} ({int((yp==1).sum())} pos / {int((yp==0).sum())} neg)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train)); tpred = np.zeros(len(test))
for tr, va in skf.split(Xtr, y):
    X_aug = pd.concat([Xtr.iloc[tr], Xp], ignore_index=True)
    for c in cat_cols:
        X_aug[c] = X_aug[c].astype("category")
    y_aug = np.r_[y[tr], yp]
    p = dict(objective="binary", metric="auc", verbosity=-1, seed=SEED, deterministic=True,
             force_row_wise=True, bagging_freq=1, **LGBM_P)
    m = lgb.train(p, lgb.Dataset(X_aug, y_aug), num_boost_round=20000,
                  valid_sets=[lgb.Dataset(Xtr.iloc[va], y[va])], callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[va] = m.predict(Xtr.iloc[va], num_iteration=m.best_iteration)
    tpred += m.predict(Xte, num_iteration=m.best_iteration) / 5
auc = roc_auc_score(y, oof)
print(f"v19 pseudo-label: CV {auc:.5f} | vs v6: {auc-0.76151:+.5f}")
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": tpred}).to_csv(OUT / "submission_v19.csv", index=False)
np.save(OUT / "oof_v19.npy", oof)
(OUT / "results.json").write_text(json.dumps({"pseudo_rows": int(mask.sum()), "cv_auc": round(auc, 5),
                                              "delta_vs_v6": round(auc - 0.76151, 5)}, indent=1))
print("OK")
