"""Exp v17: TE-OOF con smoothing para ORGANIZATION_TYPE (58) y OCCUPATION_TYPE (18),
sobre LGBM-tuneado. Ref v6: CV 0.76151. Umbral features: +0.003."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v17_te_oof"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
LGBM_P = json.loads(Path("experiments/v4_optuna/best_params.json").read_text())["best_params"]
TE_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
SMOOTH = 20.0

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# TE-OOF con smoothing: (sum + smooth*global)/(count + smooth)
gm = y.mean()
te_oof = {c: np.zeros(len(train)) for c in TE_COLS}
te_test = {c: np.zeros(len(test)) for c in TE_COLS}
for tr, va in skf.split(Xtr, y):
    dtr = Xtr.iloc[tr]
    for c in TE_COLS:
        st = pd.DataFrame({"c": dtr[c], "y": y[tr]}).groupby("c")["y"].agg(["sum", "count"])
        m = (st["sum"] + SMOOTH * gm) / (st["count"] + SMOOTH)
        te_oof[c][va] = Xtr[c].iloc[va].map(m).fillna(gm).values
        te_test[c] += Xte[c].map(m).fillna(gm).values / 5
for c in TE_COLS:
    Xtr[f"{c}_TE"] = te_oof[c]; Xte[f"{c}_TE"] = te_test[c]
    auc_te = roc_auc_score(y, te_oof[c])
    print(f"TE {c} standalone OOF AUC: {auc_te:.5f}")

for c in Xtr.select_dtypes(include=["object", "str"]).columns:
    Xtr[c] = Xtr[c].astype("category"); Xte[c] = Xte[c].astype("category")

oof = np.zeros(len(train)); tpred = np.zeros(len(test))
for tr, va in skf.split(Xtr, y):
    p = dict(objective="binary", metric="auc", verbosity=-1, seed=SEED, deterministic=True,
             force_row_wise=True, bagging_freq=1, **LGBM_P)
    m = lgb.train(p, lgb.Dataset(Xtr.iloc[tr], y[tr]), num_boost_round=20000,
                  valid_sets=[lgb.Dataset(Xtr.iloc[va], y[va])], callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[va] = m.predict(Xtr.iloc[va], num_iteration=m.best_iteration)
    tpred += m.predict(Xte, num_iteration=m.best_iteration) / 5
auc = roc_auc_score(y, oof)
print(f"v17 (v1 + TE-OOF): CV {auc:.5f} | vs v6: {auc-0.76151:+.5f}")
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": tpred}).to_csv(OUT / "submission_v17.csv", index=False)
np.save(OUT / "oof_v17.npy", oof)
(OUT / "results.json").write_text(json.dumps({"cv_auc": round(auc, 5), "delta_vs_v6": round(auc - 0.76151, 5)}, indent=1))
print("OK")
