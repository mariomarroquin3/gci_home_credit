"""Exp v21: pseudo-labeling round-2 (etiquetador = blend v20) + comparacion de umbrales.
LGBM-tuned single-seed. Ref: v19 round-1 (p<0.03, etiquetador v12) = 0.76215.
Si round-2 mejora -> v22 full pipeline round-2. Si no, linea cerrada."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v21_pseudo_round2"); OUT.mkdir(parents=True, exist_ok=True)
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
cat_cols = Xtr.select_dtypes(include=["object", "str"]).columns.tolist()
for c in cat_cols:
    Xtr[c] = Xtr[c].astype("category"); Xte[c] = Xte[c].astype("category")

p_v20 = pd.read_csv("experiments/v20_pseudo_pipeline/submission_v20.csv")["TARGET"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
res = {}
for tag, thr in (("r2_t03", 0.03), ("r2_t05", 0.05), ("r2_t02", 0.02)):
    mask = p_v20 < thr
    Xp = Xte[mask].copy(); yp = np.zeros(int(mask.sum()), dtype=int)
    oof = np.zeros(len(train))
    for tr, va in skf.split(Xtr, y):
        Xa = pd.concat([Xtr.iloc[tr], Xp], ignore_index=True)
        for c in cat_cols: Xa[c] = Xa[c].astype("category")
        ya = np.r_[y[tr], yp]
        p = dict(objective="binary", metric="auc", verbosity=-1, seed=SEED, deterministic=True,
                 force_row_wise=True, bagging_freq=1, **LGBM_P)
        m = lgb.train(p, lgb.Dataset(Xa, ya), num_boost_round=20000,
                      valid_sets=[lgb.Dataset(Xtr.iloc[va], y[va])], callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = m.predict(Xtr.iloc[va], num_iteration=m.best_iteration)
    auc = roc_auc_score(y, oof)
    res[tag] = {"thr": thr, "rows": int(mask.sum()), "cv": round(auc, 5), "delta_vs_v19": round(auc - 0.76215, 5)}
    print(f"{tag}: thr={thr} rows={int(mask.sum())} CV {auc:.5f} (vs v19 r1: {auc-0.76215:+.5f})", flush=True)
    np.save(OUT / f"oof_{tag}.npy", oof)
(OUT / "results.json").write_text(json.dumps(res, indent=1))
print("OK")
