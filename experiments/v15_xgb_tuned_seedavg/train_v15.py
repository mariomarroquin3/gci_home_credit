"""Exp v15: XGB-tuneado (v14) seed-avg 5 seeds + blend con lgbm/cb seed-avg de v12.
Ref: blend v12 = 0.76460. XGB-tuned single-seed (v14) = 0.76297."""
import itertools
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v15_xgb_tuned_seedavg"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
XP = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", nthread=-1,
          **json.loads(Path("experiments/v14_optuna_xgb/best_params.json").read_text()))

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
X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True).reindex(columns=X.columns, fill_value=0)

oof = np.zeros((len(SEEDS), len(train)))
tpred = np.zeros((len(SEEDS), len(test)))
for si, seed in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        p = dict(XP, seed=seed)
        m = xgb.train(p, xgb.DMatrix(X.iloc[tr], y[tr]), num_boost_round=20000,
                      evals=[(xgb.DMatrix(X.iloc[va], y[va]), "va")], early_stopping_rounds=300, verbose_eval=False)
        oof[si, va] = m.predict(xgb.DMatrix(X.iloc[va]), iteration_range=(0, m.best_iteration + 1))
        tpred[si] += m.predict(xgb.DMatrix(X_test), iteration_range=(0, m.best_iteration + 1)) / 5
    print(f"seed {seed}: {roc_auc_score(y, oof[si]):.5f}", flush=True)

oof_x = oof.mean(axis=0); test_x = tpred.mean(axis=0)
np.save(OUT / "oof_xgb_tuned_seedavg.npy", oof_x)
np.save(OUT / "test_xgb_tuned_seedavg.npy", test_x)
print(f"xgb-tuned seed-avg OOF: {roc_auc_score(y, oof_x):.5f}")

names = ["lgbm", "cb", "xgb"]
O = {"lgbm": np.load("experiments/v12_seedavg/oof_lgbm_seedavg.npy"),
     "cb": np.load("experiments/v12_seedavg/oof_cb_seedavg.npy"), "xgb": oof_x}
T = {"lgbm": np.load("experiments/v12_seedavg/test_lgbm_seedavg.npy"),
     "cb": np.load("experiments/v12_seedavg/test_cb_seedavg.npy"), "xgb": test_x}

def score(w):
    return roc_auc_score(y, sum(wi * O[k] for wi, k in zip(w, names)))
best = (0, None)
for w in itertools.product(range(0, 21), repeat=3):
    if sum(w) != 20:
        continue
    w = tuple(x / 20 for x in w)
    s = score(w)
    if s > best[0]:
        best = (s, w)
w = list(best[1]); bs = best[0]
imp = True
while imp:
    imp = False
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for d in (0.01, -0.01):
                w2 = list(w); w2[i] += d; w2[j] -= d
                if min(w2) < 0:
                    continue
                s = score(w2)
                if s > bs + 1e-7:
                    w, bs = w2, s; imp = True
print(f"blend v15: {bs:.5f} {dict(zip(names, [round(x,3) for x in w]))} (vs v12: {bs-0.76460:+.5f})")

pred = sum(wi * T[k] for wi, k in zip(w, names))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v15.csv", index=False)
(OUT / "results.json").write_text(json.dumps({
    "xgb_tuned_seedavg": round(roc_auc_score(y, oof_x), 5),
    "blend_cv": round(bs, 5), "weights": dict(zip(names, [round(x, 4) for x in w])),
    "delta_vs_v12": round(bs - 0.76460, 5)}, indent=1))
print("OK")
