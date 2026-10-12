"""Exp v22: TabPFN v2 como diversificador (ultima familia sin probar).
5-fold SKF seed 42, subsample 10k por fit (limite TabPFN). Ref blend v20: 0.76488."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
warnings.filterwarnings("ignore")

OUT = Path("files/exp_v22"); OUT.mkdir(parents=True, exist_ok=True)

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
id_test = test["SK_ID_CURR"].values
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
for c in cat_cols:  # ordinal: tabpfn 2.0.9 incompatible con string dtypes de pandas 2.3
    both = pd.concat([X[c], X_test[c]]).astype("object").where(lambda s: s.notna(), "MISSING").astype(str)
    codes = pd.Categorical(both).codes
    X[c] = codes[:len(X)]; X_test[c] = codes[len(X):]

from tabpfn import TabPFNClassifier
rng = np.random.RandomState(42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(train)); tpred = np.zeros(len(test))
for fi, (tr, va) in enumerate(skf.split(X, y)):
    sub = rng.choice(tr, size=min(10000, len(tr)), replace=False)
    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    clf.fit(X.iloc[sub], y[sub])
    oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
    tpred += clf.predict_proba(X_test)[:, 1] / 5
    print(f"fold {fi}: OOF parcial {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
auc = roc_auc_score(y, oof)
np.save(OUT / "oof_tabpfn.npy", oof); np.save(OUT / "test_tabpfn.npy", tpred)
print(f"TabPFN OOF: {auc:.5f}")

names = ["lgbm", "cb", "xgb", "tabpfn"]
O = {"lgbm": np.load("files/exp_v20/oof_lgbm_pl_seedavg.npy"), "cb": np.load("files/exp_v20/oof_cb_pl_seedavg.npy"),
     "xgb": np.load("files/exp_v20/oof_xgb_pl_seedavg.npy"), "tabpfn": oof}
T = {"lgbm": np.load("files/exp_v20/test_lgbm_pl_seedavg.npy"), "cb": np.load("files/exp_v20/test_cb_pl_seedavg.npy"),
     "xgb": np.load("files/exp_v20/test_xgb_pl_seedavg.npy"), "tabpfn": tpred}
import itertools
def score(w):
    return roc_auc_score(y, sum(wi * O[k] for wi, k in zip(w, names)))
best = (0, None)
for w in itertools.product(range(0, 21), repeat=4):
    if sum(w) != 20: continue
    w = tuple(x / 20 for x in w); s = score(w)
    if s > best[0]: best = (s, w)
w = list(best[1]); bs = best[0]
imp = True
while imp:
    imp = False
    for i in range(4):
        for j in range(4):
            if i == j: continue
            for d in (0.01, -0.01):
                w2 = list(w); w2[i] += d; w2[j] -= d
                if min(w2) < 0: continue
                s = score(w2)
                if s > bs + 1e-7: w, bs = w2, s; imp = True
print(f"blend 4mod: {bs:.5f} {dict(zip(names, [round(x,3) for x in w]))} (vs v20: {bs-0.76488:+.5f})")
pred = sum(wi * T[k] for wi, k in zip(w, names))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v22.csv", index=False)
(OUT / "results.json").write_text(json.dumps({"tabpfn_oof": round(auc, 5), "blend_cv": round(bs, 5),
    "weights": dict(zip(names, [round(x, 4) for x in w])), "delta_vs_v20": round(bs - 0.76488, 5)}, indent=1))
print("OK")
