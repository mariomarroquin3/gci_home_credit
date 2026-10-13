"""Exp v27: cuantificar fuga de pseudo-labels (revision PR #14).
Variante A (global, protocolo v19/v20): labeler = predicciones test de los 5 folds completos.
Variante B (anidado honesto): labeler de cada fold entrenado SOLO en ese fold-train.
Proxy LGBM-tuned seed 42. Refs: v6 0.76151 | v19 (A) 0.76215 | re-check 0.76106.
Si A >> B: la fuga es real y hay que cambiar la produccion."""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
OUT = Path("files/exp_v27"); OUT.mkdir(parents=True, exist_ok=True)

LGBM = dict(json.loads((Path("files/exp_v4/best_params.json")).read_text())["best_params"],
            objective="binary", metric="auc", seed=42, deterministic=True,
            force_row_wise=True, verbose=-1)

train = pd.read_csv("files/data/train.csv")
test = pd.read_csv("files/data/test.csv")
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

full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category"); X_test[c] = X_test[c].astype("category")

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))

def fit_lgbm(Xtr, ytr, Xva, yva):
    return lgb.train(LGBM, lgb.Dataset(Xtr, ytr), num_boost_round=20000,
                     valid_sets=[lgb.Dataset(Xva, yva)],
                     callbacks=[lgb.early_stopping(300, verbose=False)])

# ---------- baseline v6 (sin PL) ----------
oof_base = np.zeros(len(X))
test_pred_folds = np.zeros((5, len(X_test)))
best_iters = []
for fi, (tr, va) in enumerate(folds):
    m = fit_lgbm(X.iloc[tr], y[tr], X.iloc[va], y[va])
    oof_base[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    test_pred_folds[fi] = m.predict(X_test, num_iteration=m.best_iteration)
    best_iters.append(m.best_iteration)
auc_base = roc_auc_score(y, oof_base)
print(f"baseline: {auc_base:.5f}", flush=True)

# ---------- A: PL global (v19/v20) ----------
p_test_global = test_pred_folds.mean(axis=0)
mask_g = p_test_global < 0.03
Xp_g = X_test[mask_g]; yp_g = np.zeros(int(mask_g.sum()), dtype=int)
oof_A = np.zeros(len(X))
for fi, (tr, va) in enumerate(folds):
    Xtr = pd.concat([X.iloc[tr], Xp_g], ignore_index=True)
    ytr = np.r_[y[tr], yp_g]
    for c in Xtr.select_dtypes(include=["object"]).columns:
        Xtr[c] = Xtr[c].astype("category")
    m = fit_lgbm(Xtr, ytr, X.iloc[va], y[va])
    oof_A[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
auc_A = roc_auc_score(y, oof_A)
print(f"A global PL ({int(mask_g.sum())} filas): {auc_A:.5f} ({auc_A-auc_base:+.5f})", flush=True)

# ---------- B: PL anidado por fold (honesto) ----------
oof_B = np.zeros(len(X))
ns = []
for fi, (tr, va) in enumerate(folds):
    # labeler entrenado SOLO en tr (inner split 90/10 para early stopping)
    rng = np.random.RandomState(42)
    inner = rng.permutation(len(tr))
    cut = int(0.9 * len(tr))
    itr, iva = tr[inner[:cut]], tr[inner[cut:]]
    lab = fit_lgbm(X.iloc[itr], y[itr], X.iloc[iva], y[iva])
    p_te = lab.predict(X_test, num_iteration=lab.best_iteration)
    mask = p_te < 0.03
    ns.append(int(mask.sum()))
    Xtr = pd.concat([X.iloc[tr], X_test[mask]], ignore_index=True)
    ytr = np.r_[y[tr], np.zeros(int(mask.sum()), dtype=int)]
    for c in Xtr.select_dtypes(include=["object"]).columns:
        Xtr[c] = Xtr[c].astype("category")
    m = fit_lgbm(Xtr, ytr, X.iloc[va], y[va])
    oof_B[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    print(f"  fold {fi}: pseudo {ns[-1]} | parcial B {roc_auc_score(y[va], oof_B[va]):.5f}", flush=True)
auc_B = roc_auc_score(y, oof_B)
print(f"B nested PL ({ns}): {auc_B:.5f} ({auc_B-auc_base:+.5f})", flush=True)
print(f"FUGA estimada A-B: {auc_A-auc_B:+.5f}", flush=True)

(OUT / "results.json").write_text(json.dumps({
    "baseline": round(auc_base, 5), "A_global_pl": round(auc_A, 5), "B_nested_pl": round(auc_B, 5),
    "delta_A": round(auc_A - auc_base, 5), "delta_B": round(auc_B - auc_base, 5),
    "leak_A_minus_B": round(auc_A - auc_B, 5), "nested_counts": ns}, indent=1))
np.save(OUT / "oof_base.npy", oof_base); np.save(OUT / "oof_A.npy", oof_A); np.save(OUT / "oof_B.npy", oof_B)
print("DONE_V27", flush=True)
