"""Exp v31: features no supervisadas (KMeans) — linea "clustering" (pregunta de Mario 2026-10-21).
Legal: cluster_id es feature automatica y reproducible (NO hand-labeling). Fit SOLO en train.
KMeans k=8 y k=16 sobre numericas estandarizadas (NaN->mediana solo para el clustering).
Features: km8_id (cat), km16_id (cat), km8_dist (num). LGBM-tuned, 5 seeds x 5 folds.
Pareado vs baseline v28-off (misma build+params+particiones): [0.76106, 0.76100, 0.76032, 0.76120, 0.76071].
Prior: negativo (familia vecinos 0/3: KNN-500; TE +0.00015; senal concentrada en EXT_SOURCE)."""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = Path("files/exp_v31"); OUT.mkdir(parents=True, exist_ok=True)

LGBM = dict(json.loads((Path("files/exp_v4/best_params.json")).read_text())["best_params"],
            objective="binary", metric="auc", deterministic=True,
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

# --- clustering: SOLO numericas, fit en train, transform test ---
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
med = X[num_cols].median()
sc = StandardScaler().fit(X[num_cols].fillna(med))
Ztr = sc.transform(X[num_cols].fillna(med))
Zte = sc.transform(X_test[num_cols].fillna(med))
for k in (8, 16):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Ztr)
    X[f"km{k}_id"] = km.predict(Ztr)
    X_test[f"km{k}_id"] = km.predict(Zte)
    if k == 8:
        X["km8_dist"] = np.linalg.norm(Ztr - km.cluster_centers_[X["km8_id"]], axis=1)
        X_test["km8_dist"] = np.linalg.norm(Zte - km.cluster_centers_[X_test["km8_id"]], axis=1)
X["km8_id"] = X["km8_id"].astype("category"); X["km16_id"] = X["km16_id"].astype("category")
X_test["km8_id"] = X_test["km8_id"].astype("category"); X_test["km16_id"] = X_test["km16_id"].astype("category")
for c in X.select_dtypes(include=["object"]).columns:
    X[c] = X[c].astype("category"); X_test[c] = X_test[c].astype("category")
print(f"features: {X.shape[1]} (+3 cluster)", flush=True)

BASE = [0.76106, 0.761, 0.76032, 0.7612, 0.76071]  # v28 off, seeds 42-46
aucs = []
for seed in [42, 43, 44, 45, 46]:
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        m = lgb.train(dict(LGBM, seed=seed), lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=20000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    aucs.append(roc_auc_score(y, oof))
    print(f"seed {seed}: {aucs[-1]:.5f} (base {BASE[len(aucs)-1]:.5f}, {aucs[-1]-BASE[len(aucs)-1]:+.5f})", flush=True)

d = np.array(aucs) - np.array(BASE)
t, p = stats.ttest_rel(aucs, BASE)
print(f"media {np.mean(aucs):.5f} vs base {np.mean(BASE):.5f} -> {d.mean():+.5f} (sd {d.std():.5f}, t={t:.2f}, p={p:.4f})", flush=True)
(OUT / "results.json").write_text(json.dumps({
    "aucs_cluster": [round(a, 5) for a in aucs], "aucs_base_v28off": BASE,
    "delta_mean": round(float(d.mean()), 5), "delta_sd": round(float(d.std()), 5),
    "t": round(float(t), 2), "p": round(float(p), 4)}, indent=1))
print("DONE_V31", flush=True)
