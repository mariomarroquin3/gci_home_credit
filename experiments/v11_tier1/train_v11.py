"""Exp v11: features Tier-1 de soluciones ganadoras Home Credit 2018 sobre LGBM tuneado.
- KNN500_TC: media del TARGET de los 500 vecinos mas cercanos (EXT1/2/3 + CREDIT + ANNUITY), OOF.
- IMPL_RATE / IMPL_DUR: tasa de interes implicita (anual %) por duraciones candidatas (5to lugar 2018).
- EXT3_DIV: AMT_CREDIT, AMT_ANNUITY, AMT_INCOME_TOTAL divididos por EXT_SOURCE_3.
- AGE_INT + REGION_RATING como categorica.
Referencia: v6 LGBM tuneado CV 0.76151. Umbral adopcion features: +0.003."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

SEED = 42
OUT = Path("experiments/v11_tier1"); OUT.mkdir(parents=True, exist_ok=True)
PARAMS = json.loads(Path("experiments/v4_optuna/best_params.json").read_text())["best_params"]
KNN_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_ANNUITY"]


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
    # Tier-1: divisiones por EXT_SOURCE_3
    df["CREDIT_EXT3"] = df["AMT_CREDIT"] / df["EXT_SOURCE_3"]
    df["ANNUITY_EXT3"] = df["AMT_ANNUITY"] / df["EXT_SOURCE_3"]
    df["INCOME_EXT3"] = df["AMT_INCOME_TOTAL"] / df["EXT_SOURCE_3"]
    df["GOODS_EXT3"] = df["AMT_GOODS_PRICE"] / df["EXT_SOURCE_3"]
    # AGE_INT y region como categorica
    df["AGE_INT"] = np.floor(df["AGE_YEARS"]).astype("Int64")
    df["REGION_RATING_CAT"] = df["REGION_RATING_CLIENT"].astype("str")
    return df


def implied_rate(credit, annuity, n):
    """Tasa mensual implicita resolviendo annuity = credit * r / (1-(1+r)^-n) por Newton vectorizado."""
    a = np.maximum(annuity, 1.0); c = np.maximum(credit, 1.0)
    r = np.full(len(c), 0.01)
    for _ in range(60):
        f = c * r / (1 - (1 + r) ** -n) - a
        fp = c * ((1 - (1 + r) ** -n) - r * n * (1 + r) ** (-n - 1)) / (1 - (1 + r) ** -n) ** 2
        r = np.clip(r - f / np.where(np.abs(fp) < 1e-12, 1e-12, fp), 1e-6, 0.5)
    return r


def add_implied(df):
    c = df["AMT_CREDIT"].values; a = df["AMT_ANNUITY"].values
    durs = [6, 12, 18, 24, 30, 36, 48, 60, 84, 120]
    best_err = np.full(len(df), np.inf); best_rate = np.full(len(df), np.nan); best_n = np.full(len(df), np.nan)
    for n in durs:
        r = implied_rate(c, a, n)
        est = c * r / (1 - (1 + r) ** -n)
        err = np.abs(est - a) / a
        m = err < best_err
        best_err[m] = err[m]; best_rate[m] = r[m]; best_n[m] = n
    df["IMPL_RATE"] = best_rate * 12 * 100   # tasa anual %
    df["IMPL_DUR"] = best_n
    df["IMPL_ERR"] = best_err
    return df


train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values
full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
full = add_implied(full)
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
id_test = test["SK_ID_CURR"].values
for c in X.select_dtypes(include=["object", "str"]).columns:
    X[c] = X[c].astype("category"); X_test[c] = X_test[c].astype("category")

# ---------- KNN-500 target-mean (OOF) ----------
print("computando KNN500...", flush=True)
Zk = full[KNN_COLS].copy()
for c in KNN_COLS:
    Zk[c] = Zk[c].fillna(Zk[c].median())
Z = StandardScaler().fit_transform(Zk.values)
Ztr, Zte = Z[:len(train)], Z[len(train):]
knn500_tr = np.zeros(len(train)); knn500_te = np.zeros(len(test))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (tr, va) in enumerate(skf.split(Ztr, y)):
    nn = NearestNeighbors(n_neighbors=501, algorithm="kd_tree", n_jobs=-1).fit(Ztr[tr])
    _, idx = nn.kneighbors(Ztr[va])
    knn500_tr[va] = y[tr][idx].mean(axis=1)
    print(f"  knn fold {fold} ok (AUC parcial {roc_auc_score(y[va], knn500_tr[va]):.5f})", flush=True)
nn_full = NearestNeighbors(n_neighbors=500, algorithm="kd_tree", n_jobs=-1).fit(Ztr)
_, idx_te = nn_full.kneighbors(Zte)
knn500_te = y[idx_te].mean(axis=1)
print(f"KNN500 AUC standalone (train OOF): {roc_auc_score(y, knn500_tr):.5f}", flush=True)
X["KNN500_TC"] = knn500_tr; X_test["KNN500_TC"] = knn500_te

# ---------- LGBM ----------
params = dict(objective="binary", metric="auc", verbosity=-1, seed=SEED, deterministic=True,
              force_row_wise=True, bagging_freq=1, **PARAMS)
oof = np.zeros(len(train)); pred = np.zeros(len(test)); folds = []
for fold, (tr, va) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr]); dva = lgb.Dataset(X.iloc[va], y[va])
    m = lgb.train(params, dtr, num_boost_round=20000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    pred += m.predict(X_test, num_iteration=m.best_iteration) / 5
    fa = roc_auc_score(y[va], oof[va]); folds.append(round(fa, 5))
    print(f"fold {fold}: {fa:.5f} (iter {m.best_iteration})", flush=True)

cv = roc_auc_score(y, oof)
print(f"v11 CV: {cv:.5f} (delta vs v6: {cv - 0.76151:+.5f})")
np.save(OUT / "oof.npy", oof); np.save(OUT / "test_pred.npy", pred)
imp = pd.DataFrame({"feature": X.columns, "gain": m.feature_importance("gain")}).sort_values("gain", ascending=False)
imp.to_csv(OUT / "importance.csv", index=False)
(OUT / "results.json").write_text(json.dumps({"cv_auc": round(cv, 5), "delta_vs_v6": round(cv - 0.76151, 5),
    "folds": folds, "knn500_auc": round(roc_auc_score(y, knn500_tr), 5),
    "top10": imp.head(10).to_dict("records")}, indent=1))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v11.csv", index=False)
print("OK guardado")
