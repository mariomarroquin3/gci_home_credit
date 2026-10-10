"""Exp v5: CatBoost y XGBoost sobre features v1 (branch exp/diversificadores-v5).
Misma CV (5-fold, seed 42). CatBoost con categóricas nativas; XGBoost con one-hot."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v5_diversificadores")
OUT.mkdir(parents=True, exist_ok=True)


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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}

# ---------- CatBoost ----------
from catboost import CatBoostClassifier, Pool
oof_cb = np.zeros(len(train)); pred_cb = np.zeros(len(test))
Xc, Xtc = X.copy(), X_test.copy()
for c in cat_cols:
    Xc[c] = Xc[c].astype("str"); Xtc[c] = Xtc[c].astype("str")
cat_idx = [Xc.columns.get_loc(c) for c in cat_cols]
for fold, (tr, va) in enumerate(skf.split(Xc, y)):
    m = CatBoostClassifier(iterations=8000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                           random_strength=1.0, bagging_temperature=0.5, od_wait=300,
                           od_type="Iter", eval_metric="AUC", random_seed=SEED, verbose=False)
    m.fit(Xc.iloc[tr], y[tr], cat_features=cat_idx,
          eval_set=(Xc.iloc[va], y[va]), use_best_model=True)
    oof_cb[va] = m.predict_proba(Xc.iloc[va])[:, 1]
    pred_cb += m.predict_proba(Xtc)[:, 1] / 5
    print(f"CB fold {fold}: {roc_auc_score(y[va], oof_cb[va]):.5f} (iter {m.best_iteration_})")
cv_cb = roc_auc_score(y, oof_cb)
results["catboost"] = {"cv_auc": round(cv_cb, 5)}
print(f"CatBoost CV: {cv_cb:.5f}")

# ---------- XGBoost ----------
import xgboost as xgb
Xx = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
Xtx = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True)
Xtx = Xtx.reindex(columns=Xx.columns, fill_value=0)
oof_x = np.zeros(len(train)); pred_x = np.zeros(len(test))
xparams = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
               max_depth=5, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=5.0, seed=SEED, nthread=-1, tree_method="hist")
for fold, (tr, va) in enumerate(skf.split(Xx, y)):
    dtr = xgb.DMatrix(Xx.iloc[tr], y[tr]); dva = xgb.DMatrix(Xx.iloc[va], y[va])
    m = xgb.train(xparams, dtr, num_boost_round=8000, evals=[(dva, "v")],
                  early_stopping_rounds=200, verbose_eval=False)
    oof_x[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
    pred_x += m.predict(xgb.DMatrix(Xtx), iteration_range=(0, m.best_iteration + 1)) / 5
    print(f"XGB fold {fold}: {roc_auc_score(y[va], oof_x[va]):.5f} (iter {m.best_iteration})")
cv_x = roc_auc_score(y, oof_x)
results["xgboost"] = {"cv_auc": round(cv_x, 5)}
print(f"XGBoost CV: {cv_x:.5f}")

np.save(OUT / "oof_catboost.npy", oof_cb); np.save(OUT / "oof_xgboost.npy", oof_x)
np.save(OUT / "test_catboost.npy", pred_cb); np.save(OUT / "test_xgboost.npy", pred_x)
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
