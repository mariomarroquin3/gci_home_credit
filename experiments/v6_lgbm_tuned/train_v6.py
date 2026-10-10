"""Exp v6: LightGBM con params Optuna (best_params.json) + features v1.
5-fold seed 42. Guarda oof_lgbm_tuned.npy, test_lgbm_tuned.npy, submission_v6.csv."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v6_lgbm_tuned")
OUT.mkdir(parents=True, exist_ok=True)

best = json.loads(Path("experiments/v4_optuna/best_params.json").read_text())
PARAMS = dict(objective="binary", metric="auc", seed=SEED, deterministic=True,
              force_row_wise=True, verbose=-1, bagging_freq=1, **best["best_params"])


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
    for c in df.select_dtypes(include=["object", "str"]).columns:
        df[c] = df[c].astype("category")
    return df


train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values
full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
cat_feats = [c for c in X.columns if str(X[c].dtype) == "category"]

oof = np.zeros(len(train)); pred = np.zeros(len(test))
imps = pd.DataFrame({"feature": X.columns})
for fold, (tr, va) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_feats)
    dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_feats, reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=8000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    pred += m.predict(X_test, num_iteration=m.best_iteration) / 5
    imps[f"gain_{fold}"] = m.feature_importance("gain")
    print(f"fold {fold}: AUC {roc_auc_score(y[va], oof[va]):.5f} (iter {m.best_iteration})")

cv = roc_auc_score(y, oof)
print(f"CV AUC: {cv:.5f}")
np.save(OUT / "oof_lgbm_tuned.npy", oof)
np.save(OUT / "test_lgbm_tuned.npy", pred)
imps["gain_mean"] = imps[[c for c in imps.columns if c.startswith("gain_")]].mean(axis=1)
imps.sort_values("gain_mean", ascending=False).to_csv(OUT / "importance.csv", index=False)
sub = pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"].values, "TARGET": pred})
sub.to_csv(OUT / "submission_v6.csv", index=False)
(OUT / "results.json").write_text(json.dumps({"cv_auc": round(cv, 5), "params": best["best_params"]}, indent=2))
print("Submission:", OUT / "submission_v6.csv")
