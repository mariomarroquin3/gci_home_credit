"""Exp v1 (branch exp/ratios-v1): ratios + tiempo + flags de nulos + anomalia DAYS_EMPLOYED.
CV: StratifiedKFold 5-fold, semillas del proyecto. Compara contra baseline 0.7581."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v3_tuning")
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
              min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, lambda_l1=0.1, lambda_l2=5.0, min_split_gain=0.01,
              seed=SEED, deterministic=True, force_row_wise=True, verbose=-1)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # anomalia
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    # flags de nulos (>30%)
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE"]:
        df[f"{c}_ISNA"] = df[c].isna().astype(int)
    # ratios financieros
    df["CREDIT_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["ANNUITY_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["INCOME_PER_FAM"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    # tiempo
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_BIRTH_PCT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    df["PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25
    # buro y contacto
    bur = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    df["BUREAU_REQ_TOTAL"] = df[bur].sum(axis=1, min_count=1)
    df["PHONE_FLAGS_SUM"] = df[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE"]].sum(axis=1)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
    # categoricas -> category dtype (LightGBM nativo)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("category")
    return df


train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values

full = pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True)
full = build(full)
X = full.iloc[: len(train)].reset_index(drop=True)
X_test = full.iloc[len(train):].reset_index(drop=True)
X = X.drop(columns=["SK_ID_CURR"])
id_test = X_test["SK_ID_CURR"].values
X_test = X_test.drop(columns=["SK_ID_CURR"])
cat_feats = [c for c in X.columns if str(X[c].dtype) == "category"]
print(f"Features: {X.shape[1]} ({len(cat_feats)} categoricas nativas)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
scores, best_iters = [], []

for fold, (tr, va) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_feats)
    dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_feats, reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    test_pred += m.predict(X_test, num_iteration=m.best_iteration) / 5
    s = roc_auc_score(y[va], oof[va])
    scores.append(s); best_iters.append(m.best_iteration)
    print(f"Fold {fold}: AUC={s:.5f} (iter {m.best_iteration})")

cv = roc_auc_score(y, oof)
res = {"cv_auc_oof": round(cv, 5), "fold_scores": [round(s, 5) for s in scores],
       "fold_mean": round(float(np.mean(scores)), 5), "fold_std": round(float(np.std(scores)), 5),
       "best_iters": best_iters, "baseline": 0.7581, "delta": round(cv - 0.7581, 5),
       "n_features": int(X.shape[1])}
print(json.dumps(res, indent=2))

np.save(OUT / "oof.npy", oof)
np.save(OUT / "test_pred.npy", test_pred)
sub = pd.DataFrame({"SK_ID_CURR": id_test.astype(int), "TARGET": test_pred})
sub.to_csv(OUT / "submission_v3.csv", index=False)

imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
imp.head(30).to_csv(OUT / "importance_top30.csv")
print("\nTop 15 features (gain):"); print(imp.head(15).to_string())
(OUT / "results.json").write_text(json.dumps(res, indent=2))
assert len(sub) == 61500 and sub["TARGET"].between(0, 1).all()
print("\nSubmission OK:", sub.shape)
