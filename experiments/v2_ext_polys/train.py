"""Exp v2 (branch exp/ext-polys-v2): v1 + polinomicas EXT_SOURCE + stats EXT + TE-OOF
(ORGANIZATION_TYPE, OCCUPATION_TYPE) + flag BUREAU_YEAR_ISNA.
Misma CV y params que v1 para comparacion limpia."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v2_ext_polys")
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
              min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, seed=SEED, deterministic=True,
              force_row_wise=True, verbose=-1)
TE_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
TE_M = 100.0  # smoothing


def build_static(df: pd.DataFrame) -> pd.DataFrame:
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
    # EXT_SOURCE polinomicas y stats
    e1, e2, e3 = df["EXT_SOURCE_1"], df["EXT_SOURCE_2"], df["EXT_SOURCE_3"]
    df["EXT1_SQ"], df["EXT2_SQ"], df["EXT3_SQ"] = e1**2, e2**2, e3**2
    df["EXT1_EXT2"], df["EXT1_EXT3"], df["EXT2_EXT3"] = e1*e2, e1*e3, e2*e3
    df["EXT1_EXT2_EXT3"] = e1*e2*e3
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_MEAN"] = ext.mean(axis=1)
    df["EXT_STD"] = ext.std(axis=1)
    df["EXT_MIN"] = ext.min(axis=1)
    df["EXT_MAX"] = ext.max(axis=1)
    df["EXT_NAN_COUNT"] = ext.isna().sum(axis=1)
    return df


def te_fit(s: pd.Series, y: np.ndarray) -> dict:
    df = pd.DataFrame({"c": s.astype("str"), "y": y})
    g = df.groupby("c")["y"].agg(["mean", "count"])
    prior = y.mean()
    return {c: (n * m_ + TE_M * prior) / (n + TE_M)
            for c, (m_, n) in g[["mean", "count"]].iterrows()}


train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values

full = pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True)
full = build_static(full)
X = full.iloc[: len(train)].reset_index(drop=True)
X_test = full.iloc[len(train):].reset_index(drop=True)
id_test = X_test["SK_ID_CURR"].values
X = X.drop(columns=["SK_ID_CURR"]); X_test = X_test.drop(columns=["SK_ID_CURR"])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
scores, best_iters = [], []

for fold, (tr, va) in enumerate(skf.split(X, y)):
    Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    Xte = X_test.copy()
    prior = y[tr].mean()
    for c in TE_COLS:
        m = te_fit(Xtr[c], y[tr])
        for d in (Xtr, Xva, Xte):
            d[f"TE_{c}"] = d[c].astype("str").map(m).fillna(prior)
    for d in (Xtr, Xva, Xte):
        for c in d.select_dtypes(include=["object", "str"]).columns:
            d[c] = d[c].astype("category")
    cat_feats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    dtr = lgb.Dataset(Xtr, y[tr], categorical_feature=cat_feats)
    dva = lgb.Dataset(Xva, y[va], categorical_feature=cat_feats, reference=dtr)
    mdl = lgb.train(PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = mdl.predict(Xva, num_iteration=mdl.best_iteration)
    test_pred += mdl.predict(Xte, num_iteration=mdl.best_iteration) / 5
    s = roc_auc_score(y[va], oof[va])
    scores.append(s); best_iters.append(mdl.best_iteration)
    print(f"Fold {fold}: AUC={s:.5f} (iter {mdl.best_iteration})")

cv = roc_auc_score(y, oof)
res = {"cv_auc_oof": round(cv, 5), "fold_scores": [round(s, 5) for s in scores],
       "fold_mean": round(float(np.mean(scores)), 5), "fold_std": round(float(np.std(scores)), 5),
       "best_iters": best_iters, "baseline": 0.7581, "v1": 0.75827,
       "delta_vs_baseline": round(cv - 0.7581, 5), "delta_vs_v1": round(cv - 0.75827, 5),
       "n_features": int(Xtr.shape[1])}
print(json.dumps(res, indent=2))

np.save(OUT / "oof.npy", oof)
np.save(OUT / "test_pred.npy", test_pred)
pd.DataFrame({"SK_ID_CURR": id_test.astype(int), "TARGET": test_pred}).to_csv(
    OUT / "submission_v2.csv", index=False)
imp = pd.Series(mdl.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
imp.head(40).to_csv(OUT / "importance_top40.csv")
print("\nTop 20 features (gain):"); print(imp.head(20).to_string())
(OUT / "results.json").write_text(json.dumps(res, indent=2))
print("\nSubmission OK:", (len(test_pred),))
