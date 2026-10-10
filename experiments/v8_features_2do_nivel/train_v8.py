"""Exp v8: features de 2do nivel sobre v1 + params Optuna LGBM.
CV 5-fold seed 42. Compara contra v6 (0.76151). Nuevas: stats EXT, logs, interacciones,
agregados de numericas por categorias (sin target -> sin leakage), razones cruzadas."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v8_features_2do_nivel")
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(objective="binary", metric="auc", seed=SEED, deterministic=True,
              force_row_wise=True, verbose=-1, bagging_freq=1,
              **json.loads(Path("experiments/v4_optuna/best_params.json").read_text())["best_params"])


def build(df):
    df = df.copy()
    # --- v1 ---
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
    # --- v8: segundo nivel ---
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_MEAN"] = ext.mean(axis=1)
    df["EXT_STD"] = ext.std(axis=1)
    df["EXT_MIN"] = ext.min(axis=1)
    df["EXT_MAX"] = ext.max(axis=1)
    df["EXT_RANGE"] = df["EXT_MAX"] - df["EXT_MIN"]
    df["EXT_PROD"] = ext.prod(axis=1, min_count=3)
    df["EXT_1x2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    df["EXT_2x3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_1x3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
    # logs (colas largas)
    df["LOG_INCOME"] = np.log1p(df["AMT_INCOME_TOTAL"])
    df["LOG_CREDIT"] = np.log1p(df["AMT_CREDIT"])
    df["LOG_ANNUITY"] = np.log1p(df["AMT_ANNUITY"])
    df["LOG_GOODS"] = np.log1p(df["AMT_GOODS_PRICE"])
    # cruzadas de carga financiera
    df["ANNUITY_EMPLOYED"] = df["AMT_ANNUITY"] * df["EMPLOYED_YEARS"]
    df["CREDIT_BY_AGE"] = df["AMT_CREDIT"] / df["AGE_YEARS"]
    df["INCOME_BY_AGE"] = df["AMT_INCOME_TOTAL"] / df["AGE_YEARS"]
    df["EXT_MEAN_CREDIT"] = df["EXT_MEAN"] * df["LOG_CREDIT"]
    df["EXT_MEAN_ANNUITY_INC"] = df["EXT_MEAN"] * df["ANNUITY_INCOME"]
    # riesgo documental y de contacto
    doc = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOC_SUM"] = df[doc].sum(axis=1)
    contact = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
               "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    df["CONTACT_FLAGS"] = df[[c for c in contact if c in df.columns]].sum(axis=1)
    # region / vivienda
    if "REGION_RATING_CLIENT_W_CITY" in df.columns:
        df["REGION_RATING_SUM"] = df["REGION_RATING_CLIENT"] + df["REGION_RATING_CLIENT_W_CITY"]
        df["REGION_RATING_PROD"] = df["REGION_RATING_CLIENT"] * df["REGION_RATING_CLIENT_W_CITY"]
    df["POP_CREDIT"] = df["REGION_POPULATION_RELATIVE"] * df["AMT_CREDIT"]
    df["AGE_POP"] = df["REGION_POPULATION_RELATIVE"] * df["AGE_YEARS"]
    # frecuencias recientes
    df["ID_PUBLISH_YEARS"] = -df["DAYS_ID_PUBLISH"] / 365.25
    df["BUREAU_PER_YEAR"] = df["BUREAU_REQ_TOTAL"] / (df["REGISTRATION_YEARS"] + 1)
    return df


train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values
full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
# agregados por categoria (sin target): media de numericas por ORGANIZATION_TYPE y OCCUPATION_TYPE
agg_specs = {
    "ORGANIZATION_TYPE": [("EXT_MEAN", "mean"), ("CREDIT_INCOME", "mean"), ("AMT_CREDIT", "median")],
    "OCCUPATION_TYPE": [("EXT_MEAN", "mean"), ("AMT_INCOME_TOTAL", "median")],
    "CODE_GENDER": [("AMT_CREDIT", "mean")],
}
for col, specs in agg_specs.items():
    for src, func in specs:
        name = f"{col}_{src}_{func}"
        mapping = getattr(full.groupby(col)[src], func)()
        full[name] = full[col].map(mapping)
for c in full.select_dtypes(include=["object", "str"]).columns:
    full[c] = full[c].astype("category")

X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
cat_feats = [c for c in X.columns if str(X[c].dtype) == "category"]
print(f"Features: {X.shape[1]} (v1: 51)")

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
    print(f"fold {fold}: {roc_auc_score(y[va], oof[va]):.5f} (iter {m.best_iteration})")

cv = roc_auc_score(y, oof)
print(f"CV AUC: {cv:.5f} (v6 ref: 0.76151, delta {cv - 0.76151:+.5f})")
np.save(OUT / "oof.npy", oof); np.save(OUT / "test_pred.npy", pred)
imps["gain_mean"] = imps[[c for c in imps.columns if c.startswith("gain_")]].mean(axis=1)
imps.sort_values("gain_mean", ascending=False).to_csv(OUT / "importance.csv", index=False)
pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"].values, "TARGET": pred}).to_csv(OUT / "submission_v8.csv", index=False)
(OUT / "results.json").write_text(json.dumps({"cv_auc": round(cv, 5), "n_features": X.shape[1],
                                              "delta_vs_v6": round(cv - 0.76151, 5)}, indent=2))
