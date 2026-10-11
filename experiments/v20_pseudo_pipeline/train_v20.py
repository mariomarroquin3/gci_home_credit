"""Exp v20: pipeline v12 (5 seeds x LGBM-t/CB-t/XGB-t) con pseudo-labeling (v19) en los 3 modelos.
Pseudo rows: test con p<0.03 del blend v12 (todas negativas, 15156). OOF solo sobre train -> honesto.
Ref: blend v12 = 0.76460."""
import itertools
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v20_pseudo_pipeline"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
LGBM_P = json.loads(Path("experiments/v4_optuna/best_params.json").read_text())["best_params"]
CB_P = json.loads(Path("experiments/v9_optuna_cb/best_params.json").read_text())
XGB_P = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", nthread=-1,
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
for c in cat_cols:
    X[c] = X[c].astype("category"); X_test[c] = X_test[c].astype("category")
Xc, Xc_test = X.astype({c: "str" for c in cat_cols}), X_test.astype({c: "str" for c in cat_cols})
Xx = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
Xx_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True).reindex(columns=Xx.columns, fill_value=0)

p_blend = pd.read_csv("experiments/v12_seedavg/submission_v12.csv")["TARGET"].values
mask = p_blend < 0.03
yp = np.zeros(int(mask.sum()), dtype=int)
def aug(Xd, idx):
    A = pd.concat([Xd.iloc[idx], Xd.iloc[len(train):][mask] if len(Xd) > len(train) else X_test.iloc[mask]], ignore_index=True)
    return A, np.r_[y[idx], yp]
# versiones de pseudo por encoding
Xp_cat = X_test[mask].copy(); Xp_str = Xc_test[mask].copy(); Xp_oh = Xx_test[mask].copy()

models = ["lgbm", "cb", "xgb"]
OOF = {m: np.zeros((len(SEEDS), len(train))) for m in models}
TP = {m: np.zeros((len(SEEDS), len(test))) for m in models}
for si, seed in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        # LGBM
        Xa = pd.concat([X.iloc[tr], Xp_cat], ignore_index=True)
        for c in cat_cols: Xa[c] = Xa[c].astype("category")
        ya = np.r_[y[tr], yp]
        p = dict(objective="binary", metric="auc", verbosity=-1, seed=seed, deterministic=True,
                 force_row_wise=True, bagging_freq=1, **LGBM_P)
        m = lgb.train(p, lgb.Dataset(Xa, ya), num_boost_round=20000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(300, verbose=False)])
        OOF["lgbm"][si, va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        TP["lgbm"][si] += m.predict(X_test, num_iteration=m.best_iteration) / 5
        # CatBoost
        Xb = pd.concat([Xc.iloc[tr], Xp_str], ignore_index=True)
        yb = np.r_[y[tr], yp]
        cb = CatBoostClassifier(iterations=8000, learning_rate=CB_P["learning_rate"], depth=CB_P["depth"],
                                l2_leaf_reg=CB_P["l2_leaf_reg"], random_strength=CB_P["random_strength"],
                                bagging_temperature=CB_P["bagging_temperature"], min_data_in_leaf=CB_P["min_data_in_leaf"],
                                loss_function="Logloss", eval_metric="AUC", random_seed=seed,
                                od_type="Iter", od_wait=300, verbose=False, thread_count=-1)
        cb.fit(Pool(Xb, yb, cat_features=cat_cols), eval_set=Pool(Xc.iloc[va], y[va], cat_features=cat_cols))
        OOF["cb"][si, va] = cb.predict_proba(Xc.iloc[va])[:, 1]
        TP["cb"][si] += cb.predict_proba(Xc_test)[:, 1] / 5
        # XGBoost
        Xd = pd.concat([Xx.iloc[tr], Xp_oh], ignore_index=True)
        yd = np.r_[y[tr], yp]
        xm = xgb.train(dict(XGB_P, seed=seed), xgb.DMatrix(Xd, yd), num_boost_round=20000,
                       evals=[(xgb.DMatrix(Xx.iloc[va], y[va]), "va")], early_stopping_rounds=300, verbose_eval=False)
        OOF["xgb"][si, va] = xm.predict(xgb.DMatrix(Xx.iloc[va]), iteration_range=(0, xm.best_iteration + 1))
        TP["xgb"][si] += xm.predict(xgb.DMatrix(Xx_test), iteration_range=(0, xm.best_iteration + 1)) / 5
    msg = {m: round(roc_auc_score(y, OOF[m][si]), 5) for m in models}
    print(f"seed {seed}: {msg}", flush=True)
    np.save(OUT / "oof_partial.npy", np.stack([OOF[m][:si+1] for m in models]))

O = {}; T = {}
for m in models:
    O[m] = OOF[m].mean(axis=0); T[m] = TP[m].mean(axis=0)
    np.save(OUT / f"oof_{m}_pl_seedavg.npy", O[m]); np.save(OUT / f"test_{m}_pl_seedavg.npy", T[m])
    print(f"{m} PL seed-avg OOF: {roc_auc_score(y, O[m]):.5f}")

def score(w):
    return roc_auc_score(y, sum(wi * O[k] for wi, k in zip(w, models)))
best = (0, None)
for w in itertools.product(range(0, 21), repeat=3):
    if sum(w) != 20: continue
    w = tuple(x / 20 for x in w); s = score(w)
    if s > best[0]: best = (s, w)
w = list(best[1]); bs = best[0]
imp = True
while imp:
    imp = False
    for i in range(3):
        for j in range(3):
            if i == j: continue
            for d in (0.01, -0.01):
                w2 = list(w); w2[i] += d; w2[j] -= d
                if min(w2) < 0: continue
                s = score(w2)
                if s > bs + 1e-7: w, bs = w2, s; imp = True
print(f"blend v20 (PL): {bs:.5f} {dict(zip(models, [round(x,3) for x in w]))} (vs v12: {bs-0.76460:+.5f})")
pred = sum(wi * T[k] for wi, k in zip(w, models))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v20.csv", index=False)
(OUT / "results.json").write_text(json.dumps({
    "per_model_pl_seedavg": {m: round(roc_auc_score(y, O[m]), 5) for m in models},
    "blend_cv": round(bs, 5), "weights": dict(zip(models, [round(x, 4) for x in w])),
    "delta_vs_v12": round(bs - 0.76460, 5)}, indent=1))
print("OK")
