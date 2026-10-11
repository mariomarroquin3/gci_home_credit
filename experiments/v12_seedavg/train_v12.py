"""Exp v12: seed averaging 5 seeds x 3 modelos (LGBM-tuned, CB-tuned, XGB) + blend hill-climbing.
Cada seed: SKF(5, seed). OOF promediado por modelo sigue siendo OOF honesto (fila nunca en train).
Ref: blend v10 single-seed CV 0.76414."""
import itertools
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v12_seedavg"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
LGBM_P = json.loads(Path("files/exp_v4/best_params.json").read_text())["best_params"]
CB_P = json.loads(Path("files/exp_v9/best_params.json").read_text())
XGB_P = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
             max_depth=5, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=5.0, tree_method="hist", nthread=-1)


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
Xc, Xtc = X.copy(), X_test.copy()
for c in cat_cols:
    Xc[c] = Xc[c].astype("str"); Xtc[c] = Xtc[c].astype("str")
cat_idx = [Xc.columns.get_loc(c) for c in cat_cols]
Xx = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
Xtx = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True).reindex(columns=Xx.columns, fill_value=0)

oof = {m: np.zeros((len(SEEDS), len(train))) for m in ("lgbm", "cb", "xgb")}
tpred = {m: np.zeros((len(SEEDS), len(test))) for m in ("lgbm", "cb", "xgb")}

for si, seed in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        # LGBM
        p = dict(objective="binary", metric="auc", verbosity=-1, seed=seed, deterministic=True,
                 force_row_wise=True, bagging_freq=1, **LGBM_P)
        m = lgb.train(p, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=20000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(300, verbose=False)])
        oof["lgbm"][si, va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        tpred["lgbm"][si] += m.predict(X_test, num_iteration=m.best_iteration) / 5
        # CatBoost
        cb = CatBoostClassifier(iterations=8000, od_wait=300, od_type="Iter", eval_metric="AUC",
                                random_seed=seed, verbose=False, learning_rate=CB_P["learning_rate"],
                                depth=CB_P["depth"], l2_leaf_reg=CB_P["l2_leaf_reg"],
                                random_strength=CB_P["random_strength"],
                                bagging_temperature=CB_P["bagging_temperature"],
                                min_data_in_leaf=CB_P["min_data_in_leaf"])
        cb.fit(Xc.iloc[tr], y[tr], cat_features=cat_idx, eval_set=(Xc.iloc[va], y[va]), use_best_model=True)
        oof["cb"][si, va] = cb.predict_proba(Xc.iloc[va])[:, 1]
        tpred["cb"][si] += cb.predict_proba(Xtc)[:, 1] / 5
        # XGBoost
        xp = dict(XGB_P, seed=seed)
        m_x = xgb.train(xp, xgb.DMatrix(Xx.iloc[tr], y[tr]), num_boost_round=20000,
                        evals=[(xgb.DMatrix(Xx.iloc[va], y[va]), "va")], early_stopping_rounds=300, verbose_eval=False)
        oof["xgb"][si, va] = m_x.predict(xgb.DMatrix(Xx.iloc[va]), iteration_range=(0, m_x.best_iteration + 1))
        tpred["xgb"][si] += m_x.predict(xgb.DMatrix(Xtx), iteration_range=(0, m_x.best_iteration + 1)) / 5
    s_auc = {m: round(roc_auc_score(y, oof[m][si]), 5) for m in oof}
    print(f"seed {seed}: {s_auc}", flush=True)
    np.save(OUT / "oof_partial.npy", np.stack([oof[m][:si + 1] for m in ("lgbm", "cb", "xgb")]))

oof_avg = {m: oof[m].mean(axis=0) for m in oof}
test_avg = {m: tpred[m].mean(axis=0) for m in oof}
for m in ("lgbm", "cb", "xgb"):
    np.save(OUT / f"oof_{m}_seedavg.npy", oof_avg[m])
    np.save(OUT / f"test_{m}_seedavg.npy", test_avg[m])
    print(f"{m} seed-avg OOF: {roc_auc_score(y, oof_avg[m]):.5f}")

names = ["lgbm", "cb", "xgb"]
def score(w):
    return roc_auc_score(y, sum(wi * oof_avg[k] for wi, k in zip(w, names)))
best = (0, None)
for w in itertools.product(range(0, 21), repeat=3):
    if sum(w) != 20:
        continue
    w = tuple(x / 20 for x in w)
    s = score(w)
    if s > best[0]:
        best = (s, w)
w = list(best[1]); bs = best[0]
imp = True
while imp:
    imp = False
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for d in (0.01, -0.01):
                w2 = list(w); w2[i] += d; w2[j] -= d
                if min(w2) < 0:
                    continue
                s = score(w2)
                if s > bs + 1e-7:
                    w, bs = w2, s; imp = True
print(f"blend seed-avg: {bs:.5f} {dict(zip(names, [round(x,3) for x in w]))} (vs v10 0.76414: {bs-0.76414:+.5f})")

pred = sum(wi * test_avg[k] for wi, k in zip(w, names))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v12.csv", index=False)
(OUT / "results.json").write_text(json.dumps({
    "per_model_seedavg": {m: round(roc_auc_score(y, oof_avg[m]), 5) for m in names},
    "blend_cv": round(bs, 5), "weights": dict(zip(names, [round(x, 4) for x in w])),
    "delta_vs_v10": round(bs - 0.76414, 5), "seeds": SEEDS}, indent=1))
print("OK guardado")
