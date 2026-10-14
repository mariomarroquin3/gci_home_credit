"""Exp v29: v12 pipeline + restricciones monotonas EXT_SOURCE (lgbm+xgb) — test a nivel BLEND.
5 seeds x 5 folds x 3 modelos. Pareado vs files/exp_v12/oof_partial.npy (mismas seeds/pesos).
Adoptar a produccion solo si blend mejora sistematicamente (5/5 seeds, p<0.05)."""
import itertools
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
OUT = Path("files/exp_v29"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
MONO = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
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


def cb_str(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("object").where(out[c].notna(), "MISSING").astype(str)
    return out


train = pd.read_csv("files/data/train.csv")
test = pd.read_csv("files/data/test.csv")
y = train["TARGET"].values
full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
id_test = test["SK_ID_CURR"].values
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category"); X_test[c] = X_test[c].astype("category")
Xc, Xtc = cb_str(X, cat_cols), cb_str(X_test, cat_cols)
cat_idx = [Xc.columns.get_loc(c) for c in cat_cols]
Xx = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
Xtx = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True).reindex(columns=Xx.columns, fill_value=0)
mono_lgbm = [-1 if c in MONO else 0 for c in X.columns]
mono_xgb = "(" + ",".join(map(str, [-1 if c in MONO else 0 for c in Xx.columns])) + ")"

oof = {m: np.zeros((len(SEEDS), len(train))) for m in ("lgbm", "cb", "xgb")}
tpred = {m: np.zeros((len(SEEDS), len(test))) for m in ("lgbm", "cb", "xgb")}

for si, seed in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        p = dict(objective="binary", metric="auc", verbosity=-1, seed=seed, deterministic=True,
                 force_row_wise=True, bagging_freq=1, monotone_constraints=mono_lgbm,
                 monotone_constraints_method="advanced", **LGBM_P)
        m = lgb.train(p, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=20000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(300, verbose=False)])
        oof["lgbm"][si, va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        tpred["lgbm"][si] += m.predict(X_test, num_iteration=m.best_iteration) / 5
        cb = CatBoostClassifier(iterations=8000, od_wait=300, od_type="Iter", eval_metric="AUC",
                                random_seed=seed, verbose=False, learning_rate=CB_P["learning_rate"],
                                depth=CB_P["depth"], l2_leaf_reg=CB_P["l2_leaf_reg"],
                                random_strength=CB_P["random_strength"],
                                bagging_temperature=CB_P["bagging_temperature"],
                                min_data_in_leaf=CB_P["min_data_in_leaf"])
        cb.fit(Xc.iloc[tr], y[tr], cat_features=cat_idx, eval_set=(Xc.iloc[va], y[va]), use_best_model=True)
        oof["cb"][si, va] = cb.predict_proba(Xc.iloc[va])[:, 1]
        tpred["cb"][si] += cb.predict_proba(Xtc)[:, 1] / 5
        xp = dict(XGB_P, seed=seed, monotone_constraints=mono_xgb)
        m_x = xgb.train(xp, xgb.DMatrix(Xx.iloc[tr], y[tr]), num_boost_round=20000,
                        evals=[(xgb.DMatrix(Xx.iloc[va], y[va]), "va")], early_stopping_rounds=300, verbose_eval=False)
        oof["xgb"][si, va] = m_x.predict(xgb.DMatrix(Xx.iloc[va]), iteration_range=(0, m_x.best_iteration + 1))
        tpred["xgb"][si] += m_x.predict(xgb.DMatrix(Xtx), iteration_range=(0, m_x.best_iteration + 1)) / 5
    print(f"seed {seed}: {{m: round(roc_auc_score(y, oof[m][si]), 5) for m in oof}}", flush=True)
    np.save(OUT / "oof_partial.npy", np.stack([oof[m][:si + 1] for m in ("lgbm", "cb", "xgb")]))

oof_avg = {m: oof[m].mean(axis=0) for m in oof}
test_avg = {m: tpred[m].mean(axis=0) for m in oof}
for m in ("lgbm", "cb", "xgb"):
    np.save(OUT / f"oof_{m}_seedavg.npy", oof_avg[m])
    np.save(OUT / f"test_{m}_seedavg.npy", test_avg[m])
    print(f"{m} seed-avg OOF: {roc_auc_score(y, oof_avg[m]):.5f}", flush=True)

names = ["lgbm", "cb", "xgb"]
def score_on(mats, w):
    return roc_auc_score(y, sum(wi * mats[k] for wi, k in zip(w, names)))
best = (0, None)
for w in itertools.product(range(0, 21), repeat=3):
    if sum(w) != 20:
        continue
    w = tuple(x / 20 for x in w)
    s = score_on(oof_avg, w)
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
                s = score_on(oof_avg, w2)
                if s > bs + 1e-7:
                    w, bs, imp = w2, s, True
print(f"blend v29: {bs:.5f} {dict(zip(names, [round(x,3) for x in w]))} (vs v12 0.76460: {bs-0.76460:+.5f})", flush=True)

# pareado por seed con pesos fijos de v12 (.22/.43/.35) contra oof_partial de v12
W = [0.22, 0.43, 0.35]
v12 = np.load("files/exp_v12/oof_partial.npy")  # (3, 5, n)
a12 = [roc_auc_score(y, sum(wi * v12[mi, si] for mi, wi in enumerate(W))) for si in range(5)]
a29 = [roc_auc_score(y, sum(wi * oof[m][si] for m, wi in zip(names, W))) for si in range(5)]
d = np.array(a29) - np.array(a12)
t, pv = stats.ttest_rel(a29, a12)
print(f"pareado pesos fijos v12: {['%.5f'%x for x in a12]}", flush=True)
print(f"pareado pesos fijos v29: {['%.5f'%x for x in a29]}", flush=True)
print(f"delta media {d.mean():+.5f} sd {d.std():.5f} t={t:.2f} p={pv:.4f}", flush=True)

pred = sum(wi * test_avg[k] for wi, k in zip(w, names))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v29.csv", index=False)
(OUT / "results.json").write_text(json.dumps({
    "per_model_seedavg": {m: round(roc_auc_score(y, oof_avg[m]), 5) for m in names},
    "blend_cv": round(bs, 5), "weights": dict(zip(names, [round(x, 4) for x in w])),
    "delta_vs_v12": round(bs - 0.76460, 5),
    "paired_fixed_w": {"v12": [round(a, 5) for a in a12], "v29": [round(a, 5) for a in a29],
                       "delta_mean": round(float(d.mean()), 5), "delta_sd": round(float(d.std()), 5),
                       "t": round(float(t), 2), "p": round(float(pv), 4)}}, indent=1))
print("DONE_V29", flush=True)
