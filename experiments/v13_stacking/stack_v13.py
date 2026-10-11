"""Exp v13: stacking sobre OOF seed-averaged (v12) + knn500 como 4o input opcional.
Evaluacion honesta: cross_val_predict del meta-modelo (logreg, 5-fold seed 42).
Ref: blend v12 = 0.76460."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from pathlib import Path

OUT = Path("experiments/v13_stacking"); OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv("data/raw/train.csv"); y = train["TARGET"].values
id_test = pd.read_csv("data/raw/test.csv")["SK_ID_CURR"].values
names = ["lgbm", "cb", "xgb"]
O = {m: np.load(f"experiments/v12_seedavg/oof_{m}_seedavg.npy") for m in names}
T = {m: np.load(f"experiments/v12_seedavg/test_{m}_seedavg.npy") for m in names}
K_tr = np.load("experiments/v11_tier1/knn500_tr.npy"); K_te = np.load("files/exp_v11b/knn500_te.npy")

skf = StratifiedKFold(5, shuffle=True, random_state=42)
res = {}

def stack_eval(Xs, tag):
    meta = LogisticRegression(C=1.0, max_iter=2000)
    oof_meta = cross_val_predict(meta, Xs, y, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof_meta)
    meta.fit(Xs, y)
    return auc, meta

# 1) stack 3 modelos (probs)
X3 = np.column_stack([O[m] for m in names])
auc3, meta3 = stack_eval(X3, "3mod")
res["stack_3mod"] = round(auc3, 5)

# 2) stack 3 modelos + knn500
X4 = np.column_stack([X3, K_tr])
auc4, meta4 = stack_eval(X4, "4mod")
res["stack_4mod_knn"] = round(auc4, 5)

# 3) rank-average simple de los 3 (referencia robusta)
R = np.mean([rankdata(O[m]) for m in names], axis=0)
res["rankavg_3mod"] = round(roc_auc_score(y, R), 5)

# 4) rank-average 3 + knn
R4 = np.mean([rankdata(O[m]) for m in names] + [rankdata(K_tr)], axis=0)
res["rankavg_4mod"] = round(roc_auc_score(y, R4), 5)

res["ref_blend_v12"] = 0.76460
best_key = max((k for k in res if k != "ref_blend_v12"), key=lambda k: res[k])
res["best"] = best_key
res["delta_best_vs_v12"] = round(res[best_key] - 0.76460, 5)
print(json.dumps(res, indent=1))

# submission del mejor
if best_key == "stack_3mod":
    pred = meta3.predict_proba(np.column_stack([T[m] for m in names]))[:, 1]
elif best_key == "stack_4mod_knn":
    pred = meta4.predict_proba(np.column_stack([T[m] for m in names] + [K_te]))[:, 1]
elif best_key == "rankavg_3mod":
    pred = np.mean([rankdata(T[m]) for m in names], axis=0)
    pred = (pred - pred.min()) / (pred.max() - pred.min())
else:
    pred = np.mean([rankdata(T[m]) for m in names] + [rankdata(K_te)], axis=0)
    pred = (pred - pred.min()) / (pred.max() - pred.min())
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v13.csv", index=False)
(OUT / "results.json").write_text(json.dumps(res, indent=1))
print("OK")
