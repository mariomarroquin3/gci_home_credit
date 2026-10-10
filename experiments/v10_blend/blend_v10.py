"""Exp v10: blend LGBM-tuned + CB-tuned(v9) + XGBoost.
Coarse simplex 0.05 -> hill climbing 0.01. Compara vs blend v7 (CV 0.76366)."""
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v10_blend"); OUT.mkdir(parents=True, exist_ok=True)
y = pd.read_csv("data/raw/train.csv")["TARGET"].values
test_ids = pd.read_csv("data/raw/test.csv")["SK_ID_CURR"].values

models = {
    "lgbm": (np.load("experiments/v6_lgbm_tuned/oof_lgbm_tuned.npy"), np.load("experiments/v6_lgbm_tuned/test_lgbm_tuned.npy")),
    "cb": (np.load("experiments/v9_optuna_cb/oof_cb_tuned.npy"), np.load("experiments/v9_optuna_cb/test_cb_tuned.npy")),
    "xgb": (np.load("experiments/v5_diversificadores/oof_xgboost.npy"), np.load("experiments/v5_diversificadores/test_xgboost.npy")),
}
for k, (o, _) in models.items():
    print(f"{k}: OOF {roc_auc_score(y, o):.5f}")

names = list(models)
oofs = [models[k][0] for k in names]

def score(w):
    return roc_auc_score(y, sum(wi * o for wi, o in zip(w, oofs)))

# 1) simplex grueso 0.05
best = (0, None)
for w in itertools.product(range(0, 21), repeat=3):
    if sum(w) != 20:
        continue
    w = tuple(x / 20 for x in w)
    s = score(w)
    if s > best[0]:
        best = (s, w)
print(f"coarse: {best[0]:.5f} {dict(zip(names, best[1]))}")

# 2) hill climbing 0.01
w = list(best[1]); bs = best[0]; step = 0.01
improved = True
while improved:
    improved = False
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for d in (step, -step):
                w2 = list(w); w2[i] += d; w2[j] -= d
                if min(w2) < 0:
                    continue
                s = score(w2)
                if s > bs + 1e-7:
                    w, bs = w2, s; improved = True
print(f"hill: {bs:.5f} {dict(zip(names, [round(x,3) for x in w]))}")

# 3) promedio simple de referencia
s_avg = score([1/3]*3)
print(f"avg 1/3: {s_avg:.5f}")

w = [round(x, 4) for x in w]
pred = sum(wi * models[k][1] for wi, k in zip(w, names))
sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": pred})
sub.to_csv(OUT / "submission_v10_blend.csv", index=False)
(OUT / "results.json").write_text(json.dumps({
    "cv_auc": round(bs, 5), "weights": dict(zip(names, w)),
    "avg_simple": round(s_avg, 5), "ref_blend_v7": 0.76366}, indent=1))
print(f"delta vs v7: {bs - 0.76366:+.5f}")
print(f"sub range [{pred.min():.4f}, {pred.max():.4f}] rows {len(sub)}")
