"""Blend v7: pesos en simplex sobre OOF de LGBM-tuneado + CatBoost + XGBoost.
Grid 0.05. Regla del proyecto: blend adoptable si suma >= +0.001 sobre el mejor individual."""
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 42
OUT = Path("experiments/v7_blend")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values

names = ["lgbm", "catboost", "xgboost"]
oofs = [np.load(f) for f in ["experiments/v6_lgbm_tuned/oof_lgbm_tuned.npy",
                             "experiments/v5_diversificadores/oof_catboost.npy",
                             "experiments/v5_diversificadores/oof_xgboost.npy"]]
tests = [np.load(f) for f in ["experiments/v6_lgbm_tuned/test_lgbm_tuned.npy",
                              "experiments/v5_diversificadores/test_catboost.npy",
                              "experiments/v5_diversificadores/test_xgboost.npy"]]
indiv = {n: round(roc_auc_score(y, o), 5) for n, o in zip(names, oofs)}
print("Individuales:", indiv)

# correlacion de errores (diversidad)
for i, j in itertools.combinations(range(3), 2):
    c = np.corrcoef(oofs[i], oofs[j])[0, 1]
    print(f"corr({names[i]},{names[j]}) = {c:.4f}")

best = (None, -1)
step = 0.05
grid = [x / 20 for x in range(21)]
for wl in grid:
    for wc in grid:
        wx = 1 - wl - wc
        if wx < -1e-9:
            continue
        s = roc_auc_score(y, wl * oofs[0] + wc * oofs[1] + wx * oofs[2])
        if s > best[1]:
            best = ((wl, wc, wx), s)
w, cv_blend = best
print(f"Mejor blend: w={w} -> CV {cv_blend:.5f}")

blend_test = sum(wi * ti for wi, ti in zip(w, tests))
sub = pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"].values, "TARGET": blend_test})
sub.to_csv(OUT / "submission_v7_blend.csv", index=False)
res = {"individuales": indiv, "pesos": dict(zip(names, w)), "cv_blend": round(cv_blend, 5),
       "delta_vs_mejor": round(cv_blend - max(indiv.values()), 5)}
(OUT / "results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
