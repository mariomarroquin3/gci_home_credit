"""Exp v16: adversarial validation train vs test.
Clasificador LGBM distingue train(0) de test(1). AUC ~0.5 = sin shift (linea cerrada).
Si AUC alto -> features con drift identificadas por importancia (candidatas a drop/weight)."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("files/exp_v16"); OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
cols = [c for c in train.columns if c not in ("TARGET", "SK_ID_CURR")]
X = pd.concat([train[cols], test[cols]], ignore_index=True)
for c in X.select_dtypes(include=["object", "str"]).columns:
    X[c] = X[c].astype("category")
y_adv = np.r_[np.zeros(len(train)), np.ones(len(test))]

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y_adv)); imp = np.zeros(X.shape[1])
p = dict(objective="binary", metric="auc", verbosity=-1, seed=42, learning_rate=0.05,
         num_leaves=31, feature_fraction=0.8)
for tr, va in skf.split(X, y_adv):
    m = lgb.train(p, lgb.Dataset(X.iloc[tr], y_adv[tr]), num_boost_round=1000,
                  valid_sets=[lgb.Dataset(X.iloc[va], y_adv[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    imp += m.feature_importance("gain") / 5
auc = roc_auc_score(y_adv, oof)
top = sorted(zip(X.columns, imp), key=lambda t: -t[1])[:12]
print(f"ADVERSARIAL AUC = {auc:.5f}")
for c, g in top:
    print(f"  {c}: {g:.0f}")
(OUT / "results.json").write_text(json.dumps({"adv_auc": round(auc, 5),
    "top_drift_features": [(c, round(float(g), 1)) for c, g in top]}, indent=1))
