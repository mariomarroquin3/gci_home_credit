"""v37: stacking de nivel 2 sobre OOF de los 5 modelos (+ features clave) vs blend por pesos.

Regresión logística sobre logit(OOF) + features estandarizadas; se evalúa con CV 5-fold sobre el
OOF (el stacker nunca ve la fila que predice). Barato: no compite por CPU con los GBDT.
Uso: python -m experiments.final.stack "run_*_v35_s4*"
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

from src.blend import apply_weights, fast_auc, hill_climb_weights
from src.config import OUTPUT_DIR, TARGET
from src.data import load_data
from src.features import build

runs = sorted(OUTPUT_DIR.glob(sys.argv[1]))
models = ["lgbm", "cb", "xgb", "lgbm_init", "lgbm_linear"]
train, test, _ = load_data()
y = train[TARGET].values
oof = {m: np.mean([np.load(r / f"oof_{m}.npy") for r in runs], axis=0) for m in models}
L = np.column_stack([np.log(oof[m] / (1 - oof[m])) for m in models])
tr, _, _ = build(train, test, aux=False)
FEATS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "RATE_GUESS", "ka_n", "CREDIT_ANNUITY", "AGE_YEARS"]
Fx = tr[FEATS].astype(float)
Fx = pd.concat([Fx.fillna(Fx.median()), Fx.isna().astype(float).add_suffix("_NA")], axis=1)
Fx = ((Fx - Fx.mean()) / (Fx.std() + 1e-9)).values


def cv_lr(X, C=1.0, spline=False):
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
        Xa, Xb = X[a], X[b]
        if spline:
            st = SplineTransformer(n_knots=6, degree=3).fit(Xa)
            Xa, Xb = st.transform(Xa), st.transform(Xb)
        m = LogisticRegression(C=C, max_iter=2000).fit(Xa, y[a])
        out[b] = m.decision_function(Xb)
    return fast_auc(y, out)


def cv_hc():
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(L, y):
        w, _ = hill_climb_weights({m: oof[m][a] for m in models}, y[a], score_fn=fast_auc)
        out[b] = apply_weights({m: oof[m][b] for m in models}, w)
    return fast_auc(y, out)


print("runs", [r.name for r in runs])
print(f"hill-climb anidado      {cv_hc():.5f}")
print(f"LR logits               {cv_lr(L):.5f}")
print(f"LR logits + feats       {cv_lr(np.c_[L, Fx]):.5f}")
print(f"LR logits + interacc.   {cv_lr(np.c_[L, Fx, L.mean(1, keepdims=True) * Fx]):.5f}")
print(f"LR spline logits+feats  {cv_lr(np.c_[L, Fx], C=0.1, spline=True):.5f}")
