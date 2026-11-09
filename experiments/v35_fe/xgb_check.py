"""v35: ¿la ganancia de EXT_PRED/RES pasa a XGB? 5 folds, 1 seed, lr 0.05 (screening)."""
import sys
import time

import numpy as np
import pandas as pd

import src.models as M
from src.blend import fast_auc
from src.config import ID_COL, TARGET
from src.data import load_data
from src.features import build
from src.structure import FoldFeatures
from experiments.v35_fe.screen import oof_regress

M.XGB_PARAMS = dict(M.XGB_PARAMS, learning_rate=0.05)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
t0 = time.time()
train, test, _ = load_data()
trb, teb, cats = build(train, test, aux=False)
feats = [c for c in trb.columns if c not in (ID_COL, TARGET)]
X0 = pd.concat([trb[feats], teb[feats]], ignore_index=True)
y = train[TARGET].values
n = len(train)
ext = pd.DataFrame(index=X0.index)
drop = [c for c in X0.columns if c.startswith("EXT_SOURCE")]
for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    p = oof_regress(X0.drop(columns=drop), X0[c].values.astype(float))
    ext[f"{c}_PRED"] = p
    ext[f"{c}_RES"] = X0[c].values - p
tg = {"AMT_INCOME_TOTAL": (np.log(X0.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL", "ANNUITY_INCOME", "CREDIT_INCOME", "INCOME_PER_FAM"]),
      "AMT_ANNUITY": (np.log(X0.AMT_ANNUITY), ["AMT_ANNUITY", "ANNUITY_INCOME"] + [c for c in X0.columns if c.startswith(("RATE_", "TERM_", "CA_R", "CREDIT_ANNUITY"))]),
      "DAYS_EMPLOYED": (X0.DAYS_EMPLOYED, ["DAYS_EMPLOYED", "EMPLOYED_YEARS", "EMPLOYED_BIRTH_PCT", "kb_n", "kc_n"]),
      "AMT_GOODS_PRICE": (np.log(X0.AMT_GOODS_PRICE), ["AMT_GOODS_PRICE", "CREDIT_GOODS", "DOWNPAYMENT"])}
for c, (v, dr) in tg.items():
    v = np.asarray(v, float)
    ext[f"{c}_RES"] = v - oof_regress(X0.drop(columns=dr), v)
ff = FoldFeatures(train, test, y)
for name, X in (("base", X0), ("ext", pd.concat([X0, ext], axis=1))):
    oof, _ = M.run_xgb(X.iloc[:n], y, X.iloc[n:].reset_index(drop=True), cats, [seed], 5, fold_fn=ff)
    print(name, f"{fast_auc(y, oof[0]):.5f}", f"{time.time()-t0:.0f}s", flush=True)
