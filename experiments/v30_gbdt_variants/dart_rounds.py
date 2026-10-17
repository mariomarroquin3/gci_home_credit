"""Rondas fijas para lgbm_dart: un split 80/20 aparte (seed 999), AUC cada 100 rondas."""
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ID_COL, LGBM_PARAMS, TARGET
from src.data import load_data
from src.features import build
from src.models import _monotone
from src.variants import LGBM_VARIANTS

tr, te, cats = build(*load_data()[:2])
f = [c for c in tr.columns if c not in (ID_COL, TARGET)]
a, b = train_test_split(tr, test_size=0.2, stratify=tr[TARGET], random_state=999)
p = dict(LGBM_PARAMS, **LGBM_VARIANTS["lgbm_dart"], seed=999, monotone_constraints=_monotone(f),
         monotone_constraints_method="advanced")
res = {}
m = lgb.train(p, lgb.Dataset(a[f], a[TARGET]), 3000, valid_sets=[lgb.Dataset(b[f], b[TARGET])],
              callbacks=[lgb.record_evaluation(res), lgb.log_evaluation(100)])
auc = res["valid_0"]["auc"]
best = max(range(len(auc)), key=auc.__getitem__)
print(f"mejor ronda {best + 1}: AUC {auc[best]:.5f}")
