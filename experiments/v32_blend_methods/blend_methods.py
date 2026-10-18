"""Exp v32: métodos de blend sobre los miembros v30 (LGBM+CB+XGB+lgbm_init+lgbm_linear).

Compara, con pesos elegidos en CV anidado sobre filas (mismos folds que src.validate):
  hc_prob    hill-climbing sobre probabilidades (producción v30)
  hc_rank    hill-climbing sobre ranks normalizados
  hc_logit   hill-climbing sobre logits
  cs_prob    Caruana ensemble selection con reemplazo (sobre probabilidades)
  cs_rank    Caruana con reemplazo sobre ranks
  csb_rank   Caruana con bagging de modelos (Caruana et al. 2006) sobre ranks

  python experiments/v32_blend_methods/blend_methods.py outputs/run_<...>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.blend import fast_auc, hill_climb_weights  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.data import load_data  # noqa: E402
from src.validate import NESTED_FOLDS, NESTED_SEED  # noqa: E402

MEMBERS = ["lgbm", "cb", "xgb", "lgbm_init", "lgbm_linear"]
CS_ITERS = 200
CS_BAGS, CS_BAG_FRAC = 25, 0.6


def to_rank(p):
    return rankdata(p) / len(p)


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


TRANSFORMS = {"prob": lambda p: p, "rank": to_rank, "logit": to_logit}


def caruana(mats, y, iters=CS_ITERS, members=None):
    """Ensemble selection con reemplazo; init = mejor modelo solo. Devuelve pesos (conteos/total)."""
    names = members or list(mats)
    counts = dict.fromkeys(names, 0)
    best = max(names, key=lambda m: fast_auc(y, mats[m]))
    counts[best] = 1
    cur = mats[best].copy()
    best_s, best_counts = fast_auc(y, cur), dict(counts)
    for k in range(1, iters):
        s_cand = {m: fast_auc(y, (cur * k + mats[m]) / (k + 1)) for m in names}
        m = max(s_cand, key=s_cand.get)
        counts[m] += 1
        cur = (cur * k + mats[m]) / (k + 1)
        if s_cand[m] > best_s + 1e-9:
            best_s, best_counts = s_cand[m], dict(counts)
    tot = sum(best_counts.values())
    return {m: c / tot for m, c in best_counts.items()}


def caruana_bagged(mats, y, seed=0):
    rng = np.random.RandomState(seed)
    names = list(mats)
    k = max(2, int(round(CS_BAG_FRAC * len(names))))
    acc = dict.fromkeys(names, 0.0)
    for _ in range(CS_BAGS):
        sub = list(rng.choice(names, k, replace=False))
        for m, w in caruana(mats, y, members=sub).items():
            acc[m] += w / CS_BAGS
    return acc


def fit(method, mats, y):
    kind, tf = method.split("_")
    if kind == "hc":
        return hill_climb_weights(mats, y, score_fn=fast_auc)[0]
    if kind == "cs":
        return caruana(mats, y)
    return caruana_bagged(mats, y)


def nested(method, raw, y):
    tf = TRANSFORMS[method.split("_")[1]]
    mats = {m: tf(p) for m, p in raw.items()}  # transformación sin labels -> sin fuga
    pred = np.zeros(len(y))
    ws = []
    skf = StratifiedKFold(n_splits=NESTED_FOLDS, shuffle=True, random_state=NESTED_SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        w = fit(method, {m: o[tr] for m, o in mats.items()}, y[tr])
        pred[va] = sum(w[m] * mats[m][va] for m in w)
        ws.append({m: round(v, 3) for m, v in w.items()})
    return roc_auc_score(y, pred), ws


METHODS = ["hc_prob", "hc_rank", "hc_logit", "cs_prob", "cs_rank", "cs_logit", "csb_rank"]


def main():
    run = Path(sys.argv[1])
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    methods = sys.argv[3].split(",") if len(sys.argv) > 3 else METHODS
    train, _, _ = load_data()
    y = train[TARGET].values
    oof = {m: np.load(run / f"oof_{m}_perseed_pass1.npy") for m in MEMBERS}
    n = oof[MEMBERS[0]].shape[0]
    idx = list(range(n)) if seeds is None else seeds
    res = {}
    for s in idx + ["avg"]:
        raw = {m: (o.mean(axis=0) if s == "avg" else o[s]) for m, o in oof.items()}
        res[s] = {}
        for meth in methods:
            auc, ws = nested(meth, raw, y)
            res[s][meth] = {"auc": auc, "w": ws[0]}
            print(f"seed {s} {meth:9s} {auc:.5f}  Δ {auc - res[s][methods[0]]['auc']:+.5f}  w_fold0 {ws[0]}",
                  flush=True)
    (run / "v32_blend_methods.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
