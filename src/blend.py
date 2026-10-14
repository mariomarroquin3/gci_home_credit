"""Hill-climbing de pesos sobre OOF (ensemble selection, Caruana et al. 2004)."""
import itertools

import numpy as np
from sklearn.metrics import roc_auc_score


def fast_auc(y, p):
    """AUC por ranks (sin corrección de empates; preds continuas). ~4x más rápido que sklearn."""
    r = np.empty(len(p))
    r[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def hill_climb_weights(oof_dict, y, grid=20, fine_step=0.01, score_fn=roc_auc_score):
    """Pesos en simplex que maximizan AUC del blend sobre OOF. Determinista."""
    names = list(oof_dict)
    mats = [oof_dict[k] for k in names]

    def score(w):
        return score_fn(y, sum(wi * m for wi, m in zip(w, mats)))

    best_s, best_w = -1.0, None
    for combo in itertools.product(range(grid + 1), repeat=len(names)):
        if sum(combo) != grid:
            continue
        w = tuple(c / grid for c in combo)
        s = score(w)
        if s > best_s:
            best_s, best_w = s, list(w)
    w, s = best_w, best_s
    improved = True
    while improved:
        improved = False
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                for d in (fine_step, -fine_step):
                    w2 = list(w)
                    w2[i] += d
                    w2[j] -= d
                    if min(w2) < 0:
                        continue
                    s2 = score(w2)
                    if s2 > s + 1e-7:
                        w, s, improved = w2, s2, True
    return dict(zip(names, [round(x, 4) for x in w])), s


def apply_weights(test_dict, weights):
    return sum(w * test_dict[k] for k, w in weights.items())
