"""Validación de selección final: Repeated StratifiedKFold pareado sobre corridas de src.train.

Lee los OOF por seed que guarda src.train (outputs/run_*/oof_{modelo}_perseed_pass1.npy).
Cada seed es un StratifiedKFold distinto, así que n_seeds x n_folds = RSKF.

  python -m src.validate outputs/run_A               # resumen de una corrida
  python -m src.validate outputs/run_A outputs/run_B # comparación pareada A vs B (mismas seeds)

Reporta:
  - AUC por modelo y por blend: media ± std entre seeds (variabilidad por partición de folds).
  - Blend honesto: pesos ajustados con CV anidado sobre filas (los pesos nunca ven las filas
    con las que se evalúan) vs pesos in-sample (como reporta src.train) vs pesos iguales.
  - Pareado: diferencia por seed B - A; mismo split en ambas corridas -> se cancela el ruido
    de partición y queda el efecto del cambio.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.blend import hill_climb_weights
from src.config import TARGET
from src.data import load_data

MODELS = ["lgbm", "cb", "xgb"]
NESTED_FOLDS = 5
NESTED_SEED = 2026


def load_run(run_dir: Path):
    oof = {m: np.load(run_dir / f"oof_{m}_perseed_pass1.npy") for m in MODELS}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return oof, metrics


def blend(mats, weights):
    return sum(w * mats[m] for m, w in weights.items())


def nested_blend_auc(oof_avg, y):
    """AUC del blend con pesos elegidos fuera de la fila evaluada (CV anidado sobre filas)."""
    pred = np.zeros(len(y))
    fold_w = []
    skf = StratifiedKFold(n_splits=NESTED_FOLDS, shuffle=True, random_state=NESTED_SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        w, _ = hill_climb_weights({m: o[tr] for m, o in oof_avg.items()}, y[tr])
        pred[va] = blend({m: o[va] for m, o in oof_avg.items()}, w)
        fold_w.append(w)
    return roc_auc_score(y, pred), fold_w


def summarize(oof, y, weights):
    n_seeds = next(iter(oof.values())).shape[0]
    eq = {m: 1 / len(MODELS) for m in MODELS}
    rows = {}
    for m in MODELS:
        rows[m] = [roc_auc_score(y, oof[m][s]) for s in range(n_seeds)]
    rows["blend_w"] = [roc_auc_score(y, blend({m: oof[m][s] for m in MODELS}, weights)) for s in range(n_seeds)]
    rows["blend_eq"] = [roc_auc_score(y, blend({m: oof[m][s] for m in MODELS}, eq)) for s in range(n_seeds)]
    return rows


def fmt(v):
    v = np.asarray(v)
    return f"{v.mean():.5f} ± {v.std(ddof=1) if len(v) > 1 else 0:.5f}"


def report_run(name, oof, metrics, y):
    w = metrics["weights"]
    n_seeds = next(iter(oof.values())).shape[0]
    print(f"== {name}: seeds {metrics['seeds']} x {metrics['n_folds']} folds (pseudo_rows={metrics['pseudo_rows']})")
    rows = summarize(oof, y, w)
    print("  por seed (media ± std entre seeds):")
    for k, v in rows.items():
        print(f"    {k:9s} {fmt(v)}   {[round(x, 5) for x in v]}")
    avg = {m: o.mean(axis=0) for m, o in oof.items()}
    eq = {m: 1 / len(MODELS) for m in MODELS}
    _, in_auc = hill_climb_weights(avg, y)
    nested, fold_w = nested_blend_auc(avg, y)
    print(f"  seed-avg ({n_seeds} seeds):")
    for m in MODELS:
        print(f"    {m:9s} {roc_auc_score(y, avg[m]):.5f}")
    print(f"    blend pesos in-sample   {in_auc:.5f}  (lo que reporta src.train, optimista)")
    print(f"    blend pesos anidados    {nested:.5f}  (honesto)")
    print(f"    blend pesos iguales     {roc_auc_score(y, blend(avg, eq)):.5f}")
    print(f"    pesos por fold anidado: {fold_w}")
    return rows, {"in_sample": in_auc, "nested": nested, "equal": roc_auc_score(y, blend(avg, eq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    train, _, _ = load_data()
    y = train[TARGET].values

    loaded = [(r, *load_run(r)) for r in args.runs]
    results = [report_run(r.name, oof, met, y) for r, oof, met in loaded]

    if len(loaded) == 2:
        (ra, _, ma), (rb, _, mb) = loaded
        if ma["seeds"] != mb["seeds"] or ma["n_folds"] != mb["n_folds"]:
            raise SystemExit("Comparación pareada requiere mismas seeds y folds en ambas corridas.")
        print(f"== pareado {rb.name} - {ra.name}")
        for k in results[0][0]:
            d = np.array(results[1][0][k]) - np.array(results[0][0][k])
            print(f"    {k:9s} Δ {fmt(d)}   B gana {int((d > 0).sum())}/{len(d)} seeds")
        for k in ("nested", "equal"):
            print(f"    seed-avg blend {k:6s} Δ {results[1][1][k] - results[0][1][k]:+.5f}")


if __name__ == "__main__":
    main()
