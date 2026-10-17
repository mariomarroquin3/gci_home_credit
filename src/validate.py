"""Validación de selección final: Repeated StratifiedKFold pareado sobre corridas de src.train.

Lee los OOF por seed que guarda src.train (outputs/run_*/oof_{modelo}_perseed_pass1.npy).
Cada seed es un StratifiedKFold distinto, así que n_seeds x n_folds = RSKF.

  python -m src.validate outputs/run_A               # resumen de una corrida
  python -m src.validate outputs/run_A outputs/run_B # comparación pareada A vs B (mismas seeds)
  python -m src.validate outputs/run_A --add lgbm_et # ¿mejora el blend un miembro extra? (src.variants)

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

from src.blend import fast_auc, hill_climb_weights
from src.config import TARGET
from src.data import load_data

MODELS = ["lgbm", "cb", "xgb"]  # base; una corrida con EXTRA_MODELS trae más miembros en metrics["weights"]
NESTED_FOLDS = 5
NESTED_SEED = 2026


def load_run(run_dir: Path):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    members = list(dict.fromkeys(MODELS + list(metrics["weights"])))
    oof = {m: np.load(run_dir / f"oof_{m}_perseed_pass1.npy") for m in members}
    return oof, metrics


def blend(mats, weights):
    return sum(w * mats[m] for m, w in weights.items())


def nested_blend_auc(oof_avg, y):
    """AUC del blend con pesos elegidos fuera de la fila evaluada (CV anidado sobre filas)."""
    pred = np.zeros(len(y))
    fold_w = []
    skf = StratifiedKFold(n_splits=NESTED_FOLDS, shuffle=True, random_state=NESTED_SEED)
    for tr, va in skf.split(np.zeros(len(y)), y):
        w, _ = hill_climb_weights({m: o[tr] for m, o in oof_avg.items()}, y[tr], score_fn=fast_auc)
        pred[va] = blend({m: o[va] for m, o in oof_avg.items()}, w)
        fold_w.append(w)
    return roc_auc_score(y, pred), fold_w


def nested_auc_members(oof, y, members, s=None):
    """Blend anidado de `members` usando la seed s (None = seed-avg)."""
    mats = {m: (oof[m][s] if s is not None else oof[m].mean(axis=0)) for m in members}
    pred = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=NESTED_FOLDS, shuffle=True, random_state=NESTED_SEED)
    ws = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        w, _ = hill_climb_weights({m: o[tr] for m, o in mats.items()}, y[tr], score_fn=fast_auc)
        pred[va] = blend({m: o[va] for m, o in mats.items()}, w)
        ws.append(w)
    return roc_auc_score(y, pred), ws


def report_add(run_dir, y, candidates):
    """Pareado por seed: blend anidado base vs base + candidato. Criterio: gana en todas las seeds."""
    oof, _ = load_run(run_dir)
    n_seeds = oof[MODELS[0]].shape[0]
    base = [nested_auc_members(oof, y, MODELS, s)[0] for s in range(n_seeds)]
    base_avg = nested_auc_members(oof, y, MODELS)[0]
    print(f"== base {MODELS}: blend anidado por seed {[round(b, 5) for b in base]}  seed-avg {base_avg:.5f}")
    out = {}
    for group in candidates:
        extra = group.split("+")
        for c in extra:
            oof[c] = np.load(run_dir / f"oof_{c}_perseed_pass1.npy")
        solo = [roc_auc_score(y, oof[extra[0]][s]) for s in range(n_seeds)]
        corr = np.mean([np.corrcoef(oof[extra[0]][s], oof["xgb"][s])[0, 1] for s in range(n_seeds)])
        new = [nested_auc_members(oof, y, MODELS + extra, s)[0] for s in range(n_seeds)]
        new_avg, ws = nested_auc_members(oof, y, MODELS + extra)
        d = np.array(new) - np.array(base)
        wins = int((d > 0).sum())
        print(f"  +{group:20s} solo {fmt(solo)} corr_xgb {corr:.4f} | Δ por seed {[round(float(x), 5) for x in d]} "
              f"gana {wins}/{n_seeds} | seed-avg {new_avg:.5f} (Δ {new_avg - base_avg:+.5f}) "
              f"peso medio {np.mean([w[extra[-1]] for w in ws]):.2f}")
        out[group] = {"delta_seeds": d.round(6).tolist(), "wins": wins, "seed_avg": new_avg,
                      "delta_seed_avg": new_avg - base_avg}
    return out


def summarize(oof, y, weights):
    n_seeds = next(iter(oof.values())).shape[0]
    members = list(oof)
    eq = {m: 1 / len(members) for m in members}
    rows = {}
    for m in members:
        rows[m] = [roc_auc_score(y, oof[m][s]) for s in range(n_seeds)]
    rows["blend_w"] = [roc_auc_score(y, blend({m: oof[m][s] for m in members}, weights)) for s in range(n_seeds)]
    rows["blend_eq"] = [roc_auc_score(y, blend({m: oof[m][s] for m in members}, eq)) for s in range(n_seeds)]
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
    eq = {m: 1 / len(avg) for m in avg}
    _, in_auc = hill_climb_weights(avg, y, score_fn=fast_auc)
    nested, fold_w = nested_blend_auc(avg, y)
    print(f"  seed-avg ({n_seeds} seeds):")
    for m in avg:
        print(f"    {m:9s} {roc_auc_score(y, avg[m]):.5f}")
    print(f"    blend pesos in-sample   {in_auc:.5f}  (lo que reporta src.train, optimista)")
    print(f"    blend pesos anidados    {nested:.5f}  (honesto)")
    print(f"    blend pesos iguales     {roc_auc_score(y, blend(avg, eq)):.5f}")
    print(f"    pesos por fold anidado: {fold_w}")
    return rows, {"in_sample": in_auc, "nested": nested, "equal": roc_auc_score(y, blend(avg, eq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--add", type=str, default="",
                    help="candidatos separados por coma; 'a+b' prueba a y b juntos")
    args = ap.parse_args()
    train, _, _ = load_data()
    y = train[TARGET].values
    if args.add:
        report_add(args.runs[0], y, args.add.split(","))
        return

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
