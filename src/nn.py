"""Redes tabulares como miembros extra del blend (exp v31): RealMLP y TabM (pytabkit).

Mismos folds que src.models (StratifiedKFold por seed) -> OOF por seed pareado con la base.
Guarda oof_{nombre}_perseed_pass1.npy / test_{nombre}_perseed.npy en el directorio de una
corrida de src.train, para evaluarlas con `python -m src.validate <run> --add <nombre>`.

  python -m src.nn outputs/run_X --only realmlp,tabm

Modelos (defaults tuneados por los autores, sin HPO propio):
  realmlp  RealMLP-TD (Holzmüller et al. 2024)
  tabm     TabM-D, k=32 miembros con pesos compartidos (Gorishniy et al. 2025)
Como los GBDT, la mejor época se elige sobre el fold de validación (métrica AUC).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import ID_COL, TARGET
from src.data import load_data
from src.features import build
from src.models import _folds

NN_MODELS = ["realmlp", "tabm"]


def prep_nn(X, X_test, cat_cols):
    """Sin NaN/inf: numéricas con mediana de train + indicador de nulo; categóricas 'MISSING'."""
    X, X_test = X.copy(), X_test.copy()
    num_cols = [c for c in X.columns if c not in cat_cols]
    for df in (X, X_test):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    for c in num_cols:
        if X[c].isna().any() or X_test[c].isna().any():
            med = X[c].median()
            if f"{c}_ISNA" not in X.columns:
                X[f"{c}_NA"], X_test[f"{c}_NA"] = X[c].isna().astype(float), X_test[c].isna().astype(float)
            X[c], X_test[c] = X[c].fillna(med), X_test[c].fillna(med)
    for df in (X, X_test):
        for c in cat_cols:
            df[c] = df[c].astype("object").where(df[c].notna(), "MISSING").astype(str)
        for c in df.columns:
            if c not in cat_cols:
                df[c] = df[c].astype(np.float32)
    return X, X_test


def make_model(name, seed):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
    common = dict(device="cpu", random_state=seed, n_threads=int(os.environ.get("NN_THREADS", 4)), val_metric_name="1-auc_ovr", verbosity=0)
    if name == "realmlp":
        return RealMLP_TD_Classifier(**common)
    if name == "tabm":
        return TabM_D_Classifier(**common)
    raise ValueError(name)


def run_nn(name, X, y, X_test, cat_cols, seeds, n_folds, ckpt_dir=None, only_seeds=None):
    """Checkpoint por fold en ckpt_dir (una corrida completa en CPU tarda horas): se reanuda."""
    oof = np.full((len(seeds), len(X)), np.nan)
    tpred = np.zeros((len(seeds), len(X_test)))
    for si, seed in enumerate(seeds):
        if only_seeds and seed not in only_seeds:
            continue
        for fi, (tr, va) in enumerate(_folds(y, n_folds, seed)):
            ck = ckpt_dir / f"{name}_s{seed}_f{fi}.npz" if ckpt_dir else None
            if ck and ck.exists():
                d = np.load(ck)
                oof[si, va], tp = d["va"], d["test"]
            else:
                torch.manual_seed(seed * 100 + fi)
                m = make_model(name, seed * 100 + fi)
                m.fit(X.iloc[tr], y[tr], X_val=X.iloc[va], y_val=y[va], cat_col_names=cat_cols)
                oof[si, va] = m.predict_proba(X.iloc[va])[:, 1]
                tp = m.predict_proba(X_test)[:, 1]
                if ck:
                    np.savez(ck, va=oof[si, va], test=tp)
            tpred[si] += tp / n_folds
            print(f"    {name} seed {seed} fold {fi}: {roc_auc_score(y[va], oof[si, va]):.5f}", flush=True)
        print(f"  {name} seed {seed}: {roc_auc_score(y, oof[si]):.5f}", flush=True)
    return oof, tpred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="corrida de src.train (define seeds y folds)")
    ap.add_argument("--only", type=str, default=",".join(NN_MODELS))
    ap.add_argument("--rows", type=int, default=0, help="submuestra para smoke test (0 = todo)")
    ap.add_argument("--seeds", type=str, default="", help="subconjunto de seeds de la corrida (screening)")
    args = ap.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    seeds, n_folds = metrics["seeds"], metrics["n_folds"]

    train, test, _ = load_data()
    if args.rows:
        train = train.sample(args.rows, random_state=0).reset_index(drop=True)
        test = test.head(1000)
    train, test, cat_cols = build(train, test)
    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    y = train[TARGET].values
    X, X_test = prep_nn(train[features], test[features], cat_cols)
    for name in args.only.split(","):
        print(f"== {name} ({len(seeds)} seeds x {n_folds} folds, {X.shape[1]} cols)", flush=True)
        ckpt = None if args.rows else args.run / "nn_ckpt"
        if ckpt:
            ckpt.mkdir(exist_ok=True)
        only = [int(x) for x in args.seeds.split(",") if x]
        oof, tpred = run_nn(name, X, y, X_test, cat_cols, seeds, n_folds, ckpt, only)
        if not args.rows and not only:
            np.save(args.run / f"oof_{name}_perseed_pass1.npy", oof)
            np.save(args.run / f"test_{name}_perseed.npy", tpred)


if __name__ == "__main__":
    main()
