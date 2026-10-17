"""Variantes GBDT descorrelacionadas como miembros extra del blend (exp v30).

Mismos folds que src.models (StratifiedKFold por seed) -> OOF por seed comparable y pareado.
Guarda oof_{nombre}_perseed_pass1.npy / test_{nombre}_perseed.npy en el directorio de una
corrida de src.train, para evaluarlas con `python -m src.validate <run> --add <nombre>`.

  python -m src.variants outputs/run_X                     # todas las variantes
  python -m src.variants outputs/run_X --only lgbm_et,xgb_d2

Variantes (idea: otro sesgo inductivo, no otro óptimo del mismo modelo):
  lgbm_dart    boosting DART, rondas fijas (sin early stopping)
  lgbm_et      extra_trees (cortes aleatorios), hojas grandes, feature_fraction bajo
  lgbm_et2     extra_trees con los params Optuna de LGBM
  lgbm_linear  linear_tree (regresión lineal en cada hoja)
  xgb_d2       XGBoost con max_depth=2 (interacciones de orden 2 solamente)
  lgbm_init    LightGBM con init_score = logit OOF de una regresión logística (folds anidados)
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from src.config import ES_ROUNDS, ID_COL, LGBM_PARAMS, MAX_ROUNDS, TARGET, XGB_PARAMS
from src.data import load_data
from src.features import build
from src.models import _folds, _monotone

LGBM_VARIANTS = {
    "lgbm_dart": dict(boosting="dart", learning_rate=0.05, drop_rate=0.1, skip_drop=0.5, max_drop=50),
    "lgbm_et": dict(extra_trees=True, num_leaves=127, max_depth=-1, feature_fraction=0.3,
                    learning_rate=0.02, min_child_samples=100),
    "lgbm_et2": dict(extra_trees=True),  # mismos params Optuna, solo cortes aleatorios
    "lgbm_linear": dict(linear_tree=True, linear_lambda=1.0, num_leaves=15, max_depth=-1,
                        learning_rate=0.02),
    "lgbm_init": dict(learning_rate=0.01),
}
DART_ROUNDS = 1000  # mejor ronda 964 en split aparte (seed 999): experiments/v30_gbdt_variants/dart_rounds.py
XGB_VARIANTS = {
    "xgb_d2": dict(max_depth=2, learning_rate=0.02),
}
ALL = list(LGBM_VARIANTS) + list(XGB_VARIANTS)


def _logreg(cat_cols, num_cols, seed):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              QuantileTransformer(output_distribution="normal", subsample=200_000,
                                                  random_state=seed)), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat_cols),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.1, max_iter=2000))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logreg_margins(X, y, X_va, X_test, cat_cols, seed, inner_folds=5):
    """Logit de LogReg: OOF interno para train (anidado), modelo completo para val/test."""
    num_cols = [c for c in X.columns if c not in cat_cols]
    Xs = X.copy()
    for c in cat_cols:  # OneHotEncoder no acepta Categorical con NaN mezclado
        Xs[c] = Xs[c].astype("object").where(Xs[c].notna(), "MISSING")
    tr_m = np.zeros(len(X))
    for itr, iva in StratifiedKFold(inner_folds, shuffle=True, random_state=seed).split(Xs, y):
        m = _logreg(cat_cols, num_cols, seed).fit(Xs.iloc[itr], y[itr])
        tr_m[iva] = _logit(m.predict_proba(Xs.iloc[iva])[:, 1])
    m = _logreg(cat_cols, num_cols, seed).fit(Xs, y)

    def prep(df):
        df = df.copy()
        for c in cat_cols:
            df[c] = df[c].astype("object").where(df[c].notna(), "MISSING")
        return _logit(m.predict_proba(df)[:, 1])

    return tr_m, prep(X_va), prep(X_test)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run_lgbm_variant(name, X, y, X_test, cat_cols, seeds, n_folds):
    oof = np.zeros((len(seeds), len(X)))
    tpred = np.zeros((len(seeds), len(X_test)))
    over = LGBM_VARIANTS[name]
    for si, seed in enumerate(seeds):
        for tr, va in _folds(y, n_folds, seed):
            params = dict(LGBM_PARAMS, **over, seed=seed)
            if not over.get("linear_tree"):  # linear_tree no soporta monotonía
                params.update(monotone_constraints=_monotone(X.columns),
                              monotone_constraints_method="advanced")
            init_tr = init_va = init_te = None
            if name == "lgbm_init":
                init_tr, init_va, init_te = logreg_margins(X.iloc[tr], y[tr], X.iloc[va], X_test,
                                                           cat_cols, seed)
            dtr = lgb.Dataset(X.iloc[tr], y[tr], init_score=init_tr)
            if over.get("boosting") == "dart":
                m = lgb.train(params, dtr, num_boost_round=DART_ROUNDS)
                it = None
            else:
                m = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS["lgbm"],
                              valid_sets=[lgb.Dataset(X.iloc[va], y[va], init_score=init_va)],
                              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
                it = m.best_iteration
            if name == "lgbm_init":
                oof[si, va] = _sigmoid(m.predict(X.iloc[va], num_iteration=it, raw_score=True) + init_va)
                tpred[si] += _sigmoid(m.predict(X_test, num_iteration=it, raw_score=True) + init_te) / n_folds
            else:
                oof[si, va] = m.predict(X.iloc[va], num_iteration=it)
                tpred[si] += m.predict(X_test, num_iteration=it) / n_folds
        print(f"  {name} seed {seed}: {roc_auc_score(y, oof[si]):.5f}", flush=True)
    return oof, tpred


def run_xgb_variant(name, X, y, X_test, cat_cols, seeds, n_folds):
    Xd = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    Xd_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True).reindex(columns=Xd.columns, fill_value=0)
    mono = "(" + ",".join(map(str, _monotone(Xd.columns))) + ")"
    dtest = xgb.DMatrix(Xd_test)
    oof = np.zeros((len(seeds), len(X)))
    tpred = np.zeros((len(seeds), len(X_test)))
    for si, seed in enumerate(seeds):
        for tr, va in _folds(y, n_folds, seed):
            dva = xgb.DMatrix(Xd.iloc[va], y[va])
            m = xgb.train(dict(XGB_PARAMS, **XGB_VARIANTS[name], seed=seed, monotone_constraints=mono),
                          xgb.DMatrix(Xd.iloc[tr], y[tr]), num_boost_round=MAX_ROUNDS["xgb"],
                          evals=[(dva, "va")], early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
            rng = (0, m.best_iteration + 1)
            oof[si, va] = m.predict(dva, iteration_range=rng)
            tpred[si] += m.predict(dtest, iteration_range=rng) / n_folds
        print(f"  {name} seed {seed}: {roc_auc_score(y, oof[si]):.5f}", flush=True)
    return oof, tpred


def run_variant(name, X, y, X_test, cat_cols, seeds, n_folds):
    fn = run_xgb_variant if name in XGB_VARIANTS else run_lgbm_variant
    return fn(name, X, y, X_test, cat_cols, seeds, n_folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="corrida de src.train (define seeds y folds)")
    ap.add_argument("--only", type=str, default=",".join(ALL))
    args = ap.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    seeds, n_folds = metrics["seeds"], metrics["n_folds"]

    train, test, _ = load_data()
    train, test, cat_cols = build(train, test)
    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X, y, X_test = train[features], train[TARGET].values, test[features]
    for name in args.only.split(","):
        print(f"== {name} ({len(seeds)} seeds x {n_folds} folds)", flush=True)
        oof, tpred = run_variant(name, X, y, X_test, cat_cols, seeds, n_folds)
        np.save(args.run / f"oof_{name}_perseed_pass1.npy", oof)
        np.save(args.run / f"test_{name}_perseed.npy", tpred)


if __name__ == "__main__":
    main()
