"""Wrappers de entrenamiento por seed con early stopping y pseudo-rows opcionales.

Cada función devuelve (oof_por_seed, test_por_seed) con shape (n_seeds, n_rows).
El OOF es honesto: cada fila de train solo se predice con modelos que no la vieron;
las pseudo-filas (de test) nunca entran al OOF.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

from src.config import CB_PARAMS, ES_ROUNDS, LGBM_PARAMS, MAX_ROUNDS, XGB_PARAMS


def _folds(y, n_folds, seed):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y)


def run_lgbm(X, y, X_test, seeds, n_folds, X_pseudo=None, y_pseudo=None, save_dir=None):
    oof = np.zeros((len(seeds), len(X)))
    tpred = np.zeros((len(seeds), len(X_test)))
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(_folds(y, n_folds, seed)):
            X_tr, y_tr = X.iloc[tr], y[tr]
            if X_pseudo is not None:
                X_tr = pd.concat([X_tr, X_pseudo], ignore_index=True)
                y_tr = np.r_[y_tr, y_pseudo]
                for c in X_tr.select_dtypes(include=["object", "string"]).columns:
                    X_tr[c] = X_tr[c].astype("category")
            params = dict(LGBM_PARAMS, seed=seed)
            m = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=MAX_ROUNDS["lgbm"],
                          valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
            oof[si, va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            tpred[si] += m.predict(X_test, num_iteration=m.best_iteration) / n_folds
            if save_dir:
                m.save_model(str(save_dir / f"lgbm_s{seed}_f{fi}.txt"))
    return oof, tpred


def _cb_str(df, cat_cols):
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype("object").where(out[c].notna(), "MISSING").astype(str)
    return out


def run_cb(X, y, X_test, cat_cols, seeds, n_folds, X_pseudo=None, y_pseudo=None, save_dir=None):
    Xs = _cb_str(X, cat_cols)
    Xs_test = _cb_str(X_test, cat_cols)
    oof = np.zeros((len(seeds), len(X)))
    tpred = np.zeros((len(seeds), len(X_test)))
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(_folds(y, n_folds, seed)):
            X_tr, y_tr = Xs.iloc[tr], y[tr]
            if X_pseudo is not None:
                X_tr = pd.concat([X_tr, _cb_str(X_pseudo, cat_cols)], ignore_index=True)
                y_tr = np.r_[y_tr, y_pseudo]
            m = CatBoostClassifier(**CB_PARAMS, random_seed=seed, thread_count=-1)
            m.fit(Pool(X_tr, y_tr, cat_features=cat_cols),
                  eval_set=Pool(Xs.iloc[va], y[va], cat_features=cat_cols))
            oof[si, va] = m.predict_proba(Xs.iloc[va])[:, 1]
            tpred[si] += m.predict_proba(Xs_test)[:, 1] / n_folds
            if save_dir:
                m.save_model(str(save_dir / f"cb_s{seed}_f{fi}.cbm"))
    return oof, tpred


def run_xgb(X, y, X_test, cat_cols, seeds, n_folds, X_pseudo=None, y_pseudo=None, save_dir=None):
    Xd = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    Xd_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=True).reindex(columns=Xd.columns, fill_value=0)
    Xd_pseudo = Xd_test.iloc[X_pseudo.index].reset_index(drop=True) if X_pseudo is not None else None
    oof = np.zeros((len(seeds), len(X)))
    tpred = np.zeros((len(seeds), len(X_test)))
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(_folds(y, n_folds, seed)):
            X_tr, y_tr = Xd.iloc[tr], y[tr]
            if Xd_pseudo is not None:
                X_tr = pd.concat([X_tr, Xd_pseudo], ignore_index=True)
                y_tr = np.r_[y_tr, y_pseudo]
            m = xgb.train(dict(XGB_PARAMS, seed=seed), xgb.DMatrix(X_tr, y_tr),
                          num_boost_round=MAX_ROUNDS["xgb"],
                          evals=[(xgb.DMatrix(Xd.iloc[va], y[va]), "va")],
                          early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
            oof[si, va] = m.predict(xgb.DMatrix(Xd.iloc[va]), iteration_range=(0, m.best_iteration + 1))
            tpred[si] += m.predict(xgb.DMatrix(Xd_test), iteration_range=(0, m.best_iteration + 1)) / n_folds
            if save_dir:
                m.save_model(str(save_dir / f"xgb_s{seed}_f{fi}.json"))
    return oof, tpred
