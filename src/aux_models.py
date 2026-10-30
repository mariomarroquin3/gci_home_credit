"""Modelos auxiliares sin target (exp v35): predicción y residuo de variables clave.

Una regresión LGBM con OOF sobre train+test (sin TARGET) predice cada variable a partir del
resto. La predicción imputa EXT_SOURCE faltantes (EXT_SOURCE_1 falta en ~56% de filas) y el
residuo mide cuánto se aleja el cliente de lo esperado para su perfil.
Screening LGBM (2 seeds): EXT +0.0015; EXT + residuos de ingreso/anualidad/empleo/precio +0.0019.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
AUX_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
                  deterministic=True, force_row_wise=True)
AUX_ROUNDS = 600


def oof_regress(X, target, folds=5, seed=0):
    """Predicción OOF de target (NaN = sin dato) a partir de X; filas sin dato: media de los folds."""
    has = ~np.isnan(target)
    idx, miss = np.where(has)[0], np.where(~has)[0]
    pred = np.full(len(target), np.nan)
    fill = np.zeros(len(miss))
    for a, b in KFold(folds, shuffle=True, random_state=seed).split(idx):
        m = lgb.train(AUX_PARAMS, lgb.Dataset(X.iloc[idx[a]], target[idx[a]]), AUX_ROUNDS)
        pred[idx[b]] = m.predict(X.iloc[idx[b]])
        if len(miss):
            fill += m.predict(X.iloc[miss]) / folds
    pred[miss] = fill
    return pred


def add_aux(X: pd.DataFrame) -> pd.DataFrame:
    """X = features de build (train+test concatenados, sin ID ni TARGET). Devuelve columnas nuevas."""
    f = pd.DataFrame(index=X.index)
    no_ext = X.drop(columns=[c for c in X.columns if c.startswith("EXT_SOURCE")])
    for c in EXT:
        p = oof_regress(no_ext, X[c].values.astype(float))
        f[f"{c}_PRED"] = p
        f[f"{c}_RES"] = X[c].values - p
    rate = [c for c in X.columns if c.startswith(("RATE_", "TERM_", "CA_R", "CREDIT_ANNUITY"))]
    targets = {  # variable -> (valor, columnas que la contienen y se excluyen del predictor)
        "AMT_INCOME_TOTAL": (np.log(X.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL", "ANNUITY_INCOME", "CREDIT_INCOME",
                                                          "INCOME_PER_FAM"]),
        "AMT_ANNUITY": (np.log(X.AMT_ANNUITY), ["AMT_ANNUITY", "ANNUITY_INCOME"] + rate),
        "DAYS_EMPLOYED": (X.DAYS_EMPLOYED, ["DAYS_EMPLOYED", "EMPLOYED_YEARS", "EMPLOYED_BIRTH_PCT", "kb_n", "kc_n"]),
        "AMT_GOODS_PRICE": (np.log(X.AMT_GOODS_PRICE), ["AMT_GOODS_PRICE", "CREDIT_GOODS", "DOWNPAYMENT"]),
    }
    for c, (v, drop) in targets.items():
        v = np.asarray(v, float)
        f[f"{c}_RES"] = v - oof_regress(X.drop(columns=drop), v)
    return f
