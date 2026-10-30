"""v34: señal estructural. Clientes repetidos (claves invariantes al tiempo) + tasa implícita.

DAYS_* son relativos a la fecha de solicitud, así que DAYS_BIRTH - DAYS_ID_PUBLISH y
DAYS_BIRTH - DAYS_REGISTRATION son constantes para la misma persona en solicitudes distintas.
Features de grupo con target: solo labels del fold de train (sin fuga). Screening LGBM lr 0.03.
Uso: python -m experiments.v34_structure.screen [--seeds 42,43,44] [--sets base,dup,...]
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.blend import fast_auc
from src.config import ID_COL, LGBM_PARAMS, TARGET
from src.data import load_data
from src.features import build

KEYS = {"ka": ["CODE_GENDER", "B_ID", "B_REG"], "kb": ["CODE_GENDER", "B_REG", "B_EMP"],
        "kc": ["CODE_GENDER", "B_ID", "B_EMP"]}
TERMS = np.array([6, 10, 12, 18, 24, 30, 36, 42, 48, 54, 60])


def group_ids(al):
    al = al.copy()
    al["B_ID"] = al.DAYS_BIRTH - al.DAYS_ID_PUBLISH
    al["B_REG"] = al.DAYS_BIRTH - al.DAYS_REGISTRATION
    al["B_EMP"] = (al.DAYS_BIRTH - al.DAYS_EMPLOYED).where(al.DAYS_EMPLOYED != 365243)
    out = {}
    for k, cols in KEYS.items():
        g = al.groupby(cols, dropna=True, observed=True).ngroup()
        g[al[cols].isna().any(axis=1).values] = -1
        out[k] = g.values
    return out


def static_group_feats(al, gids):
    """Sin target: tamaño de grupo, orden temporal y diferencias con el resto del grupo."""
    f = pd.DataFrame(index=al.index)
    for k, g in gids.items():
        s = pd.Series(g)
        size = s.map(s.value_counts()).to_numpy(copy=True)
        size[g < 0] = 1
        f[f"{k}_n"] = size
        if k == "ka":
            df = pd.DataFrame({"g": g, "t": -al.DAYS_BIRTH.values, "c": al.AMT_CREDIT.values})
            df.loc[g < 0, "g"] = -np.arange(1, (g < 0).sum() + 1) - 10**7
            grp = df.groupby("g")
            f["ka_rank"] = grp.t.rank().values  # orden de la solicitud (t = edad en días)
            f["ka_dt_first"] = (df.t - grp.t.transform("min")).values
            f["ka_dt_last"] = (grp.t.transform("max") - df.t).values
            f["ka_credit_rel"] = (df.c / grp.c.transform("mean")).values
    return f


def target_group_feats(gids, y_lab, lab_mask):
    """y_lab: target donde lab_mask (labels usables); devuelve suma/cuenta de OTROS miembros."""
    f = {}
    for k, g in gids.items():
        valid = g >= 0
        yl = np.where(lab_mask, y_lab, 0.0)
        cnt = pd.Series(lab_mask.astype(float)).groupby(g).transform("sum").values
        sm = pd.Series(yl).groupby(g).transform("sum").values
        cnt_o = cnt - lab_mask.astype(float)
        sm_o = sm - yl
        cnt_o[~valid] = 0
        sm_o[~valid] = 0
        f[f"{k}_nlab"] = cnt_o
        f[f"{k}_ypos"] = sm_o
        f[f"{k}_ymean"] = np.where(cnt_o > 0, sm_o / np.maximum(cnt_o, 1), np.nan)
    return pd.DataFrame(f)


def rate_feats(df):
    """Tasa mensual implícita para cada plazo estándar; plazo más plausible = tasa en rango típico."""
    c, a = df.AMT_CREDIT.values, df.AMT_ANNUITY.values
    ratio = c / a  # = (1-(1+r)^-n)/r
    rates = np.full((len(df), len(TERMS)), np.nan)
    for j, n in enumerate(TERMS):
        lo, hi = np.full(len(df), 1e-6), np.full(len(df), 0.2)
        for _ in range(50):
            mid = (lo + hi) / 2
            pv = (1 - (1 + mid) ** -n) / mid
            lo = np.where(pv > ratio, mid, lo)
            hi = np.where(pv > ratio, hi, mid)
        r = (lo + hi) / 2
        r[ratio >= n] = np.nan  # plazo imposible (tasa negativa)
        rates[:, j] = r
    out = pd.DataFrame(index=df.index)
    # plazo mínimo factible y tasa con ese plazo; plazo cuya tasa anual cae más cerca de 25%
    feas = ~np.isnan(rates)
    jmin = np.where(feas.any(1), feas.argmax(1), -1)
    out["TERM_MIN"] = np.where(jmin >= 0, TERMS[np.clip(jmin, 0, None)], np.nan)
    out["RATE_AT_MIN"] = np.where(jmin >= 0, rates[np.arange(len(df)), np.clip(jmin, 0, None)], np.nan)
    d = np.abs(np.nan_to_num(rates, nan=9) * 12 - 0.25)
    jb = d.argmin(1)
    out["TERM_GUESS"] = TERMS[jb]
    out["RATE_GUESS"] = rates[np.arange(len(df)), jb]
    out["RATE_12"] = rates[:, 2]
    out["RATE_24"] = rates[:, 4]
    out["RATE_36"] = rates[:, 6]
    return out


def prod_static(df):
    """Producto implícito: el ratio crédito/anualidad repite valores exactos por (plazo, tasa)."""
    r = (df.AMT_CREDIT / df.AMT_ANNUITY)
    out = pd.DataFrame(index=df.index)
    for d in (3, 2, 1):
        k = r.round(d)
        out[f"CA_R{d}_CNT"] = k.map(k.value_counts()).values
    return out


FREQ_COLS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL", "EXT_SOURCE_1",
             "EXT_SOURCE_2", "EXT_SOURCE_3", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH",
             "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"]


def freq_feats(df):
    out = pd.DataFrame(index=df.index)
    for c in FREQ_COLS:
        out[f"{c}_CNT"] = df[c].map(df[c].value_counts()).values
    k = (df.AMT_CREDIT / df.AMT_GOODS_PRICE).round(4)
    out["CG_R4_CNT"] = k.map(k.value_counts()).values
    k = df.AMT_CREDIT.astype(str) + "_" + df.AMT_ANNUITY.astype(str)
    out["CA_PAIR_CNT"] = k.map(k.value_counts()).values
    return out


def twin_feats(df, gids):
    """Media de otros miembros del grupo ka (sin target): completa EXT_SOURCE faltantes del gemelo."""
    g = gids["ka"].copy()
    g[g < 0] = -np.arange(1, (g < 0).sum() + 1) - 10**7
    out = pd.DataFrame(index=df.index)
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_INCOME_TOTAL"]:
        v = df[c].values
        has = ~np.isnan(v)
        sm = pd.Series(np.where(has, v, 0)).groupby(g).transform("sum").values - np.where(has, v, 0)
        ct = pd.Series(has.astype(float)).groupby(g).transform("sum").values - has
        out[f"TW_{c}"] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
        out[f"{c}_FILL"] = df[c].fillna(out[f"TW_{c}"]).values
    return out


def te_fold(key, y_all, lab, tr_idx, seed, m=30, inner=5):
    """TE suavizado: filas de train con OOF interno; resto con todo el fold de train."""
    key = pd.Series(key)
    prior = y_all[lab].mean()
    def enc(fit_idx, apply_idx):
        st = pd.DataFrame({"k": key.values[fit_idx], "y": y_all[fit_idx]}).groupby("k").y.agg(["sum", "count"])
        s = key.iloc[apply_idx].map(st["sum"]).fillna(0).values
        c = key.iloc[apply_idx].map(st["count"]).fillna(0).values
        return (s + m * prior) / (c + m)
    out = np.empty(len(key))
    rest = np.setdiff1d(np.arange(len(key)), tr_idx)
    out[rest] = enc(tr_idx, rest)
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(tr_idx, y_all[tr_idx]):
        out[tr_idx[b]] = enc(tr_idx[a], tr_idx[b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--sets", default="base,static,dup,rate,all")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    train, test, _ = load_data()
    n = len(train)
    raw = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    gids = group_ids(raw)
    stat = static_group_feats(raw, gids)
    rate = rate_feats(raw)
    prod = prod_static(raw)
    freq = freq_feats(raw)
    twin = twin_feats(raw, gids)
    ca3 = (raw.AMT_CREDIT / raw.AMT_ANNUITY).round(3).values
    trb, teb, cats = build(train, test, aux=False)
    feats = [c for c in trb.columns if c not in (ID_COL, TARGET)]
    X0 = pd.concat([trb[feats], teb[feats]], ignore_index=True)
    y = train[TARGET].values
    y_all = np.r_[y, np.zeros(len(test))]
    params = dict(LGBM_PARAMS, learning_rate=0.03)
    res = {s: [] for s in args.sets.split(",")}
    for seed in map(int, args.seeds.split(",")):
        oof = {s: np.zeros(n) for s in res}
        for tr, va in StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(np.zeros(n), y):
            lab = np.zeros(len(X0), bool)
            lab[tr] = True
            tg = target_group_feats(gids, y_all, lab)
            pte = pd.DataFrame({"CA_R3_TE": te_fold(ca3, y_all, lab, tr, seed)})
            for s in res:
                parts = [X0]
                if s.startswith("all"): parts += [stat, tg, rate]
                if s in ("allp", "allpt"): parts.append(prod)
                if s in ("allpt", "allptf"): parts.append(pte)
                if s == "allptf": parts += [prod, freq]
                if s == "allptt": parts += [prod, pte, twin]
                if s == "lf": parts += [stat, rate, prod]
                if s == "lft": parts += [stat, rate, prod, twin]
                if s in ("static", "all"): parts.append(stat)
                if s in ("dup", "all"): parts += [stat, tg]
                if s in ("rate", "all"): parts.append(rate)
                if s == "dupy": parts.append(tg)
                X = pd.concat(parts, axis=1)
                X = X.loc[:, ~X.columns.duplicated()]
                m = lgb.train(dict(params, seed=seed), lgb.Dataset(X.iloc[tr], y[tr]), 20000,
                              valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                oof[s][va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        for s in res:
            res[s].append(fast_auc(y, oof[s]))
            print(seed, s, f"{res[s][-1]:.5f}", flush=True)
    base = np.array(res.get("base", [np.nan]))
    for s, v in res.items():
        print(f"{s:8s} mean {np.mean(v):.5f}  delta vs base {np.mean(np.array(v) - base):+.5f}")


if __name__ == "__main__":
    main()
