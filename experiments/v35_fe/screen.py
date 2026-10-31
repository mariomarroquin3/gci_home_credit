"""v35: más señal sobre v34 (clientes repetidos, tasa implícita, dominio). Screening LGBM lr 0.03.

Base = features de producción (features.build + FoldFeatures). Cada set añade columnas.
Uso: python -m experiments.v35_fe.screen --seeds 42,43 --sets base,uf,gd,...
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.blend import fast_auc
from src.config import ID_COL, LGBM_PARAMS, TARGET
from src.data import load_data
from src.features import build
from src.structure import FoldFeatures, _singletons, group_ids


def union_groups(gids):
    """Componentes conexas sobre las claves ka/kb/kc (una persona aunque falte un campo)."""
    n = len(next(iter(gids.values())))
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for g in gids.values():
        s = pd.Series(np.arange(n))[g >= 0].groupby(g[g >= 0])
        for idx in s.apply(list):
            r0 = find(idx[0])
            for i in idx[1:]:
                ri = find(i)
                if ri != r0:
                    parent[ri] = r0
    root = np.array([find(i) for i in range(n)])
    sz = pd.Series(root).map(pd.Series(root).value_counts()).values
    return np.where(sz > 1, root, -1)


def group_static(raw, g, pre):
    gs = _singletons(g)
    t = -raw.DAYS_BIRTH.values
    df = pd.DataFrame({"g": gs, "t": t})
    f = pd.DataFrame(index=raw.index)
    s = pd.Series(gs)
    f[f"{pre}_n"] = s.map(s.value_counts()).values
    grp = df.groupby("g")
    f[f"{pre}_rank"] = grp.t.rank().values
    f[f"{pre}_dt_first"] = (df.t - grp.t.transform("min")).values
    f[f"{pre}_dt_last"] = (grp.t.transform("max") - df.t).values
    return f


DELTA_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL", "AMT_CREDIT",
              "AMT_ANNUITY", "AMT_REQ_CREDIT_BUREAU_YEAR", "CNT_CHILDREN", "REGION_RATING_CLIENT"]


def group_deltas(raw, g):
    """Diferencias con la solicitud anterior/siguiente del mismo cliente (sin target)."""
    gs = _singletons(g)
    order = np.lexsort((np.arange(len(raw)), -raw.DAYS_BIRTH.values, gs))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    gso = gs[order]
    prev_ok = np.r_[False, gso[1:] == gso[:-1]]
    next_ok = np.r_[gso[1:] == gso[:-1], False]
    f = pd.DataFrame(index=raw.index)
    t = (-raw.DAYS_BIRTH.values)[order]
    f["GD_dt_prev"] = np.where(prev_ok, t - np.r_[np.nan, t[:-1]], np.nan)[inv]
    f["GD_dt_next"] = np.where(next_ok, np.r_[t[1:], np.nan] - t, np.nan)[inv]
    ph = (raw.DAYS_BIRTH - raw.DAYS_LAST_PHONE_CHANGE).values[order]  # fecha absoluta del cambio
    f["GD_phone_same_prev"] = np.where(prev_ok, (ph == np.r_[np.nan, ph[:-1]]).astype(float), np.nan)[inv]
    for c in DELTA_COLS:
        v = raw[c].values.astype(float)[order]
        f[f"GD_{c}_dprev"] = np.where(prev_ok, v - np.r_[np.nan, v[:-1]], np.nan)[inv]
        f[f"GD_{c}_dnext"] = np.where(next_ok, np.r_[v[1:], np.nan] - v, np.nan)[inv]
    ct = pd.Series(raw.NAME_CONTRACT_TYPE.values[order])
    f["GD_type_prev_same"] = np.where(prev_ok, (ct.values == ct.shift(1).values).astype(float), np.nan)[inv]
    return f


class TimeTarget:
    """Target de otros miembros separado en anteriores / posteriores en el tiempo (por fold)."""

    def __init__(self, g, t, y_all):
        self.g, self.t, self.y = g, t, y_all

    def __call__(self, lab):
        g, t = self.g, self.t
        df = pd.DataFrame({"g": g, "t": t, "lab": lab.astype(float), "yl": np.where(lab, self.y, 0.0),
                           "i": np.arange(len(g))})
        df = df[g >= 0]
        out = {k: np.full(len(g), np.nan) for k in ["TT_prev_n", "TT_prev_y", "TT_next_n", "TT_next_y",
                                                     "TT_gap_pos"]}
        for k in ["TT_prev_n", "TT_prev_y", "TT_next_n", "TT_next_y"]:
            out[k][:] = 0
        m = df.merge(df, on="g", suffixes=("", "_o"))
        m = m[m.i != m.i_o]
        prev = m[m.t_o < m.t]
        nxt = m[m.t_o > m.t]
        for name, part in (("prev", prev), ("next", nxt)):
            a = part.groupby("i")[["lab_o", "yl_o"]].sum()
            out[f"TT_{name}_n"][a.index] = a.lab_o.values
            out[f"TT_{name}_y"][a.index] = a.yl_o.values
        pos = m[m.yl_o > 0]
        if len(pos):
            gp = (pos.t - pos.t_o).abs().groupby(pos.i).min()
            out["TT_gap_pos"][gp.index] = gp.values
        return pd.DataFrame(out)


def oof_regress(Xall, target, seed=0, folds=5, lr=0.05, rounds=600):
    """Predicción OOF (sobre train+test, sin TARGET) de una columna a partir del resto."""
    has = ~np.isnan(target)
    pred = np.full(len(target), np.nan)
    idx = np.where(has)[0]
    p = dict(objective="regression", learning_rate=lr, num_leaves=63, min_child_samples=100,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed,
             deterministic=True, force_row_wise=True)
    nofit = np.zeros(len(target))
    for a, b in KFold(folds, shuffle=True, random_state=seed).split(idx):
        m = lgb.train(p, lgb.Dataset(Xall.iloc[idx[a]], target[idx[a]]), rounds)
        pred[idx[b]] = m.predict(Xall.iloc[idx[b]])
        if (~has).any():
            nofit[~has] += m.predict(Xall.iloc[np.where(~has)[0]]) / folds
    pred[~has] = nofit[~has]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--sets", default="base,uf,gd,tt,ext,rr")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--aux", action="store_true", help="base con modelos auxiliares v35")
    args = ap.parse_args()
    t0 = time.time()
    train, test, _ = load_data()
    n = len(train)
    raw = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    trb, teb, cats = build(train, test, aux=args.aux)
    feats = [c for c in trb.columns if c not in (ID_COL, TARGET)]
    X0 = pd.concat([trb[feats], teb[feats]], ignore_index=True)
    y = train[TARGET].values
    y_all = np.r_[y, np.zeros(len(test))]
    ff = FoldFeatures(train, test, y)
    gids = group_ids(raw)
    uf = union_groups(gids)
    t = -raw.DAYS_BIRTH.values
    sets = args.sets.split(",")
    S = {}
    if any("uf" in s for s in sets):
        S["uf"] = group_static(raw, uf, "uf")
    if any("gd" in s for s in sets):
        S["gd"] = group_deltas(raw, uf)
    if any("ext" in s for s in sets):
        f = pd.DataFrame(index=raw.index)
        drop = [c for c in X0.columns if c.startswith("EXT_SOURCE") or "_y" in c]
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            p = oof_regress(X0.drop(columns=drop), raw[c].values.astype(float))
            f[f"{c}_PRED"] = p
            f[f"{c}_RES"] = raw[c].values - p
        S["ext"] = f
    EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    if any(k in ("extx", "extxp") for s in sets for k in s.split("+")):
        f = pd.DataFrame(index=raw.index)
        for c in EXT:
            p = oof_regress(X0.drop(columns=[c, f"{c}_ISNA"], errors="ignore"), raw[c].values.astype(float))
            f[f"{c}_PREDX"] = p
            f[f"{c}_RESX"] = raw[c].values - p
        S["extx"] = f
        S["extxp"] = f[[c for c in f.columns if "PREDX" in c]]
    if "ext" in S:
        S["extp"] = S["ext"][[c for c in S["ext"].columns if "PRED" in c]]
        S["extr"] = S["ext"][[c for c in S["ext"].columns if "RES" in c]]
    if any("res" in s.split("+") for s in sets):
        f = pd.DataFrame(index=raw.index)
        tg = {"AMT_INCOME_TOTAL": (np.log(raw.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL", "ANNUITY_INCOME", "CREDIT_INCOME", "INCOME_PER_FAM"]),
              "AMT_ANNUITY": (np.log(raw.AMT_ANNUITY), ["AMT_ANNUITY", "ANNUITY_INCOME"] + [c for c in X0.columns if c.startswith(("RATE_", "TERM_", "CA_R", "CREDIT_ANNUITY"))]),
              "DAYS_EMPLOYED": (X0.DAYS_EMPLOYED, ["DAYS_EMPLOYED", "EMPLOYED_YEARS", "EMPLOYED_BIRTH_PCT", "kb_n", "kc_n"]),
              "AMT_GOODS_PRICE": (np.log(raw.AMT_GOODS_PRICE), ["AMT_GOODS_PRICE", "CREDIT_GOODS", "DOWNPAYMENT"])}
        for c, (v, dr) in tg.items():
            v = np.asarray(v, float)
            p = oof_regress(X0.drop(columns=dr), v)
            f[f"{c}_RES"] = v - p
        S["res"] = f
    has = lambda k: any(k in s.split("+") for s in sets)
    AUXC = [c for c in X0.columns if c.endswith(("_PRED", "_RES"))]
    if has("res2"):
        f = pd.DataFrame(index=raw.index)
        cr = [c for c in X0.columns if c.startswith(("RATE_", "TERM_", "CA_R", "CREDIT_"))]
        tg = {"AMT_CREDIT": (np.log(raw.AMT_CREDIT), ["AMT_CREDIT", "DOWNPAYMENT", "ka_credit_rel"] + cr),
              "REGION_POPULATION_RELATIVE": (raw.REGION_POPULATION_RELATIVE, ["REGION_POPULATION_RELATIVE"]),
              "DAYS_REGISTRATION": (raw.DAYS_REGISTRATION, ["DAYS_REGISTRATION", "REGISTRATION_YEARS"]),
              "DAYS_ID_PUBLISH": (raw.DAYS_ID_PUBLISH, ["DAYS_ID_PUBLISH"]),
              "DAYS_LAST_PHONE_CHANGE": (raw.DAYS_LAST_PHONE_CHANGE, ["DAYS_LAST_PHONE_CHANGE", "PHONE_CHANGE_YEARS"]),
              "OWN_CAR_AGE": (raw.OWN_CAR_AGE, ["OWN_CAR_AGE", "OWN_CAR_AGE_ISNA"]),
              "AMT_REQ_CREDIT_BUREAU_YEAR": (raw.AMT_REQ_CREDIT_BUREAU_YEAR, ["AMT_REQ_CREDIT_BUREAU_YEAR", "BUREAU_REQ_TOTAL", "AMT_REQ_CREDIT_BUREAU_YEAR_ISNA"])}
        for c, (v, dr) in tg.items():
            v = np.asarray(v, float)
            f[f"{c}_RES2"] = v - oof_regress(X0.drop(columns=dr + AUXC), v)
        S["res2"] = f
    if has("nares"):
        f = pd.DataFrame(index=raw.index)
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "OCCUPATION_TYPE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
            v = raw[c].isna().values.astype(float)
            dr = [k for k in X0.columns if k.startswith(c)] + AUXC
            f[f"{c}_NARES"] = v - oof_regress(X0.drop(columns=dr), v)
        S["nares"] = f
    if has("extb"):
        f = pd.DataFrame(index=raw.index)
        Xn = X0.drop(columns=[c for c in X0.columns if c.startswith("EXT_SOURCE")] + AUXC)
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            p = oof_regress(Xn, raw[c].values.astype(float), folds=10, lr=0.02, rounds=2000)
            f[f"{c}_PREDB"] = p
            f[f"{c}_RESB"] = raw[c].values - p
        S["extb"] = f
    if has("extagg"):
        f = pd.DataFrame(index=raw.index)
        E = np.column_stack([X0[c].values for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]).astype(float)
        P = np.column_stack([X0[f"{c}_PRED"].values for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]).astype(float)
        F = np.where(np.isnan(E), P, E)
        f["EXT_MEAN"] = np.nanmean(E, 1)
        f["EXT_STD"] = np.nanstd(E, 1)
        f["EXT_PROD_F"] = F.prod(1)
        f["EXT_MEAN_F"] = F.mean(1)
        f["EXT_MIN_F"] = F.min(1)
        f["EXT_WMEAN_F"] = (2 * F[:, 0] + 3 * F[:, 1] + 4 * F[:, 2]) / 9
        for i, c in enumerate(["E1", "E2", "E3"]):
            f[f"{c}_FILL"] = F[:, i]
        f["EXT_F_x_CA"] = f["EXT_MEAN_F"] * X0["CREDIT_ANNUITY"].values
        f["EXT_F_x_AGE"] = f["EXT_MEAN_F"] * X0["AGE_YEARS"].values
        S["extagg"] = f
    if any("rr" in s for s in sets):
        f = pd.DataFrame(index=raw.index)
        rate_cols = [c for c in X0.columns if c.startswith(("RATE_", "TERM_", "CA_R", "CREDIT_ANNUITY"))]
        base_cols = [c for c in X0.columns if c not in rate_cols and c not in ("AMT_ANNUITY",)]
        p = oof_regress(X0[base_cols], X0["RATE_GUESS"].values.astype(float))
        f["RATE_PRED"] = p
        f["RATE_RES"] = X0["RATE_GUESS"].values - p
        f["RATE_RANK_TERM"] = X0.groupby(["TERM_GUESS", "NAME_CONTRACT_TYPE"], observed=True).RATE_GUESS.rank(pct=True).values
        S["rr"] = f
    print(f"prep {time.time()-t0:.0f}s", flush=True)
    tt = TimeTarget(uf, t, y_all)
    params = dict(LGBM_PARAMS, learning_rate=0.03)
    res = {s: [] for s in sets}
    for seed in map(int, args.seeds.split(",")):
        oof = {s: np.zeros(n) for s in res}
        for tr, va in StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(np.zeros(n), y):
            lab = np.zeros(len(X0), bool)
            lab[tr] = True
            F_tr, F_te = ff(tr, seed)
            F = pd.concat([F_tr, F_te], ignore_index=True)
            need_t = any("tt" in s or "uy" in s for s in sets)
            T = tt(lab) if need_t else None
            if any("uy" in s for s in sets):
                cnt = pd.Series(lab.astype(float)).groupby(uf).transform("sum").values - lab
                sm = pd.Series(np.where(lab, y_all, 0.)).groupby(uf).transform("sum").values - np.where(lab, y_all, 0.)
                cnt[uf < 0] = 0
                sm[uf < 0] = 0
                UY = pd.DataFrame({"uf_nlab": cnt, "uf_ypos": sm,
                                   "uf_ymean": np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)})
            for s in sets:
                parts = [X0, F]
                for k in s.split("+"):
                    if k in S:
                        parts.append(S[k])
                    elif k == "tt":
                        parts.append(T)
                    elif k == "uy":
                        parts.append(UY)
                X = pd.concat(parts, axis=1)
                X = X.loc[:, ~X.columns.duplicated()]
                m = lgb.train(dict(params, seed=seed), lgb.Dataset(X.iloc[tr], y[tr]), 20000,
                              valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                oof[s][va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        for s in res:
            res[s].append(fast_auc(y, oof[s]))
            print(seed, s, f"{res[s][-1]:.5f}", f"{time.time()-t0:.0f}s", flush=True)
    base = np.array(res.get("base", [np.nan]))
    for s, v in res.items():
        d = np.array(v) - base
        print(f"{s:16s} mean {np.mean(v):.5f}  delta {np.mean(d):+.5f}  ({' '.join(f'{x:+.5f}' for x in d)})")


if __name__ == "__main__":
    main()
