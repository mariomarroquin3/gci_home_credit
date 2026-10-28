"""Señal estructural (exp v34): clientes repetidos, tasa implícita y producto.

- Clientes repetidos: DAYS_* son relativos a la fecha de solicitud, así que
  DAYS_BIRTH - DAYS_ID_PUBLISH / - DAYS_REGISTRATION / - DAYS_EMPLOYED son constantes para la
  misma persona en solicitudes distintas. ~3-4% de filas comparten clave (azar: ~1%).
- Tasa implícita: AMT_CREDIT / AMT_ANNUITY = (1 - (1+r)^-n) / r para plazos estándar n.
- Producto: el ratio crédito/anualidad repite valores exactos por (plazo, tasa) del producto.

Features sin target (add_structure) entran en features.build. Las que usan target
(FoldFeatures: target de otros miembros del grupo, TE del ratio) se calculan dentro de cada
fold solo con labels del fold de train; test usa las mismas labels del fold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

KEYS = {"ka": ["CODE_GENDER", "B_ID", "B_REG"], "kb": ["CODE_GENDER", "B_REG", "B_EMP"],
        "kc": ["CODE_GENDER", "B_ID", "B_EMP"]}
TERMS = np.array([6, 10, 12, 18, 24, 30, 36, 42, 48, 54, 60])
TE_SMOOTH = 30


def group_ids(raw: pd.DataFrame) -> dict:
    """Id de grupo por clave invariante (-1 si falta algún campo). raw = train+test sin limpiar."""
    al = pd.DataFrame({"CODE_GENDER": raw.CODE_GENDER.values,
                       "B_ID": (raw.DAYS_BIRTH - raw.DAYS_ID_PUBLISH).values,
                       "B_REG": (raw.DAYS_BIRTH - raw.DAYS_REGISTRATION).values,
                       "B_EMP": (raw.DAYS_BIRTH - raw.DAYS_EMPLOYED).where(raw.DAYS_EMPLOYED != 365243).values})
    out = {}
    for k, cols in KEYS.items():
        g = al.groupby(cols, dropna=True, observed=True).ngroup().to_numpy(copy=True)
        g[al[cols].isna().any(axis=1).values] = -1
        out[k] = g
    return out


def _singletons(g):
    g = g.copy()
    g[g < 0] = -np.arange(1, (g < 0).sum() + 1) - 10**7
    return g


def _rates(raw):
    ratio = (raw.AMT_CREDIT / raw.AMT_ANNUITY).values
    rates = np.full((len(raw), len(TERMS)), np.nan)
    for j, n in enumerate(TERMS):  # bisección: el valor presente decrece con la tasa
        lo, hi = np.full(len(raw), 1e-6), np.full(len(raw), 0.2)
        for _ in range(50):
            mid = (lo + hi) / 2
            above = (1 - (1 + mid) ** -n) / mid > ratio
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
        r = (lo + hi) / 2
        r[ratio >= n] = np.nan  # plazo imposible (tasa negativa)
        rates[:, j] = r
    return ratio, rates


def add_structure(raw: pd.DataFrame) -> pd.DataFrame:
    """Features sin target sobre train+test concatenados (mismo orden que raw)."""
    f = pd.DataFrame(index=raw.index)
    gids = group_ids(raw)
    for k, g in gids.items():
        s = pd.Series(g)
        size = s.map(s.value_counts()).to_numpy(copy=True)
        size[g < 0] = 1
        f[f"{k}_n"] = size
    df = pd.DataFrame({"g": _singletons(gids["ka"]), "t": -raw.DAYS_BIRTH.values, "c": raw.AMT_CREDIT.values})
    grp = df.groupby("g")
    f["ka_rank"] = grp.t.rank().values  # orden de la solicitud dentro del cliente
    f["ka_dt_first"] = (df.t - grp.t.transform("min")).values
    f["ka_dt_last"] = (grp.t.transform("max") - df.t).values
    f["ka_credit_rel"] = (df.c / grp.c.transform("mean")).values

    ratio, rates = _rates(raw)
    feas = ~np.isnan(rates)
    jmin = np.clip(feas.argmax(1), 0, None)
    has = feas.any(1)
    idx = np.arange(len(raw))
    f["TERM_MIN"] = np.where(has, TERMS[jmin], np.nan)
    f["RATE_AT_MIN"] = np.where(has, rates[idx, jmin], np.nan)
    jb = np.abs(np.nan_to_num(rates, nan=9) * 12 - 0.25).argmin(1)  # plazo con tasa anual ~25%
    f["TERM_GUESS"] = TERMS[jb]
    f["RATE_GUESS"] = rates[idx, jb]
    for n in (12, 24, 36):
        f[f"RATE_{n}"] = rates[:, list(TERMS).index(n)]
    for d in (3, 2, 1):
        k = pd.Series(ratio).round(d)
        f[f"CA_R{d}_CNT"] = k.map(k.value_counts()).values
    return f


class FoldFeatures:
    """Features con target por fold: fn(tr_idx, seed) -> (F_train, F_test)."""

    def __init__(self, train_raw: pd.DataFrame, test_raw: pd.DataFrame, y):
        raw = pd.concat([train_raw.drop(columns="TARGET", errors="ignore"), test_raw], ignore_index=True)
        self.n = len(train_raw)
        self.gids = group_ids(raw)
        self.ca3 = pd.Series((raw.AMT_CREDIT / raw.AMT_ANNUITY).round(3).values)
        self.y_all = np.r_[np.asarray(y, float), np.zeros(len(test_raw))]

    def _group(self, lab):
        f = {}
        yl = np.where(lab, self.y_all, 0.0)
        for k, g in self.gids.items():
            cnt = pd.Series(lab.astype(float)).groupby(g).transform("sum").values - lab
            sm = pd.Series(yl).groupby(g).transform("sum").values - yl
            cnt[g < 0] = 0
            sm[g < 0] = 0
            f[f"{k}_nlab"] = cnt
            f[f"{k}_ypos"] = sm
            f[f"{k}_ymean"] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        return pd.DataFrame(f)

    def _te(self, tr, seed, inner=5):
        """TE suavizado del ratio exacto: OOF interno para filas de train del fold."""
        prior = self.y_all[tr].mean()
        key = self.ca3

        def enc(fit_idx, apply_idx):
            st = pd.DataFrame({"k": key.values[fit_idx], "y": self.y_all[fit_idx]}).groupby("k").y.agg(["sum", "count"])
            s = key.iloc[apply_idx].map(st["sum"]).fillna(0).values
            c = key.iloc[apply_idx].map(st["count"]).fillna(0).values
            return (s + TE_SMOOTH * prior) / (c + TE_SMOOTH)

        out = np.empty(len(key))
        rest = np.setdiff1d(np.arange(len(key)), tr)
        out[rest] = enc(tr, rest)
        for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(tr, self.y_all[tr]):
            out[tr[b]] = enc(tr[a], tr[b])
        return out

    def __call__(self, tr, seed):
        lab = np.zeros(len(self.y_all), bool)
        lab[tr] = True
        F = self._group(lab)
        F["CA_R3_TE"] = self._te(np.asarray(tr), seed)
        return F.iloc[:self.n].reset_index(drop=True), F.iloc[self.n:].reset_index(drop=True)
