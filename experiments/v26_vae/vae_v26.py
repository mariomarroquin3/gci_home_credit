"""Exp v26: VAE para augmentacion de clase minoritaria (estilo notebooks Kaggle).
Por fold: VAE entrenado SOLO en defaulters del fold-train -> 10k sinteticos label=1.
Proxy LGBM-tuned, CV honesto. Refs: v6 0.76151 / re-check 0.76106. Umbral +0.003."""
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
OUT = Path("files/exp_v26"); OUT.mkdir(parents=True, exist_ok=True)

LGBM = dict(json.loads((Path("files/exp_v4/best_params.json")).read_text())["best_params"],
            objective="binary", metric="auc", seed=42, deterministic=True,
            force_row_wise=True, verbose=-1)

train = pd.read_csv("files/data/train.csv")
test = pd.read_csv("files/data/test.csv")
y = train["TARGET"].values

# builder v1 (igual que v23-25)
import importlib.util
spec = importlib.util.spec_from_file_location("f", "files/exp_v23_25/feat_v23_25.py")
# no importamos el modulo (ejecutaria); copiamos build inline:
def build(df):
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        df[f"{c}_ISNA"] = df[c].isna().astype(int)
    df["CREDIT_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["ANNUITY_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["INCOME_PER_FAM"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_BIRTH_PCT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    df["PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25
    bur = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    df["BUREAU_REQ_TOTAL"] = df[bur].sum(axis=1, min_count=1)
    df["PHONE_FLAGS_SUM"] = df[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE"]].sum(axis=1)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
    return df

full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")

# matriz numerica para el VAE: cats -> codigos, NaN -> media del fold (scaler)
Xn = X.copy()
for c in cat_cols:
    Xn[c] = Xn[c].cat.codes.astype(float)
Xn = Xn.astype(float)


class VAE(nn.Module):
    def __init__(self, d, z=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU())
        self.mu = nn.Linear(48, z); self.lv = nn.Linear(48, z)
        self.dec = nn.Sequential(nn.Linear(z, 48), nn.ReLU(), nn.Linear(48, 96), nn.ReLU(), nn.Linear(96, d))

    def forward(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv


def train_vae(Xd, epochs=15, bs=1024):
    sc = StandardScaler()
    Xs = torch.tensor(sc.fit_transform(np.nan_to_num(Xd, nan=0.0)), dtype=torch.float32)
    vae = VAE(Xs.shape[1]); opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xs))
        for i in range(0, len(Xs), bs):
            xb = Xs[perm[i:i + bs]]
            rec, mu, lv = vae(xb)
            loss = ((rec - xb) ** 2).mean() + 1e-3 * (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return vae, sc


skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
N_SYN = 10000
for fi, (tr, va) in enumerate(skf.split(X, y)):
    pos_idx = tr[y[tr] == 1]
    vae, sc = train_vae(Xn.iloc[pos_idx].values)
    with torch.no_grad():
        z = torch.randn(N_SYN, 16)
        syn = sc.inverse_transform(vae.dec(z).numpy())
    X_aug = np.vstack([Xn.iloc[tr].values, syn])
    y_aug = np.r_[y[tr], np.ones(N_SYN)]
    Xa = pd.DataFrame(X_aug, columns=Xn.columns)
    for c in cat_cols:
        Xa[c] = pd.Categorical(Xa[c].round().astype(int))  # codigos ordinales -> categorias enteras
    for c in Xn.columns:
        if c not in cat_cols:
            Xa[c] = Xa[c].where(Xa[c].notna(), np.nan)
    m = lgb.train(LGBM, lgb.Dataset(Xa, y_aug), num_boost_round=20000,
                  valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    print(f"fold {fi}: parcial {roc_auc_score(y[va], oof[va]):.5f}", flush=True)

auc = roc_auc_score(y, oof)
print(f"VAE-aug CV: {auc:.5f} ({auc-0.76106:+.5f} vs re-check / {auc-0.76151:+.5f} vs v6)", flush=True)
(OUT / "results.json").write_text(json.dumps({"vae_aug_cv": round(auc, 5),
    "delta_vs_recheck": round(auc - 0.76106, 5), "delta_vs_v6": round(auc - 0.76151, 5),
    "n_synthetic": N_SYN, "epochs": 15}, indent=1))
np.save(OUT / "oof_vae_aug.npy", oof)
print("DONE_V26", flush=True)
