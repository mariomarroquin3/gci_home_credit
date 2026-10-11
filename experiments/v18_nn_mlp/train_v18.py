"""Exp v18: NN-MLP con embeddings de categoricas, 5-fold x 3 seeds (bagging), diversificador.
Numericas: imputar mediana + RobustScaler. Categoricas: embedding (card<=64 -> dim 6/12).
Ref blend v12: 0.76460. Se evalua solo y como 4o componente del blend."""
import itertools
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("experiments/v18_nn_mlp"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]
torch.manual_seed(42); np.random.seed(42)
DEV = "cpu"

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

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
y = train["TARGET"].values.astype(np.float32)
full = build(pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True))
X = full.iloc[:len(train)].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
X_test = full.iloc[len(train):].drop(columns=["SK_ID_CURR"]).reset_index(drop=True)
id_test = test["SK_ID_CURR"].values

cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
# encodings globales (train+test para cardinalidad completa)
cat_maps = {c: {v: i + 1 for i, v in enumerate(pd.concat([X[c], X_test[c]]).astype("str").unique())} for c in cat_cols}
def enc_cats(df):
    return np.stack([df[c].astype("str").map(cat_maps[c]).fillna(0).values.astype(np.int64) for c in cat_cols], 1)
# numericas: mediana + quantile clip + zscore (fit en TODO train+test; sin leakage de target)
allnum = pd.concat([X[num_cols], X_test[num_cols]])
med = allnum.median(); q1 = allnum.quantile(0.001); q99 = allnum.quantile(0.999)
mu = allnum.mean(); sd = allnum.std().replace(0, 1)
def enc_nums(df):
    v = df[num_cols].fillna(med).clip(q1, q99, axis=1)
    return ((v - mu) / sd).values.astype(np.float32)

Xn_tr, Xn_te = enc_nums(X), enc_nums(X_test)
Xc_tr, Xc_te = enc_cats(X), enc_cats(X_test)
cards = [len(cat_maps[c]) + 1 for c in cat_cols]
embs = [min(12, max(4, (c + 1) // 8)) for c in cards]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.em = nn.ModuleList([nn.Embedding(c, e) for c, e in zip(cards, embs)])
        d = Xn_tr.shape[1] + sum(embs)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, xn, xc):
        z = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.em)], 1)
        return self.mlp(z).squeeze(-1)

def train_one(seed, tr_idx, va_idx):
    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    pos_w = torch.tensor([(y == 0).sum() / (y == 1).sum()], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.tensor(Xn_tr[tr_idx]); Xct = torch.tensor(Xc_tr[tr_idx]); yt = torch.tensor(y[tr_idx])
    Xv = torch.tensor(Xn_tr[va_idx]); Xcv = torch.tensor(Xc_tr[va_idx])
    best_auc, best_state, bad = 0, None, 0
    n = len(tr_idx); bs = 4096
    for ep in range(60):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(Xt[j], Xct[j]), yt[j]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv, Xcv)).numpy()
        a = roc_auc_score(y[va_idx], pv); sched.step(1 - a)
        if a > best_auc:
            best_auc, best_state, bad = a, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        oof_p = torch.sigmoid(net(Xv, Xcv)).numpy()
        te_p = torch.sigmoid(net(torch.tensor(Xn_te), torch.tensor(Xc_te))).numpy()
    return oof_p, te_p

oof = np.zeros((len(SEEDS), len(train))); tpred = np.zeros((len(SEEDS), len(test)))
for si, seed in enumerate(SEEDS):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(Xn_tr, y):
        o, t = train_one(seed * 100 + si, tr, va)
        oof[si, va] = o; tpred[si] += t / 5
    print(f"seed {seed}: NN OOF {roc_auc_score(y, oof[si]):.5f}", flush=True)
oof_nn = oof.mean(axis=0); test_nn = tpred.mean(axis=0)
np.save(OUT / "oof_nn_seedavg.npy", oof_nn); np.save(OUT / "test_nn_seedavg.npy", test_nn)
auc_nn = roc_auc_score(y, oof_nn)
print(f"NN seed-avg OOF: {auc_nn:.5f}")

names = ["lgbm", "cb", "xgb", "nn"]
O = {"lgbm": np.load("experiments/v12_seedavg/oof_lgbm_seedavg.npy"), "cb": np.load("files/exp_v12/oof_cb_seedavg.npy"),
     "xgb": np.load("experiments/v15_xgb_tuned_seedavg/oof_xgb_tuned_seedavg.npy"), "nn": oof_nn}
T = {"lgbm": np.load("experiments/v12_seedavg/test_lgbm_seedavg.npy"), "cb": np.load("files/exp_v12/test_cb_seedavg.npy"),
     "xgb": np.load("experiments/v15_xgb_tuned_seedavg/test_xgb_tuned_seedavg.npy"), "nn": test_nn}
def score(w):
    return roc_auc_score(y, sum(wi * O[k] for wi, k in zip(w, names)))
best = (0, None)
for w in itertools.product(range(0, 21), repeat=4):
    if sum(w) != 20:
        continue
    w = tuple(x / 20 for x in w)
    s = score(w)
    if s > best[0]:
        best = (s, w)
w = list(best[1]); bs = best[0]
imp = True
while imp:
    imp = False
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            for d in (0.01, -0.01):
                w2 = list(w); w2[i] += d; w2[j] -= d
                if min(w2) < 0:
                    continue
                s = score(w2)
                if s > bs + 1e-7:
                    w, bs = w2, s; imp = True
print(f"blend 4mod: {bs:.5f} {dict(zip(names, [round(x,3) for x in w]))} (vs v12: {bs-0.76460:+.5f})")
pred = sum(wi * T[k] for wi, k in zip(w, names))
pd.DataFrame({"SK_ID_CURR": id_test, "TARGET": pred}).to_csv(OUT / "submission_v18.csv", index=False)
(OUT / "results.json").write_text(json.dumps({"nn_seedavg": round(auc_nn, 5), "blend_cv": round(bs, 5),
    "weights": dict(zip(names, [round(x, 4) for x in w])), "delta_vs_v12": round(bs - 0.76460, 5)}, indent=1))
print("OK")
