import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import ID_COL, N_FOLDS, OUTPUT_DIR, SEED, SUBMISSION_DIR, TARGET, seed_everything
from src.data import load_data
from src.features import build

PARAMS = {
    # Optuna 60 trials (exp/optuna-v4): CV AUC 0.76145 (+0.00336 vs baseline 0.75809)
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.008135825913706748,
    "num_leaves": 37,
    "min_child_samples": 197,
    "feature_fraction": 0.40362837123334017,
    "bagging_fraction": 0.9525103687282102,
    "bagging_freq": 1,
    "lambda_l1": 0.00011657904410727821,
    "lambda_l2": 3.0181881636817223,
    "min_split_gain": 0.22728540197274544,
    "max_depth": 9,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def main():
    seed_everything()
    train, test, sample = load_data()
    train, test, _ = build(train, test)
    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X, y, X_test = train[features], train[TARGET], test[features]

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    importance = pd.Series(0.0, index=features)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
        dva = lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx])
        model = lgb.train(
            PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        importance += pd.Series(model.feature_importance("gain"), index=features) / N_FOLDS
        print(f"fold {fold}: AUC {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f} (iter {model.best_iteration})")

    cv_auc = roc_auc_score(y, oof)
    print(f"CV AUC: {cv_auc:.5f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    sub = sample[[ID_COL]].merge(pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred}), on=ID_COL, how="left")
    sub_path = SUBMISSION_DIR / f"sub_{stamp}_cv{cv_auc:.5f}.csv"
    sub.to_csv(sub_path, index=False)
    importance.sort_values(ascending=False).to_csv(OUTPUT_DIR / f"importance_{stamp}.csv")
    with open(OUTPUT_DIR / "experiments.jsonl", "a") as f:
        f.write(json.dumps({"time": stamp, "cv_auc": cv_auc, "n_features": len(features),
                            "params": PARAMS, "submission": sub_path.name}) + "\n")
    print(f"Submission: {sub_path}")


if __name__ == "__main__":
    main()
