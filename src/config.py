import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GCI_DATA_DIR", ROOT / "data" / "raw"))
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_DIR = ROOT / "submissions"

SEED = 42
SEEDS = [42, 43, 44, 45, 46]  # seed-averaging (exp/seedavg-v12)
N_FOLDS = 10  # 10 vs 5 folds: +0.0010 CV (LGBM, 3 seeds de folds, revision 2026-10-13)
ID_COL = "SK_ID_CURR"
TARGET = "TARGET"

# Pseudo-labeling (exp/pseudo-pipeline-v20): 1 sola ronda, solo negativos muy confiables
PSEUDO_MAX_PROB = 0.03

# Hiperparámetros tuneados por Optuna (una branch por estudio, ver experiments/)
LGBM_PARAMS = {
    # exp/optuna-v4 (60 trials): CV 0.76145
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.008135825913706748, "num_leaves": 37, "min_child_samples": 197,
    "feature_fraction": 0.40362837123334017, "bagging_fraction": 0.9525103687282102,
    "bagging_freq": 1, "lambda_l1": 0.00011657904410727821, "lambda_l2": 3.0181881636817223,
    "min_split_gain": 0.22728540197274544, "max_depth": 9,
    "deterministic": True, "force_row_wise": True, "verbose": -1,
}
CB_PARAMS = {
    # exp/optuna-cb-v9 (26 trials): CV 0.76243
    "iterations": 8000, "learning_rate": 0.02929, "depth": 6, "l2_leaf_reg": 13.52,
    "random_strength": 0.2955, "bagging_temperature": 0.1934, "min_data_in_leaf": 168,
    "loss_function": "Logloss", "eval_metric": "AUC", "od_type": "Iter", "od_wait": 300,
    "verbose": False, "allow_writing_files": False,
}
XGB_PARAMS = {
    # exp/optuna-xgb-v14 (45 trials): CV 0.76297
    "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
    "learning_rate": 0.006704715664362213, "max_depth": 3, "min_child_weight": 14,
    "subsample": 0.9174841990499791, "colsample_bytree": 0.5037589701822436,
    "reg_alpha": 8.778854079749578, "reg_lambda": 3.009713063266187, "gamma": 0.2565501232341565,
}
MAX_ROUNDS = {"lgbm": 20000, "xgb": 20000}
ES_ROUNDS = 300

# Pesos del blend por hill-climbing sobre OOF (exp/pseudo-pipeline-v20)
BLEND_WEIGHTS = {"lgbm": 0.29, "cb": 0.37, "xgb": 0.34}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
