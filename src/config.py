import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GCI_DATA_DIR", ROOT / "data" / "raw"))
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_DIR = ROOT / "submissions"

SEED = 42
N_FOLDS = 5
ID_COL = "SK_ID_CURR"
TARGET = "TARGET"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
