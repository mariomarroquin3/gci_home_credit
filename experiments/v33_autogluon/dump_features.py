"""Exporta las features de src.features a CSV para el entorno de AutoGluon (pandas 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import ID_COL, TARGET, OUTPUT_DIR  # noqa: E402
from src.data import load_data  # noqa: E402
from src.features import build  # noqa: E402

out = OUTPUT_DIR / "v33_ag"
out.mkdir(parents=True, exist_ok=True)
train, test, sample = load_data()
train, test, _ = build(train, test)
train.drop(columns=[ID_COL]).to_csv(out / "train_feat.csv", index=False)
test.drop(columns=[ID_COL]).to_csv(out / "test_feat.csv", index=False)
print(train.shape, test.shape, train[TARGET].mean())
