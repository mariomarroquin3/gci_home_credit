# GCI Global 2026 · Home Credit Default Risk

Predicción de probabilidad de impago (`TARGET`). Métrica: ROC-AUC. Entrega: CSV `SK_ID_CURR,TARGET` en Omnicampus.
Cierre: **2026-11-20 01:00 UTC**. Objetivo: Outstanding (código reproducible del CSV final).

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
Copia `train.csv`, `test.csv` y `sample_submission.csv` a `data/raw/` (no se versionan).

## Uso
```bash
python -m src.train            # CV 5-fold LightGBM -> outputs/ + submissions/
```
EDA en `notebooks/01_eda.ipynb`.

## Estructura
```
src/config.py     rutas, SEED, seed_everything
src/data.py       carga de datos
src/features.py   limpieza y features
src/train.py      CV + predicción + submission
notebooks/        EDA y experimentos
submissions/      CSVs (solo final_submission.csv se versiona)
refs.md           fuentes de código citadas
```

## Reglas clave
- Sin datos externos; sin hand-labeling (todo sale del modelo).
- Seeds fijas (`src/config.py`). Citar código ajeno en `refs.md`.
- Cuenta la **última** submission.
