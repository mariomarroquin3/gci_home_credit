# Exp v20 — pseudo-labeling en pipeline completa (5 seeds x 3 modelos) — NUEVO MEJOR

Pseudo rows: test p<0.03 del blend v12 (15,156, todas negativas). OOF honesto.

| Modelo | v12 (sin PL) | v20 (con PL) | delta |
|---|---|---|---|
| LGBM-t | 0.76273 | 0.76327 | +0.00054 |
| CB-t | 0.76343 | 0.76364 | +0.00021 |
| XGB-t | 0.76362 | 0.76391 | +0.00029 |

**Blend v20: CV 0.76488** (w .29/.37/.34) = **+0.00028 vs v12** → nuevo mejor pipeline medido.
Submission: `submission_v20.csv`. Estimado LB ~0.7675 (calibracion +0.003).
