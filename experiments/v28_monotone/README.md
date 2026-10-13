# Exp v28 — restricciones monotonas EXT_SOURCE (revision PR #14) — POSITIVO SISTEMATICO (componente)

LGBM-tuned, 5 seeds x 5 folds, con/sin monotone(-1) en EXT_SOURCE_1/2/3:

| seed | off | on | delta |
|---|---|---|---|
| 42 | 0.76106 | 0.76117 | +0.00011 |
| 43 | 0.76100 | 0.76133 | +0.00033 |
| 44 | 0.76032 | 0.76075 | +0.00043 |
| 45 | 0.76120 | 0.76157 | +0.00037 |
| 46 | 0.76071 | 0.76073 | +0.00002 |

**media +0.00025 (sd 0.00016), t=3.18, p=0.0335** — 5/5 seeds positivo. Efecto real pero pequeno
(umbral de tuning es +0.002; esto es regularizacion con prior de dominio: P(default) decrece con
EXT_SOURCE). Wiring correcto heredado del PR #14 (lgbm monotone_constraints + advanced; xgb
monotone string). Falta validar a nivel BLEND (v29, pareado vs oof_partial de v12) y decidir CB.
