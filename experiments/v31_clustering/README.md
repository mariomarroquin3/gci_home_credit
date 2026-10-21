# Exp v31 — clustering no supervisado (KMeans) — NEGATIVO significativo — linea cerrada

Pregunta: ¿estructura no supervisada (cluster_id como feature) suma? Reglamento: PERMITIDO
(es automatico/reproducible/aplicable a datos nuevos — no es hand-labeling). Fit solo en train.

Setup: KMeans k=8/16 sobre numericas estandarizadas (NaN->mediana solo p/ clustering), features
km8_id + km16_id (cat) + km8_dist. LGBM-tuned, 5 seeds x 5 folds, pareado vs baseline v28-off.

| seed | base | +clusters | delta |
|---|---|---|---|
| 42 | 0.76106 | 0.76069 | -0.00037 |
| 43 | 0.76100 | 0.76066 | -0.00034 |
| 44 | 0.76032 | 0.75971 | -0.00061 |
| 45 | 0.76120 | 0.76108 | -0.00012 |
| 46 | 0.76071 | 0.75998 | -0.00073 |

**media -0.00043 (sd 0.00022), t=-4.03, p=0.0157** — negativo en 5/5 seeds.

Conclusion: los clusters redescubren estratos de EXT_SOURCE que los arboles ya particionan mejor
de forma supervisada; el ruido extra de particion solo resta. Familia "estructura de vecindad/
no supervisada" queda 0/4 (KNN-500 x3, KMeans x1). No reabrir. GMM queda descartado por
pertenecer a la misma familia con el mismo mecanismo.
