# Exp v11 — Features Tier-1 (soluciones ganadoras 2018) — NEGATIVO

Linea probada sobre LGBM-tuneado (ref v6: CV 0.76151, 5-fold seed 42):

| Variante | Contenido | CV | Delta |
|---|---|---|---|
| v11b | v1 + KNN500 target-mean (1er lugar 2018) | 0.76033 | **-0.00118** |
| v11c | v1 + tasa interes implicita + div EXT3 + AGE_INT/region-cat (5to lugar 2018) | 0.76057 | **-0.00094** |
| v11 | todo junto | 0.75980 | **-0.00171** |

## Hallazgos
- KNN500 standalone OOF AUC = 0.71537 (fortisimo como feature individual) y fue #1 por ganancia (216k, 2.4x EXT_SOURCE_2)... pero el CV BAJO: el subset ya incluye EXT1/2/3+CREDIT+ANNUITY en el modelo, el KNN es redundante y su componente target-mean actua como pseudo-label ruidoso (varianza entre folds 0.755-0.765).
- En 2018 (dataset multi-tabla) el KNN aportaba señal ortogonal; en el subset de 1 tabla no.
- Las features de tasa implicita/div EXT3 tampoco aportan: mismo veredicto que v8 (ruido en espacio chico).

## Decision
Linea CERRADA como negativa documentada (0/5 en feature engineering; ganan tuning + ensemble).
Los npy knn500 quedan en sandbox (files/exp_v11b/) por si se quiere como feature para stacking despues.
