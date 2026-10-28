# Experimento v34 — señal estructural: clientes repetidos, tasa implícita y producto

Código de producción: `src/structure.py` (features sin target en `features.build`; features con
target por fold vía `FoldFeatures`, pasado como `fold_fn` a todos los modelos). Screening: `screen.py`.

## Hallazgos
- Orden de filas, SK_ID_CURR y duplicados train-test: sin señal.
- **Clientes repetidos**: DAYS_* son relativos a la fecha de solicitud, así que
  DAYS_BIRTH - DAYS_ID_PUBLISH, DAYS_BIRTH - DAYS_REGISTRATION y DAYS_BIRTH - DAYS_EMPLOYED son
  constantes para la misma persona. Con género, ~3-4% de filas comparten clave (azar: ~1%).
  Si el gemelo hizo default, tasa de default 25-41% (base 8%); si no, ~5%.
- **Tasa implícita**: AMT_CREDIT / AMT_ANNUITY = (1 - (1+r)^-n) / r para plazos estándar n.
- **Producto**: el ratio crédito/anualidad repite valores exactos (20.000 en 16k filas); frecuencia
  + target encoding suavizado del ratio redondeado (OOF interno dentro del fold de train).

## Screening (LGBM lr 0.03, 5-fold, seeds 42/43)
| set | AUC media |
|---|---|
| base | 0.76055 |
| + grupos sin target | 0.76153 |
| + tasa implícita | 0.76308 |
| todo sin target | 0.76515 |
| + target de gemelos y TE del ratio | **0.76656** |
| + frecuencias de valores exactos (rechazado) | 0.76622 |
| + EXT_SOURCE del gemelo (rechazado) | 0.76635 |

## Pipeline completo (5 modelos + blend, 5-fold, pareado contra main)
| seed | main | v34 | Δ |
|---|---|---|---|
| 42 | 0.76414 | 0.77007 | +0.00593 |
| 43 | 0.76369 | 0.76929 | +0.00560 |
| 44 | 0.76327 | 0.76911 | +0.00584 |

Gana 3/3 seeds, Δ medio +0.0058 (ruido de CV ~0.0004). Comando: `python -m src.train --seeds <s> --folds 5`.
