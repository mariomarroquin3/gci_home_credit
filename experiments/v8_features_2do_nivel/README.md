# Experimento v8 — Features de 2do nivel (RESULTADO NEGATIVO, documentado)

Hipotesis: stats EXT, logs, interacciones y agregados por categoria sumarian sobre v1.
Misma CV (5-fold, seed 42), mismos params Optuna. Referencia v6 = 0.76151.

| Variante | n_features | CV | delta |
|---|---|---|---|
| v8 (todo: 30 nuevas) | 81 | 0.76090 | -0.00061 |
| v8b (-DOC_SUM, -LOG_INCOME, -CONTACT_FLAGS) | 78 | 0.76069 | -0.00082 |
| v8c (v8b - agregados por categoria) | 72 | 0.76089 | -0.00062 |

## Aprendizajes
- EXT_MEAN quedo como feature #1 por ganancia (129k) y EXT_MEAN_CREDIT #2: la senal existe.
- Pero el CV no sube: con params tuneados para 51 features, el ruido extra cuesta mas que la senal.
- DOC_SUM, CONTACT_FLAGS y los agregados por categoria no aportan (ganancia ~0-5k).
- Decision: v8d prueba SOLO el bloque EXT (6 features). Si tampoco suma, se descarta
  la linea y el peso pasa a tuning de CatBoost/XGBoost + stacking (Sprint 4).
