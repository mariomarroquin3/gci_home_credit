# Exp v23-v25 — deep feature engineering — NEGATIVO x3

Proxy LGBM-tuneado (protocolo v6). Ref: v6 CV 0.76151; re-check en entorno actual: 0.76106
(drift -0.00045 por pandas 2.3.3 en manejo de categoricas; deltas contra ambas refs).

| Candidato | CV | vs v6 | vs re-check |
|---|---|---|---|
| v23 missingness por fila (NAN_COUNT/EXT/BUREAU/DAYS/PCT) | 0.76083 | -0.00068 | -0.00023 |
| v24 frequency encoding (ORG, OCC, ORGxOCC) | 0.76067 | -0.00084 | -0.00039 |
| v25 seleccion por permutacion (drop 12 feats perm-neg) | 0.76117 | -0.00034 | +0.00011 |

v25: las 12 features con perm-importance "negativa" tienen valores <= 0.00036 = ruido de un fold;
quitarlas no cambia nada (empate). Las perm-importances reales confirman la jerarquia:
EXT_SOURCE_2/3 >> CREDIT_ANNUITY >> EXT_SOURCE_1 > CODE_GENDER/DAYS_ID_PUBLISH/EDUCATION...

**Conclusion: feature engineering queda 0/9. El set v1 (51 features) + GBM stack + pseudo-labeling
es el techo medido de esta tabla. No se adopta nada; se documenta y se cierra la linea.**

Nota tecnica: pandas 2.3.3 rompe `select_dtypes(include=["str"])` y altera CV en ~0.0004 —
los numeros historicos de experiments/ se midieron con pandas 2.2.x (entorno del repo, requirements).
