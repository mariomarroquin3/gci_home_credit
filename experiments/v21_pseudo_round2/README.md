# Exp v21 — pseudo-labeling round-2 — PLANO (linea cerrada)

Etiquetador = blend v20. LGBM-tuned single-seed. Ref: round-1 (v19) = 0.76215.

| Umbral | rows | CV | vs round-1 |
|---|---|---|---|
| p<0.02 | 10,811 | 0.76182 | -0.00033 |
| p<0.03 | 17,673 | 0.76205 | -0.00010 |
| p<0.05 | 28,473 | 0.76223 | +0.00008 |

Conclusion: la ganancia de pseudo-labeling se captura en la ronda 1; iterar solo recicla ruido.
Una sola ronda con p<0.03 es el punto optimo. No hay v22.
