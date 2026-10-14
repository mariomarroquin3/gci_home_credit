# Exp v29 — monotonas a nivel blend — EMPATE (dilucion) — produccion sigue = v12

Pipeline v12 completa (5 seeds x 3 modelos) con monotone(-1) EXT_SOURCE_1/2/3 en LGBM y XGB:

| | v12 | v29 | delta |
|---|---|---|---|
| LGBM-t seed-avg | 0.76273 | 0.76303 | **+0.00030** (replica v28) |
| CB-t seed-avg | 0.76343 | 0.76343 | 0 (sin cambios) |
| XGB seed-avg | 0.76362 | 0.76343 | -0.00019 |
| **Blend** | **0.76460** | **0.76462** | **+0.00002 empate** |

Pareado por seed con pesos fijos (.22/.43/.35): +0.00007 (sd 0.0001), t=1.44, **p=0.2227 n.s.**

Conclusion: el efecto real del componente LGBM se diluye en el blend; en XGB no aporta.
No cumple umbrales -> NO se adopta. La monotonia queda como opcion de seguridad para la
corrida final de noviembre (protege contra splits no-monotonos raros en datos privados),
decision diferida. Linea de "regularizacion por priors" queda medida y cerrada.
