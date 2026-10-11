# Exp v12 — seed averaging (5 seeds x 3 modelos) + blend

Ref: blend v10 (single-seed) = 0.76414.

| Modelo | single-seed (seed 42) | seed-avg OOF | delta |
|---|---|---|---|
| LGBM-tuned | 0.76151 | 0.76273 | +0.00122 |
| CB-tuned | 0.76243 | 0.76343 | +0.00100 |
| XGB | 0.76186 | 0.76336 | +0.00150 |

**Blend 5-seeds: CV 0.76460** (pesos lgbm .22 / cb .43 / xgb .35) = **+0.00046 vs v10**.
Bajo el umbral de adopcion de blend (+0.001) pero es la mejor pipeline medida; ademas reduce varianza
de la submission (alineado con la regla RSKF de seleccion final).
Seeds: 42,43,44,45,46 (cada seed con su propio SKF-5; OOF sigue siendo honesto).
