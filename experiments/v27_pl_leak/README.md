# Exp v27 — fuga de pseudo-labels — CONFIRMADA (revision PR #14)

El CV de v20 (0.76488) estaba inflado por fuga de SELECCION: los pseudo-labels se elegian con
predicciones de test de modelos pass-1 que habian entrenado sobre el 100% de train (incluido el
fold de validacion). Protocolo correcto = labeler anidado por fold (entrenado solo en fold-train).

| Protocolo (LGBM-tuned, seed 42, 5-fold) | CV | delta |
|---|---|---|
| Baseline sin PL | 0.76106 | — |
| A: PL global (v19/v20) | 0.76185 | +0.00079 |
| B: PL anidado honesto | 0.76089 | -0.00017 |

**Fuga = +0.00096. Efecto honesto del PL ~ 0 (negativo).** Replica los numeros del PR #14
(+0.0004..+0.0009 global; -0.0001..-0.0003 anidado) con protocolo independiente.

Consecuencias:
- v20 NO es mejor que v12: CV honesto ~0.7646 (o 0.7644 con PL anidado). El analisis pareado
  5-seeds (t=9.5) media una fuga sistematica, no senal.
- Produccion = pipeline v12 (pass 1). PL queda FUERA (efecto honesto ~0, coste y riesgo altos).
- v19/v20/v21 se corrigen con nota en sus READMEs.
