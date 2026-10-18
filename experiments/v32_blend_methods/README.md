# Experimento v32 — blend por rangos y Caruana ensemble selection

Idea (research/Métodos nuevos para ganar GCI.md): cambiar el método de blend de los 5 miembros v30
(LGBM, CB, XGB, lgbm_init, lgbm_linear) en vez de añadir modelos.

## Protocolo
- OOF: `python -m src.train --seeds 42,43,44 --folds 5 --tag v32base` (RSKF 3x5; in-sample 0.76448).
- `python experiments/v32_blend_methods/blend_methods.py outputs/run_<...>_v32base`
- Pesos siempre elegidos en CV anidado de 5 folds sobre filas (NESTED_SEED de src.validate).
  Las transformaciones (rank, logit) no usan labels.
- Criterio: mejorar en seed 42 y luego ganar en las 3 seeds.

## Resultados (AUC blend anidado)

| método | seed 42 | Δ vs hc_prob | seed 43 | Δ |
|---|---|---|---|---|
| hc_prob (producción v30) | 0.76409 | — | 0.76360 | — |
| hc_rank | 0.76403 | -0.00006 | 0.76362 | +0.00002 |
| hc_logit | 0.76404 | -0.00006 | 0.76360 | -0.00000 |
| cs_prob (Caruana, reemplazo, 200 it.) | 0.76409 | -0.00000 | — | — |
| cs_rank | 0.76403 | -0.00006 | — | — |
| cs_logit | 0.76404 | -0.00005 | — | — |
| csb_rank (Caruana + bagging 25×60%) | 0.76401 | -0.00008 | — | — |

**Rechazado.** Ningún método mejora seed 42; el resto de seeds no se completó (corte por tiempo)
porque el criterio ya falla. Caruana converge a los mismos pesos que el hill-climbing (con 5
miembros muy correlacionados el simplex es plano); ranks/logits pierden ~0.00006.
`main` no cambia.
