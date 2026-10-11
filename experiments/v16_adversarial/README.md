# Exp v16 — adversarial validation train vs test — SIN SHIFT

LGBM distinguiendo train(0) vs test(1): **AUC 0.51746 ~ 0.5** → no hay drift exploitable.
Consecuencia: (1) no hay truco de sample-weighting/drop-features disponible;
(2) confirma estructuralmente la calibracion CV->LB (+0.003 estable).
