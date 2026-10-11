# Exp v15 — XGB-tuneado seed-avg + re-blend — NEUTRO

- XGB-tuned (v14) 5-seed OOF: **0.76362** (vs XGB-default seed-avg 0.76336: +0.00026)
- Blend con lgbm/cb seed-avg de v12: **0.76459** (w .25/.39/.36) = **-0.00001 vs v12** (empate)

Conclusion: la mejora del componente XGB se diluye en el blend (alta correlacion entre GBMs).
El trio {LGBM-t, CB-t, XGB-t} x features v1 x seed-avg tiene techo ~0.7646.
Siguiente: diversificadores de otra familia (TabPFN / NN) y chequeo estructural (adversarial validation).
