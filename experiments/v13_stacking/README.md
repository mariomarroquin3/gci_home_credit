# Exp v13 — stacking sobre OOF seed-averaged — NO ADOPTADO

Meta-modelo LogisticRegression evaluado honesto (cross_val_predict, 5-fold seed 42):

| Variante | CV | vs blend v12 (0.76460) |
|---|---|---|
| stack 3 modelos | 0.76452 | -0.00008 (empate) |
| stack 3 + knn500 | 0.76297 | -0.00163 |
| rank-avg 3 | 0.76451 | -0.00009 |
| rank-avg 3 + knn500 | 0.76065 | -0.00395 |

Conclusion: con 3 modelos correlacionados fuertes, el stacking no supera al blend ponderado.
knn500 como input de stack TAMBIEN resta (3ra prueba negativa de la linea).
