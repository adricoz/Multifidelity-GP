# Tolérances de fabrication de la meilleure section du 8e run (03/11/2026)

**Méthode** (`scripts/robustness_checks.py tolerance`) : 5 perturbations gaussiennes lisses des 16 poids de
Kulfan par amplitude, mises à l'échelle pour que le défaut maximal de surface vaille 0,05 % c (0,35 mm) ou
0,15 % c (1,05 mm), corde 0,70 m. VPP sur le surrogate (section projetée sur la PCA), puis AVL × XFOIL avec les
poids non projetés aux équilibres (517 calculs), vitesse corrigée au premier ordre : V − (μ − y_HF)·dV/dC.
Moyenne sur 98 % de la probabilité de `profile_v2`. Référence : 13,5661 nd.

| Cas | Amplitude | Ixx(actuelle) / Ixx(section) | Δ vitesse (nd) |
|---|---|---|---|
| 1 | 0,35 mm | 0,982 | −0,0015 |
| 2 | 0,35 mm | 0,983 | −0,0018 |
| 3 | 0,35 mm | 1,003 | +0,0018 |
| 4 | 0,35 mm | 0,990 | −0,0006 |
| 5 | 0,35 mm | 0,991 | +0,0004 |
| 6 | 1,05 mm | 1,049 | +0,0066 |
| 7 | 1,05 mm | 1,000 | −0,0001 |
| 8 | 1,05 mm | 0,970 | −0,0026 |
| 9 | 1,05 mm | 1,021 | +0,0028 |
| 10 | 1,05 mm | 0,996 | −0,0016 |

(section non perturbée : 0,996, soit Ixx = 1,004 × Ixx(actuelle), sur la contrainte Ixx ≥ Ixx(actuelle) du run ;
un rapport > 1 veut dire une section plus souple que l'actuelle, donc hors contrainte)

- Les écarts sont de l'ordre de ±0,002 nd à 0,35 mm et de −0,003 / +0,007 nd à 1,05 mm.
- Ils suivent la souplesse : Δv ≈ +0,12 nd par unité du rapport Ixx(actuelle)/Ixx(section) (corrélation 0,96,
  résidu 0,0007 nd). Une section plus souple va plus vite (effet FSI), c'est le prix de la contrainte Ixx.
- Les cas qui respectent la contrainte (rapport ≤ 1 : 1, 2, 4, 5, 7, 8, 10) perdent de 0 à 0,003 nd, le meilleur
  gagne +0,0004 nd (bruit). Les seuls cas plus rapides (3, 6, 9) sont plus souples que la contrainte.
- À raideur égale, la forme perturbée ne perd presque rien : l'optimum est un plateau, pas un pic, et il est bien
  tenu par la contrainte Ixx.

Figures : [`figures/index.html`](figures/index.html) (`scripts/robustness_figures.py`).

Fichiers : `results/cases.csv` (une condition par ligne), `results/summary.json`, `run.log`.
