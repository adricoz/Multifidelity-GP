# Tirages a posteriori sur le 8e run (03/11/2026)

Test §6.3 de la note de Le Maître (`optimisation_foil_GP_VPP_v01`) : le classement des sections tient-il
dans l'incertitude du surrogate ?

**Méthode** (`scripts/robustness_checks.py posterior`) : 940 points d'ancrage (états d'équilibre des 4 sections
et un nuage Sobol de 4 attitudes autour de chacun). On tire conjointement les 6 efforts du niveau haut en ces
points, on conditionne le co-krigeage sur ces valeurs (fonction GP cohérente), puis on refait la VPP des 4 sections
avec cette fonction. 20 tirages, même bruit à l'échelle σ et 2σ (variance sous-estimée d'environ ×2 loin des
données au 8e run). Moyenne sur 98 % de la probabilité de `profile_v2`.

| Section | Nominal (nd) | σ : moyenne ± écart-type | 2σ : moyenne ± écart-type |
|---|---|---|---|
| best | 13,5665 | 13,5664 ± 0,0004 | 13,5663 ± 0,0008 |
| région 1 | 13,5596 | 13,5594 ± 0,0010 | 13,5591 ± 0,0021 |
| best du run 7 | 13,5494 | 13,5495 ± 0,0020 | 13,5497 ± 0,0039 |
| départ | 13,3089 | 13,3080 ± 0,0029 | 13,3072 ± 0,0056 |

- P(best est la meilleure) = 1 aux deux échelles (20/20), regret de garder best = 0.
- best − région 1 = +0,0070 ± 0,0010 nd (σ), +0,0072 ± 0,0019 nd (2σ) ; best − run 7 = +0,017 ± 0,004 nd ;
  best − départ = +0,259 ± 0,006 nd.
- Limite : ce test couvre l'incertitude du GP sur AVL × XFOIL, pas l'écart entre AVL × XFOIL et la réalité (CFD).

Figures : [`figures/index.html`](figures/index.html) (`scripts/robustness_figures.py`).

Fichiers : `results/draws.csv` (un tirage par ligne), `results/summary.json`, `run.log`.
