# Jumeau numérique de l'optimisation de section

Compte rendu écrit le 12/11/2026 à 17:51 par `report.py`.

## Le principe

On remplace les codes aéro par une fonction **connue** : la moyenne du modèle final du run du 11/11 (niveau 1 = NPLLT × NeuralFoil, niveau 2 = AVL × XFOIL). Sur cette « vérité », qui ne coûte rien, on peut chercher l'optimum **directement** (montée sous contraintes avec la vraie VPP, chaque pas vérifié par la VPP). On relance ensuite notre méthode complète depuis zéro, comme sur un run réel, et l'on regarde si elle retrouve cet optimum.

Critère fixé avant de lancer : la section trouvée par la méthode doit être à moins de **0,02 nd** du meilleur optimum de la recherche directe, en vitesse vraie.

## Verdict

- Vitesse perdue par la méthode sur le meilleur optimum de référence : **-0,0456 nd** (critère : 0,02 nd) → **réussi**.
- Distance entre les deux sections : 2,52 score,

Référence de comparaison : `reference_from_best`. Run de la méthode : `2026-11-12_1743_smoke_twin`.

| Section | Famille | Écart de vitesse vraie (nd) | Distance (scores) | Conditions résolues |
|---|---|---|---|---|
| `method_region_0` | méthode sur le jumeau | +0,0686 | 3,37 | 4 |
| `method_current_section` | section de départ | +0,0686 | 3,37 | 4 |
| `truth_run_current_section` | run réel du 11/11 | +0,0686 | 3,37 | 4 |
| `method_best` | méthode sur le jumeau | +0,0456 | 2,52 | 4 |
| `reference_from_best` | recherche directe (référence) | +0,0000 | 0,00 | 4 |
| `truth_run_region_0` | run réel du 11/11 | -0,0633 | 1,92 | 4 |
| `truth_run_region_3` | run réel du 11/11 | -0,0770 | 2,02 | 4 |
| `truth_run_region_1` | run réel du 11/11 | -0,0812 | 2,30 | 4 |
| `truth_run_best` | run réel du 11/11 | -0,0863 | 1,25 | 4 |

## La recherche directe

| Départ | Arrêt | Itérations | Déplacement (scores) | Gain de vitesse vraie (nd) | Durée (min) |
|---|---|---|---|---|---|
| `best` | max_iter | 2 | 1,25 | +0,0701 | 2 |

Lecture : si des départs différents arrivent à la même vitesse vraie, la vérité n'a qu'un sommet (ou une crête) ; s'ils diffèrent, elle a plusieurs bassins et la colonne « écart » du tableau du verdict dit lesquels comptent.

## Limites

- La vérité est elle-même un krigeage : plus lisse que les vrais codes, et de la même famille que le modèle qui l'apprend. Le test juge **l'algorithme** (plan initial, ciblage des points, criblage, redémarrages, convergence), pas l'hypothèse de régularité.
- La recherche directe est une montée locale depuis quelques départs : elle peut manquer un bassin qu'aucun départ ne voit.
- Les vitesses sont comparées sur les conditions résolues pour les deux sections.

## Figures

- [verdict.html](figures_smoke/verdict.html)
- [carte_sections.html](figures_smoke/carte_sections.html)
- [methode_convergence.html](figures_smoke/methode_convergence.html)
- [recherche_directe.html](figures_smoke/recherche_directe.html)

## Refaire

```
..\.venv-mds\Scripts\python twin_experiments\reference.py --workers 12
..\.venv-mds\Scripts\python twin_experiments\run_twin.py --time-limit-min 330
..\.venv-mds\Scripts\python twin_experiments\compare.py --workers 24
..\.venv-mds\Scripts\python twin_experiments\report.py
```

Fichiers : `twin.py` (vérité, remplaçant des codes), `reference.py` (recherche directe), `run_twin.py` (méthode complète, `scripts/run_real_vpp.py` inchangé), `compare.py` (vitesses vraies), `twin_runs/` (dossiers des runs sur le jumeau).
