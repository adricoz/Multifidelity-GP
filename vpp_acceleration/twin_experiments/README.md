# Jumeau numérique de l'optimisation de section

Compte rendu écrit le 13/11/2026 à 15:11 par `report.py`.

## Le principe

On remplace les codes aéro par une fonction **connue** : la moyenne du modèle final du run du 11/11 (niveau 1 = NPLLT × NeuralFoil, niveau 2 = AVL × XFOIL). Sur cette « vérité », qui ne coûte rien, on peut chercher l'optimum **directement** (montée sous contraintes avec la vraie VPP, chaque pas vérifié par la VPP). On relance ensuite notre méthode complète depuis zéro, comme sur un run réel, et l'on regarde si elle retrouve cet optimum.

Critère fixé avant de lancer : la section trouvée par la méthode doit être à moins de **0,02 nd** du meilleur optimum de la recherche directe, en vitesse vraie.

Comment lire le résultat :

- **Perte ≤ 0,02 nd** : l'algorithme (plan initial, ciblage, criblage, redémarrages, convergence) retrouve l'optimum d'une fonction dont on connaît la réponse.
- **Perte plus grande, section éloignée** : la méthode s'est arrêtée dans un autre bassin ; le criblage ou les redémarrages ne suffisent pas.
- **Section éloignée mais même vitesse** : la vérité a une crête plate ; l'optimum n'est pas unique, ce qui n'est pas un défaut.
- **Départs de la recherche directe à des vitesses différentes** : la vérité a plusieurs bassins ; le tableau dit lesquels comptent.
- Les lignes « sections de borne haute maximale » sont les sections du certificat (test 2) résolues par la VPP sur la vérité : si aucune ne bat la référence, l'endroit le plus prometteur selon le modèle ne cache rien.
- La ligne « run réel sans CFD » est la section de l'autre run jugée exactement par le modèle du 11/11 (jugement croisé, test 3).

## Verdict

- Vitesse perdue par la méthode sur le meilleur optimum de référence : **+0,0019 nd** (critère : 0,02 nd) → **réussi**.
- Distance entre les deux sections : 1,53 score,

Référence de comparaison : `reference_from_best`. Run de la méthode : `2026-11-13_0930_twin_method`.

| Section | Famille | Écart de vitesse vraie (nd) | Distance (scores) | Conditions résolues |
|---|---|---|---|---|
| `reference_from_best` | recherche directe (référence) | +0,0000 | 0,00 | 48 |
| `reference_from_random` | recherche directe (référence) | -0,0006 | 2,29 | 48 |
| `nocfd_best` | run réel sans CFD | -0,0008 | 0,51 | 48 |
| `reference_from_region_3` | recherche directe (référence) | -0,0010 | 0,43 | 48 |
| `reference_from_region_0` | recherche directe (référence) | -0,0017 | 2,25 | 48 |
| `method_best` | méthode sur le jumeau | -0,0019 | 1,53 | 48 |
| `reference_from_current_section` | recherche directe (référence) | -0,0020 | 0,63 | 48 |
| `reference_from_region_1` | recherche directe (référence) | -0,0028 | 0,51 | 48 |
| `truth_run_region_0` | run réel du 11/11 | -0,0040 | 1,97 | 48 |
| `nocfd_region_3` | run réel sans CFD | -0,0047 | 1,28 | 48 |
| `method_region_3` | méthode sur le jumeau | -0,0049 | 1,45 | 48 |
| `truth_run_best` | run réel du 11/11 | -0,0050 | 1,43 | 48 |
| `truth_run_region_1` | run réel du 11/11 | -0,0051 | 1,91 | 48 |
| `truth_run_region_3` | run réel du 11/11 | -0,0051 | 1,87 | 48 |
| `method_region_2` | méthode sur le jumeau | -0,0053 | 1,57 | 48 |
| `method_region_0` | méthode sur le jumeau | -0,0072 | 2,50 | 48 |
| `nocfd_region_2` | run réel sans CFD | -0,0078 | 2,42 | 48 |
| `nocfd_region_0` | run réel sans CFD | -0,0106 | 2,60 | 48 |
| `certificate_k3_2` | sections de borne haute maximale | -0,0652 | 5,24 | 48 |
| `certificate_k2_5` | sections de borne haute maximale | -0,0744 | 4,08 | 48 |
| `certificate_k2_4` | sections de borne haute maximale | -0,0793 | 3,46 | 48 |
| `certificate_k3_3` | sections de borne haute maximale | -0,0879 | 5,52 | 48 |
| `certificate_k3_1` | sections de borne haute maximale | -0,1116 | 6,07 | 48 |
| `method_current_section` | section de départ | -0,3157 | 3,93 | 48 |
| `truth_run_current_section` | run réel du 11/11 | -0,3157 | 3,93 | 48 |

## La recherche directe

| Départ | Arrêt | Itérations | Déplacement (scores) | Gain de vitesse vraie (nd) | Durée (min) |
|---|---|---|---|---|---|
| `best` | radius | 15 | 1,43 | +0,0278 | 75 |
| `current_section` | radius | 17 | 3,76 | +0,3144 | 76 |
| `region_0` | radius | 16 | 2,10 | +0,0205 | 61 |
| `region_1` | radius | 13 | 2,09 | +0,0121 | 35 |
| `region_3` | radius | 18 | 2,14 | +0,0185 | 39 |
| `random` | radius | 16 | 3,73 | +0,1091 | 33 |

Lecture : si des départs différents arrivent à la même vitesse vraie, la vérité n'a qu'un sommet (ou une crête) ; s'ils diffèrent, elle a plusieurs bassins et la colonne « écart » du tableau du verdict dit lesquels comptent.

## Limites

- La vérité est elle-même un krigeage : plus lisse que les vrais codes, et de la même famille que le modèle qui l'apprend. Le test juge **l'algorithme** (plan initial, ciblage des points, criblage, redémarrages, convergence), pas l'hypothèse de régularité.
- La recherche directe est une montée locale depuis quelques départs : elle peut manquer un bassin qu'aucun départ ne voit.
- Les vitesses sont comparées sur les conditions résolues pour les deux sections.

## Figures

- [verdict.html](figures/verdict.html)
- [carte_sections.html](figures/carte_sections.html)
- [methode_convergence.html](figures/methode_convergence.html)
- [recherche_directe.html](figures/recherche_directe.html)

## Refaire

```
..\.venv-mds\Scripts\python twin_experiments\reference.py --workers 12
..\.venv-mds\Scripts\python twin_experiments\run_twin.py --time-limit-min 330
..\.venv-mds\Scripts\python twin_experiments\compare.py --workers 24
..\.venv-mds\Scripts\python twin_experiments\report.py
```

Fichiers : `twin.py` (vérité, remplaçant des codes), `reference.py` (recherche directe), `run_twin.py` (méthode complète, `scripts/run_real_vpp.py` inchangé), `compare.py` (vitesses vraies), `twin_runs/` (dossiers des runs sur le jumeau).
