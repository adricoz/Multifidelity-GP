# 8e run : profil d'usage `profile_v2`, boîte à l'échelle 3 (seule contrainte : Ixx)

Le 29/10 de 10 h 02 à 12 h 51, sur toute la machine (28 + 28 processus, CFD arrêtée et WSL éteint). La boucle est
limitée à 140 min, à la demande de l'utilisateur (3 h au plus). Fichiers : [`config.json`](config.json),
[`run.log`](run.log), [`results/summary.txt`](results/summary.txt), [`results/optima.json`](results/optima.json),
[`results/profile_profile_v2.csv`](results/profile_profile_v2.csv), [`figures/index.html`](figures/index.html).

**Ce qu'on teste.** Deux décisions du 29/10, prises avec le collègue :
- **le profil d'usage** : [`profiles/profile_v2.json`](../../profiles/profile_v2.json). C'est `profile_v1` corrigé :
  - TWS est limité à [5, 17] nd, parce qu'en dessous de 5 nd ce n'est pas un cas d'usage ;
  - les mêmes bosses à 40° et 80°, avec la même queue ;
  - 12 conditions placées à la main sur les bords du domaine, à 1 % chacune : TWS 5 et 17 pour TWA 40, 60, 80
    et 100 ; TWA 32 et 110 pour TWS 8 et 14 ;
  - le creux de la vallée, TWA 60 et TWS 11, à 2 % ;
  - 49 conditions en tout ;
- **ne pas restreindre la section** : `--box-scale 3`. Seules restent les contraintes physiques : Ixx ≥ Ixx actuel,
  pas de croisement des surfaces, confiance NeuralFoil.
  - Jusqu'à l'échelle 2, la boîte reste ce qui limite.
  - À l'échelle 3, l'absence de croisement commence à couper (94 % des tirages passent).
  - La PCA à 10 modes garde 99,87 % de la variance.

Les autres réglages sont ceux du 7e run : niveaux NPLLT × NF et AVL × XFOIL itéré, même graine de criblage, mêmes
critères de convergence et de redémarrage.

**Avant le run** :
- La VPP a été testée sur les 49 conditions : 47 sont résolues, soit 98 % de la probabilité (contre 79 % pour
  `profile_v1`). Les échecs sont à 60° et 80° à 17 nd.
- Un test court ([`../2026-10-29_0943_smoke_profile_v2_box3`](../2026-10-29_0943_smoke_profile_v2_box3)) n'a eu
  aucun échec de code dans la boîte à l'échelle 3.

## En bref

- **Le meilleur résultat jusqu'ici, mais le gain sur le 7e run est petit.**

  | section | vitesse sur `profile_v2` [nd] | gain sur la section de départ |
  | --- | --- | --- |
  | ce run, région 0 (retenue) | 13,566 | +0,256 nd |
  | ce run, région 1 | 13,558 | +0,249 nd |
  | 7e run | 13,545 | +0,235 nd |
  | 6e run | 13,496 | +0,187 nd |
  | 5e run | 13,490 | +0,181 nd |
  | départ (section admissible la plus proche de l'actuelle) | 13,310 | — |

  Les vitesses sont validées contre AVL × XFOIL et moyennées sur les 47 conditions résolues pour toutes les sections.
  Sans les conditions en butée à 24 nd (44 conditions, 91 % du poids), l'écart avec le 7e run est de +0,049 nd.
- **La boîte n'est plus active.**
  - La meilleure section a besoin de l'échelle 2,13 sur 3, et la région 1 de 1,70.
  - Elle est loin de la limite de confiance NeuralFoil : 0,93 à 0,96 pour un seuil à 0,90.
  - Ixx est sur la contrainte : 1,004 × Ixx(actuelle).
  - C'est donc la contrainte d'Ixx et l'objectif qui fixent la section, plus la boîte.
- **La cambrure baisse encore, et elle recule.**
  - La moyenne des poids de cambrure vaut 0,052, contre 0,069 au 7e run et 0,108 pour la section actuelle.
  - La cambrure maximale passe à 2,2 % à 55 % de corde, contre 2,4 % à 42 % au 7e run et 4,2 % à 33 % pour
    l'actuelle.
  - L'épaisseur maximale reste à 12,5 %, mais elle recule à 30 % de corde, contre 27 % pour l'actuelle.
- **Un plateau plutôt qu'un pic.**
  - Les deux régions sont à 1,8 l'une de l'autre en scores PCA, mais à 0,008 nd l'une de l'autre en vitesse.
  - Le 7e run est entre les deux, à 1,4–1,5 de chacune.
  - La région 1 n'avait pas fini de converger à la limite de temps.
- **⚠ L'extrado n'est plus convexe.**
  - Sa courbure change de signe à 69 % et à 91 % de corde : la surface est légèrement creuse entre les deux.
  - On le voit dans les poids de l'extrado : 0,154, puis 0,085, puis 0,165 vers le bord de fuite.
  - Ni la section actuelle ni celle du 7e run n'ont ça. La famille n'impose aucune régularité.
  - **À regarder avant de croire au gain** : est-ce fabricable, et le gain tient-il en CFD ? Le surrogate est exact
    contre AVL × XFOIL (0,0006 nd RMS), donc ce n'est pas une erreur du surrogate, mais une question de fidélité
    d'AVL × XFOIL sur cette forme.
- **Les gains par allure, sur la section de départ et sur le 7e run :**

  | allure | sur le départ | sur le 7e run |
  | --- | --- | --- |
  | près (bosse à 40°) | +0,05 nd | −0,01 nd |
  | travers (bosse à 80°) | +0,34 nd | +0,03 nd |
  | queue du portant | +0,52 nd | +0,08 nd |
  | conditions des bords | +0,38 nd | +0,12 nd |

  - Les gains viennent du travers et de la queue par vent moyen à fort : +0,4 à +1,4 nd entre 12 et 15 nd.
  - **Au près serré par vent fort, ce run perd** : à 35°–40° et 15,4 nd, −0,16 à −0,20 nd sur le départ et −0,10 à
    −0,12 nd sur le 7e run. La section moins cambrée échange du près serré contre du travers.
- **Le surrogate est juste à la meilleure section** : erreur de vitesse contre AVL × XFOIL de 0,0006 nd RMS (au plus
  0,002 nd).

## Points ouverts

- **La butée à 24 nd de la VPP** reste au travers à 15,5–17 nd : 77°, 83° et 92° à 15,5 nd, 80° et 100° à 17 nd.
  - Elle vient de MDS : les RS de coque ne sont calées que jusqu'à 24 nd (`_SOLVER_SPEED_KTS = (6.0, 24.0)` dans
    `IntensSY_config.py`).
  - Ces conditions ne départagent pas les sections : le gain au travers par vent fort est sous-estimé.
  - Il faudrait de nouvelles données de coque pour la relever.
- **La forme de l'extrado** : faut-il une contrainte de convexité (ou de régularité des poids) dans la famille ?
- **Le compromis près / travers** dépend directement des poids du profil d'usage. Si le près serré par vent fort
  compte plus que `profile_v2` ne le dit, l'optimum change.
