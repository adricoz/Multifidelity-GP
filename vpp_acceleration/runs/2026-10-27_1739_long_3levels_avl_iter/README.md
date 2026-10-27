# 6e run : trois niveaux de fidélité sur bdfoil_v1, AVL itéré

Le 27/10, de 17 h 39 à 0 h 48 (la boucle s'arrête à 6 h 21, puis viennent la validation et l'analyse sur la
distribution). Fichiers : [`config.json`](config.json), [`run.log`](run.log),
[`results/summary.txt`](results/summary.txt), [`results/optima.json`](results/optima.json),
[`results/profile_guess_bimodal.json`](results/profile_guess_bimodal.json), [`figures/index.html`](figures/index.html).

**Ce qu'on teste.** Deux choses : le surrogate généralisé à L niveaux (ici 3), et le moteur bdfoil_v1
(`pipelines/CHANGEMENTS_2026-10-27_bdfoil_v1.md`), dont le couplage itératif de la portance d'AVL (J8,
`coupling = "induced_angle"`, la dernière version poussée par Antoine). La question posée : retombe-t-on sur la même
section qu'avant ?

**Les niveaux.**
1. NPLLT × NeuralFoil xxxlarge, sans coupure de confiance.
2. NPLLT × XFOIL.
3. AVL × XFOIL itéré : la référence du VPP.

Les coûts passés au critère sont 1 / 2 / 2,5. Le plan de départ fait 256 / 64 / 32 points, tous réussis.

**Les autres réglages** sont ceux des 4e et 5e runs :
- même boîte et même distribution supposée ;
- même graine de criblage (875768254) ;
- `--converge 10 0.01 --restarts 2 --min-gain 0.002 --sensitivity-median 3` ;
- limite de 360 min.

Les meilleurs profils des runs 4 et 5 sont validés en fin de run (`--compare`).

## En bref

- **Oui, c'est la même section.** On compare les trois meilleurs profils, tous validés contre la nouvelle référence
  (AVL × XFOIL itéré). Les vitesses sont corrigées au premier ordre et moyennées sur les 13 conditions résolues pour
  toutes les sections (72 % de la probabilité).

  | profil | vitesse [nd] | écart au meilleur de ce run |
  | --- | --- | --- |
  | ce run | 13,0877 | — |
  | 4e run | 13,0912 | +0,0035 nd |
  | 5e run | 13,0826 | −0,0051 nd |
  | profil actuel | 12,9292 | −0,159 nd |

  Le gain sur le profil actuel (+0,16 nd) est le même qu'aux runs 4 et 5.
- **La distance en scores PCA le confirme.**

  | paire | distance |
  | --- | --- |
  | ce run – 5e run | 0,95 |
  | ce run – 4e run | 1,30 |
  | 4e run – 5e run | 1,21 |
  | profil actuel – 4e run | 5,9 |

  L'écart entre les poids de Kulfan vaut au plus 0,016 avec le 4e run. Les trois profils tombent dans le même bassin.
  Leurs écarts sont du même ordre que l'écart entre les deux runs précédents.
- **Les trois régions convergent au même niveau.**
  - La région 0 converge à 204 min.
  - La région 1 part d'une section à 3,75 des optima et converge à 304 min.
  - La région 2 part à 3,42 et tourne jusqu'à la limite.
  - Optima finaux, sur 78 % de la probabilité : 12,8393 / 12,8387 / 12,8201 nd. C'est la région 0 qui est retenue.
- **Le surrogate est juste sur la référence.** Au meilleur profil, l'erreur de vitesse contre AVL × XFOIL vaut 0,0002 nd
  RMS, avec |z| ≤ 2 partout.
- **Le niveau 2 n'apporte rien en précision ici.** Sur les sections validées, l'erreur de vitesse contre AVL × XFOIL
  vaut 0,010 nd RMS pour NPLLT × XFOIL. C'est autant que pour NPLLT × NeuralFoil (0,0097 nd).
- **Le niveau 2 n'est pas moins cher non plus.** Il coûte presque autant qu'AVL itéré :

  | niveau | temps par point |
  | --- | --- |
  | 1 | 0,6 s |
  | 2 | 11,3 s |
  | 3 | 14,9 s |

  Les coûts 1 / 2 / 2,5 sous-estiment donc fortement l'écart entre le niveau 1 et les deux autres. Le temps de code se
  répartit en 636 s, 5 910 s et 2 956 s. Le niveau 2 en a pris deux fois plus que la référence, pour une information
  que l'écart BF–HF ne montre pas meilleure.
- **Le run est plus lent que le 5e** (convergence à 204 min contre 92). La cause principale est le coût de code des
  niveaux 2 et 3 (AVL itéré).

## À décider

- Garder le niveau 2 ? Une seule trajectoire ne suffit pas pour trancher.
- Recaler les coûts sur les temps mesurés, par exemple 1 / 19 / 25.
- Faire un run de contrôle à deux niveaux (NF / AVL itéré) avec les mêmes réglages. Ce serait la comparaison propre.
