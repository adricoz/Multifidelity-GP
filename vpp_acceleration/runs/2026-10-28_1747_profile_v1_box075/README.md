# 7e run : profil d'usage `profile_v1`, boîte élargie à 0,75

Le 28/10 de 17 h 47 au 29/10 à 2 h 38 (la boucle s'arrête à 1 h 45, puis viennent la validation et l'analyse sur la
distribution). Fichiers : [`config.json`](config.json), [`run.log`](run.log),
[`results/summary.txt`](results/summary.txt), [`results/optima.json`](results/optima.json),
[`results/profile_profile_v1.json`](results/profile_profile_v1.json), [`figures/index.html`](figures/index.html).

**Ce qu'on teste.** Deux changements par rapport aux runs 4 à 6 :
- **la distribution d'usage** : [`profiles/profile_v1.json`](../../profiles/profile_v1.json), faite avec l'éditeur
  `Qwen-Projects/Bimodal-Sailing-Discretizer` ;
  - deux bosses à TWA 40 et 80 ± 10, avec une vallée lisse entre les deux ;
  - une queue large jusqu'à ~100° ;
  - 3 conditions placées à la main à 60° ;
  - TWS quasi uniforme de 1 à 20 nd ;
  - 39 conditions, contre 18 pour `guess_bimodal` ;
- **la boîte des sections** : `--box-scale 0.75` au lieu de 0,5, pour voir si l'optimum part ailleurs.

**Les niveaux** sont ceux de la décision du 28/10 : NPLLT × NeuralFoil et AVL × XFOIL itéré (défaut, sans `--levels`).
Les autres réglages sont ceux du 6e run :
- même graine de criblage ;
- `--converge 10 0.01 --restarts 2 --min-gain 0.002 --sensitivity-median 3` ;
- plan de départ 256 / 64 ;
- limite de 720 min.

Le run a tourné sur 8 + 8 processus à côté de la CFD du maillage fin (14 processus) : **ses durées ne sont pas
comparables** à celles des autres runs.

**Avant le run**, la VPP a été testée sur les 39 conditions avec le surrogate du 6e run, pour deux sections.
- Elle converge sur 31 d'entre elles, soit 78,8 % de la probabilité.
- Les échecs sont à TWS 3,5 nd (près) et 17,4 nd (près 53°, travers).

## En bref

- **Un nouveau bassin, meilleur sur ce profil.** Les trois régions convergent au même niveau, à moins de 1,4 l'une
  de l'autre en scores PCA. Elles sont à 2,0–2,5 des meilleures sections des runs 5 et 6, qui ne sont qu'à 0,64 l'une
  de l'autre.

  | section | vitesse sur `profile_v1` [nd] | gain sur la section actuelle |
  | --- | --- | --- |
  | ce run | 13,104 | +0,208 nd |
  | 6e run | 13,053 | +0,156 nd |
  | 5e run | 13,045 | +0,149 nd |
  | section actuelle | 12,900 | — |

  Les vitesses sont validées contre AVL × XFOIL et moyennées sur les 31 conditions résolues pour toutes les sections
  (79 % de la probabilité).
- **La section est encore moins cambrée.** La moyenne des poids de cambrure vaut 0,069, contre 0,082 au 6e run et
  0,108 pour la section actuelle. C'est la même tendance que depuis le 2e run.
- **Ixx est toujours sur la contrainte** : 1,001 × Ixx(actuelle).
- **La meilleure section est sur le bord de la nouvelle boîte.**
  - Il faut exactement l'échelle 0,75 pour la contenir, et 11 de ses 16 poids sont hors de l'ancienne boîte (0,5).
  - Les meilleures sections des runs 5 et 6 étaient déjà au bord de la leur : 0,486 et 0,497 pour 0,5.
  - **La boîte est donc active.** Le gain vient en partie de son élargissement, sans qu'on puisse le séparer de l'effet
    du profil : il faudrait pour cela un run `profile_v1` en boîte 0,5.
- **Les gains par allure** :
  - surtout au travers et dans la queue à 12,6–15 nd : +0,35 à +1,0 nd ;
  - presque nuls au près à 8–13 nd (−0,04 à +0,03 nd).
- **Le surrogate est juste à la meilleure section** : erreur de vitesse contre AVL × XFOIL de 0,0003 nd RMS, avec
  |z| ≤ 2 partout.
- **Les régions** :
  - la région 0 converge à 168 min ;
  - la région 1 part à 6,06 des optima et converge à 280 min ;
  - la région 2 part à 3,11 et converge à 448 min ; c'est elle qui est retenue.

## Points à corriger

- **Butée de vitesse de la VPP.** Au travers à 17,4 nd (68° à 92°), toutes les sections donnent exactement
  24,000 nd : c'est une butée, pas un équilibre.
  - Dans la recherche, seule la condition 92°/17,4 nd est comptée, pour ~2,6 % du poids. Elle ne départage pas les
    sections, mais elle entre dans les « 79 % ».
  - Il faut relever la borne de vitesse de la VPP, ou retirer ces vents au travers du profil.
- **Vents extrêmes du profil.** Ramener TWS de [1, 20] à environ [5, 17] nd éviterait les conditions sans équilibre.
