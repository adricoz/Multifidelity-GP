# 2e vrai run : la meilleure section autour du profil actuel, à Ixx au moins égal

22/10/2026, 11 h 31 → 12 h 54 (82 min) · [`config.json`](config.json) · [`run.log`](run.log) ·
[`results/summary.txt`](results/summary.txt)

**En bref.** En partant du profil actuel du C-foil, la recherche trouve un profil **moins cambré**, aussi épais (12,5 %
contre 12,25 %) et un peu plus raide (Ixx +2,4 %). Avec la chaîne NPLLT × NeuralFoil / AVL × XFOIL et la VPP MDS, il
gagne **+0,18 nd de vitesse moyenne sur les 5 conditions (+0,95 %)**. Le gain tient dans chaque condition, de +0,02 nd à
45/12 à +0,35 nd à 90/14.

## Ce qui a tourné

Comme le premier run ([`../2026-10-22_1021_real_1h/README.md`](../2026-10-22_1021_real_1h/README.md)), avec ces
changements :

* **La famille.** C'est une boîte de **demi-largeur** Dim-Reduction **centrée sur le profil actuel** (épaisseur MC2 et
  cambrure de référence, 12,25 %), avec :
  * des surfaces sans croisement ;
  * **Ixx ≥ Ixx du profil actuel** (6,72·10⁻⁵, corde unité) ;
  * NeuralFoil fiable.
  Tout est vérifié **sur le profil décodé**, celui que voient les codes. La projection sur 10 modes faisait perdre un
  peu d'Ixx à 24 % des profils.
* **Le départ de la recherche :** le profil actuel, admissible une fois décodé, avec le même Ixx.
* **Les réglages :** k = 10 (99,9 % de la variance) ; départ 128 BF + 32 HF ; sensibilités une mise à jour sur trois ;
  points en échec exclus ; boucle arrêtée à 80 min.

## Ce qui s'est passé

Figure : [`figures/convergence.html`](figures/convergence.html).

* **Le volume :** 373 pas ; 471 calculs BF (89 s de code) et 132 calculs HF (20 min de code) ; plan final de 458 BF et
  121 HF. **Aucun échec** dans la boucle ; seulement 3 au départ.
* **Les niveaux :** l'algorithme a fait 283 choix BF et 90 choix HF.
* **Le placement :** tous les points ajoutés sont près des états VPP, à 0,8 demi-largeur de nuage en médiane
  ([`figures/points.html`](figures/points.html)).
* **La recherche :** la meilleure section a changé 6 fois, puis **la vitesse plafonne vers 18,91 nd dès 20 min**. La
  recherche a convergé dans ce budget.
* **Le temps :** les mises à jour de l'optimiseur ont pris 29 % de la boucle (24 min), contre 42 % au premier run.

## Le résultat, mesuré de façon cohérente

**Le chiffre de départ du résumé (18,616 nd) n'est pas la bonne référence.** C'est la vitesse du profil actuel vue par
le *premier* surrogate, avant tout raffinement. La comparaison juste prend les deux profils avec le surrogate *final*,
puis corrige chaque état de l'écart mesuré contre AVL, au premier ordre :

  V_AVL ≈ V_surrogate − Σ_k (∂V/∂C_k)(Ĉ_k − C_k^AVL)

Source : [`results/validation.csv`](results/validation.csv).

| Condition (TWA/TWS) | Profil actuel | Meilleur profil | Gain |
| --- | --- | --- | --- |
| 45/12 | 12,118 nd | 12,136 nd | +0,018 |
| 45/16 | 18,393 nd | 18,541 nd | +0,148 |
| 60/16 | 20,927 nd | 21,149 nd | +0,222 |
| 75/16 | 22,844 nd | 22,991 nd | +0,147 |
| 90/14 | 19,383 nd | 19,735 nd | +0,352 |
| **moyenne** | **18,733 nd** | **18,911 nd** | **+0,177 nd (+0,95 %)** |

Avec le surrogate final seul, le gain moyen vaut +0,214 nd. La correction contre AVL le ramène à +0,177 nd.

**La précision du surrogate final** :

* au meilleur profil, raffiné : 0,005 nd d'erreur de vitesse RMS contre AVL, et 0,02 à 0,4 % d'erreur par coefficient ;
* au profil actuel, peu revisité après le départ : 0,07 nd RMS, dont 0,15 nd à 90/14, où l'état dépasse un peu son
  incertitude annoncée.

Le gain corrigé est donc fiable à quelques centièmes de nœud près, et à l'écart de modèle AVL près.

**Le profil** ([`figures/sections.html`](figures/sections.html), poids Kulfan dans
[`results/best_section.json`](results/best_section.json)) :

* 12,53 % d'épaisseur à x/c 0,27, la même position que l'actuel ;
* Ixx de 6,88·10⁻⁵, soit +2,4 % ;
* **moins de cambrure** : premier score PCA à −4 ; l'extrados et l'intrados descendent tous deux, l'intrados davantage.

Le surrogate prédit moins de portance à attitude égale, et la VPP retrouve l'équilibre avec moins de traînée.

**Cohérence avec la VPP de production.** Au profil actuel et à 45/16, notre chaîne donne 18,39 nd. La VPP de production,
avec la RS FSI du bateau, donne 18,41 nd.

## Limites

* Le gain est **vu par AVL × XFOIL** : un AVL rigide, dont la portance ne voit que la ligne de cambrure. Ton rapport du
  15/10 recommande NPLLT × XFOIL pour le trim. **Un profil moins cambré est justement le cas où AVL et un modèle
  visqueux peuvent diverger.** Le gain est à confirmer, avec NPLLT × XFOIL puis par quelques calculs CFD aux états VPP
  des deux profils.
* La correction contre AVL est au premier ordre. Je n'ai pas résolu la VPP avec AVL dans la boucle : environ 240 appels
  au foil par point, soit 30 min par condition.
* Cpmin est mal prédit (5 à 17 %) : c'est à surveiller si la cavitation entre en jeu.
* Le surrogate est rigide, sans flexion ; la RS de production est en FSI.
