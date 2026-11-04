# 3e vrai run : NPLLT × XFOIL, foil flexible, calculs en parallèle

22/10/2026, 15 h 19 → 16 h 30 (70 min) · [`config.json`](config.json) · [`run.log`](run.log) ·
[`results/summary.txt`](results/summary.txt)

**En bref.** Le run refait la recherche du 2e run avec un modèle plus fidèle, NPLLT × XFOIL. Il ajoute la déformation du
foil flexible et utilise les 32 cœurs. Il retrouve **le même profil** que le 2e run, moins cambré. Le gain sur le profil
actuel tient, et grandit un peu :

* **+0,24 nd de vitesse moyenne sur les 5 conditions** (+1,3 %) vu par NPLLT × XFOIL ;
* **+0,22 nd** vu par AVL × XFOIL ;
* contre +0,18 nd au 2e run, avec AVL et sans déformation.

Le meilleur profil a aussi **un Cpmin bien meilleur** : −0,87 contre −2,57 pour le profil actuel, avec NPLLT × XFOIL.
Sa marge à la cavitation est donc nettement plus grande.

## Ce qui a changé par rapport au 2e run

La famille de sections, le départ et les conditions sont ceux du 2e run :

* une boîte de demi-largeur Dim-Reduction centrée sur le profil actuel ;
* Ixx ≥ Ixx actuel ;
* NeuralFoil fiable ;
* départ = profil actuel.

Trois choses changent.

1. **Le niveau HF est NPLLT × XFOIL.** C'est la ligne portante du niveau BF, avec des polaires XFOIL au lieu de
   NeuralFoil. Ta recommandation du 15/10 pour le trim est celle-ci : sa portance voit la polaire visqueuse, alors que
   celle d'AVL ne voit que la ligne de cambrure. AVL × XFOIL ne sert plus qu'à la validation finale.
   * **Le coût :** une polaire XFOIL par nouvelle section (2 à 13 s), puis 0,1 s par attitude.
   * **Une mesure préalable**, sur le profil actuel : NeuralFoil xxxlarge et XFOIL donnent les mêmes forces à 0,1 %
     près. Les deux niveaux ne diffèrent vraiment que sur Cpmin (−0,9 contre −2,0), car NeuralFoil ne voit pas le pic de
     bord d'attaque.
   * **Le Cpmin de NPLLT × XFOIL** est défini comme celui d'AVL : 20e percentile, sur les bandes, du cpmin de la polaire
     XFOIL à l'incidence locale.
2. **Le foil est flexible** (`vpp_coupling.FSICorrection`). Les codes et le surrogate restent rigides. La VPP ajoute la
   correction de déformée de la RS FSI de production (bdFoil `CFoil_fsi.ipynb`, modèle de poutre « supported ») :
     δ_k(attitude, V) = (Ixx actuel / Ixx section) · b_k(attitude) · V²
   * **La loi en V².** À attitude fixée, l'effet de la déformée sur la RS suit V² : à 1,3 % près au pire sur les
     forces, entre 7 et 28 nd. b_k est la pente de la RS contre V².
   * **Le rapport d'Ixx.** Un profil plus raide se déforme moins : la flexion va en 1/EI, la torsion en 1/GJ, et les
     deux sont pris proportionnels à Ixx dans la famille.
   * **Le contrôle.** « Partie rigide de la RS + δ » redonne la VPP de production à 0,0005 nd près à 45/16, et à
     0,007 nd près à 90/14. C'est un test de `tests/test_vpp_coupling.py`.
   * **L'effet.** La déformée **fait gagner** de la vitesse : +0,075 nd à 45/16 et +0,31 nd à 90/14 par rapport au foil
     rigide.
   * **L'entrée vitesse.** MDS passe la vitesse au foil dès que la RS déclare l'entrée `Speed_kts`.
   * **Bending et Twisting** sont des sorties de rapport. Elles n'entrent pas dans l'équilibre.
3. **Les calculs tournent en parallèle** (Ryzen 9 5950X, 32 cœurs logiques) :
   * **les points par lots de 16.** Chacun est choisi par le critère comme si les précédents avaient déjà été calculés
     (« kriging believer »). La variance d'un GP ne dépend pas des valeurs observées, donc ce choix est exact pour
     notre critère. Les 16 calculs tournent ensuite dans 28 processus ;
   * **les solves VPP** d'une mise à jour et des sensibilités, dans 30 processus MDS. Chaque worker refait le solve de
     base avant ses différences finies : sans cela, le premier test court donnait dV/dFx = −32 au lieu de +83, à cause
     de la dépendance au chemin de MDS ;
   * **les 7 co-krigeages**, ajustés et évalués en fils ;
   * **la VPP ne calcule que la moyenne du surrogate**, sans sa variance, en O(n²) par appel ;
   * **les hyperparamètres** sont réajustés quand le plan a grossi de 25 %.

## Ce qui s'est passé

Figures : [`figures/convergence.html`](figures/convergence.html),
[`figures/points.html`](figures/points.html).

* **Le volume.**
  * Plan de départ : 256 BF + 64 HF, en 1,4 min.
  * Boucle : 816 choix en 51 lots, contre 373 pas au 2e run.
  * Plan final : 952 BF et **313 HF** (121 au 2e run).
  * Échecs : 3 BF et 3 HF sur 1 320 calculs (NPLLT non convergé, polaire XFOIL ratée).
* **Le débit.** Environ 12 choix par minute, contre 4,7 au 2e run. Le choix d'un lot de 16 a coûté de 25 s au début à
  100 s à la fin : le conditionnement et les gains croissent en n². C'est désormais le goulot, et non plus les codes.
  Les mises à jour de la VPP ont pris 4 à 20 s, contre 27 à 72 s au 2e run.
* **La recherche.** La meilleure section a changé 6 fois. Elle est fixée dès la mise à jour 14 (13 min), puis la vitesse
  plafonne à 19,074 nd jusqu'à la fin.
* **Une sensibilité aberrante.** À 46 min, dV/dFx a valu 201 au lieu de 111 à 45/16 pendant trois mises à jour, puis
  est revenu à sa valeur : un solve parti sur un autre équilibre. Cela n'a pas d'effet visible sur la recherche, déjà
  fixée.

## Le résultat

Les vitesses viennent de la VPP avec le surrogate final et la FSI. Chaque état est ensuite corrigé au premier ordre de
l'écart mesuré contre le code. Source : [`results/validation.csv`](results/validation.csv) et
[`figures/speeds.html`](figures/speeds.html).

| Condition | Actuel, NPLLT × XFOIL | Meilleur, NPLLT × XFOIL | Gain | Gain vu par AVL × XFOIL | Cpmin actuel → meilleur |
| --- | --- | --- | --- | --- | --- |
| 45/12 | 12,127 nd | 12,150 nd | +0,023 | +0,023 | −1,89 → −0,87 |
| 45/16 | 18,383 nd | 18,567 nd | +0,184 | +0,166 | −2,07 → −0,83 |
| 60/16 | 21,005 nd | 21,293 nd | +0,288 | +0,258 | −2,61 → −0,82 |
| 75/16 | 23,067 nd | 23,370 nd | +0,303 | +0,267 | −2,95 → −0,86 |
| 90/14 | 19,555 nd | 19,959 nd | +0,404 | +0,372 | −3,35 → −0,97 |
| **moyenne** | **18,827 nd** | **19,068 nd** | **+0,241 nd** | **+0,217 nd** | **−2,57 → −0,87** |

* **Le meilleur profil est celui du 2e run, à l'identique.** Il est décrit dans
  [`../2026-10-22_1131_real_1h30_current_box/README.md`](../2026-10-22_1131_real_1h30_current_box/README.md) et dans
  [`figures/sections.html`](figures/sections.html) : 12,53 % d'épaisseur, Ixx +2,4 %, moins de cambrure.
  * **Pourquoi la même section :** les sections candidates du criblage sont tirées avec la même graine dans la même
    famille, donc les deux runs ont criblé **les mêmes réserves de sections**. Parmi elles, le modèle visqueux avec
    déformée choisit le même profil que le modèle AVL rigide, à la même mise à jour.
  * **C'est une confirmation**, pas un hasard : le choix tient au changement de modèle 3D et à l'ajout de la FSI.
* **La précision du surrogate final** :
  * contre NPLLT × XFOIL : 0,010 nd RMS au meilleur profil et 0,005 nd au profil actuel ;
  * par coefficient : 0,02 à 0,3 % d'erreur, 0,5 à 1,4 % sur Cpmin (5 à 17 % au 2e run) ;
  * aux états validés, |z| ≤ 2 partout.
* **AVL contre NPLLT × XFOIL.** AVL donne 0,03 à 0,06 nd de plus que NPLLT × XFOIL. Le gain entre les deux profils
  diffère de 0,02 nd, soit 10 %.
* **L'effet de la FSI** sur le profil actuel, avec AVL : 18,885 nd avec déformation, contre 18,733 nd au 2e run sans
  déformation, soit +0,15 nd.
* **Ce qui change à l'équilibre.** Chaque profil est pris à son propre état VPP. Le meilleur profil porte 3 % de Fy en
  moins et a 3 à 5 % de traînée (|Fx|) en moins. La VPP l'équilibre avec 0,1 à 0,2° de rake en plus et un yaw plus
  élevé d'environ 0,4°. Moins cambré, il travaille à un Cpmin bien moins négatif. Je n'ai pas regardé les distributions
  de Cp pour dire où passe le pic.

## Sur une distribution d'usage (hypothèse)

L'analyse a été faite après coup, avec `scripts/profile_analysis.py --profile profiles/guess_bimodal.json`.
Elle ne change pas la recherche : la VPP, avec le surrogate final et la FSI, est résolue aux 18 nœuds d'une
distribution bimodale **supposée**, en attendant la vraie :

* un mode près, TWA 45 ± 4° ;
* un mode portant, TWA 135 ± 7° ;
* le même poids pour les deux, TWS 13 ± 3 nd.

Sources : [`results/profile_guess_bimodal.json`](results/profile_guess_bimodal.json) et
[`figures/profile_guess_bimodal.html`](figures/profile_guess_bimodal.html).

* **Le gain du meilleur profil : +0,093 nd** sur les nœuds que la VPP résout pour les deux profils, soit 72 % de la
  probabilité. Il vaut **+0,076 nd au près** et **+0,13 nd au portant**. Sur les 5 conditions du run, centrées sur le
  reaching, il valait +0,24 nd. Le profil est meilleur à chacun des 13 nœuds communs, de +0,015 nd (près, 10 nd de
  vent) à +0,19 nd (45°, 15,5 nd). Mais le gain dépend nettement du domaine d'usage : il est faible au près par petit
  temps.
* **Ces chiffres ont été corrigés le 26/10.**
  * La première analyse (+0,099 nd, dont +0,085 au près) résolvait chaque nœud depuis le seul point de départ fixe de
    la VPP. Celui-ci peut converger sur un équilibre lent mais « réussi » par vent de 16 nd.
  * L'analyse est maintenant faite avec des départs multiples, en gardant l'équilibre le plus rapide
    (`profile_analysis.robust_solve`, `CHANGEMENTS_2026-10-23.md`).
  * Ici, l'écart reste faible (−0,006 nd), parce que les équilibres lents touchaient les deux profils à peu près de la
    même façon. Au run 4, il changeait le signe du gain.
* **Des nœuds sans équilibre.** La VPP ne résout pas le portant profond par petit temps : la gîte bute sur sa
  borne de 0°. C'est vrai aussi avec la RS de production, donc ce n'est pas le surrogate. C'est un réglage de
  la VPP à revoir (bornes, configuration du foil au portant) avant d'exploiter le mode portant.
* **L'incertitude.** L'écart type du surrogate vaut environ 0,006 nd aux nœuds résolus. C'est l'estimation
  du modèle lui-même : les états au portant n'ont pas été validés contre les codes.

## Limites

* **La recherche ne choisit que parmi des sections tirées au hasard.** En 52 mises à jour, elle a criblé environ
  13 000 sections de la famille, puis résolu la VPP complète sur les 3 meilleures de chaque mise à jour. Le meilleur
  profil est le meilleur de ces tirages, et non un optimum local au sens strict.
  * **Première suite :** un raffinement local autour de lui, par un criblage dans un voisinage qui se resserre, ou un
    gradient sur le surrogate.
  * **Seconde suite :** un run avec une autre graine.
* **La FSI est une correction de premier ordre.** Elle reprend la déformée de la RS de production, calculée pour le
  profil actuel, à charges égales, et la met à l'échelle de l'Ixx. Un calcul FSI des deux profils, avec bdFoil
  `simulation_fsi`, la vérifierait.
* **La validation reste entre modèles de ligne portante et de surface portante.** Quelques calculs CFD aux états VPP des
  deux profils trancheraient : 10 par profil donneraient un ordre de grandeur, d'après l'étude 4 du rapport.
* **Une sensibilité ponctuelle aberrante** (voir plus haut) : on pourrait prendre la médiane de deux solves, ou rejeter
  une valeur loin de la précédente.
