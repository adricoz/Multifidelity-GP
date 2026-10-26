# 4e vrai run : recherche sur la distribution d'usage, avec recherche locale de la section

Du 23/10 à 17 h 02 au 24/10 à 1 h 22 (8 h 20, dont 8 h 10 de boucle). Fichiers : [`config.json`](config.json),
[`run.log`](run.log), [`results/summary.txt`](results/summary.txt) (réécrit le 26/10 avec la comparaison corrigée),
[`figures/index.html`](figures/index.html).

**En bref.** La recherche cible maintenant la distribution d'usage **supposée**, `profiles/guess_bimodal.json` :
- un mode près (TWA 45 ± 4°) et un mode portant (TWA 135 ± 7°), de même poids ;
- TWS 13 ± 3 nd ;
- soit 18 conditions.

Elle trouve un profil **encore moins cambré** que celui des runs 2 et 3, de **12,54 % d'épaisseur, avec un Ixx exactement
à la contrainte**. Ses gains de vitesse moyenne pondérée, sur les conditions que la VPP résout pour toutes les sections
(72 % de la probabilité) :
- **sur le profil actuel : +0,16 nd** (+0,10 au près, +0,26 au portant) ;
- **sur le meilleur profil des runs 2 et 3 : +0,056 nd.**

## Ce qui a tourné

C'est la chaîne du run 3 : HF = NPLLT × XFOIL, AVL en validation, FSI, lots de 16 choix, 28 + 30 processus, plan de départ
256 BF + 64 HF, même boîte de sections. S'y ajoutent les changements du 23/10 (voir
[`../../CHANGEMENTS_2026-10-23.md`](../../CHANGEMENTS_2026-10-23.md)) :
- **la recherche sur les 18 nœuds de la distribution,** pondérés par leur probabilité ;
- **la recherche locale :** la moitié des sections criblées autour de la titulaire, et des nuages qui varient aussi la
  section ;
- **une graine de criblage propre** (875768254) ;
- **les candidats proches seulement ;**
- **des hyperparamètres figés au-delà de 2 500 points.**

## Ce qui s'est passé

- **Le volume.** 1 920 choix en 120 lots. Le plan final compte **2 029 BF et 1 408 HF**, contre 952 et 313 au run 3. Les
  hyperparamètres ont été réestimés jusqu'à 2 098 points, puis figés.
- **La recherche a convergé en environ 1 h 15.** Il y a eu 23 changements de titulaire dans les 30 premières mises à jour
  (jusqu'à 74 min), avec un gain médian de 0,027 nd par changement. Les 25 changements suivants valent chacun 0,006 nd
  au plus, et 0,0002 nd en médiane : la recherche locale ne fait plus que des micro-ajustements, sans effet sur la
  vitesse.
- **La région de confiance a suivi la règle prévue.**
  - Elle s'est élargie jusqu'à 3 au début, quand la titulaire progressait.
  - Elle s'est resserrée ensuite, jusqu'au rayon minimal de 0,1 à partir de 3 h 43.
  - Les sections tirées localement ont occupé les 3 places du trio de concurrentes dès les premières mises à jour.
- **Le risque de « malédiction du vainqueur » ne s'est pas matérialisé.** Les changements tardifs sont minuscules, et non
  des sauts surestimés.

## Le résultat

La comparaison est cohérente : le surrogate final, avec la vitesse de chaque état corrigée au premier ordre de l'écart
mesuré contre le code. On prend la moyenne pondérée sur les 13 conditions résolues pour les trois sections, soit 72 % de
la probabilité. Les 5 autres (le portant profond, et 141° par petit temps) n'ont pas d'équilibre VPP, même avec la RS de
production. Source : [`results/validation.csv`](results/validation.csv).

| Section | Vitesse, corrigée NPLLT × XFOIL | Vitesse, corrigée AVL × XFOIL |
| --- | --- | --- |
| profil actuel | 12,923 nd | 12,924 nd |
| meilleur des runs 2 et 3 | 13,028 nd | 13,023 nd |
| **meilleur du run 4** | **13,084 nd** | **13,079 nd** |
| gain run 4 / actuel | **+0,161 nd** | **+0,156 nd** |

**Le gain par condition** (NPLLT × XFOIL) :
- **positif partout** ;
- **faible au près par petit temps :** +0,008 à +0,030 nd à 10 et 13 nd de vent ;
- **net à 16 nd au près :** +0,26 à +0,30 nd ;
- **fort au portant :** +0,17 à +0,42 nd.

**L'analyse sur la distribution** ([`results/profile_guess_bimodal.json`](results/profile_guess_bimodal.json),
[`figures/profile_guess_bimodal.html`](figures/profile_guess_bimodal.html)) donne **+0,152 nd** : +0,103 au près, +0,262
au portant. Le gain est positif aux 13 nœuds, de +0,006 à +0,42 nd. Le meilleur profil des runs 2 et 3 y gagne +0,100 nd.

**Une correction de cette analyse.** Faite en fin de run, elle donnait **−0,23 nd**, ce qui était faux. Elle résolvait
chaque nœud depuis le seul point de départ fixe de la VPP. À 48/16, celui-ci menait le meilleur profil sur un équilibre
lent mais « réussi » : 15,7 nd, gîte 2,7°, voiles réduites à 0,35. La validation, partie de l'équilibre de la titulaire,
trouve 19,2 nd, avec gîte 5,9° et voiles pleines. L'analyse est refaite le 26/10 avec des départs multiples
(`profile_analysis.robust_solve`), et elle rejoint la validation.

**Le profil** ([`figures/sections.html`](figures/sections.html), poids dans
[`results/best_section.json`](results/best_section.json)) :

| | Épaisseur max | Position | Ixx / actuel | Cambrure (poids moyen) |
| --- | --- | --- | --- | --- |
| profil actuel | 12,25 % | 0,27 c | — | 0,108 |
| meilleur des runs 2 et 3 | 12,53 % | 0,27 c | +2,4 % | 0,090 |
| **meilleur du run 4** | **12,54 %** | **0,26 c** | **+0,0 %** | **0,081** |

La tendance se confirme : **moins de cambrure fait aller plus vite**, jusqu'à ce que la contrainte de raideur
(Ixx ≥ Ixx actuel) arrête la recherche. L'optimum est sur cette frontière. Relâcher ou durcir la contrainte change donc
directement le résultat : c'est un choix de conception, pas un détail numérique.

**La précision du surrogate.**
- **Contre NPLLT × XFOIL :** 0,0001 nd RMS au meilleur profil, très raffiné, et 0,008 nd au profil actuel (0,022 au
  pire).
- **Contre AVL :** 0,018 à 0,022 nd RMS.
- **L'écart type annoncé est un peu optimiste :** |z| ≤ 2 sur 78 à 83 % des états du meilleur et de l'actuel, au lieu
  d'environ 95 %.

## Limites

- **La distribution est une hypothèse.** Le gain dépend nettement du domaine d'usage : il est fort au portant et à 16 nd
  au près, faible au près par petit temps.
- **28 % de la probabilité n'a pas d'équilibre VPP :** le portant profond par petit temps, gîte bloquée à 0°. C'est un
  réglage de la VPP à revoir (bornes, configuration du foil au portant).
- **Le premier solve de la titulaire part du point de départ fixe de la VPP,** comme l'analyse fautive. Les solves
  suivants partent de ses états. Un équilibre lent pourrait donc se propager : je propose de donner les départs
  multiples à ce premier solve, ce qui n'est pas fait sans ton accord.
- **Le gain reste vu par la ligne portante et AVL.** La CFD de validation viendra après la reprise de l'étude de
  maillage (`../2026-10-23_weekend/README.md`).
