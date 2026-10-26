# Week-end du 23/10/2026 : le run long, puis l'étude de maillage CFD

Lancé le vendredi 23/10 à 17 h 02, sans surveillance, comme demandé : le run long d'abord, puis le test CFD, en 2 jours
au plus. Tout est enchaîné par [`orchestrate.ps1`](orchestrate.ps1), qui tourne détaché de la session Claude et garde
le PC éveillé (`SetThreadExecutionState`, comme le lanceur bdcfd ; le plan d'alimentation n'est pas modifié). Son
journal est [`orchestrate.log`](orchestrate.log), et l'échéance est le dimanche 25/10 à 16 h 02.

## Ce qui s'est passé (lecture du 26/10)

**L'enchaînement.**
- Le run long a tourné du 23/10 17 h 02 au 24/10 1 h 22.
- L'étude CFD a tourné ensuite, de 1 h 22 à 12 h 35 (11,2 h), et ses 6 cas ont abouti.
- Rien n'a été sauté, et il restait du temps sur l'échéance.

La lecture du run long est dans son propre `README.md`. Les conclusions de l'étude CFD suivent. Les chiffres sont dans
[`cfd/report.md`](cfd/report.md), corrigé le 26/10 : la première version donnait des GCI pour des grandeurs qui ne
convergent pas.

1. **La traînée est convergée, en partie par compensation.**
   - La traînée totale change de −0,64 % de G1 à G2, puis de −0,21 % de G2 à G3 (GCI de G2 : 0,34 %).
   - La traînée de pression baisse (−2,4 % puis −1,3 %) pendant que le frottement monte (+1,7 % puis +1,1 %).
   - Leurs GCI propres, environ 3 %, disent la vraie incertitude.
2. **La portance ne converge pas.**
   - √(Fy² + Fz²) gagne +0,99 % puis +1,04 % : les écarts ne diminuent pas (R = 1,11 pour Fy, 1,04 pour Mz, 0,93 et
     0,94 pour My et Fz).
   - Il n'y a pas de GCI possible : la série n'est pas dans le régime asymptotique.
   - Mx converge lentement (GCI de G2 : 3,3 %).
3. **Les rapports de raffinement réels ne sont que de 1,20 et 1,24,** sous le 1,3 recommandé. Le nombre de cellules
   croît moins vite que le cube du rapport de fond.
4. **Les couches sont la cause la plus probable du défaut.**
   - Leur couverture baisse quand on raffine : 77,9, 75,2 puis 72,3 %. L'épaisseur des couches est absolue (4,5 mm),
     alors que les mailles de surface rétrécissent, et snappy en retire davantage.
   - Le y+ moyen n'est donc pas resté constant (108, 94, 86, au lieu d'environ 50).
   - La zone proche de la paroi change d'un maillage à l'autre, ce qui empêche une convergence propre. C'est une
     hypothèse, à confirmer par l'essai suivant.
5. **La résolution de paroi n'a pas été testée comme prévu.**
   - On a obtenu des y+ de 83, 94 et 121, au lieu de 30, 50 et 100. Le y+ moyen est dominé par les zones sans couches,
     où il monte jusqu'à 1 900.
   - Dans cette plage, la traînée varie de moins de 0,5 %. Un vrai y+ bas reste à tester.
6. **Le domaine est trop petit.**
   - En doublant le domaine, Fx monte de 1,25 %, Fy de 0,42 %, Fz de 0,61 % et Mx de 0,98 %.
   - En vitesse, cela fait −0,037 nd : **plus que l'effet du maillage** (G2 → G3 : −0,011 nd ; G1 → G2 : −0,013 nd).
   - Ces équivalents vitesse viennent des sensibilités du run 3 à 45/16, et ne sont qu'un ordre de grandeur.
7. **700 itérations suffisent :** la dérive de Fx sur les 200 dernières reste sous 4·10⁻⁵. Les durées sont de 1 h pour
   G1, 1 h 45 pour G2 et 3 h 40 pour G3.

**La suite proposée pour la reprise :**
1. **Réparer les couches :** `thicknessModel firstAndRelativeFinal`, première couche absolue (y+ fixé), dernière couche
   relative à la maille ; réglage de `addLayersControls` ; couverture visée d'au moins 95 %, vérifiée sur G1.
2. **Agrandir le domaine** au moins à × 2.
3. **Refaire G1/G2/G3** avec ces corrections, avec un rapport de raffinement d'au moins 1,3 sur les cellules et pas
   seulement sur le fond.
4. **Suivre la répartition du y+** sur la surface, et non sa moyenne.

## 1. Le run long (≈ 8 h 30)

**Dossier :** `runs/2026-10-23_1702_real_search_profile_local`. Ses figures sont dans `figures/index.html`, et sa
sortie brute dans [`long_run.out`](long_run.out).

C'est le run 3, avec les changements du 23/10 expliqués dans
[`../../CHANGEMENTS_2026-10-23.md`](../../CHANGEMENTS_2026-10-23.md) :
- **la recherche sur la distribution d'usage supposée** (`profiles/guess_bimodal.json`, 18 nœuds) ;
- **la recherche locale** : la moitié des sections criblées est tirée autour de la titulaire, et les nuages varient
  aussi la section ;
- **une graine de criblage tirée au hasard,** notée dans `config.json` ;
- **les candidats proches seulement ;**
- **des hyperparamètres figés au-delà de 2 500 points.**

Le reste ne change pas : HF = NPLLT × XFOIL, FSI, lots de 16, 28 + 30 processus, plan de départ 256 BF + 64 HF. La
boucle s'arrête à 480 min.

À la fin :
- **la validation** du meilleur profil, du profil actuel et du meilleur profil des runs 2 et 3, avec NPLLT × NeuralFoil,
  NPLLT × XFOIL et AVL, aux états VPP des 18 nœuds ;
- **l'analyse sur la distribution** (`results/profile_guess_bimodal.json`), et les figures avec leur index.

## 2. L'étude de maillage CFD (≈ 13 h, après le run long)

Elle est faite par le script [`cfd_mesh_study.sh`](cfd_mesh_study.sh), qui tourne dans WSL Ubuntu-24.04 avec OpenFOAM
v2512 sur 16 processus. Les cas sont dans `/home/bdcfd/bdcfd/meshstudy_20261023/`, et les résultats dans
[`cfd/`](cfd) :
- `results.csv`, une ligne par cas ;
- `cfd.log` ;
- `report.md`, le GCI et les tableaux, écrit par [`cfd_mesh_report.py`](cfd_mesh_report.py) ;
- la figure `figures/cfd_mesh_study.html`.

**Le cas de départ.** C'est le calcul fin du 20/10, fait par le pipeline bdfoil_v1 : même STL (le C-foil à une
attitude, 11,8 m/s), kOmegaSST, loi de paroi de Spalding, plan de symétrie à z = 0, raffinements de surface, de distance
et de sillage (`wakeBox`, niveau 4).

**Ce qui change, et pourquoi.**
- **Des couches de prismes en taille absolue,** pour un y+ constant dans la série. Le 20/10, les couches étaient
  relatives à la maille : y+ est passé de 217 à 96 entre les deux maillages. Le frottement a bougé de 22 % et la
  traînée de pression de −9 %, si bien que la traînée totale ne bougeait que de 2,4 %, par compensation. Une série où
  le régime de paroi change ne mesure pas la convergence en maillage.
- **Un rapport de raffinement de 1,3, obtenu par le maillage de fond** (blockMesh). Les niveaux de snappy sont relatifs
  au fond, donc toutes les tailles de maille changent dans le même rapport. Un niveau snappy de plus (rapport 2)
  mènerait à environ 16 M cellules pour le maillage fin.
- **700 itérations :** 200 au premier ordre, puis 500 au second. Le 20/10, les efforts étaient convergés à 10⁻⁴ dès
  l'itération 380, mais le calcul avait tourné jusqu'à 2 470. La dérive de Fx sur les 200 dernières itérations est
  enregistrée pour le vérifier.

**Les cas, par ordre de priorité.** Un cas ne démarre que s'il tient dans le temps restant.

| Cas | Maillage de fond | Couches (1re, nombre) | y+ visé | Rôle | Durée estimée |
| --- | --- | --- | --- | --- | --- |
| G2_medium_yp50 | 54 × 39 × 25 | 0,27 mm × 8 | ≈ 50 | série GCI, maillage moyen (candidat de production) | 2 h |
| G1_coarse_yp50 | 42 × 30 × 19 | 0,27 mm × 8 | ≈ 50 | série GCI, grossier | 1 h |
| G3_fine_yp50 | 70 × 51 × 33 | 0,27 mm × 8 | ≈ 50 | série GCI, fin | 4,5 h |
| Y100_medium | 54 × 39 × 25 | 0,54 mm × 6 | ≈ 100 | résolution de paroi | 2 h |
| Y30_medium | 54 × 39 × 25 | 0,16 mm × 10 | ≈ 30 | résolution de paroi | 2,2 h |
| D2_coarse_yp50 | 84 × 60 × 38 (domaine × 2) | 0,27 mm × 8 | ≈ 50 | domaine (champ lointain) | 1,3 h |

Le y+ est celui que donne OpenFOAM, au centre de la première maille. On en tire la hauteur de couche par
Δy = 2 y+ ν / u_τ, avec u_τ = 0,444 m/s (formule de plaque plane à mi-corde).

**Comment lire les résultats.**
1. **Le GCI** (Celik et al., 2008, procédure ASME). On calcule l'ordre de convergence observé, la valeur extrapolée de
   Richardson et la marge d'erreur du maillage moyen et du fin. Cela se fait pour la traînée totale, de pression et de
   frottement, et pour Fy, Fz et les moments. **Le critère de choix** du maillage de production : l'incertitude
   convertie en vitesse par les sensibilités VPP doit rester sous environ 0,02 nd, soit 10 % du gain cherché. Cela
   correspond à environ 1,5 % sur Cx et Cz, et 1 % sur Mx. Le rapport donne l'équivalent vitesse de chaque écart.
2. **La résolution de paroi.** Si la traînée varie de moins de 2 % entre y+ 30 et y+ 100, on garde les lois de paroi
   (y+ ≈ 50). Sinon, il faudra une couche limite résolue (y+ ≈ 1, environ 35 couches), qui n'est pas dans cette étude.
3. **Le domaine.** On attend moins de 0,5 % d'écart entre D2 et G1.
4. **La couverture des couches** (`layer_percent`). Elle valait 72 à 82 % le 20/10. Si elle reste basse, il faudra
   régler `addLayersControls` avant la production, car des couches manquantes biaisent la traînée.

## Limites et précautions

- **Un seul état :** le STL du 20/10, à une seule attitude. Je n'ai pas le dépôt bdfoil_v1 sur ce PC, donc pas de
  génération de STL pour nos sections ou nos états VPP. Le pilote aux états VPP viendra après, avec ce pipeline.
- **Pas de y+ ≈ 1 en 3D :** l'ajout de 35 couches fines par snappy sur ce foil est risqué sans surveillance.
- **La machine est partagée :** ton collègue lance parfois des calculs CFD la nuit sur ce PC. S'il en lance ce week-end,
  les durées s'allongent, et la garde de temps saute les derniers cas.
- **Un dossier à supprimer :** `test_report/` dans ce dossier ne contient que des données fictives. Il m'a servi à
  tester `cfd_mesh_report.py`, et la session n'a pas pu le supprimer.
