# Le passage du pipeline à bdfoil_v1 (27/10/2026)

**Ce qui change.** `pipelines/bdtoolbox_foil`, et donc `vpp_acceleration/src/foil_codes.py`, ne s'appuient plus sur
bdFoil core (AVL, XFOIL) ni sur bdFoil/NonPlanarSolver (ligne portante). Ils utilisent désormais **bdfoil_v1**, le
successeur de `foil.py` écrit par Antoine (dépôt `bdGit-56780/bdfoil_v1`, cloné dans `Code-Adri/bdfoil_v1`).

L'ancien moteur reste disponible sous l'étiquette git **`pre-bdfoil-v1`**. Les études `studies/cfoil_*` y ont été
calculées : [`../studies/README.md`](../studies/README.md).

Aucun fichier de bdfoil_v1 n'est modifié. Le paquet est importé depuis son dossier `src/`, sans installation dans
l'environnement.

## Ce qui a été fait

| Fichier | Changement |
| --- | --- |
| `bdtoolbox_foil/bridge.py` | localise bdfoil_v1 : clé `paths.bdfoil_v1_root`, variable `MFEGO_BDFOIL_V1_ROOT`, sinon `Code-Adri/bdfoil_v1` ; règle `BDFOIL_SOFT` pour les exécutables ; note le commit de bdfoil_v1 dans le rapport d'environnement |
| `bdtoolbox_foil/solvers.py` | un seul backend 3D, `BdfoilV1Backend` (solveur `avl` ou `npllt`, polaires `xfoil` ou `neuralfoil`), qui construit un cas bdfoil_v1 par évaluation (`Study.from_dict` → `Session.evaluate`) ; backend 2D `XfoilBackend` sur `bdfoil_v1.xfoil` ; suppression des anciens `NpLltBackend`, `AvlCoreBackend` et du correctif XFOIL de NonPlanarSolver |
| `bdtoolbox_foil/simulator.py`, `config.py` | erreurs de bdfoil_v1 (`CoreError` = évaluation ratée, `ExeError` = arrêt), clé de chemin `bdfoil_v1_root` |
| `configs/foil3d_cfoil_kulfan_3levels.json` | niveaux NP-LLT × NeuralFoil, NP-LLT × XFOIL, AVL × XFOIL (bdfoil_v1 n'a que le modèle NeuralFoil xxxlarge) |
| `vpp_acceleration/src/foil_codes.py` | mêmes trois niveaux, Cpmin pris dans bdfoil_v1 (XFOIL) ou recalculé avec NeuralFoil à l'incidence de chaque panneau (bdfoil_v1 n'en donne pas avec NeuralFoil) |
| `tests/test_pipeline_bdfoil_v1.py` | 8 tests : le cas construit passe la validation de bdfoil_v1, la correspondance des sorties, le contrôle de décrochage hors racine, les échecs ; et une évaluation réelle par niveau (`slow`) |

## Les choix, et ce qui diffère du cas C-foil d'Antoine

Pour travailler sur la même base qu'Antoine, le pipeline garde les conventions de bdfoil_v1 partout où rien d'autre
n'est décidé :
- eau de mer à 15 °C ;
- grille de Reynolds à 8 nœuds par décade ;
- polaires XFOIL de −12 à +12° par pas de 0,5°, n_crit 1, transition à 10 % ;
- `npane` 250, `niter` 100, `nchord` 8 ;
- paroi de coque en z = 0, surface libre 0,654 m au-dessus.

Trois choix diffèrent de son cas `cases/cfoil_r9800_fsi.json`. Ce sont des options du backend.

| Choix | Pipeline | Cas d'Antoine | Pourquoi |
| --- | --- | --- | --- |
| vitesse | fixe : celle qui donne Re = 5·10⁶ sur la corde de 0,7 m, **8,49 m/s** | une liste de vitesses (9,3 à 14,4 m/s) | le surrogate n'a pas la vitesse en entrée, et la VPP multiplie les coefficients par ½ρSV² |
| traînée de vagues | **désactivée** | activée | un terme qui dépend du Froude serait faux à vitesse fixe |
| Cpmin du foil | **20e percentile** des bandes | 1er percentile | valeur de la RS de production (`CP_MIN_PERC = 20`) |

## Ce que le changement fait aux résultats

Vérifié sur le profil actuel du C-foil, aux attitudes (cant, rake, yaw) = (1, 0, 1), (4 ; 0,5 ; 3) et (6, 1, 5) degrés.
Les écarts sont relatifs, nouveau moteur moins ancien.

| Niveau | Fx | Fy | Fz | Mx | My | Mz | Cpmin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NP-LLT × NeuralFoil | +0,2 à +1,5 % | −1,4 à −0,2 % | +0,1 à +1,2 % | −0,4 à +0,2 % | 0 à +0,5 % | +0,2 à +1,4 % | −0,5 à −0,2 % |
| NP-LLT × XFOIL | +0,2 à +1,5 % | −1,4 à −0,2 % | +0,1 à +1,2 % | −0,4 à +0,2 % | 0 à +0,5 % | +0,2 à +1,4 % | −1,4 à **+36 %** |
| AVL × XFOIL | −0,2 à −0,1 % | −0,1 % | +0,1 % | −0,1 % | **+7 à +16 %** | **−5 à −6 %** | −0,3 à +0,2 % |

- **Les forces bougent de 1,5 % au plus.** Fx étant négatif, +1,5 % veut dire un peu moins de traînée.
- **Les moments My et Mz d'AVL** comprennent maintenant la part visqueuse ; ceux de foil.py ne gardaient que la part
  induite. My d'AVL se rapproche ainsi de celui de la ligne portante : l'écart de 16 % entre les deux, noté le 22/10,
  venait en bonne partie de cette convention.
- **Le Cpmin de NP-LLT × XFOIL** est maintenant lu dans la polaire XFOIL au cl et au Re de chaque bande. Avant, il était
  pris à l'incidence locale du point de contrôle. À la petite attitude (1, 0, 1), il passe de −1,82 à −1,16 ; ailleurs,
  l'écart est de 1 à 6 %.
- **Le temps** : 0,2 à 2 s par évaluation NP-LLT et environ 1 s pour AVL, une fois les polaires en cache. Le premier
  appel d'une section calcule ses 2 polaires XFOIL, environ 15 s.

**Conséquence pour vpp_acceleration.** Le surrogate du 5e run a été construit avec l'ancien moteur. Un nouveau run
partira donc d'un plan neuf, calculé avec bdfoil_v1. Les résultats des runs 1 à 5 restent valables pour l'ancien moteur.

## Ce qui reste à savoir

- **La racine contre la paroi.** bdfoil_v1 avertit que, sur le C-foil, les efforts de la ligne portante dérivent de 0,1 à
  0,5 % par doublement du nombre de panneaux. C'est une limite connue (décision d'Antoine du 19/10 : pas de correction).
- **La NeuralFoil de bdfoil_v1** ajuste une section de Kulfan aux coordonnées du profil, et elle n'a que le modèle
  xxxlarge. La hiérarchie NeuralFoil xxsmall / xxlarge de l'ancienne config 3D n'existe donc plus.

## Les tests faits, sans calcul coûteux

Un seul processus à la fois, numba limité à 2 fils, pendant qu'une CFD tournait dans WSL.
- La suite rapide : 172 tests, tous verts.
- 3 évaluations réelles (`pytest -m slow`, 25 s), plus le test 2D XFOIL.
- 18 évaluations de comparaison entre l'ancien et le nouveau moteur, environ 1 min au total.
