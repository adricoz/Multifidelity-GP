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

## La coupure de confiance de NeuralFoil (décision du 27/10)

**Le problème.** bdfoil_v1 coupe ses polaires NeuralFoil là où la confiance de NeuralFoil est sous 0,90 : c'est le seuil
`MIN_CONFIDENCE` de `neuralfoil_polars.py`, décision J7-2 d'Antoine.
- **Sur les sections du C-foil à Re 5·10⁶,** cette confiance tombe à 0,4–0,5 vers −4°. La polaire commence alors à −1,5°
  ou −0,5°.
- **Conséquence au smoke du 27/10** (`vpp_acceleration/runs/2026-10-27_1636_smoke_bdfoil_v1`) : les bandes à incidence
  légèrement négative sortaient de la polaire, et **16 évaluations NP-LLT × NeuralFoil sur 66 (24 %) ont échoué** pour
  « décrochage ».
- **Ces points ne décrochaient pas :** à cant 5, rake 1, yaw 0, l'incidence médiane des panneaux est de −0,2°.
- **L'ancien pipeline** n'avait pas de coupure et aucun échec.

**La décision.** `BdfoilV1Backend(..., neuralfoil_min_confidence=0.0)` pour le niveau BF de `foil_codes.py`. Le seuil
est changé pour le processus seulement : bdfoil_v1 n'est pas modifié, et ses polaires NeuralFoil en mémoire sont
vidées quand le seuil change. Sans l'option, le seuil reste celui de bdfoil_v1.

**Ce que cela change, mesuré.** 30 points du smoke : les 16 qui échouaient et 14 qui passaient. Pour chacun, le BF sans
coupure, et le HF NP-LLT × XFOIL comme référence. L'écart de vitesse est l'écart des coefficients multiplié par les
sensibilités ∂V/∂C de la VPP, à 45/12 et 45/16.

| | 14 points où la coupure passait | 16 points où elle échouait |
| --- | --- | --- |
| BF sans coupure contre BF à 0,90 | **identiques** (écart 0,000 %) | — (pas de valeur à 0,90) |
| écart BF – HF, médiane : Fx / Fy, Fz, Mx / My / Mz | 0,36 % / 0,2 % / 0,2 % / 0,1 % | **2,6 %** / 0,2 % / **1,0 %** / 0,4 % |
| écart de vitesse BF – HF au premier ordre | médiane 0,005 nd, max 0,015 nd | médiane **0,036 nd**, max **0,39 nd** |
| Cpmin BF contre HF | 40 % | 61 % |

**Ce qu'il faut en retenir.**
1. **Là où la coupure passait, rien ne change.** Abaisser le seuil ne modifie que les points qui échouaient.
2. **Là où elle échouait, le BF est moins juste, surtout en traînée.** NeuralFoil sous-estime la traînée de 4 à 25 % aux
   yaws négatifs (−1 à −4°), alors que la portance et les moments latéraux restent à 0,2–0,4 % du HF. Or **les états
   portants de la VPP sont dans cette zone** : au 4e run, Yaw_Eff y vaut de −2,2 à −3,8° avec un cant proche de 0. C'est
   là que le 4e run avait trouvé ses plus gros gains (+0,17 à +0,42 nd).
3. **La corrélation BF/HF, sur laquelle repose le co-krigeage, reste forte.** La droite HF = ρ·BF + b ajuste encore
   Fx avec R² = 0,989, et les forces latérales avec R² = 0,99999. Le reste, que le terme δ du co-krigeage doit apprendre,
   est 2,4 fois plus grand en traînée aux points qui échouaient (0,14 écart type contre 0,06).
4. **Pour les résultats, cela veut dire :**
   - **Les vitesses finales ne sont pas biaisées par le BF.** Le surrogate prédit le niveau HF, et la validation compare
     au HF.
   - **Le BF est moins utile au portant.** Le critère de raffinement y verra une variance plus grande, et il y placera
     plus de HF. Le coût d'un run pour la même précision au portant sera donc un peu plus élevé.
   - **Un éventuel défaut se verrait.** Si le portant restait sous-raffiné, l'erreur du surrogate y serait de l'ordre de
     ce reste, et la validation finale contre le HF (RMS par état) l'afficherait.
   - **Garder la coupure serait pire.** On n'aurait aucune information BF au portant, et tout le travail y reposerait
     sur le HF.
5. **Le Cpmin du BF reste faible de toute façon** (40 à 60 % d'écart à XFOIL). Le Cpmin du surrogate tient à ses points
   HF.

**Si le portant pose problème au prochain run**, l'alternative est de prendre NP-LLT × XFOIL comme BF au portant, ou
partout (coût 2 au lieu de 1). C'est à décider sur les chiffres de validation.

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
