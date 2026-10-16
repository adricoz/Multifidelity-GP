# mfego v2 — un modèle de substitution qualifiable, comparé à la concurrence

16/10/2026 · dossier `benchmarks/surrogate_v2/` · suite de [`RAPPORT_COKRIGING.md`](../cokriging/RAPPORT_COKRIGING.md) et
[`RAPPORT_PRIORS.md`](../priors_all/RAPPORT_PRIORS.md) (état v1)

> **Note v3.** Le défaut de `CoKriging` est maintenant la v3 : ρ appris avec une prior log-normale, sans borne ρ ≤ 5
> (voir [`RAPPORT_V3.md`](../surrogate_v3/RAPPORT_V3.md)). Les scripts de ce dossier fixent les options v2
> explicitement et redonnent les chiffres ci-dessous.

Les figures interactives sont dans [`figures/`](figures/) (à ouvrir dans un navigateur ; thème sombre automatique). Les
résultats bruts sont dans [`results/`](results/). Dans les tableaux, NF = NeuralFoil.

---

## 1. Résumé

* **Implémenté** : les cinq points validés.
  * **(1) Qualification sans jeu de test** : leave-one-out (LOO) en forme fermée, Q², NLPD et couvertures.
  * **(2) Correction du biais de ρ** : centrage du niveau inférieur et prior N(1, 2²).
  * **(4) Normalisation des entrées** dans les modèles.
  * **(6) Noyau Matérn 5/2.**
  * **(7) Validation sur un vrai couple** : NeuralFoil → XFOIL, sans le pipeline bdFoil.
  * Quelques clés ajoutées au pipeline.
  * 161 tests, dont 35 nouveaux ; tout passe (1 test ignoré, numba absent).
* **Qualité du modèle de substitution (`CoKriging`, plans imbriqués) : v2 meilleur que v1, au niveau ou au-dessus de la
  concurrence.**
  * `CoKriging` v2 bat v1 sur 25 graines sur 30 en RMSE et 27 à 28 sur 30 en NLPD, pour chacun des 3 problèmes.
  * **Le gain vient surtout du noyau Matérn 5/2.** La correction de ρ aide nettement Hartmann, un peu l'airfoil
    NF/NF, et rien sur le cas réel NF/XFOIL. Elle y est neutre sur plan imbriqué et nuisible sur plan non imbriqué
    (NLPD +0,14, RMSE +5 % par rapport à v1 avec Matérn ; § 3).
  * **Hartmann 6D**, n_HF = 40 : Q² 0,56 contre 0,43 (v1) et 0,43 (emukit MAP) ; NLPD 0,13 contre 0,60 et 0,24.
    **Ce résultat dépend de la borne ρ ≤ 5** : à cette taille, ρ̂ y est en butée dans 70 % des ajustements. Avec
    N(1, 1²), on obtiendrait Q² 0,55 et NLPD 0,17, toujours devant (§ 3).
  * **NeuralFoil → XFOIL**, n_HF = 20 : RMSE relative 0,12, soit Q² 0,985, contre 0,14 pour BoTorch et 0,15 pour
    emukit MAP ; NLPD 2,41, la meilleure.
  * Ajustement en 0,1 à 1,6 s, contre 1,3 à 8 s pour BoTorch, 1,7 à 20 s pour SMT et 3 à 20 s pour emukit.
* **Qualification.**
  * La LOO classique est optimiste sur plan imbriqué : elle dépasse le Q² du jeu de test dans 63 à 83 % des cas, car la
    valeur BF au point retiré reste connue.
  * La **LOO par emplacement** (défaut v2) retire toutes les observations au même point. Elle n'est sans biais
    qu'en moyenne sur les tailles. Avec 5 points HF, elle est pessimiste sur les airfoils (−0,17 et −0,12 en Q²). Sur
    Hartmann à n_HF = 20, elle est optimiste (+0,26, 9 cas sur 10).
  * Sur Hartmann, elle déclare le modèle bien calibré alors qu'il ne l'est pas sur le jeu de test : les points HF ne
    tombent pas dans les puits étroits où il se trompe.
  * **Sur des données d'EGO (points groupés), toute LOO est très optimiste** : Q² ≈ 0,95–0,99 contre 0,18–0,91 sur le
    test.
  * La LOO est donc un contrôle rapide, pas une certification : il faut garder 5 à 10 points HF de validation (§ 6).
* **Optimisation : la v2 fait moins bien que la v1 sur Hartmann.** NN-MF-EGO v2 ne bat NN-MF-EGO v1 que sur 1 graine
  sur 10.
  * Les deux changements y contribuent : Matérn seul et la correction de ρ seule perdent aussi contre v1 (sur 6 et 8
    graines sur 10). Avec 10 graines, la part de chacun ne peut pas être isolée.
  * La correction de ρ (ρ̂ ≈ 2,4 au lieu de 0,27) déplace le budget vers la BF. La fonction de mérite (Sacher éq. 24)
    pondère l'information d'un niveau bas par ρ² : 5,8 au lieu de 0,07.
  * Résultat : 86 % des évaluations sont des évaluations BF, pour 38 % du coût (médianes sur les graines). Sur 6
    graines, les évaluations HF tombent de 13 à 7–9. Sur 3 graines, l'EGO n'évalue aucune BF.
  * Sur les deux airfoils (ρ ≈ 1), aucune différence.
* **Défauts appliqués (après la relecture indépendante).**
  * `CoKriging`, le modèle de substitution sur plan imbriqué, prend les défauts v2.
  * `MultifidelityModel`, le modèle de l'EGO, utilisé sur des plans quelconques, et le pipeline bdFoil gardent le ρ de
    v1 (non centré, sans prior). Les options v2 s'activent par argument ou dans le JSON.
* **Décision prise en dehors de la règle écrite d'avance.**
  * La règle de sélection désignait « ρ centré sans prior ». Sur Hartmann, ce choix laisse ρ̂ sur la borne arbitraire
    ρ ≤ 5 dans 68 % des ajustements.
  * J'ai retenu la prior N(1, 2²) que tu avais choisie (écart de NLPD moyen : −0,32 contre −0,40). Elle ramène cette
    part à 27 %, mais à 40 % sur plan imbriqué : **elle ne règle pas le problème invoqué.**
  * N(1, 1²) le règle presque (8 %, 10 % sur plan imbriqué) : résultats identiques sur les airfoils, NLPD un peu moins
    bonne sur Hartmann (§ 3). **C'est ma recommandation, à valider.**

---

## 2. Ce qui a changé dans le code

| Point | Fichiers | Changement |
|---|---|---|
| (2) biais de ρ | `mfego/src/surrogate_models.py` | `rho_centering=True` : le niveau inférieur est centré dans le profil de ρ, la constante du résidu suit ρ. Groupe de prior `"rho": ("normal", μ, σ)` en forme fermée : ρ̂ = (FᵀK⁻¹y + μ/σ²) / (FᵀK⁻¹F + 1/σ²). Prior par défaut N(1, 2²), **seulement avec le centrage**. Défaut de `CoKriging` ; `MultifidelityModel` garde `rho_centering=False` (§ 7). |
| (4) normalisation | `surrogate_models.py`, `cokriging.py` | Les GP reçoivent x mis à l'échelle sur [0, 1] avec les bornes de `ExperimentData` (`scale_to_unit`), enregistrées dans le JSON (`input_bounds`). Pour des bornes [0, 1] (exemples, pipeline), rien ne change. |
| (6) Matérn 5/2 | `mfego/src/kernels.py` | `Matern52Kernel`, même interface, gradient analytique ; défaut de `CoKriging`. |
| (1) qualification | `cokriging.py`, nouveau `mfego/src/validation.py` | `CoKriging.leave_one_out(level, by_location=True)` en forme fermée, par blocs ; `CoKriging.from_multifidelity(model, data)`, conditionnement exact d'un `MultifidelityModel` sur un plan quelconque ; `prediction_scores(y, mean, var)` : RMSE relative, Q², NLPD, couvertures 50/90/95/99 %, moyenne quadratique de z (`z_rms`, idéal 1 ; elle détecte aussi un biais de la moyenne). |
| pipeline | `pipelines/bdtoolbox_foil/config.py`, `run.py` | Clés `kernel`, `rho_centering`, `priors`, vérifiées. **Défauts = comportement v1** (§ 7). |
| (7) problème réel | `benchmarks/cokriging/airfoil_problem.py` | `AirfoilXfoilMF` : BF = NeuralFoil `xxxlarge`, HF = XFOIL (`xfoil.exe` de bdToolbox, appelé par `asb.XFoil`). |

**Compatibilité.**

* Les options v1 restent disponibles : `rho_centering=False` (défaut de `MultifidelityModel`), qui retire aussi la prior
  sur ρ, avec `SquaredExponentialKernel`.
* Un fichier JSON écrit en v1 est relu avec ce comportement : pas de bornes d'entrée, pas de prior sur ρ, constante à
  ρ = 1. Il prédit à l'identique, y compris pour des bornes autres que [0, 1] (testé).
* Pour des bornes [0, 1] (exemples, pipeline), le monofidélité, ρ fixé (`estimate_rho=False`) et les options v1
  donnent exactement le même modèle qu'avant (testé).
* Pour d'autres bornes, la normalisation des entrées, qui ne se désactive pas, change le modèle ajusté. C'est voulu :
  les priors et les bornes de recherche supposent des entrées dans [0, 1].

**Nouveaux tests** (35) :
* forme fermée de ρ̂ avec prior, égale à la minimisation numérique ;
* cas synthétique décalé de +50 : la v1 tire ρ vers 1, la v2 retrouve ρ = 2 ;
* identité à ρ fixé ;
* Matérn : formule, gradient, `check_grad` de la NLL ;
* normalisation : équivariance sur [0, 10] et [2, 12], aller-retour JSON ;
* LOO par observation et par emplacement, égales au calcul par force brute à 1e-8 près ;
* `prediction_scores` ;
* lecture des fichiers v1 de `MultifidelityModel` (entrées brutes sur [0, 10]) et de `CoKriging` ;
* défauts de ρ des deux classes ;
* `from_multifidelity`, qui garde les options du modèle d'origine ;
* clés du pipeline, jusqu'au modèle construit par `run.py`.

---

## 3. Sélection des nouveaux défauts

Script `run_selection.py`, figure [`selection.html`](figures/selection.html).

* **Données :** 3 problèmes, plans imbriqués et non imbriqués, 3 tailles, 10 graines, soit 1 800 ajustements du modèle
  récursif. Sur plan imbriqué, ce modèle égale `CoKriging`.
* **Règle fixée avant le calcul :** minimiser l'écart moyen de NLPD à v1 SE sur les 3 problèmes, sans dégrader la RMSE
  de plus de 5 % sur aucun.

| Variante | dNLPD Hartmann | dNLPD airfoil NF/NF | dNLPD airfoil NF/XFOIL | dNLPD moyen | Rapport de RMSE (H · NF · XFOIL) |
|---|---|---|---|---|---|
| ρ centré, sans prior, Matérn *(règle)* | −0,72 | −0,27 | −0,20 | −0,40 | 0,92 · 0,83 · 0,94 |
| **ρ centré, N(1, 2²), Matérn (retenu)** | −0,50 | −0,27 | −0,20 | −0,32 | 0,94 · 0,83 · 0,94 |
| ρ centré, N(1, 1²), Matérn | −0,37 | −0,27 | −0,20 | −0,28 | 0,96 · 0,83 · 0,94 |
| v1 (ρ non centré), Matérn | −0,25 | −0,21 | −0,31 | −0,25 | 1,00 · 0,90 · 0,91 |
| ρ centré, N(1, 0,5²), Matérn | −0,26 | −0,27 | −0,21 | −0,25 | 0,98 · 0,83 · 0,94 |
| ρ centré, sans prior, SE | −0,57 | −0,06 | +0,16 | −0,16 | 0,94 · 0,92 · 1,05 |
| v1 SE (référence) | 0 | 0 | 0 | 0 | 1 · 1 · 1 |

**Lecture.**

* **Matérn 5/2 gagne partout.** À traitement de ρ égal, il améliore l'écart de NLPD de 0,15 à 0,36 selon le problème
  et la variante, et la RMSE de 0 à 10 %. Exception : la fonction analytique de Forrester, où le noyau SE reste
  meilleur. L'exemple `main.py` garde donc SE.
* **Le centrage de ρ dépend du problème.** Ci-dessous, ρ centré avec N(1, 2²) comparé à v1 avec Matérn, graine par
  graine (30 ajustements par case) :

  | Problème | Plans imbriqués : victoires RMSE / NLPD · ΔNLPD moyen | Plans non imbriqués : victoires RMSE / NLPD · ΔNLPD moyen · rapport de RMSE |
  |---|---|---|
  | Hartmann | 27 / 24 · −0,38 | 23 / 23 · −0,16 · 0,94 |
  | airfoil NF/NF | 21 / 24 · −0,06 | 16 / 22 · −0,05 · 0,98 |
  | airfoil NF/XFOIL | 10 / 13 · +0,02 | 9 / 7 · +0,14 · 1,05 |

  * Il aide nettement Hartmann et un peu l'airfoil NF/NF.
  * Sur le cas réel NF/XFOIL, il est neutre sur plan imbriqué et nuisible sur plan non imbriqué. Les niveaux y sont
    presque identiques (ρ ≈ 1). **Sur ce cas, tout le gain de v2 vient du noyau Matérn.**
  * C'est une seconde raison, après l'EGO (§ 7), de garder le ρ de v1 dans `MultifidelityModel`, qui travaille sur des
    plans quelconques. `CoKriging` ne travaille que sur plans imbriqués, où le centrage aide ou est neutre.
* **Pourquoi je n'ai pas suivi la règle à la lettre.**
  * Sans prior, ρ̂ est sur la borne ρ ≤ 5 de mfego dans 68 % des ajustements de Hartmann (90 % sur plan imbriqué). Le
    gain vient en partie de cette borne arbitraire : avec des bornes ±50, ρ̂ monte à 6–23 et la RMSE se dégrade
    (RAPPORT_COKRIGING § 7).
  * J'ai retenu la prior N(1, 2²), la régularisation que tu avais choisie. **Elle ne règle pas ce problème** :

  | Prior sur ρ (Matérn, centré) | ρ̂ en butée, Hartmann : tous plans · imbriqués · imbriqués n_HF = 40 | Hartmann imbriqué n_HF = 40 : RMSE · NLPD | ΔNLPD moyen (3 problèmes) |
  |---|---|---|---|
  | aucune *(règle)* | 68 % · 90 % · 90 % | 0,66 · 0,13 | −0,40 |
  | **N(1, 2²) (retenu)** | 27 % · 40 % · 70 % | 0,66 · 0,13 | −0,32 |
  | N(1, 1²) | 8 % · 10 % · 30 % | 0,67 · 0,17 | −0,28 |
  | N(1, 0,5²) | 0 % · 0 % · 0 % | 0,69 · 0,40 | −0,25 |
  | v1 Matérn (pour mémoire) | 0 % | 0,75 · 0,34 | −0,25 |

  * Quand ρ̂ est en butée, la prior ne change rien : la borne fixe ρ, donc le terme de prior est constant. Sur les 12
    ajustements imbriqués de Hartmann en butée avec et sans prior, l'écart de NLPD est inférieur à 1e-6.
  * **Le résultat de Hartmann à n_HF = 40 dépend donc de la borne arbitraire**, avec ou sans N(1, 2²). Hartmann a un
    « vrai » ρ constant grand, à cause d'une relation BF/HF très non linéaire.
  * Sur les deux airfoils, N(1, 1²) et N(1, 2²) donnent les mêmes médianes (ρ ≈ 1).
  * **Recommandation : N(1, 1²).** Il répond à la raison invoquée (8 % en butée) pour une perte faible sur Hartmann et
    nulle sur les airfoils. Ce choix est à valider par toi, puis à benchmarker. Il ne change qu'une ligne de
    `DEFAULT_PRIORS`.

---

## 4. Protocole du benchmark final

Script `run_benchmark.py`.

* **Problèmes**, tous à entrées dans [0, 1]^d, 2 niveaux, coûts 1 et 10 :
  * **Hartmann 6D** de Sacher et al. ;
  * **airfoil NF/NF** : NACA 4 chiffres continu, 4D (cambrure, position, épaisseur, incidence), Cd en points de traînée,
    NeuralFoil `xxsmall` → `xxxlarge` ;
  * **airfoil NF/XFOIL** : mêmes sections, NeuralFoil `xxxlarge` → **XFOIL**, Re = 5·10⁵, n_crit = 1, transition forcée
    à 10 %. XFOIL prend 0,2 s par point. Aux points HF des plans, la corrélation BF/HF est de 0,99 et l'écart relatif
    médian de 0,8 % (16 % au plus), soit un écart quadratique de 13 % de l'écart-type de la HF. C'est réaliste,
    puisque NeuralFoil est entraîné sur XFOIL. 2 échecs XFOIL sur 700 calculs HF des plans ; 598 points de test
    valides sur 600.
* **Plans imbriqués**, n_BF = 4 n_HF, 10 graines. Les plans et les jeux de test sont calculés une seule fois et mis en
  cache (`results/designs.pkl`). Les points dont le calcul a échoué sont retirés pour toutes les bibliothèques.
* **Modèles :**
  * mfego v2 `CoKriging` (défauts v2) et mfego v1 `CoKriging` (options v1) ;
  * GPy/emukit AR(1) en estimation jointe MLE et MAP ;
  * SMT MFK ;
  * **BoTorch SingleTaskMultiFidelityGP** ;
  * HF seule : mfego v2 et BoTorch SingleTaskGP.
* **Données de l'infilling :** NN-MF-EGO v1 et v2 jusqu'au coût d'un plan imbriqué, sur les 3 problèmes.
* **Optimisation de Hartmann à budget 250**, protocole de `priors_all` : NN-MF-EGO v1 et v2, SF-EGO v2, BoTorch qLogEI.

---

## 5. Précision et calibration sur plans imbriqués

Figures : [`nested_accuracy.html`](figures/nested_accuracy.html) · [`calibration.html`](figures/calibration.html).
Médianes sur 10 graines, dans chaque case : **RMSE relative · NLPD · couverture 95 %** (Q² = 1 − RMSE_rel²). Temps
d'ajustement médian d'une exécution, sur 32 cœurs en parallèle (variable de ± 10 % d'une exécution à l'autre).

**Hartmann 6D**

| Modèle | n_HF = 10 | n_HF = 20 | n_HF = 40 | temps (n_HF = 40) |
|---|---|---|---|---|
| **mfego v2 CoKriging** | **0,92** · 2,09 · 0,85 | **0,82** · **0,59** · 0,89 | **0,66** · **0,13** · 0,92 | 1,56 s |
| mfego v1 CoKriging | 0,95 · 2,88 · 0,82 | 0,84 · 1,37 · 0,86 | 0,76 · 0,60 · 0,87 | 2,07 s |
| GPy/emukit AR(1), MAP joint | 0,95 · **1,24** · 0,85 | 0,99 · 1,31 · 0,85 | 0,76 · 0,24 · 0,91 | 16,1 s |
| GPy/emukit AR(1), MLE joint | 1,15 · 6,36 · 0,63 | 1,19 · 2,58 · 0,79 | 0,87 · 0,80 · 0,90 | 19,6 s |
| BoTorch MF-GP | 1,00 · 2,52 · 0,83 | 0,92 · 2,87 · 0,78 | 0,84 · 3,06 · 0,80 | 8,0 s |
| SMT MFK | 1,49 · 4,21 · 0,82 | 1,22 · 2,91 · 0,76 | 1,01 · 0,67 · 0,92 | 19,7 s |
| mfego v2, HF seule | 0,98 · 2,37 · 0,85 | 0,86 · 0,83 · 0,88 | 0,81 · 0,42 · 0,91 | 0,07 s |
| BoTorch SingleTaskGP, HF seule | 1,02 · 7,15 · 0,71 | 0,92 · 2,21 · 0,80 | 0,86 · 1,68 · 0,83 | 0,59 s |

**Airfoil NF/NF**

| Modèle | n_HF = 5 | n_HF = 10 | n_HF = 20 | temps (n_HF = 20) |
|---|---|---|---|---|
| **mfego v2 CoKriging** | **0,37** · 3,38 · 0,93 | 0,26 · 3,06 · 0,91 | 0,21 · 2,80 · 0,94 | 0,21 s |
| mfego v1 CoKriging | 0,39 · 3,46 · 0,89 | 0,32 · 3,43 · 0,85 | 0,26 · 3,18 · 0,90 | 0,27 s |
| GPy/emukit AR(1), MAP joint | 0,37 · **3,36** · 0,95 | **0,25** · 3,07 · 0,89 | 0,21 · 3,19 · 0,84 | 8,5 s |
| BoTorch MF-GP | 0,43 · 3,46 · 0,97 | 0,26 · **3,03** · 0,95 | **0,20** · **2,78** · 0,94 | 2,3 s |
| SMT MFK | 0,39 · 7,85 · 0,48 | 0,33 · 8,75 · 0,49 | 0,29 · 4,28 · 0,74 | 3,2 s |
| mfego v2, HF seule | 0,74 · 4,25 · 0,96 | 0,46 · 3,87 · 0,98 | 0,27 · 3,20 · 0,97 | 0,04 s |

**Airfoil NF/XFOIL (le cas réel)**

| Modèle | n_HF = 5 | n_HF = 10 | n_HF = 20 | temps (n_HF = 20) |
|---|---|---|---|---|
| **mfego v2 CoKriging** | 0,29 · **3,15** · 0,94 | 0,20 · **2,83** · 0,87 | **0,12** · **2,41** · 0,90 | 0,21 s |
| mfego v1 CoKriging | 0,29 · 3,31 · 0,88 | 0,23 · 3,59 · 0,79 | 0,15 · 3,09 · 0,83 | 0,26 s |
| GPy/emukit AR(1), MAP joint | 0,29 · 3,18 · 0,95 | **0,19** · 3,28 · 0,82 | 0,15 · 3,45 · 0,76 | 8,8 s |
| GPy/emukit AR(1), MLE joint | **0,28** · 5,30 · 0,68 | 0,20 · 3,82 · 0,74 | 0,14 · 3,77 · 0,74 | 10,9 s |
| BoTorch MF-GP | 0,35 · 3,39 · 0,96 | 0,22 · 2,86 · 0,91 | 0,14 · 2,67 · 0,88 | 1,5 s |
| SMT MFK | 0,32 · 6,04 · 0,61 | 0,23 · 6,02 · 0,64 | 0,18 · 4,80 · 0,66 | 3,1 s |
| mfego v2, HF seule | 0,75 · 4,26 · 0,97 | 0,46 · 3,87 · 0,97 | 0,26 · 3,19 · 0,94 | 0,04 s |

Les lignes emukit MLE et BoTorch HF seule de NF/NF figurent dans `results/summary.txt`.

**Graines où mfego v2 fait mieux** (toutes tailles ; 30 ajustements par problème, RMSE / NLPD) :

| contre | Hartmann | airfoil NF/NF | airfoil NF/XFOIL |
|---|---|---|---|
| mfego v1 | 25 / 27 | 25 / 27 | 25 / 28 |
| GPy/emukit MAP | 20 / 22 | 16 / 18 | 18 / 24 |
| BoTorch MF-GP | 29 / 27 | 17 / 14 | 27 / 22 |
| SMT MFK | 30 / 26 | 24 / 30 | 21 / 30 |
| HF seule (mfego v2) | 27 / 22 | 29 / 30 | 30 / 29 |

**Lecture.**

* **v2 contre v1** : le gain est net et régulier, environ 25 graines sur 30 sur chaque problème. Sur NF/XFOIL, il
  vient du noyau Matérn (§ 3).
  * ρ̂ v2 (médianes) : 2,9 à 5 sur Hartmann, en butée dans 10 à 70 % des ajustements selon n_HF ; 0,78–0,91 sur NF/NF ;
    ≈ 1,0 sur NF/XFOIL.
  * En v1, il vaut 0,95 à 1,01 partout.
* **Contre la concurrence** :
  * sur Hartmann, mfego v2 est le meilleur à n_HF = 20 et 40. À n_HF = 10, emukit MAP a une meilleure NLPD (1,24
    contre 2,09) ;
  * sur le cas réel NF/XFOIL, mfego v2 a la meilleure NLPD à toutes les tailles. En RMSE, il est au niveau des autres
    à n_HF = 5 et 10, et le meilleur à n_HF = 20 ;
  * sur NF/NF, il est à égalité avec emukit MAP et BoTorch MF-GP, ce dernier étant un peu mieux calibré (NLPD meilleure
    sur 16 graines sur 30) ;
  * SMT MFK est le plus mal calibré (couverture 0,48–0,74 sur les airfoils).
* **Calibration** : couverture à 95 % de 0,85 à 0,94 pour mfego v2. Sur les airfoils, la moyenne quadratique de z
  (idéal 1) vaut 1,0 à 1,5. Sur Hartmann, la variance reste sous-estimée dans les puits (1,2 à 2,5) : c'est la limite
  d'un noyau stationnaire.
* **Intérêt du multifidélité** sur le cas réel : à 10 points HF, Q² 0,96 contre 0,79 en HF seule. Même 20 points HF
  seuls (Q² 0,93) n'atteignent pas ce niveau.

---

## 6. Qualification sans jeu de test : que vaut la LOO ?

Figure : [`loo_vs_test.html`](figures/loo_vs_test.html). Comparaison, ajustement par ajustement, du Q² et de la
couverture calculés par LOO aux points HF avec les mêmes scores sur le jeu de test (mfego v2, plans imbriqués).

| Problème, n_HF | Q² test | Q² LOO par emplacement (défaut) | Q² LOO par observation |
|---|---|---|---|
| Hartmann, 10 · 20 · 40 | 0,15 · 0,32 · 0,56 | 0,00 · 0,57 · 0,53 | 0,10 · 0,68 · 0,75 |
| airfoil NF/NF, 5 · 10 · 20 | 0,86 · 0,94 · 0,96 | 0,69 · 0,94 · 0,94 | 0,95 · 0,97 · 0,97 |
| airfoil NF/XFOIL, 5 · 10 · 20 | 0,92 · 0,96 · 0,99 | 0,81 · 0,97 · 0,98 | 1,00 · 0,99 · 0,98 |

**Écart Q²_LOO − Q²_test par taille** (médiane sur 10 graines, et nombre de graines où la LOO est optimiste) :

| Problème, n_HF | LOO par emplacement (défaut) | LOO par observation |
|---|---|---|
| Hartmann, 10 · 20 · 40 | −0,12 (3) · **+0,26 (9)** · −0,07 (3) | −0,05 (3) · +0,37 (9) · +0,11 (7) |
| airfoil NF/NF, 5 · 10 · 20 | **−0,17 (2)** · +0,01 (8) · −0,02 (4) | +0,10 (10) · +0,04 (9) · +0,01 (6) |
| airfoil NF/XFOIL, 5 · 10 · 20 | **−0,12 (2)** · +0,01 (6) · −0,01 (3) | +0,08 (10) · +0,03 (10) · 0,00 (5) |

Toutes tailles confondues, l'écart médian |Q²_LOO − Q²_test| par emplacement vaut 0,15 sur Hartmann, 0,04 sur NF/NF et
0,03 sur NF/XFOIL. La LOO par emplacement est optimiste dans 50 %, 47 % et 37 % des cas, contre 63 %, 83 % et 83 % pour
la LOO par observation.

**Calibration vue par la LOO et par le test** (médianes, LOO par emplacement) :

| Problème, n_HF | NLPD test · LOO | couverture 95 % test · LOO |
|---|---|---|
| Hartmann, 10 · 20 · 40 | 2,09 · −0,35 / 0,59 · −0,21 / 0,13 · −0,02 | 0,85 · 0,90 / 0,89 · 0,95 / 0,92 · 0,975 |
| airfoil NF/NF, 5 · 10 · 20 | 3,38 · 3,80 / 3,06 · 3,18 / 2,80 · 3,21 | 0,93 · 0,80 / 0,91 · 0,90 / 0,94 · 0,90 |
| airfoil NF/XFOIL, 5 · 10 · 20 | 3,15 · 3,33 / 2,83 · 2,83 / 2,41 · 2,65 | 0,94 · 1,00 / 0,87 · 0,90 / 0,90 · 0,925 |

* **La LOO par observation est optimiste**, comme prévu : la valeur BF au point retiré reste connue, si bien qu'elle ne
  teste que l'interpolation du résidu.
* **La LOO par emplacement n'est sans biais qu'en moyenne sur les tailles.**
  * Avec 5 points HF, elle est pessimiste sur les airfoils : le modèle perd alors 20 % de ses données.
  * Sur Hartmann à n_HF = 20, elle est optimiste dans 9 cas sur 10 (+0,26 en médiane).
* **Sur Hartmann, elle juge le modèle bien calibré alors qu'il ne l'est pas.** Sa NLPD est négative et sa couverture
  proche de 95 %, alors que sur le jeu de test le modèle sous-estime sa variance. Les points HF ne tombent pas dans les
  puits étroits où le modèle se trompe : la LOO ne peut pas voir ces erreurs.
* Sur les airfoils, la calibration LOO est proche de celle du test ou plus sévère.
* **Précision d'un seul ajustement** : ± 0,03 à 0,04 en Q² sur les airfoils, mais ± 0,15 sur Hartmann. La couverture LOO,
  calculée sur 5 à 40 points HF, est grossière (pas de 2,5 à 20 %).
* **Sur des données d'EGO, toute LOO est fortement optimiste** : Q² LOO 0,95 à 0,99 contre 0,18 à 0,91 sur le test (§ 7).
  Les points groupés près de l'optimum se prédisent bien entre eux, sans rien dire du reste du domaine.

**Recommandation pratique** :
* utiliser la LOO par emplacement comme **contrôle rapide** : erreurs grossières, point aberrant, modèle mal ajusté ;
* ne pas s'en servir pour affirmer qu'un modèle est bien calibré : elle peut ne pas voir les zones où il se trompe ;
* pour **certifier** un modèle de substitution, garder **5 à 10 points HF de validation indépendants**, répartis dans
  le domaine (LHS) ;
* sur des données d'optimisation, ne jamais conclure sur la LOO seule.

---

## 7. Données de l'infilling et optimisation : la v2 n'aide pas l'EGO

Figures : [`infill.html`](figures/infill.html) · [`optimization.html`](figures/optimization.html) ·
[`ego_rho.html`](figures/ego_rho.html).

**Optimisation de Hartmann 6D (budget 250, 10 graines, f − f\*)**

| Méthode | Meilleure obs. HF, médiane (moyenne) | Recommandation, médiane [quartiles] (moyenne) | Part BF |
|---|---|---|---|
| NN-MF-EGO v1 | 0,26 (0,59) | **0,25** [0,11–0,55] (0,49) | 7 % |
| NN-MF-EGO v2 | 0,32 (0,56) | 0,29 [0,19–0,44] (0,51) | 86 % |
| SF-EGO v2 | 0,52 (0,72) | 0,46 [0,29–0,98] (0,67) | — |
| BoTorch qLogEI | 0,62 (0,98) | 0,71 [0,48–1,66] (1,00) | — |

NN-MF-EGO v2 fait mieux que v1 sur 1 graine sur 10, et mieux que BoTorch sur 9 sur 10 (v1 : 8 sur 10).

**Attribution** (`explore_ego_rho.py`, 10 graines) :

| Variante | Recommandation, médiane (moyenne) | Graines où v1 fait mieux | Part BF des évaluations | ρ final |
|---|---|---|---|---|
| v1 | **0,25** (0,49) | — | 7 % | 0,27 |
| Matérn seul (ρ v1) | 0,29 (0,59) | 6 / 10 | 7 % | 0,28 |
| correction de ρ seule, avec N(1, 2²) (SE) | 0,38 (0,58) | 8 / 10 | 90 % | 2,18 |
| v2 | 0,29 (0,51) | 9 / 10 | 86 % | 2,41 |

* **La correction de ρ change la répartition du budget.**
  * La fonction de mérite de NN-MF-EGO (Sacher éq. 24) multiplie l'information d'un niveau bas par ρ². Ce facteur vaut
    5,8 avec ρ̂ ≈ 2,4, contre 0,07 en v1 (ρ̂ ≈ 0,27).
  * En v2, 86 % des évaluations sont BF, pour 38 % du coût après le plan initial (médianes). Sur 6 graines, les
    évaluations HF passent de 13 à 7–9. Sur 3 graines, l'EGO v2 n'évalue aucune BF, comme v1 sur ces graines.
* **La baisse de qualité des recommandations n'est pas attribuable à un seul changement.**
  * v1 fait mieux que chacune des trois autres variantes, Matérn seul compris (6 graines sur 10, médiane 0,29 contre
    0,25).
  * Les erreurs vont de 0,02 à 1,9 selon la graine : 10 graines ne permettent pas de séparer l'effet du noyau de celui
    de ρ.
  * La variante « correction de ρ seule » combine centrage et prior : leurs effets ne sont pas séparés non plus.
* **Sur les airfoils** (ρ ≈ 1), v1 et v2 recommandent le même optimum. Sur NF/XFOIL, les 40 recommandations valent
  Cd ≈ 110,6 points au dixième près, la précision d'affichage de XFOIL. C'est moins que le minimum du jeu de test
  (110,9) : l'optimum est facile à trouver, probablement sur une borne du domaine (non vérifié). Ce problème ne
  départage donc pas les versions.
* **Données de l'infilling sur Hartmann** : l'EGO v2 récolte plus de BF et moins de HF (81 BF / 20 HF contre 21 / 26 à
  coût 280). Le modèle obtenu est moins bon (Q² 0,25 contre 0,33).
* **Variance du modèle récursif aux points HF** (données d'EGO) : 0,1 à 9,5 % de std(y), contre 0,0 à 0,5 % avec le
  conditionnement exact (`from_multifidelity`). La correction de ρ l'accentue, car le terme ρ²σ²_BF grandit.

**Conséquence appliquée** (corrigée après la relecture indépendante) :

* le gain de v2 n'est confirmé que pour `CoKriging` sur plan imbriqué : `CoKriging` prend les défauts v2 ;
* `MultifidelityModel` sert à l'EGO et travaille sur des plans quelconques. Il garde le ρ de v1 par défaut
  (`rho_centering=False`, qui retire aussi la prior sur ρ). Les scripts EGO (`mfego/main.py`, exemples, études) ont
  donc le même comportement qu'avant la v2 ;
* le pipeline bdFoil garde le comportement v1 par défaut : `kernel: squared_exponential`, `rho_centering: false`,
  `priors: null`. Un `priors` donné sans `rho_centering: true` n'active pas la prior sur ρ ;
* les options v2 s'activent par argument (`rho_centering=True`, `Matern52Kernel`) ou dans le JSON.

---

## 8. Limites et suites

* **Fonction de mérite à revoir.** Le vrai correctif côté EGO est de revoir le terme ρ² de l'éq. 24. Deux pistes, à
  benchmarker :
  * utiliser la réduction de variance exacte du co-krigeage (`from_multifidelity`) ;
  * borner ρ dans le mérite.
* **Prior sur ρ** : σ = 2 laisse le résultat de Hartmann dépendre de la borne ρ ≤ 5. σ = 1 est recommandé, à valider
  puis à benchmarker (§ 3).
* **LOO** : utile mais approximative (§ 6). Elle ne remplace pas quelques points de validation, et elle peut juger bien
  calibré un modèle qui ne l'est pas.
* **Niveaux bas bruités** : l'estimation par niveau reste biaisée (RAPPORT_COKRIGING § 2.1).
* **Variance sous-estimée** dans les zones à forte variation (Hartmann) : c'est la limite des noyaux stationnaires.
* **Trois problèmes, 10 graines.** Le cas NF/XFOIL est réel mais ses niveaux sont proches (ρ ≈ 1). Un couple plus
  éloigné (XFOIL → CFD, ou NeuralFoil `xxsmall` → XFOIL) reste à tester.
* **Hors périmètre** (points 3 et 5) : incertitude des hyperparamètres et estimation jointe.

## Reproduction

Depuis la racine du dépôt, avec `.venv-benchmark`. Le premier lancement calcule les plans et jeux de test XFOIL (≈ 1 min
sur 32 cœurs) ; le tout prend environ 10 min.

```
.venv-benchmark\Scripts\python benchmarks\surrogate_v2\run_selection.py
.venv-benchmark\Scripts\python benchmarks\surrogate_v2\run_benchmark.py      # --quick : 3 graines
.venv-benchmark\Scripts\python benchmarks\surrogate_v2\explore_ego_rho.py
.venv-benchmark\Scripts\python benchmarks\surrogate_v2\make_figures.py
```

| Fichier | Rôle |
|---|---|
| `v2_lib.py` | Problèmes, jeux de test et plans en cache, options v1 et v2 |
| `run_selection.py` | Sélection des défauts (règle de décision dans l'en-tête) |
| `run_benchmark.py` | Plans imbriqués contre la concurrence, LOO, données de l'infilling, optimisation |
| `explore_ego_rho.py` | Attribution du changement de comportement de l'EGO |
| `make_figures.py` | Les 7 figures de `figures/` |
| `results/` | `selection.csv`, `nested.csv`, `infill.csv`, `optimization.csv`, `ego_rho.csv`, `summary.txt`, plans et jeux de test |

## Figures

| Figure | Contenu |
|---|---|
| [`selection.html`](figures/selection.html) | Sélection des défauts (NLPD et RMSE par variante et par problème) |
| [`nested_accuracy.html`](figures/nested_accuracy.html) | RMSE, NLPD et couverture contre n_HF, 6 modèles, 3 problèmes |
| [`calibration.html`](figures/calibration.html) | Couverture contre probabilité nominale ; histogramme de z |
| [`loo_vs_test.html`](figures/loo_vs_test.html) | LOO par emplacement et par observation contre jeu de test, ajustement par ajustement |
| [`infill.html`](figures/infill.html) | Modèles issus de l'EGO v1 et v2 (précision, calibrage, recommandation) |
| [`optimization.html`](figures/optimization.html) | Convergence de l'optimisation de Hartmann |
| [`ego_rho.html`](figures/ego_rho.html) | Attribution : la correction de ρ fait préférer la BF au mérite |

## Références

* **[LG]** L. Le Gratiet, thèse (2013), éq. 3.8–3.11, Prop. 3.1.
* **[S]** M. Sacher et al., IJUQ (2021), éq. 9, 24, 30–32.
* O. Dubrule, *Cross validation of kriging in a unique neighborhood*, Math. Geology 15 (1983) ; Rasmussen & Williams,
  GPML (2006) éq. 5.12 — LOO en forme fermée.
* emukit 0.5.1 / GPy 1.14.2, BoTorch 0.18.1, SMT 2.15, NeuralFoil 0.3.3, XFOIL de bdToolbox (`soft/xfoil.exe`).
