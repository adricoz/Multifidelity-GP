# mfego v3 — ρ appris comme hyperparamètre, et départage v1 / v2 / v3

16/10/2026 · dossier `benchmarks/surrogate_v3/` · suite de [`RAPPORT_V2.md`](../surrogate_v2/RAPPORT_V2.md)

Les figures interactives sont dans [`figures/`](figures/), à ouvrir dans un navigateur (thème sombre automatique). Les
résultats bruts sont dans [`results/`](results/). Dans les tableaux, NF = NeuralFoil.

---

## 1. Résumé

* **Une prior normale apprise redonne la v2.** Apprendre ρ « comme Sacher », c'est-à-dire avec les autres
  hyperparamètres dans L-BFGS-B, sous la prior normale de la v2, donne le même modèle que la v2. Pour θ fixé, la forme
  fermée de la v2 est l'optimum exact en ρ. Un test le vérifie : mêmes optimums à 1e-6 près, même ρ à 1e-3 près.
* **Ce que change la v3.** ρ suit une **prior log-normale** et est vraiment appris par L-BFGS-B, sans forme fermée,
  avec un gradient analytique.
  * ρ > 0, mode 1, bornes larges [0,001 ; 50] : c'est la prior, et non plus la borne ρ ≤ 5, qui limite ρ. Sans
    jacobien, cette prior revient à une pénalité normale sur log ρ, symétrique en facteur autour de 1 (§ 2).
  * Le centrage de la v2 est conservé.
  * Environ 40 lignes dans `surrogate_models.py` ; 8 nouveaux tests et 3 tests mis à jour ; 168 tests passent (1
    ignoré, numba absent).
* **Modèle de substitution (`CoKriging`, plans imbriqués) : la règle écrite d'avance désigne la v3 avec
  LogNormal(1, 1).** Ce choix devient le défaut de `CoKriging`.
  * **La v2 N(1, 2²) est exclue par la règle** : ρ̂ est en butée dans 40 % des ajustements de Hartmann. Parmi les
    variantes éligibles, la v3 s = 1 a le meilleur score de NLPD (−0,52, contre −0,44 pour les deux autres). Elle
    reste première avec des moyennes au lieu de médianes, et en retirant n'importe laquelle des 10 graines.
  * Elle n'est **pas meilleure que la v2 N(1, 2²)** en NLPD sur Hartmann : comparées graine par graine, c'est la v2 qui
    gagne 18 ajustements sur 30. L'avance de la v3 dans le score est un effet de médianes sur 10 graines (§ 4).
  * Sur les deux airfoils, les quatre variantes centrées sont identiques (ρ ≈ 1).
  * **Compromis sur Hartmann.**
    * La v3 a la meilleure couverture à 95 % (0,87 à 0,94), mais pas toujours la meilleure NLPD.
    * Sa RMSE est moins bonne que celle de la v2 sur 25 ajustements sur 30 : Q² 0,48 contre 0,56 à n_HF = 40.
    * Libéré de la borne, ρ̂ monte à 5,8–8,5 en médiane, et jusqu'à 25. Limiter ρ, par la borne 5 ou par une prior
      normale plus serrée, améliorait la RMSE.
  * Contre la concurrence, la v3 reste devant emukit MAP et BoTorch MF-GP sur Hartmann et NF/XFOIL, à égalité sur
    NF/NF.
* **Optimisation (NN-MF-EGO) : aucune variante ne bat la v1, la v1 reste le défaut de `MultifidelityModel` et du
  pipeline.**
  * Sur Hartmann, la v3 bat la v1 sur 2 graines sur 10.
  * Comme la v2, elle évalue 85 à 87 % de BF, car les données imposent ρ ≈ 2,4. La prior n'y change rien.
  * **Hypothèse :** le terme ρ² de la fonction de mérite (Sacher éq. 24) pousse l'EGO vers la BF. Elle n'est pas
    démontrée : Matérn seul, sans changer ρ, perd aussi contre la v1, et 10 graines ne séparent pas les effets.
* **À décider.** Si la RMSE compte plus que la calibration, la v2 N(1, 1²) est l'alternative éligible : rapport de
  RMSE de 0,96, 10 % de butée. Le changement tient en une ligne (§ 6).

---

## 2. Profiler ou apprendre ρ : même optimum, sauf si la prior change

**L'équivalence.** Pour θ fixé (longueurs, t1, t2, bruit), −log L(θ, ρ) − log p(ρ) est un polynôme du second degré
en ρ quand la prior est normale. Son minimum est la forme fermée de la v2 :

ρ̂(θ) = (FᵀK⁻¹y + μ/σ²) / (FᵀK⁻¹F + 1/σ²), tronquée aux bornes.

Donc min sur (θ, ρ) = min sur θ de [ −log L(θ, ρ̂(θ)) − log p(ρ̂(θ)) − log p(θ) ]. Le test
`test_learning_rho_with_a_normal_prior_gives_the_closed_form` minimise les deux objectifs par L-BFGS-B, à plusieurs
départs, et trouve le même minimum (1e-6 en relatif) et le même ρ (1e-3).

**La v3.** Une prior **log-normale** n'a pas de forme fermée.
* **Paramètre :** ρ devient un paramètre de L-BFGS-B, en log comme les autres hyperparamètres : vecteur
  `[log l, log t1, log t2, log bruit, log ρ]`.
* **Gradient :** d(NLL)/d log ρ = −ρ f_nᵀK⁻¹(y_n − ρ f_n), plus le gradient de la prior. Il est vérifié par
  `check_grad`.
* **Mode de la prior :** mfego fait le MAP dans l'espace des paramètres, sans jacobien. Le MAP tire donc ρ vers le
  **mode** de la prior, exp(μ − s²). La prior v3 est LogNormal(μ = s², s), de mode 1. C'est la convention déjà
  utilisée pour t1, dont la prior LogNormal(1, 1) a un mode à 1.
* **Pénalité effective :** avec μ = s², la pénalité −log p(ρ) vaut exactement (log ρ)²/(2s²) + constante. C'est une
  loi normale N(0, s²) sur log ρ : ρ = 2 et ρ = 0,5 coûtent la même chose. Pour s = 1, ρ = 25 ne coûte que 5,2 nats.
  C'est pourquoi ρ̂ peut aller aussi loin sur Hartmann.
* **Positivité :** ρ > 0. Avec une corrélation négative, la prior arrête ρ à l'intérieur de ]0,001 ; 1[ (testé).

Le tableau donne la plage de ρ où la pénalité de la prior reste sous 1,92 nat, l'équivalent de ± 1,96 écart-type.

| Prior sur ρ | Estimation | Plage où la pénalité < 1,92 nat | Bornes |
|---|---|---|---|
| N(1, 2²) (v2) | forme fermée | [−2,9 ; 4,9] | ±5 |
| N(1, 1²) | forme fermée | [−0,96 ; 2,96] | ±5 |
| LogNormal(0,25 ; 0,5) (v3, s = 0,5) | apprise | [0,38 ; 2,7] | [0,001 ; 50] |
| **LogNormal(1 ; 1) (v3, s = 1)** | apprise | [0,14 ; 7,1] | [0,001 ; 50] |

Figure : [`priors.html`](figures/priors.html).

**Code.**
* **`mfego/src/surrogate_models.py`, `GaussianProcess`, ~40 lignes `[V3]` :**
  * `check_priors` accepte la famille `"lognormal"` pour `"rho"` ;
  * nouvelle méthode `_learns_rho` ;
  * le dernier paramètre log ρ est géré dans `fit`, `negative_log_likelihood`, `neg_log_priors` et
    `_initial_guesses` ;
  * nouvelles constantes `V3_RHO_PRIOR` et `V3_RHO_BOUNDS`.
* **`mfego/src/cokriging.py` :** les défauts v3 (§ 6).
* **`MultifidelityModel` :** inchangé, l'option passe par `priors` et `rho_bounds`.

**Nouveaux tests (8)** :
* équivalence avec la forme fermée ;
* objectif et gradient avec ρ appris, avec et sans centrage ;
* ρ = 2 retrouvé (cas décalé de +50), ρ positif, bornes invalides refusées ;
* aller-retour JSON de `MultifidelityModel` (avec ρ = 2 sur Forrester) et de `CoKriging` ;
* bornes de ρ conservées dans le JSON, fichier v2 relu en v2 ;
* normalisation des entrées avec ρ appris.

**Tests mis à jour (3)** : familles acceptées pour ρ ; défauts de `CoKriging` (v3, retour à v2 ou v1 par option,
maximum de vraisemblance).

---

## 3. Protocole et règles de décision

Script `run_benchmark.py`. On réutilise les problèmes, les plans et les jeux de test en cache de la v2, sans recalcul
XFOIL. **Contrôle :** les variantes v1 et v2 N(1, 2²) redonnent exactement les chiffres de `RAPPORT_V2`, avec un écart
nul sur les 90 ajustements imbriqués et sur les 10 optimisations.

| Variante | Noyau | ρ | Prior de ρ | Bornes |
|---|---|---|---|---|
| v1 | SE | non centré, forme fermée | aucune | ±5 |
| v2 N(1, 2²) (défaut de `CoKriging` avant la v3) | Matérn 5/2 | centré, forme fermée | N(1, 2²) | ±5 |
| v2 N(1, 1²) | Matérn 5/2 | centré, forme fermée | N(1, 1²) | ±5 |
| v3 s = 0,5 | Matérn 5/2 | centré, **appris** | LogNormal(0,25 ; 0,5) | [0,001 ; 50] |
| v3 s = 1 | Matérn 5/2 | centré, **appris** | LogNormal(1 ; 1) | [0,001 ; 50] |
| v1 Matérn (EGO seulement) | Matérn 5/2 | non centré, forme fermée | aucune | ±5 |

* **A. Plans imbriqués :** n_BF = 4 n_HF, 3 tailles, 10 graines, 990 ajustements. `CoKriging` dans chaque variante,
  contre emukit MLE et MAP, SMT MFK, BoTorch MF-GP, et HF seule (mfego, BoTorch).
* **B1. Optimisation de Hartmann à budget 250 :** 10 graines, NN-MF-EGO dans chaque variante, plus SF-EGO et BoTorch
  qLogEI.
* **B2. Données de l'infilling :** NN-MF-EGO jusqu'au coût d'un plan imbriqué, 3 problèmes, 2 tailles, 10 graines.

**Règles, fixées dans l'en-tête du script avant le calcul :**
1. **Défaut de `CoKriging`.**
   * **Score :** écart moyen de NLPD à la v1, moyenne des médianes par (problème, taille).
   * **Éligibilité :** sur chaque problème, rapport de RMSE moyen ≤ 1,05 et ρ̂ en butée dans au plus 10 % des
     ajustements.
   * **Choix :** la meilleure variante éligible. Si aucune ne l'est, la v2 N(1, 2²) reste.
2. **Défaut de `MultifidelityModel` et du pipeline.** La v1 reste, sauf si une variante remplit les deux conditions :
   * elle bat la v1 sur Hartmann (B1) sur au moins 6 graines sur 10, avec une médiane meilleure ;
   * elle n'est pas pire que la v1 sur les airfoils (B2) de plus de 0,2 point de traînée, en médiane.

---

## 4. Plans imbriqués : précision, calibration, ρ

Figures : [`nested_accuracy.html`](figures/nested_accuracy.html) · [`rho.html`](figures/rho.html). Médianes sur 10
graines, dans chaque case : **RMSE relative · NLPD · couverture 95 %** (Q² = 1 − RMSE_rel²).

**Hartmann 6D**

| Modèle | n_HF = 10 | n_HF = 20 | n_HF = 40 | ρ̂ médian (10 · 20 · 40) |
|---|---|---|---|---|
| mfego v1 | 0,95 · 2,88 · 0,82 | 0,84 · 1,37 · 0,86 | 0,76 · 0,60 · 0,87 | 0,97 · 0,98 · 0,96 |
| mfego v2 N(1, 2²) | **0,92** · 2,09 · 0,85 | **0,82** · **0,59** · 0,89 | **0,66** · **0,13** · 0,92 | 2,9 · 4,9 · 5,0 (butée) |
| mfego v2 N(1, 1²) | 0,94 · 2,42 · 0,84 | 0,84 · 0,72 · 0,87 | 0,67 · 0,17 · 0,92 | 1,5 · 3,1 · 4,0 |
| mfego v3 s = 0,5 | 0,95 · 2,42 · 0,84 | 0,86 · 0,63 · 0,90 | 0,72 · 0,21 · 0,93 | 1,1 · 4,5 · 7,7 |
| **mfego v3 s = 1** | 0,96 · 1,73 · **0,87** | 0,88 · 0,66 · **0,90** | 0,72 · 0,14 · **0,94** | 5,8 · 6,0 · 8,5 |
| GPy/emukit AR(1), MAP joint | 0,95 · **1,24** · 0,85 | 0,99 · 1,31 · 0,85 | 0,76 · 0,24 · 0,91 | 11 · 6,4 · 7,4 |
| BoTorch MF-GP | 1,00 · 2,52 · 0,83 | 0,92 · 2,87 · 0,78 | 0,84 · 3,06 · 0,80 | — |

**Airfoils.** Les quatre variantes centrées donnent les mêmes médianes à 0,005 près. ρ̂ ≈ 1 est loin des bornes et
des queues des priors.

| Modèle | NF/NF, n_HF = 5 · 10 · 20 | NF/XFOIL, n_HF = 5 · 10 · 20 |
|---|---|---|
| mfego v1 | 0,39 · 3,46 · 0,89 / 0,32 · 3,43 · 0,85 / 0,26 · 3,18 · 0,90 | 0,29 · 3,31 · 0,88 / 0,23 · 3,59 · 0,79 / 0,15 · 3,09 · 0,83 |
| **mfego v2 ou v3 (centré)** | 0,37 · 3,38 · 0,93 / 0,26 · 3,06 · 0,91 / 0,21 · 2,80 · 0,94 | 0,29 · **3,15** · 0,94 / 0,20 · **2,83** · 0,87 / **0,12** · **2,41** · 0,90 |
| GPy/emukit AR(1), MAP joint | 0,37 · **3,36** · 0,95 / **0,25** · 3,07 · 0,89 / 0,21 · 3,19 · 0,84 | 0,29 · 3,18 · 0,95 / **0,19** · 3,28 · 0,82 / 0,15 · 3,45 · 0,76 |
| BoTorch MF-GP | 0,43 · 3,46 · 0,97 / 0,26 · **3,03** · 0,95 / **0,20** · **2,78** · 0,94 | 0,35 · 3,39 · 0,96 / 0,22 · 2,86 · 0,91 / 0,14 · 2,67 · 0,88 |

Les autres modèles (emukit MLE, SMT MFK, HF seule) restent derrière, comme en v2. Ils figurent dans
`results/summary.txt`.

**Graines où mfego v3 s = 1 fait mieux** (30 ajustements par problème, RMSE / NLPD) :

| contre | Hartmann | airfoil NF/NF | airfoil NF/XFOIL |
|---|---|---|---|
| mfego v1 | 13 / 24 | 25 / 27 | 25 / 28 |
| mfego v2 N(1, 2²) | **5 / 12** | 17 / 15 | 16 / 19 |
| GPy/emukit MAP | 20 / 25 | 16 / 18 | 18 / 24 |
| BoTorch MF-GP | 19 / 26 | 17 / 14 | 27 / 22 |

Sur les airfoils, v3 et v2 diffèrent de moins de 1e-3 en RMSE et de 4e-3 en NLPD, ajustement par ajustement : les
victoires de 15 à 19 sur 30 y sont du bruit numérique.

**Règle 1 appliquée**

| Variante | Score (ΔNLPD moyen) | Rapport de RMSE (H · NF · XFOIL) | ρ̂ en butée (H · NF · XFOIL) | Éligible |
|---|---|---|---|---|
| **v3 s = 1** | **−0,52** | 1,00 · 0,85 · 0,90 | 0 · 0 · 0 % | **oui → retenue** |
| v2 N(1, 2²) | −0,49 | 0,94 · 0,85 · 0,90 | **40** · 0 · 0 % | non |
| v3 s = 0,5 | −0,44 | 0,99 · 0,85 · 0,90 | 0 · 0 · 0 % | oui |
| v2 N(1, 1²) | −0,44 | 0,96 · 0,85 · 0,90 | 10 · 0 · 0 % | oui |

**Lecture.**
* **Le score place la v3 s = 1 devant la v2 N(1, 2²), mais la comparaison appariée dit l'inverse.**
  * Le score compare des médianes. À n_HF = 10, la médiane de NLPD vaut 1,73 pour la v3 contre 2,09 pour la v2.
  * Graine par graine, c'est la v2 qui fait mieux : sur 8 graines sur 10 à n_HF = 10, et sur 18 ajustements sur 30 au
    total sur Hartmann. Les distributions sur 10 graines sont bimodales, et leur médiane est instable.
  * Avec des moyennes au lieu de médianes, l'ordre s'inverse : −0,51 pour la v2, −0,48 pour la v3.
  * **La décision n'en dépend pas.** La v2 N(1, 2²) est inéligible à cause de la butée. Parmi les éligibles, la v3
    s = 1 reste première, en médianes comme en moyennes (−0,48 contre −0,44 et −0,45), et en retirant n'importe
    laquelle des 10 graines. Contre la v2 N(1, 1²), elle gagne en NLPD sur 8 graines sur 10 à n_HF = 10 et à
    n_HF = 20.
  * Pour une prochaine règle, une statistique appariée serait plus solide.
* **Sur Hartmann, la RMSE de la v3 n'est pas meilleure que celle de la v1** (rapport 1,00), contre 0,94 pour la v2.
  Limiter ρ améliorait la RMSE.
  * Dans les 12 ajustements où la v2 N(1, 2²) est en butée à 5, elle bat la v3 en RMSE 11 fois (rapport médian
    0,88). Dans les 18 autres, le rapport vaut 0,99.
  * La v2 N(1, 1²), en butée dans 10 % des cas seulement, atteint aussi 0,96 : une prior normale plus serrée limite ρ
    comme la borne.
  * Sans borne, ρ̂ monte à 5,8–8,5 en médiane, et jusqu'à 25. Il dépasse 5 dans 73 % des ajustements.
  * Hartmann a une relation BF/HF très non linéaire, qu'un ρ constant décrit mal. Les données réclament un ρ grand,
    et un ρ grand dégrade la prédiction. C'était déjà la conclusion de `RAPPORT_COKRIGING` § 7, avec des bornes ±50.
* **En revanche, la v3 s = 1 a la meilleure couverture à 95 % sur Hartmann**, à toutes les tailles et parmi tous les
  modèles (0,87 ; 0,90 ; 0,94). Sa moyenne quadratique de z vaut 2,23, 1,33 et 1,07, contre 2,46, 1,58 et 1,18 pour
  la v2. À n_HF = 10, emukit MAP fait mieux en NLPD (1,24) et en z (1,98).
* **Le résultat ne dépend plus d'une borne arbitraire** : ρ̂ n'est en butée nulle part. Sa limite vient d'une prior
  explicite, dont on peut discuter.
* **Coût :** apprendre ρ allonge un peu l'ajustement sur les airfoils (0,10 à 0,27 s contre 0,08 à 0,23 s). Sur
  Hartmann, c'est 0,14 à 2,0 s, contre 5 à 17 s pour emukit MAP et 2,5 à 10 s pour BoTorch.

---

## 5. Optimisation NN-MF-EGO

Figures : [`optimization.html`](figures/optimization.html) · [`ego.html`](figures/ego.html).

**Hartmann 6D, budget 250, 10 graines (f − f\*)**

| Méthode | Recommandation, médiane (moyenne) | Graines où elle bat v1 | Part BF des évaluations (médiane) | Part BF du coût | ρ final médian |
|---|---|---|---|---|---|
| **NN-MF-EGO v1** | **0,25** (0,49) | — | 7 % | 1 % | 0,27 |
| NN-MF-EGO v1 Matérn | 0,29 (0,59) | 4 | 7 % | 1 % | 0,28 |
| NN-MF-EGO v2 N(1, 2²) | 0,29 (0,51) | 1 | 86 % | 38 % | 2,41 |
| NN-MF-EGO v2 N(1, 1²) | 0,31 (0,50) | 3 | 82 % | 32 % | 2,36 |
| NN-MF-EGO v3 s = 0,5 | 0,27 (0,50) | 2 | 85 % | 37 % | 2,47 |
| NN-MF-EGO v3 s = 1 | 0,31 (0,54) | 2 | 87 % | 40 % | 2,37 |
| SF-EGO (Matérn) | 0,46 (0,67) | 3 | — | — | — |
| BoTorch qLogEI (SF) | 0,71 (1,00) | 2 | — | — | — |

**Airfoils (B2).** Toutes les variantes recommandent le même profil que la v1 :
* l'écart médian de Cd recommandé avec la v1 vaut 0,000 à 0,007 point ;
* sur NF/XFOIL, la recommandation médiane vaut Cd = 110,6 points pour toutes les variantes ; les runs vont de 110,6 à
  111,1 points ;
* une seule exception : une graine de la v1 Matérn sur NF/XFOIL, à 120,7 points.

**Données de l'infilling sur Hartmann (B2).** La v1 garde le meilleur modèle final : Q² 0,56 à n_HF = 40, contre 0,42
pour la v1 Matérn et 0,22 à 0,29 pour les variantes centrées. Elle garde aussi la meilleure recommandation à
n_HF = 20 : 0,10, contre 0,13 pour la v1 Matérn et 0,19 à 0,29 pour les variantes centrées.

**Règle 2 appliquée.** Aucune variante ne passe : 4 graines sur 10 au mieux (v1 Matérn), et 2 sur 10 pour la v3. **La
v1 reste le défaut de `MultifidelityModel` et du pipeline.**

**Lecture.**
* **La v3 ne corrige pas l'EGO.** Dans l'EGO, ρ final ≈ 2,4 pour les quatre variantes centrées, quelle que soit la
  prior : ce sont les données qui l'imposent. La fonction de mérite (Sacher éq. 24) multiplie l'information de la BF
  par ρ² ≈ 5,6 à 6,1, et l'EGO passe 82 à 87 % de ses évaluations en BF.
* **Hypothèse, non démontrée :** le correctif serait à chercher dans la **fonction de mérite** (RAPPORT_V2 § 8)
  plutôt que dans l'estimation de ρ. La v1 Matérn, avec le même ρ que la v1 et la même part de BF (7 %), perd aussi
  contre la v1 (médiane 0,29, 4 graines sur 10). La part du noyau et celle de ρ ne sont donc pas séparées.

---

## 6. Défauts appliqués

| Classe | Défaut | Règle |
|---|---|---|
| `CoKriging` (modèle de substitution, plans imbriqués) | **v3** : Matérn 5/2, MAP, ρ centré appris sous `V3_RHO_PRIOR` = LogNormal(1, 1), `V3_RHO_BOUNDS` = [0,001 ; 50] | 1 |
| `MultifidelityModel` (EGO), pipeline bdFoil | **v1** : ρ non centré, sans prior (SE dans le pipeline) | 2 |

* **`CoKriging`.**
  * `priors["rho"]` vaut par défaut `V3_RHO_PRIOR` si ρ est centré.
  * `rho_bounds=None` donne [0,001 ; 50] pour un ρ appris et ±5 sinon.
  * Le ρ appris n'est utilisé qu'avec le MAP. Avec `use_map=False`, ρ reste profilé dans ±5.
  * La v2 se retrouve avec `CoKriging(L, priors={"rho": ("normal", 1.0, 2.0)})`, et la v1 avec
    `CoKriging(L, SquaredExponentialKernel, rho_centering=False)` (testé).
  * Les bornes de ρ sont enregistrées dans le JSON.
  * `DEFAULT_PRIORS["rho"]` reste N(1, 2²), pour ne rien changer en silence aux autres classes.
* **Compatibilité.**
  * Un fichier `CoKriging` écrit en v2 contient sa prior normale et reste en v2 au rechargement.
  * Un fichier v1 reste en v1.
  * `CoKriging.from_multifidelity` reprend la prior de ρ du modèle d'origine.
  * Les scripts de la v2 fixent maintenant leurs options explicitement et redonnent leurs chiffres.
* **Alternative orientée RMSE.** Si l'on préfère la meilleure RMSE sur des cas de type Hartmann à la meilleure
  calibration, la v2 N(1, 1²) est éligible, avec un rapport de RMSE de 0,96 et 10 % de butée. Sur les airfoils, elle
  est identique. Le changement tient en une ligne : remplacer `V3_RHO_PRIOR` par `("normal", 1.0, 1.0)` dans
  `CoKriging`.

---

## 7. Limites et suites

* **Le ρ constant est le vrai problème de Hartmann.** Ni la borne ni la prior ne règlent le fait que la relation BF/HF
  n'est pas une homothétie. Un ρ(x) dépendant de l'entrée (Le Gratiet, ρ(x) = g(x)ᵀβ) serait la piste de fond.
* **EGO :** le terme ρ² du mérite reste à revoir, avant tout changement de défaut côté optimisation.
* **Pas de sélection sur plans non imbriqués** : ce choix a été fait au départ. En v2, le centrage était défavorable
  sur NF/XFOIL non imbriqué. `MultifidelityModel` garde donc la v1.
* **Trois problèmes et 10 graines.** L'avance de la v3 s = 1 tient à une seule case (Hartmann, n_HF = 10). Sur des cas
  réels proches des airfoils (ρ ≈ 1), le choix entre v2 et v3 ne change rien.

## Reproduction

Depuis la racine du dépôt, avec `.venv-benchmark`. Il faut compter environ 10 min, car XFOIL est appelé pendant l'EGO.

```
.venv-benchmark\Scripts\python benchmarks\surrogate_v3\run_benchmark.py      # --quick : 3 graines
.venv-benchmark\Scripts\python benchmarks\surrogate_v3\make_figures.py
```

| Fichier | Rôle |
|---|---|
| `v3_lib.py` | Variantes v1 / v2 / v3, réutilisation des plans et jeux de test de la v2 |
| `run_benchmark.py` | A, B1, B2 et les deux règles de décision (en-tête) |
| `make_figures.py` | Les 5 figures de `figures/` |
| `results/` | `nested.csv`, `optimization.csv`, `optimization_histories.json`, `infill.csv`, `summary.txt` (tableaux des règles), `console.txt` |

## Figures

| Figure | Contenu |
|---|---|
| [`priors.html`](figures/priors.html) | Les quatre priors sur ρ et la borne 5 de la v2 |
| [`rho.html`](figures/rho.html) | ρ̂ de chaque variante sur plans imbriqués, part en butée |
| [`nested_accuracy.html`](figures/nested_accuracy.html) | RMSE, NLPD et couverture contre n_HF, 5 variantes et 2 concurrents |
| [`optimization.html`](figures/optimization.html) | Convergence et recommandation de l'optimisation de Hartmann |
| [`ego.html`](figures/ego.html) | NN-MF-EGO par variante : recommandation, part BF, ρ final, airfoils |

## Références

* **[S]** M. Sacher et al., IJUQ (2021), éq. 9, 15, 24 : ρ hyperparamètre de la vraisemblance du niveau l.
* **[LG]** L. Le Gratiet, thèse (2013), éq. 3.11, 4.10 (forme fermée de ρ), Prop. 3.1.
* Benchmarks précédents : [`RAPPORT_V2.md`](../surrogate_v2/RAPPORT_V2.md),
  [`RAPPORT_COKRIGING.md`](../cokriging/RAPPORT_COKRIGING.md).
