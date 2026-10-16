# Co-krigeage conditionné à toutes les données ([LG] éq. 3.11) — plans imbriqués

> **Note v2.** Ce rapport décrit **mfego v1** (16/10/2026, matin). Le biais de ρ (§ 7) est corrigé dans la v2 (ρ centré
> et prior N(1, 2²)), qui ajoute aussi Matérn 5/2, la normalisation des entrées et la qualification par leave-one-out.
> Voir [`benchmarks/surrogate_v2/RAPPORT_V2.md`](../surrogate_v2/RAPPORT_V2.md).
>
> Relancer ces scripts aujourd'hui utilise les défauts v2. Pour retrouver les chiffres ci-dessous, il faut les options v1 :
> `priors=V1_PRIORS, rho_centering=False` et `SquaredExponentialKernel`. `forrester_checks.py` les fixe déjà.

16/10/2026 · nouvelle classe `CoKriging` (`mfego/src/cokriging.py`, modèle de substitution seulement) comparée au modèle
récursif actuel de mfego, à GPy/emukit, à SMT et aux surrogates issus de l'infilling NN-MF-EGO · dossier
`benchmarks/cokriging/`

Les figures interactives sont dans [`figures/`](figures/) (à ouvrir dans un navigateur ; plotly.js vient du CDN, thème
sombre automatique). Les résultats bruts sont dans [`results/`](results/). Exemple minimal :
[`example/cokriging/main.py`](../../example/cokriging/main.py).

---

## 1. Résumé

* **Implémenté.** `CoKriging` conditionne Z_L(x) sur **toutes** les observations, tous niveaux confondus, avec la matrice
  de covariance jointe V ([LG] éq. 3.11, généralisée à L niveaux). La prédiction suit [LG] éq. 3.8–3.10 : à
  hyperparamètres donnés, c'est la moyenne et la variance a posteriori exactes du modèle AR(1), quel que soit le bruit.
  * Les hyperparamètres sont estimés niveau par niveau en réutilisant tel quel `GaussianProcess.fit`, priors comprises,
    car sur plan imbriqué la vraisemblance se factorise ([LG] Prop. 3.1).
  * Cette factorisation suppose des niveaux inférieurs **peu bruités**. Avec une BF bruitée, le résidu du niveau l
    contient aussi ρ² × le bruit BF, et ρ̂ est biaisé vers 0 (§ 2.1).
  * Le code reste court : une fonction assemble V, une Cholesky, deux résolutions triangulaires. Le fichier fait
    283 lignes, docstrings comprises.
  * La classe refuse un plan non imbriqué avec un message explicite. `missing_points_for_nesting` liste les points à
    évaluer aux niveaux inférieurs pour le rendre imbriqué.
* **Vérifié, de trois façons :**
  1. V est égale, à 10⁻¹² près, à la matrice par blocs de [LG] éq. 3.11 écrite à la main (test) ;
  2. avec les hyperparamètres d'emukit, `CoKriging` reproduit emukit à **4·10⁻¹¹ près** sur la moyenne et à 3·10⁻¹⁰ près sur
     la variance, sur 120 ajustements (2 problèmes, 3 tailles, 10 graines, emukit MLE et MAP). C'est le même
     co-krigeage ;
  3. sur plan imbriqué, `CoKriging` et le modèle récursif actuel donnent la même chose (écart maximal de RMSE relative
     7·10⁻⁴). C'est l'équivalence attendue de [LG] Prop. 3.1. Un test unitaire la vérifie à hyperparamètres égaux.
* **Conséquence directe de 3.** Sur plan imbriqué, la nouvelle classe ne peut pas faire *mieux* que le modèle actuel :
  elle donne la même loi prédictive. Son intérêt est d'être la référence exacte du conditionnement, à hyperparamètres
  donnés.
  * Petit écart de forme relevé au passage : la variance totale du modèle récursif ajoute ρ² × le bruit BF en plus du
    bruit HF, alors que la prédiction HF ne devrait porter que le bruit HF. C'est négligeable avec la pépite par défaut.
* **La « vraie variance » du modèle récursif sur plan non imbriqué** (données de l'infilling) a été mesurée directement.
  Aux points HF observés, là où la valeur est connue, l'écart-type prédit vaut 1,4 à 5,2 % de std(y) sur Hartmann et
  0,2 à 3,3 % sur l'airfoil. Le conditionnement exact donne 0,3–0,5 % et 0,02–0,04 %, c'est-à-dire le niveau du bruit.
  L'erreur est structurelle (le terme ρ²σ²_BF(x) ne s'annule pas) mais **modeste sur le domaine**, à hyperparamètres
  égaux (§ 6) :
  * Hartmann : aucun changement ;
  * airfoil à coût 140 : RMSE 0,42 → 0,38, NLPD 3,96 → 3,65, couverture 0,77 → 0,82 ;
  * airfoil à coût 280 : quasi nul.
* **Le surrogate de l'infilling n'est pas systématiquement mauvais : il est spécialisé** (§ 6).
  * Sur l'airfoil, il est moins bon globalement que le LHS imbriqué de même coût (RMSE 0,42 contre 0,32 à coût 140)
    mais meilleur près de l'optimum (0,12 contre 0,18), et trop sûr de lui (couverture 95 % : 0,75–0,77 contre
    0,85–0,90).
  * Sur Hartmann, il est même meilleur globalement (0,66 contre 0,76 à coût 560), avec des intervalles trop larges
    (couverture 0,98–0,99). L'EGO y dépense presque tout le budget en HF, et la BF de Hartmann aide peu.
* **Ce qui limite vraiment la qualité et le calibrage : l'estimation des hyperparamètres, et surtout ρ** (§ 7).
  * mfego fixe la constante du résidu avec ρ = 1, ce qui tire ρ̂ vers 1 sur plan imbriqué (0,95–0,99 sur les deux
    problèmes). Emukit trouve 0,64–0,81 sur l'airfoil et 6 à 11 sur Hartmann.
  * Centrer le niveau inférieur corrige le biais. **Sur l'airfoil, le gain est net** : ρ̂ passe à 0,76–0,90, jamais en
    butée, et la RMSE est meilleure sur 7 à 9 graines sur 10.
  * **Sur Hartmann, c'est plus nuancé.** Le ρ centré part sur la borne 5 de mfego (80 à 100 % des ajustements) ; ce
    plafond agit comme une régularisation. Libre (bornes ±50), ρ̂ monte à 6–23 : la NLPD reste meilleure, mais la RMSE
    se dégrade (1,39 contre 0,95 à n_HF = 10). Avec une relation BF/HF aussi non linéaire, un ρ constant élevé
    amplifie les erreurs BF.
  * **Correctif proposé, non appliqué au cœur** : il faudra décider en même temps comment borner ou régulariser ρ.
* **Coût :** l'ajustement du `CoKriging` coûte celui du modèle récursif plus une Cholesky de V. Sur Hartmann, à n_HF = 40
  (200 points), il prend 4,2 s contre 3,9 s, et 11,7 s pour emukit MAP.

---

## 2. Ce qui a été implémenté

### 2.1 Le modèle (`mfego/src/cokriging.py`)

Modèle auto-régressif (Kennedy & O'Hagan ; [LG] éq. 3.1 ; [S] éq. 9) :

> Z₁ = δ₁, Z_l = ρ_(l−1) Z_(l−1) + δ_l, avec δ₁, …, δ_L des PG indépendants.

En unités brutes, δ_m a pour moyenne la constante `y_mean_m` et pour covariance c_m(x, x′) = `y_std_m`² (t1_m e^(−r²/2l²) +
t2_m), soit le noyau de mfego. On en déduit :

* Z_l = Σ_m P(m, l) δ_m, avec P(m, l) = ρ_m ρ_(m+1) ⋯ ρ_(l−1) (P(l, l) = 1, et P(m, l) = 0 si m > l) ;
* **Cov(Z_i(x), Z_j(x′)) = Σ_(m ≤ min(i, j)) P(m, i) P(m, j) c_m(x, x′)**. Pour L = 2, c'est exactement la matrice par
  blocs de [LG] éq. 3.11 :

  > V = [[c₁(D₁, D₁), ρ c₁(D₁, D₂)], [ρ c₁(D₂, D₁), ρ² c₁(D₂, D₂) + c₂(D₂, D₂)]] + bruit sur la diagonale ;

* moyenne a priori de Z_l : m_l = Σ_m P(m, l) y_mean_m ;
* prédiction ([LG] éq. 3.8–3.10) : **μ(x) = m_L + k(x)ᵀ V⁻¹ (z − m)** et **σ²(x) = c_LL(x, x) − k(x)ᵀ V⁻¹ k(x)**, plus le bruit
  du niveau si `include_noise=True`.

Correspondance avec [LG] : σ_t² R_t ↔ `y_std_t`² × noyau normalisé. La tendance β_t de [LG] est ici une constante par
niveau, `y_mean`, plus la variance de biais t2 du noyau. Ce terme constant est bayésien : son incertitude est déjà dans la
variance, sans terme de krigeage universel séparé.

**Estimation des hyperparamètres, niveau par niveau.** Sur plan imbriqué, z₁(D₂) est observé. La vraisemblance jointe
s'écrit alors p(z₁) · p(z₂ | z₁), et le second facteur ne dépend que de δ₂ ([LG] p. 91 et Prop. 3.1). On réutilise donc
`GaussianProcess.fit` sans le modifier :

* le niveau 1 est ajusté sur z₁ ;
* le niveau l est ajusté sur z_l avec `f_prev = z_(l−1)(D_l)`, c'est-à-dire les **observations** du niveau inférieur aux
  points du niveau l, et non des prédictions ; ρ est profilé ;
* les priors sont les mêmes que pour le reste de mfego (`use_map=True` par défaut, voir
  `benchmarks/priors_all/RAPPORT_PRIORS.md`).

La factorisation est exacte sans bruit au niveau inférieur. Avec la pépite par défaut (10⁻⁶ en unités normalisées), l'écart
est négligeable. **Avec une BF réellement bruitée, l'estimation par niveau devient fausse.**

* Le résidu z_l − ρ z_(l−1)(D_l) contient ρ² × le bruit BF : le bruit du niveau l est surestimé.
* Le régresseur bruité biaise ρ̂ vers 0.
* Contrôle `forrester_checks.py` (Forrester, plan imbriqué 30 / 12, bruit BF d'écart-type 0,5, HF exacte) : ρ̂ = 1,73,
  0,03 et 0,11 au lieu de 2 sur 3 plans. De plus, le vrai bruit BF dépasse ici la borne haute du bruit normalisé de mfego
  (10⁻²).

Seul le conditionnement (étape 2) reste exact. Pour des niveaux bas bruités, il faudrait l'estimation jointe sur V, comme
emukit.

**Le code**, par ordre de lecture :

| Fonction | Rôle | Lignes de code (hors docstrings) |
|---|---|---|
| `CoKriging.fit(data)` | vérifie l'imbrication, ajuste les GP niveau par niveau, appelle `condition` | 22 |
| `CoKriging.condition(xs, ys)` | assemble V, Cholesky (`safe_cholesky` existant), α = V⁻¹(z − m) | 7 |
| `_factors`, `_covariance`, `_mean` | P(m, l), Cov(Z_i, Z_j) en une boucle sur les niveaux, m_l | 12 |
| `predict_batch(x, level=None, include_noise=True)` | moyenne et variance, à n'importe quel niveau | 14 |
| `to_dict` / `from_dict`, `diagnostics` | rechargement sans réajustement, hyperparamètres de chaque niveau | 28 |
| `nested_index`, `missing_points_for_nesting` | test d'imbrication, points à compléter | 13 |

Ajout dans `mfego/src/data_management.py` : `ExperimentData.generate_nested_design(points_per_level, seed)`. Le niveau 1
est un LHS, puis chaque niveau garde un sous-ensemble bien réparti (maximin glouton, `greedy_maximin` existant) du niveau
inférieur.

### 2.2 Utilisation

```python
data = ExperimentData(bounds=[(0.0, 1.0)] * d, costs=[1.0, 10.0])
data.generate_nested_design([40, 10])          # X_2 inclus dans X_1
...                                            # évaluer les deux niveaux (data.y_dict)
model = CoKriging(2, use_map=True)             # priors sur tous les hyperparamètres
model.fit(data)
mean, var = model.predict_batch(x)             # niveau le plus haut ; level=1 pour la BF
```

Pour un plan venant de l'EGO : `missing_points_for_nesting(data, 2)` donne les points à évaluer en BF, puis `fit`.

### 2.3 Tests (`tests/test_cokriging.py`, `tests/test_benchmark_lib.py`)

* V égale la matrice par blocs de [LG] éq. 3.11 écrite à la main.
* Équivalence [LG] Prop. 3.1 : plan imbriqué, bruit ≈ 0, mêmes hyperparamètres. La moyenne et la variance sont celles de la
  formule récursive avec résidus **observés**, à 10⁻⁶ près.
* Mêmes hyperparamètres qu'un `MultifidelityModel` ajusté, sur plan imbriqué : même moyenne et même variance latente que le
  modèle récursif.
* Produits des ρ à trois niveaux (Cov(Z₁, Z₃) = ρ₁ρ₂c₁, etc.), écrits à la main.
* Avec un seul niveau, `CoKriging` donne le `GaussianProcess` de mfego.
* Interpolation aux points HF, variance ≥ 0, ρ = 2 retrouvé sur Forrester.
* Prédiction à chaque niveau.
* `include_noise` ajoute exactement le bruit du niveau.
* Plan non imbriqué refusé, puis accepté après complétion.
* Propagation de `missing_points_for_nesting` sur trois niveaux.
* Trois niveaux.
* Aller-retour JSON, priors comprises.
* `generate_nested_design`.
* Exactitude contre emukit (ignorée sans GPy/emukit).

---

## 3. Protocole

* **Problèmes**
  * **Hartmann 6D** multifidélité de [S] (éq. 30–32), `bench_lib.HartmannMF` : BF = U₁ décalé, HF = Hartmann, coûts 1 et
    10. 2000 points de test LHS.
  * **Airfoil NACA 4 chiffres** (`airfoil_problem.py`, 78 lignes autonomes, sans le pipeline bdFoil).
    * Entrées dans [0, 1]⁴ : cambrure m ∈ [0 ; 0,06], position p ∈ [0,2 ; 0,6], épaisseur t ∈ [0,08 ; 0,18] et incidence
      α ∈ [−2° ; 8°].
    * Sortie : Cd en points de traînée (10⁴ Cd), à Re = 5·10⁵, n_crit = 1 et transition forcée à 10 % (convention
      maison).
    * BF = NeuralFoil `xxsmall`, HF = NeuralFoil `xxxlarge`, coûts nominaux 1 et 10.
    * Cd HF de 112 à 267 points sur le jeu de test (99 % sous 210) ; corrélation BF/HF 0,97, écart relatif
      std(BF − HF) / std(HF) de 35 %.
    * 1000 points de test.
* **Plans imbriqués :** n_BF = 4 n_HF. Le niveau BF est un LHS, le niveau HF un sous-ensemble maximin. Hartmann :
  n_HF ∈ {10, 20, 40} ; airfoil : n_HF ∈ {5, 10, 20}. 10 graines, le même plan pour tous les modèles d'une graine.
* **Modèles :**
  * mfego `CoKriging`, MAP (priors par défaut) et MLE ;
  * mfego récursif (MAP), c'est-à-dire `MultifidelityModel` actuel ;
  * GPy/emukit `GPyLinearMultiFidelityModel` : le co-krigeage AR(1) complet de Kennedy & O'Hagan, avec **estimation
    jointe** de tous les hyperparamètres.
    * Même noyau que mfego par niveau (RBF ARD + biais constant), bruit gaussien par niveau, mêmes bornes, 5
      redémarrages.
    * Chaque niveau est centré sur sa moyenne, avec une échelle commune. Une moyenne globale donnait des ρ aberrants,
      car Hartmann a des moyennes BF et HF très différentes.
    * En MLE, ou en MAP avec les **mêmes densités** de prior que mfego. L'objectif effectif n'est pas le même pour
      autant.
      * `set_prior` remplace les bornes par la transformation positive de GPy (softplus) et ajoute son log-jacobien
        log(1 − e^(−θ)).
      * Ce jacobien est proche de ln θ pour les petites valeurs (bruit, t2) : MAP presque en espace log. Il est presque
        nul pour t1 et les grandes longueurs : MAP presque en espace θ, comme mfego.
  * SMT MFK (co-krigeage récursif de Le Gratiet) ;
  * mfego HF seule (MAP), sur les mêmes points HF.
* **Mesures :**
  * RMSE relative (RMSE / std(y)), aussi sur les 10 % meilleurs points de test (région de l'optimum) ;
  * NLPD ;
  * couverture des intervalles centraux de 10 % à 99 % ;
  * écart-type et histogramme des erreurs standardisées z = (y − μ)/σ, qui devraient suivre N(0, 1) ;
  * pour l'infilling, l'écart-type prédit aux points HF observés.
* **Infilling actuel :** NN-MF-EGO de mfego (MAP), à partir d'un plan non imbriqué (Hartmann 20 BF + 10 HF ; airfoil
  20 BF + 5 HF), jusqu'au **coût du plan imbriqué de même n_HF** (14 n_HF). On compare ensuite trois modèles sur ses
  données :
  * le modèle final de l'EGO (formule récursive) ;
  * le **conditionnement exact** des mêmes données avec les **mêmes hyperparamètres**, par la matrice V. Il n'y a pas
    d'estimation, donc pas besoin d'imbrication ;
  * un `CoKriging` ajusté après évaluation de la BF aux points HF qui en manquaient (coût ajouté compté).

---

## 4. Vérifications

Figure : [`verification.html`](figures/verification.html).

| Contrôle | Résultat |
|---|---|
| V = [LG] éq. 3.11 (L = 2, matrice par blocs écrite à la main) | écart ≤ 10⁻¹² (test unitaire) |
| `CoKriging` = formule récursive à résidus observés, plan imbriqué sans bruit ([LG] Prop. 3.1) | ≤ 10⁻⁶ (test unitaire) |
| `CoKriging` = modèle récursif mfego, 60 ajustements imbriqués | écart max de RMSE relative 7·10⁻⁴, de NLPD 0,04 |
| Hyperparamètres d'emukit dans `CoKriging`, 120 ajustements | moyenne : 4·10⁻¹¹ × std(y) ; variance latente : 3·10⁻¹⁰ (relatif), pour cond(V) jusqu'à 8·10⁷ |

Deux détails d'implémentation ont dû être alignés pour atteindre l'égalité :

* GPy ajoute une constante de 10⁻⁸ (en unités standardisées) au bruit dans son inférence (`ExactGaussianInference`) ;
* la moyenne doit être traitée de la même façon (constante par niveau).

Sans le premier point, l'écart montait à 10⁻⁴ quand le bruit était en butée basse.

---

## 5. Plans imbriqués : précision et calibration

Figures : [`nested_accuracy.html`](figures/nested_accuracy.html) · [`calibration.html`](figures/calibration.html).
Médianes sur 10 graines : **RMSE relative · NLPD · couverture 95 %**.

| Hartmann 6D | n_HF = 10 | n_HF = 20 | n_HF = 40 |
|---|---|---|---|
| **mfego CoKriging (MAP)** | 0,95 · 2,88 · 0,82 | 0,84 · 1,37 · 0,86 | 0,76 · 0,60 · 0,87 |
| mfego récursif (MAP) | 0,95 · 2,88 · 0,82 | 0,84 · 1,37 · 0,86 | 0,76 · 0,60 · 0,87 |
| mfego CoKriging (MLE) | 1,03 · 19,60 · 0,61 | 0,99 · 4,39 · 0,72 | 1,01 · 2,02 · 0,81 |
| GPy/emukit AR(1), MAP joint | 0,95 · 1,24 · 0,85 | 0,99 · 1,31 · 0,85 | 0,76 · 0,24 · 0,91 |
| GPy/emukit AR(1), MLE joint | 1,15 · 6,36 · 0,63 | 1,19 · 2,58 · 0,79 | 0,87 · 0,80 · 0,90 |
| SMT MFK | 1,49 · 4,21 · 0,82 | 1,22 · 2,91 · 0,76 | 1,01 · 0,67 · 0,92 |
| mfego HF seule (MAP) | 0,97 · 2,96 · 0,82 | 0,85 · 1,14 · 0,87 | 0,80 · 0,73 · 0,86 |

| Airfoil NACA 4 chiffres | n_HF = 5 | n_HF = 10 | n_HF = 20 |
|---|---|---|---|
| **mfego CoKriging (MAP)** | 0,39 · 3,46 · 0,89 | 0,32 · 3,43 · 0,85 | 0,26 · 3,18 · 0,90 |
| mfego récursif (MAP) | 0,39 · 3,46 · 0,89 | 0,32 · 3,43 · 0,85 | 0,26 · 3,18 · 0,90 |
| mfego CoKriging (MLE) | 0,40 · 4,03 · 0,75 | 0,32 · 3,79 · 0,85 | 0,34 · 3,54 · 0,87 |
| GPy/emukit AR(1), MAP joint | 0,37 · 3,36 · 0,95 | 0,25 · 3,07 · 0,89 | 0,21 · 3,19 · 0,84 |
| GPy/emukit AR(1), MLE joint | 0,37 · 4,62 · 0,66 | 0,27 · 3,89 · 0,78 | 0,28 · 3,51 · 0,83 |
| SMT MFK | 0,39 · 7,85 · 0,48 | 0,33 · 8,75 · 0,49 | 0,29 · 4,28 · 0,74 |
| mfego HF seule (MAP) | 0,70 · 4,14 · 0,95 | 0,48 · 3,74 · 0,95 | 0,29 · 3,29 · 0,89 |

ρ estimé (médiane) :

| | Hartmann n_HF = 10 / 20 / 40 | Airfoil n_HF = 5 / 10 / 20 |
|---|---|---|
| mfego (CoKriging et récursif) | 0,97 / 0,98 / 0,96 | 0,99 / 0,98 / 0,95 |
| GPy/emukit MAP joint | 10,97 / 6,40 / 7,41 | 0,80 / 0,81 / 0,64 |

**Lecture.**

* **Le multifidélité paie nettement sur l'airfoil**, avec une RMSE divisée par 1,8 à n_HF = 5 (0,39 contre 0,70) et par
  1,5 à n_HF = 10. Sur Hartmann, il paie peu (0,76 contre 0,80 à n_HF = 40). La relation entre U₁ et Hartmann est très
  non linéaire : la pente locale dHF/dBF va de 1,5 au fond des puits à plus de 10 ailleurs, ce qu'un ρ constant ne
  représente pas.
* **Le MAP compte plus que la formule.** `CoKriging` MLE est bien moins calibré que `CoKriging` MAP (couverture 0,61 contre
  0,82 sur Hartmann à n_HF = 10), et il en va de même pour emukit.
* **emukit MAP fait mieux que mfego** en NLPD sur Hartmann (0,24 contre 0,60 à n_HF = 40) et en RMSE sur l'airfoil (0,21
  contre 0,26 à n_HF = 20). Ce n'est pas la formule, identique (§ 4) : c'est l'estimation de ρ (§ 7).
* **Calibration** (figure `calibration.html`) :
  * sur l'airfoil, les courbes de couverture de `CoKriging` et d'emukit MAP suivent la diagonale, un peu en dessous
    (couverture 95 % : 0,85–0,90), et l'histogramme de z est proche de N(0, 1) ;
  * sur Hartmann, l'histogramme est pointu au centre avec des queues lourdes. Le modèle stationnaire surestime
    l'incertitude dans les grandes zones plates de Hartmann (≈ 0) et la sous-estime dans les puits.
  * Les couvertures 50 % / 90 % / 99 % du `CoKriging` valent 0,51 / 0,83 / 0,90 sur Hartmann et 0,43 / 0,79 / 0,94 sur
    l'airfoil (n_HF = 20 et 10).
* **Ce que veut dire « vraie variance ».** C'est la variance a posteriori exacte **à hyperparamètres fixés**. L'incertitude
  sur les hyperparamètres n'y est pas, ce qui donne une variance sous-estimée avec très peu de points HF ([LG] p. 90 et
  101). Exemple sur Forrester 1D (`forrester_checks.py`, sortie dans `results/forrester_checks.txt`, 5 plans) :
  * avec 11 points BF et seulement 4 HF, ρ̂ vaut 1,89–1,92 sur 3 plans et 0,86–0,90 sur les 2 autres (vrai ρ = 2) ; la
    couverture à 95 % tombe à 0,03–0,23 ;
  * avec 5 points HF, ρ̂ = 2,00 et la couverture vaut 0,90–0,94. L'exemple `example/cokriging/main.py` utilise ce plan.

---

## 6. Les données de l'infilling actuel (NN-MF-EGO)

Figure : [`infill.html`](figures/infill.html). Médianes sur 10 graines, à coût égal. Colonnes : points BF / HF, RMSE
relative sur tout le domaine, RMSE sur les 10 % meilleurs points, NLPD, couverture 95 %, écart-type prédit aux points HF
observés (/ std(y)).

**Hartmann, coût 280 (plan imbriqué 80 BF / 20 HF)**

| Données + modèle | BF / HF | RMSE | RMSE optimum | NLPD | Couv. 95 % | σ aux points HF |
|---|---|---|---|---|---|---|
| LHS imbriqué + CoKriging (MAP) | 80 / 20 | 0,84 | 2,32 | 1,37 | 0,86 | — |
| infill + modèle récursif (MAP) | 21 / 26 | 0,82 | 2,00 | 0,57 | 0,99 | 1,41 % |
| infill + conditionnement exact, mêmes hyperparamètres | 21 / 26 | 0,82 | 2,00 | 0,57 | 0,99 | 0,50 % |
| infill + BF aux points HF + CoKriging (coût 307) | 47 / 26 | 0,87 | 1,87 | 0,56 | 0,99 | 0,46 % |

**Hartmann, coût 560 (160 BF / 40 HF)**

| Données + modèle | BF / HF | RMSE | RMSE optimum | NLPD | Couv. 95 % | σ aux points HF |
|---|---|---|---|---|---|---|
| LHS imbriqué + CoKriging (MAP) | 160 / 40 | 0,76 | 1,94 | 0,60 | 0,87 | — |
| infill + modèle récursif (MAP) | 25 / 54 | 0,66 | 1,78 | 0,30 | 0,98 | 5,17 % |
| infill + conditionnement exact | 25 / 54 | 0,66 | 1,78 | 0,30 | 0,98 | 0,34 % |
| infill + BF aux points HF + CoKriging (coût 618) | 79 / 54 | 0,67 | 1,73 | 0,29 | 0,98 | 0,33 % |

**Airfoil, coût 140 (40 BF / 10 HF)**

| Données + modèle | BF / HF | RMSE | RMSE optimum | NLPD | Couv. 95 % | σ aux points HF |
|---|---|---|---|---|---|---|
| LHS imbriqué + CoKriging (MAP) | 40 / 10 | 0,32 | 0,18 | 3,43 | 0,85 | — |
| infill + modèle récursif (MAP) | 40 / 10 | 0,42 | 0,12 | 3,96 | 0,77 | 3,25 % |
| infill + conditionnement exact | 40 / 10 | 0,38 | 0,11 | 3,65 | 0,82 | 0,04 % |
| infill + BF aux points HF + CoKriging (coût 152) | 50 / 10 | 0,39 | 0,11 | 4,32 | 0,71 | 0,03 % |

**Airfoil, coût 280 (80 BF / 20 HF)**

| Données + modèle | BF / HF | RMSE | RMSE optimum | NLPD | Couv. 95 % | σ aux points HF |
|---|---|---|---|---|---|---|
| LHS imbriqué + CoKriging (MAP) | 80 / 20 | 0,26 | 0,12 | 3,18 | 0,90 | — |
| infill + modèle récursif (MAP) | 72 / 21 | 0,32 | 0,09 | 4,17 | 0,75 | 0,18 % |
| infill + conditionnement exact | 72 / 21 | 0,32 | 0,08 | 4,14 | 0,73 | 0,02 % |
| infill + BF aux points HF + CoKriging (coût 303) | 92 / 21 | 0,34 | 0,09 | 3,90 | 0,73 | 0,02 % |

Graines où le LHS imbriqué donne une meilleure RMSE globale que l'infilling : Hartmann 4/10 et 2/10 ; airfoil 6/10 et 7/10.

**Lecture.**

* **Airfoil : l'idée de départ se vérifie, en partie.** À coût égal, le surrogate de l'infilling est moins bon sur le
  domaine (+ 25 à 30 % de RMSE), meilleur près de l'optimum (0,12 contre 0,18), et trop sûr de lui (couverture 0,75–0,77).
  C'est le comportement attendu d'un plan construit pour l'optimisation : les points se concentrent là où le mérite est
  grand.
* **Hartmann : ce n'est pas le cas.** L'EGO met presque tout le budget en HF (26 et 54 points HF, contre 20 et 40), parce
  que la BF de Hartmann aide peu (§ 5). De plus, les variations de Hartmann sont concentrées dans ses puits, que l'EGO
  explore. Résultat : une meilleure RMSE globale, mais des intervalles trop larges (couverture 0,98–0,99).
* **L'erreur de variance du modèle récursif non imbriqué est réelle mais petite.** Aux points HF observés, il garde un
  écart-type de 0,2 à 5 % de std(y), entièrement dû au terme ρ²σ²_BF(x) : le GP BF ne « sait » pas que la valeur HF est
  connue. Le conditionnement exact le ramène au niveau du bruit. Sur le domaine, à hyperparamètres égaux, l'écart est nul
  sur Hartmann et modeste sur l'airfoil à coût 140 (RMSE 0,42 → 0,38, NLPD 3,96 → 3,65, couverture 0,77 → 0,82).
* **Compléter la BF aux points HF n'aide pas ici.**
  * La RMSE est équivalente ou un peu moins bonne (Hartmann à coût 280 : 0,87 contre 0,82), et le calibrage n'est pas
    meilleur.
  * L'expérience réestime les hyperparamètres sur plus de données ; elle ne teste donc pas directement [LG] Prop. 3.1.
  * Elle reste cohérente avec cette proposition, selon laquelle la BF n'apporte de l'information qu'*entre* les points HF.
* **Le mauvais calibrage de l'infilling vient surtout des hyperparamètres**, estimés sur un plan concentré, et non de la
  formule. Les deux directions apparaissent (trop large sur Hartmann, trop étroit sur l'airfoil).
* **ρ sur les données de l'EGO** (médianes) : 0,98–0,99 sur l'airfoil, mais 0,12 (coût 280) et 0,18 (coût 560) sur
  Hartmann.
  * Sur Hartmann, l'EGO a peu de points BF (21–25) et le modèle récursif utilise des prédictions BF aux points HF ; ρ̂ y
    est donc loin de 1.
  * Le biais vers 1 du § 7 n'est pas visible sur ces données.

---

## 7. Le biais de ρ dans mfego (exploration, cœur non modifié)

Figure : [`rho_bias.html`](figures/rho_bias.html) · script `explore_rho_centering.py` · `results/rho_centering.csv`.

**Le mécanisme.** Dans `GaussianProcess.fit`, le résidu du niveau l est normalisé avec la moyenne de y − ρ_init f_prev,
où ρ_init = 1. La constante du résidu est donc fixée pour ρ = 1, puis ρ est profilé avec cette constante, avec une seule
passe de correction. Quand le niveau inférieur a une moyenne loin de 0 (Hartmann : BF ≈ −2,6), tout ρ ≠ 1 décale le résidu
de (ρ − 1) × moyenne(f_prev). La petite variance de biais t2 ne peut pas absorber ce décalage, et ρ̂ est ramené vers 1.

**Le correctif testé** (`CenteredGP`, dans le script seulement) : centrer aussi le niveau inférieur. Le résidu devient
(y − ȳ) − ρ (f − f̄) : sa constante suit ρ, comme dans emukit avec des niveaux centrés.

Médianes sur 10 graines, plans imbriqués : **ρ · RMSE relative · NLPD**.

| Hartmann 6D | n_HF = 10 | n_HF = 20 | n_HF = 40 |
|---|---|---|---|
| mfego actuel | 0,97 · 0,95 · 2,88 | 0,98 · 0,84 · 1,37 | 0,96 · 0,76 · 0,60 |
| niveau inférieur centré, ρ ∈ [−5, 5] | 5,00 · 0,86 · 1,05 | 5,00 · 0,84 · 0,78 | 5,00 · 0,67 · 0,22 |
| niveau inférieur centré, ρ ∈ [−50, 50] | 23,05 · 1,39 · 1,09 | 6,38 · 0,95 · 1,04 | 8,45 · 0,74 · 0,20 |
| GPy/emukit MAP joint (ρ non borné) | 10,97 · 0,95 · 1,24 | 6,40 · 0,99 · 1,31 | 7,41 · 0,76 · 0,24 |

| Airfoil | n_HF = 5 | n_HF = 10 | n_HF = 20 |
|---|---|---|---|
| mfego actuel | 0,99 · 0,39 · 3,46 | 0,98 · 0,32 · 3,43 | 0,95 · 0,26 · 3,18 |
| niveau inférieur centré (ρ jamais en butée : mêmes valeurs avec ±5 et ±50) | 0,90 · 0,39 · 3,53 | 0,85 · 0,30 · 3,39 | 0,76 · 0,23 · 3,03 |
| GPy/emukit MAP joint | 0,80 · 0,37 · 3,36 | 0,81 · 0,25 · 3,07 | 0,64 · 0,21 · 3,19 |

Graines où la variante centrée améliore la RMSE / la NLPD par rapport à mfego actuel :

| | Airfoil n_HF = 5 / 10 / 20 | Hartmann n_HF = 10 / 20 / 40 |
|---|---|---|
| centré, ρ ∈ [−5, 5] | RMSE 8, 7, 9 · NLPD 5, 5, 8 | RMSE 8, 7, 9 · NLPD 9, 6, 9 |
| centré, ρ ∈ [−50, 50] | identique | RMSE 3, 3, 6 · NLPD 8, 5, 8 |

Sur Hartmann, le ρ centré est à la borne 5 dans 100 %, 80 % et 90 % des ajustements (n_HF = 10, 20, 40). Il ne l'est
jamais sur l'airfoil.

**Lecture.**

* **Airfoil :** le centrage corrige un vrai biais. ρ̂ se rapproche de l'estimation jointe d'emukit, n'est jamais en butée,
  et la RMSE s'améliore sur 7 à 9 graines sur 10.
* **Hartmann :** le gain de la variante bornée vient en partie de la borne. Libre, ρ̂ monte à 6–23, comme emukit. La NLPD
  reste meilleure, mais la RMSE se dégrade (1,39 contre 0,95 à n_HF = 10 ; 3 graines sur 10 seulement en mieux).
  * Un ρ constant ne décrit pas la relation très non linéaire entre U₁ et Hartmann (§ 5).
  * Le ρ « non biaisé » est grand, et il amplifie les erreurs du GP BF.
  * Sur ce cas, le ρ ≈ 1 de mfego est biaisé, mais il joue de fait le rôle d'une régularisation.
* **Le mécanisme est commun à `MultifidelityModel`**, qui partage `GaussianProcess.fit`. Sur les données non imbriquées de
  l'EGO de Hartmann, ρ̂ vaut pourtant 0,12–0,18 (§ 6) : l'effet dépend des données.
* **Correctif proposé (≈ 5 lignes dans `GaussianProcess._normalize` et `fit`, comme `CenteredGP`), à valider avec toi
  avant de l'appliquer.**
  * Il change le comportement de tous les modèles, donc les études déjà faites.
  * Il doit aller avec une décision sur la régularisation de ρ : borne plus serrée, ou prior normale centrée sur 1. La
    prior garde une forme fermée ; tu l'avais écartée au départ, mais ces résultats la rendent pertinente.

---

## 8. Limites

* **Plans imbriqués seulement**, comme demandé, pour l'*estimation*. Le *conditionnement* exact fonctionne sur n'importe quel
  plan, si les hyperparamètres sont donnés : c'est `exact_conditioning` dans le benchmark, avec ceux du modèle récursif.
  Une méthode `CoKriging.from_multifidelity(model)` en ferait une fonction du cœur, en ~10 lignes. Non faite, pour rester
  dans le périmètre.
* **Coût O((Σ n_l)³)** : il est négligeable ici (200 points, 4 s d'ajustement, dont la Cholesky de V) mais limite l'usage
  à quelques centaines de points.
* **ρ constant** et un noyau stationnaire, comme dans tout mfego. Sur Hartmann, la relation BF/HF est trop non linéaire
  pour un ρ constant. La NLPD reste mauvaise dans les puits.
* **Deux problèmes, 10 graines** ; l'airfoil est un cas NeuralFoil/NeuralFoil, plus proche entre niveaux qu'un vrai
  couple NeuralFoil/XFOIL.
* L'incertitude sur les hyperparamètres n'est pas propagée (§ 5).
* **Niveaux inférieurs bruités :** l'estimation par niveau est biaisée (§ 2.1). Il faudrait alors une estimation jointe
  sur V.

## 9. Suites proposées

1. **Corriger le biais de ρ** (§ 7) dans `GaussianProcess`, avec une régularisation de ρ (prior normale ou borne), puis
   relancer les deux benchmarks et une étude existante.
2. Si la vraie variance des données de l'EGO est utile en optimisation, ajouter `CoKriging.from_multifidelity(model)` et
   l'utiliser dans le mérite. Le gain mesuré ici sur le domaine est faible ; il serait surtout local, près des points HF.
3. Valider sur un couple de niveaux plus éloignés, NeuralFoil/XFOIL, à travers le pipeline existant.

---

## Reproduction

Depuis la racine du dépôt, avec l'environnement `.venv-benchmark` (GPy, emukit, NeuralFoil ; environ 2 min 30 sur 32 cœurs) :

```
.venv-benchmark\Scripts\python benchmarks\cokriging\run_cokriging_benchmark.py    # --quick : 3 graines
.venv-benchmark\Scripts\python benchmarks\cokriging\explore_rho_centering.py
.venv-benchmark\Scripts\python benchmarks\cokriging\forrester_checks.py
.venv-benchmark\Scripts\python benchmarks\cokriging\make_figures.py
```

| Fichier | Rôle |
|---|---|
| `airfoil_problem.py` | Problème NACA 4 chiffres, NeuralFoil xxsmall / xxxlarge |
| `run_cokriging_benchmark.py` | Plans imbriqués (7 modèles), exactitude contre emukit, données de l'infilling |
| `explore_rho_centering.py` | Biais de ρ (mfego actuel, niveau inférieur centré avec ρ borné à ±5 ou ±50, emukit) |
| `forrester_checks.py` | Forrester : très peu de points HF (§ 5), BF bruitée (§ 2.1) |
| `make_figures.py` | Les 5 figures de `figures/` |
| `../bench_lib.py` | `nested_doe`, `CoKrigingSurrogate`, `EmukitAR1Surrogate` (+ `to_cokriging`), `run_mfego(return_model=True)` |
| `results/` | `nested.csv`, `infill.csv`, `rho_centering.csv`, `forrester_checks.txt`, `test_set_*.npz`, `summary.txt` |

## Figures

| Figure | Contenu |
|---|---|
| [`nested_accuracy.html`](figures/nested_accuracy.html) | RMSE, NLPD et couverture contre n_HF, 6 modèles, 2 problèmes |
| [`calibration.html`](figures/calibration.html) | Couverture contre probabilité nominale ; histogramme de z contre N(0, 1) |
| [`verification.html`](figures/verification.html) | Équivalence CoKriging / récursif ; exactitude contre emukit |
| [`infill.html`](figures/infill.html) | Données de l'infilling contre LHS imbriqué ; écart-type aux points HF |
| [`rho_bias.html`](figures/rho_bias.html) | ρ estimé, RMSE et NLPD : mfego actuel, centré (ρ borné à ±5 ou ±50), emukit |

## Références

* **[LG]** L. Le Gratiet, *Multi-fidelity Gaussian process regression for computer experiments*, thèse, Université Paris
  Diderot (2013) — éq. 3.1, 3.8–3.11, p. 91, Prop. 3.1.
* **[S]** M. Sacher et al., *A Non-Nested Infilling Strategy for Multi-Fidelity based Efficient Global Optimization*,
  Int. J. Uncertainty Quantification (2021).
* M. C. Kennedy, A. O'Hagan, *Predicting the output from a complex computer code when fast approximations are available*,
  Biometrika 87 (2000).
* emukit (`GPyLinearMultiFidelityModel`, `LinearMultiFidelityKernel`) et GPy 1.14.
