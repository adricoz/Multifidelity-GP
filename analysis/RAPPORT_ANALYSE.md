# Rapport d'analyse — `mfego` (NN-MF-EGO, Sacher et al. 2021)

Branche : `fix/analyse-theorique-mfego` · Base : `main` (commit `adc0b7f`) · Code initial préservé dans
`legacy/legacy_mfego_initial/`.

Références utilisées pour tout le rapport :

* **[S]** M. Sacher, O. Le Maître, R. Duvigneau, F. Hauville, M. Durand, C. Lothodé, *A Non-Nested Infilling Strategy for
  Multi-Fidelity based Efficient Global Optimization*, Int. J. Uncertainty Quantification (2021) — `references/multifidelity_opt.pdf`.
  Les numéros d'équation « Éq. n » renvoient à cet article.
* **[LG]** L. Le Gratiet, *Multi-fidelity Gaussian process regression for computer experiments*, thèse, Univ. Paris-Diderot (2013) —
  `references/Multi_fidelity_Gaussian_process_regressi.pdf`.
* **[RW]** C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press (2006) (pratique standard
  citée pour la paramétrisation log et le gradient de la vraisemblance).

Chaque modification du code porte un commentaire `# [FIX-<ID>]` (en anglais) qui renvoie à l'identifiant de ce rapport :
`grep -rn "\[FIX-" mfego example` les liste toutes (115 occurrences dans le code Python).

---

## 0. Sécurité (hors périmètre, à traiter en premier)

Un **jeton d'accès personnel GitHub (PAT, `ghp_…`) est écrit en clair dans l'URL du remote `origin`** du dépôt
(`.git/config` : `https://ghp_…@github.com/adricoz/Multifidelity-GP.git`). Il n'apparaît dans **aucun fichier suivi** par git
(vérifié par recherche sur tout l'arbre), il n'a donc pas été publié par les commits. Il reste lisible par tout programme ou
personne ayant accès au poste. **Recommandation** : révoquer ce jeton sur GitHub (*Settings → Developer settings → Personal access
tokens*), puis remplacer l'URL par `https://github.com/adricoz/Multifidelity-GP.git` et laisser Git Credential Manager gérer
l'authentification.

---

## 1. Résumé exécutif

L'implémentation reprend fidèlement la **structure** de la méthode de Sacher (GP récursifs par niveau, résidus non imbriqués
Éq. 18, variance récursive Éq. 12, mérite Éq. 24 avec ratio de coûts et produit des ρ²). En revanche, l'audit a relevé :

1. **Des écarts théoriques** : ρ figé à 1 sans possibilité de l'estimer alors que l'article l'estime (Éq. 15) — l'estimation
   est désormais disponible (ρ profilé, mode hybride) et une étude dédiée (§10) a conduit à garder ρ = 1 par défaut ; facteur AEI (Éq. 20) absent ; valeur de
   référence de l'EI différente de l'Éq. 19 ; réduction de variance calculée avec la variance bruitée au lieu de la variance
   latente (Éq. 22, 28-29). Le legacy non imbriqué estimait ρ et contenait l'AEI : ce sont des **régressions**.
2. **Des bornes d'hyperparamètres mal posées** : bornes absolues sur des sorties non normalisées → bornes actives sur tous les
   cas sauvegardés (θ₂ bloqué à 1, θ₁ en butée, bruit en butée), et un modèle qui dépend des **unités** de y
   (erreur relative de 1 à 4 écarts-types quand y est multiplié par 10⁻³ ou 10³ ; ≤ 2·10⁻⁵ après correction).
3. **Une reconstruction des surfaces de réponse incorrecte** : le JSON associe les données de l'itération n+1 aux
   hyperparamètres de l'itération n → la surface tracée ne correspond à **aucun modèle entraîné** (écart de 1,9 écart-type de f
   mesuré) ; de plus le scatter 2D trace des **lignes** au lieu des colonnes.
4. **Des simulateurs avec des erreurs** : constantes de Hartmann 6D fausses (minimum réel −3,3975 au lieu de −3,32237, donc
   cible fausse) ; mapping niveau → modèle NeuralFoil régressé ; pénalité 1e6 injectée dans le GP.
5. **Un README incohérent avec l'environnement** : « Python 3.8+ » faux ; le code **ne s'importait pas** sous Python 3.13
   (`TypeError`) ni sous Python 3.14 sans `traitlets` (`ModuleNotFoundError`), à cause d'une dépendance cachée et mal utilisée ;
   versions minimales de NeuralFoil fausses ; API décrite inexacte.
6. **Des performances numériques faibles** : matrice de covariance en double boucle Python, gradient par différences finies
   en espace linéaire, inverse explicite → ajustement d'un GP 136 à 400× plus lent, itération EGO 22× plus lente qu'après correction.
7. **Aucune fuite mémoire** (mesuré par `tracemalloc`), mais des **défauts de robustesse** : coût du plan d'expériences jamais
   compté, NaN non filtrés, runs non reproductibles.
8. **Un point ouvert** (non introduit par les corrections) : sur le protocole de [S] §4.1, le mérite choisit très majoritairement
   le niveau le moins cher, alors que [S] rapporte une préférence pour le niveau intermédiaire (§10.5).

Les points 1 à 7 sont corrigés sur la branche (ρ : option + étude), prouvés par **63 tests pytest** et par **7 scripts d'analyse avant/après**
(`analysis/scripts/`) dont les figures interactives sont dans `analysis/figures/`.

---

## 2. Méthode

* Lecture complète du dépôt : `mfego/` (source, `main.py`, artefacts), `example/` (Hartmann, hydrofoil), `legacy/`
  (versions imbriquée et non imbriquée), `sandbox/` (notebook mono-fidélité), `charts/`, `README.md`, `.vscode/`, `.gitignore`,
  logs et JSON sauvegardés.
* Confrontation équation par équation avec [S] (§2-3, Algorithme 1, §4.1) et [LG] (§1.3, §4.2-4.4, annexe B).
* Mesures « avant/après » : chaque script de `analysis/scripts/` importe successivement le code initial
  (`legacy/legacy_mfego_initial`, avec un *shim* `traitlets` pour qu'il s'importe) et le code corrigé, sur les mêmes données.
* Environnement : Windows 11, Python 3.13.5 (Anaconda : numpy 2.4.6, scipy 1.15.3, matplotlib 3.11.2, plotly 5.24.1,
  pytest 8.3.4, neuralfoil 0.3.3, aerosandbox 4.2.10) et Python 3.14.7 (numpy 2.5.2, scipy 1.18.1, sans traitlets/pytest/plotly).
* Choix validés avec l'utilisateur : hyperparamètres par **maximum de vraisemblance uniquement** (pas de LOO, voir §11),
  normalisation de y, plotly à côté de matplotlib, régénération de tous les artefacts, legacy/sandbox non modifiés.

Reproduire toutes les mesures (depuis la racine du dépôt) :

```bash
python -m pytest                                   # 63 tests (dont 1 « slow » NeuralFoil)
python analysis/scripts/check_env.py               # axe 5
python analysis/scripts/check_theory.py            # axe 1
python analysis/scripts/check_hyperparams.py       # axe 2
python analysis/scripts/check_simulators.py        # axe 3 (≈ 2 min : carte NeuralFoil 21×21)
python analysis/scripts/check_reconstruction.py    # axe 4
python analysis/scripts/profile_perf.py            # axes 6-7 (≈ 10 min : le code initial est lent)
python analysis/scripts/rho_study.py               # étude de ρ (parallèle, plusieurs minutes)
```

---

## 3. Synthèse priorisée

| ID | Gravité | Constat (avant) | État |
|---|---|---|---|
| R6 | **Critique** | Le paquet ne s'importe pas (traitlets) sous Python 3.13 / 3.14 sans traitlets | Corrigé |
| T1 | **Critique** | ρ figé à 1 sans possibilité de l'estimer (Éq. 15) ; Forrester RMSE 4,36 au lieu de 0,19 avec ρ estimé | Option `estimate_rho` ajoutée (ρ profilé + mode hybride) ; défaut ρ = 1 conservé après étude (§10) |
| X2 | **Critique** | JSON = données n+1 + hyperparamètres n ; surface tracée ≠ modèle entraîné | Corrigé |
| E1 | **Critique** | Constantes Hartmann fausses ; minimum −3,3975 ; cible −3,32237 fausse | Corrigé |
| N7/N2 | **Majeur** | Bornes absolues, y non normalisé : bornes actives partout, modèle dépendant des unités | Corrigé |
| R4 | **Majeur** | NaN/pénalités 1e6 injectés dans le GP | Corrigé |
| R5 | **Majeur** | Coût du DOE jamais compté (64 au lieu de 114 ; 40 au lieu de 48) | Corrigé |
| T2 | Majeur | Facteur AEI (Éq. 20) absent (régression du legacy) | Corrigé |
| T3 | Majeur | f_best = min y_HF au lieu de f̂⁽ᴸ⁾(x_best) (Éq. 19) | Corrigé (+ T3b) |
| X3 | Majeur | Scatter 2D sur les lignes ; grille [0,1] codée en dur ; points BF non tracés | Corrigé |
| E2 | Majeur | Hydrofoil : mapping niveau→modèle régressé, pénalité 1e6, L codé en dur | Corrigé |
| N1/N3/N4 | Majeur | Double boucle, différences finies, inverse explicite : ajustement 136-400× plus lent | Corrigé |
| D1 | Majeur | README : versions, dépendances, API, lancement faux ou absents | Corrigé |
| T4 | Mineur → Majeur | Δσ² avec variance bruitée (+167 % d'erreur si bruit = variance) | Corrigé |
| R1 | Mineur | Runs non reproductibles (RNG global, DE sans graine) | Corrigé |
| R7 | Mineur | Point déjà évalué → itération perdue ; avec une recherche graine et un point en échec, blocage à chaque itération | Corrigé |
| R2 | Mineur | `ValueError` inaccessible ; échec de Cholesky non expliqué | Corrigé |
| R3 | Mineur | `logging.basicConfig` à l'import d'une bibliothèque | Corrigé |
| N5/N6/N8 | Mineur | Objectif discontinu (1e10), L² prédictions par point, pas de démarrage à chaud | Corrigé |
| X1 | Amélioration | Pas d'export utilisable du surrogate | Ajouté |
| D2 | Mineur | Diagramme `.mmd` décrivant des classes inexistantes | Refait |
| — | Info | Pas de LOO dans le code (MLE conforme à [S]) | Documenté (§11) |

---

## 4. Axe 1 — Cohérence avec la théorie

### 4.1 Tableau équation par équation

| Éq. [S] | Contenu | Code initial | Code corrigé |
|---|---|---|---|
| 2 | Noyau SE + constante : θ₁ exp(−Σ(xᵢ−x'ᵢ)²/2lᵢ²) + θ₂ | Conforme (`kernels.py`) | Conservé (vectorisé, N1) |
| 4-5 | Moyenne kᵀK⁻¹Y, variance κ + σ² − kᵀK⁻¹k (bruit inclus) | Conforme (inverse explicite) | Conforme (Cholesky, N4) |
| 9 | Y⁽ˡ⁾ = ρ₍ₗ₋₁₎Y⁽ˡ⁻¹⁾ + δY⁽ˡ⁾, Y⁽⁰⁾ = 0 | Structure conforme | Conservée |
| 11-12 | Moyenne et variance récursives | Conformes | Conservées (vectorisées) |
| 15-16 | Vraisemblance de niveau l en (ρ, θ, σ²) | **ρ absent (=1)** | ρ profilé avec `estimate_rho=True` (T1) ; ρ = 1 par défaut après étude (§10) |
| 18 | Résidu non imbriqué y⁽ˡ⁾ − ρ f̂⁽ˡ⁻¹⁾(X⁽ˡ⁾) | Conforme (ρ = 1) | Conforme |
| 19 | EI avec f̂⁽ᴸ⁾(x_best), x_best « effective best » sur ∪X⁽ˡ⁾ | **min des y HF observés** | Conforme (T3) |
| 20 | AEI : EI·(1 − σ_ε / √(σ̂² + σ²_ε)) | **absent** | Conforme (T2) |
| 21-22, 28-29 | Réduction de variance en x : s⁴/(s² + σ²_ε) (variance latente) | variance **bruitée** | Conforme (T4) |
| 23-24 | Mérite AEI·W_L/W_l·max(0, R²ₗΔσ²/σ²_L), R²ₗ = Π ρ² | Conforme (R² ≡ 1 car ρ = 1) | Conforme |
| Algo 1 | Ré-estimation (ρ, θ, σ²) à chaque itération | θ, σ² seulement | (ρ, θ, σ²) si `estimate_rho=True`, sinon (θ, σ²) |
| §2.1.1 | Hyperparamètres par CMA-ES (global) | L-BFGS-B 3 redémarrages | L-BFGS-B (gradient exact, log, démarrage à chaud) — écart assumé, voir §11 |

### 4.2 Détail des écarts

**T1 — ρ figé à 1.** `surrogate_models.py:130` (initial) réécrivait `rho = 1.0` à chaque `fit()` pour tous les niveaux l ≥ 2.
Or l'Éq. 15 de [S] fait de ρ₍ₗ₋₁₎ un argument de la vraisemblance du niveau l, et l'Algorithme 1 l'estime à chaque itération.
Le niveau 1 n'a pas de ρ (Y⁽⁰⁾ = 0) : l'initialisation `rhos = [1.0, …]` dans `__init__` est donc légitime et a été conservée.
Fixer ρ = 1 pour l ≥ 2 revient à un modèle de **correction additive** : δ⁽ˡ⁾ doit alors absorber (ρ_vrai − 1)·f⁽ˡ⁻¹⁾, et le
facteur R²ₗ du mérite vaut toujours 1, ce qui supprime le mécanisme de robustesse de [S] vis-à-vis des niveaux mal corrélés.
Mesure (`check_theory.py`) sur Forrester (Éq. 17, ρ vrai = 2, 10 BF / 6 HF) : profil de vraisemblance minimal en **ρ = 2,0**
(NLL 1,61) contre NLL 10,59 en ρ = 1 ; RMSE du surrogate HF : code initial **4,36**, code corrigé avec ρ = 1 (défaut) **1,81**
(gain dû à la normalisation et aux bornes seules), code corrigé avec `estimate_rho=True` **0,19** (ρ̂ = 1,96). Sur Hartmann 6D
(20 BF / 10 HF, δ = 0,05) les trois variantes sont équivalentes (RMSE 0,45-0,47) : ρ̂ = 1,18 et la limite est le nombre de
points. Voir §10 pour l'étude complète de la démarche « ρ fixé » et le choix du défaut.

**T2 — Facteur AEI absent.** L'Éq. 20 multiplie l'EI par (1 − σ_ε,L / √(σ̂²_L + σ²_ε,L)). Le code initial ne le contenait pas,
alors que `legacy/legacy_non_nested/non_nested_mf_optimizer.py:188-219` l'implémentait. Son poids (`check_theory.py`) :
0,999 pour un bruit relatif de 10⁻⁶, 0,90 pour 10⁻², 0,71 pour 10⁻¹, 0,42 pour 1. Il est donc négligeable sur les cas
déterministes mais essentiel dès qu'un niveau est bruité ([S] §4.3).

**T3 — Valeur de référence de l'EI.** L'Éq. 19 utilise f̂⁽ᴸ⁾(x_best) avec x_best ∈ ∪ₗX⁽ˡ⁾ la « current effective best
solution » de Huang et al. (réf. [25] de [S]) : x_best = argmin [f̂⁽ᴸ⁾ + σ̂⁽ᴸ⁾] sur tous les points d'apprentissage. Le code
initial utilisait le minimum des observations HF. Exemple (Forrester, DOE 10/4) : min y_HF = −0,458 contre
f̂⁽ᴸ⁾(x_best) = −0,905 (x_best = 0,135, **un point BF**) : les points basse fidélité informent bien la référence.
*Effet de bord découvert et traité (T3b)* : quand le surrogate est sûr de son optimum (souvent un point BF), le mérite s'annule
partout et le repli initial (« point aléatoire au niveau L ») gaspillait des évaluations HF. Le repli évalue désormais **d'abord
le niveau L au point x_best** (confirmation de l'optimum), puis seulement en aléatoire ; une option `stop_on_convergence` arrête
la boucle (critère d'arrêt sur l'EI suggéré en [S] §3.4).

**T4 — Réduction de variance.** Par le lemme d'inversion par blocs (Éq. 28-29), ajouter une observation bruitée en x̃ = x réduit
la variance prédictive de (κ − kᵀK⁻¹k)²/(κ + σ²_ε − kᵀK⁻¹k) = s⁴/(s² + σ²_ε), où s² est la variance **latente**. Le code initial
utilisait la variance prédictive (qui contient déjà σ²_ε), soit (s²+σ²_ε)²/(s²+2σ²_ε). Erreur relative (`check_theory.py`) :
10⁻⁶ pour un bruit relatif de 10⁻⁶, 1 % pour 10⁻², 11 % pour 10⁻¹, **+167 %** pour 1. Le test
`test_variance_reduction_matches_block_inverse_update` vérifie la nouvelle formule contre une vraie mise à jour du GP.

**T5 (information) — Approximation non imbriquée.** L'Éq. 18 remplace les observations du niveau inférieur par la moyenne
prédite f̂⁽ˡ⁻¹⁾(X⁽ˡ⁾). La formulation exacte de [LG] (annexe B.1) propage aussi la covariance ρ²s²_{Z_{l−1}}(Dₗ, Dₗ) dans la
matrice du niveau l. Le code suit [S] (choix de l'article, qui lisse le bruit des niveaux inférieurs, [S] §4.3) : conservé et
documenté.

---

## 5. Axe 2 — Bornes de recherche des hyperparamètres

**Avant** (`surrogate_models.py:67`) : l ∈ [0,01 ; 5], θ₁ ∈ [10⁻³ ; 50], θ₂ ∈ [10⁻⁶ ; 1], σ²_ε ∈ [10⁻⁸ ; 10⁻⁵], **absolues**,
sur des sorties **non normalisées**, optimisées en **espace linéaire** avec un gradient par **différences finies** (pas absolu
≈ 10⁻⁸, du même ordre que la borne basse du bruit : le gradient en σ²_ε n'avait pas de sens).

**Constats sur les JSON sauvegardés par le code initial** :
`mfego` (Forrester) : θ₂ = 1,0 (borne haute) aux deux niveaux, θ₁ ≈ 48 (près de 50), σ²_ε = 10⁻⁸ (borne basse) ;
`hydrofoil` : l ≈ 4,92-4,96 (près de 5), Var(C_d) ≈ 10⁻⁶ alors que θ₁ ≥ 10⁻³ ;
`hartmann` : l = 5 et 0,01, θ₁ = 50, θ₂ = 1 (fichier par ailleurs contaminé, §6).

**Mesures avant/après** (`check_hyperparams.py`, mêmes données) :

| Données | Code | Hyperparamètres en butée | Erreur d'échelle (y×10⁻³ / 10³) |
|---|---|---|---|
| Forrester (JSON `main`, 14 BF / 10 HF) | initial | niv. 1 : bruit bas · niv. 2 : **θ₂ haut**, bruit bas | **1,07 / 1,96** |
| | corrigé (ρ = 1, défaut) | bruit bas seulement (données déterministes : attendu) | **1·10⁻⁶** |
| | corrigé (`estimate_rho=True`) | niv. 2 : θ₁ haut (résidu exactement linéaire, cf. ci-dessous) | **2·10⁻⁷** |
| Hydrofoil (JSON `main`, 25 BF / 23 HF) | initial | niv. 2 : **l₁ haut, θ₁ bas, θ₂ bas**, bruit bas | **0,23 / 2,46** |
| | corrigé | bruit bas seulement (données déterministes : attendu) | **2·10⁻⁵** |
| Hartmann 6D (20 BF / 15 HF) | initial | niv. 1 : **les 6 lᵢ hauts, θ₁ bas, θ₂ haut, bruit haut** | **2,54 / 4,13** |
| | corrigé | quelques lᵢ hauts (dimensions jugées non pertinentes avec 15 points : comportement ARD normal) | **10⁻¹³** |

L'erreur d'échelle est max|μ(a·y)/a − μ(y)| / std(μ) : un modèle bien posé doit être équivariant (≈ 0). Avant correction,
**changer d'unités (par ex. C_d en « counts ») changeait complètement le surrogate.**

**Correctifs (N2, N7)** : sorties normalisées (centrage et réduction calculés une fois par ajustement, sur le résidu à ρ_init puis,
si ρ̂ s'en écarte, sur le résidu à ρ̂ — seconde passe à démarrage à chaud) ; bornes exprimées dans l'espace normalisé
(l ∈ [10⁻² ; 10], θ₁ ∈ [10⁻⁴ ; 10²], θ₂ ∈ [10⁻⁸ ; 10²], σ²_ε ∈ [10⁻⁸ ; 10⁻²]) et explorées **en log** ; le noyau de [S] Éq. 2
est conservé (θ₂ absorbe toujours la moyenne constante résiduelle). Le jacobien n·log s de la normalisation est constant : l'optimum
de vraisemblance n'est pas modifié.

Précision sur des points de test (500 LHS) : Forrester RMSE/std 0,064 (initial) / 0,073 (corrigé, ρ = 1) / **0,0045** (corrigé,
ρ estimé = 2,004), couverture à 95 % 0,80 / 0,85 / 0,90 ; Hartmann 1,09 / 1,10, couverture 0,41 / 0,38 (limite des données). À ρ
égal, la correction des bornes ne change donc pas la précision sur ces jeux mais supprime la dépendance aux unités et les
butées ; le gain de précision sur Forrester vient de ρ.

**Bornes encore atteintes, et pourquoi c'est sain** : (i) la borne basse du bruit sur des fonctions déterministes (l'interpolation
est l'optimum, la borne joue le rôle de *nugget*) ; (ii) θ₁ haut quand le résidu est exactement linéaire (Forrester avec ρ̂ ≈ 2 :
δ = −20(x−1)) — c'est la limite dégénérée du noyau SE (l → ∞, θ₁ → ∞) et la prédiction reste exacte (RMSE 0,0045) ;
(iii) quelques lᵢ hauts en 6D avec peu de points (dimension ignorée).

**Limite restante (documentée)** : avec très peu de points HF (n ≤ d + 3 hyperparamètres), le MLE sur-ajuste et les intervalles
sont trop étroits (Forrester 4 points HF : couverture à 95 % de 27 %). C'est une limite connue du MLE ; le mode hybride de ρ
(§10) évite d'y ajouter un ρ̂ aberrant (ρ̂ = 0,60 au lieu de 2 avec 4 points).

---

## 6. Axe 3 — Bornes et cohérence des fonctions de simulation

**Forrester (`mfego/main.py`)** : domaine [0, 1] et Éq. 17 conformes (10(x−1) = 10(x−0,5)−5, et f₂ = 2f₁ − 20(x−1) est bien la
fonction HF de Forrester) ; cible −6,02074 en x ≈ 0,7572 correcte. Défaut : un niveau ≤ 0 renvoyait `None` (validation faite
après les branches) → corrigé (E3).

**Hartmann 6D (`example/hartmann_6d`)** — E1, critique. Les constantes différaient de [S] Éq. 30 et de la définition standard :
`A[3] = [17, 8, 0.05, 10, 14, 3.5]` au lieu de `[17, 8, 0.05, 10, 0.1, 14]`, et `P[0][3] = 1244` au lieu de `124`
(`check_simulators.py`) :

| | f(x*) standard | minimum de la fonction codée | argmin |
|---|---|---|---|
| initial | −3,39572 | **−3,39750** | (0,197 ; 0,151 ; 0,486 ; 0,279 ; 0,313 ; 0,656) |
| corrigé | −3,322368 | −3,322368 | (0,20169 ; 0,15001 ; 0,47687 ; 0,27533 ; 0,31165 ; 0,6573) |

La cible du tracé de convergence (−3,32237) était donc fausse pour la fonction optimisée. Les mêmes constantes fausses sont
présentes dans `legacy/*/Hartmann6d.py` et dans le notebook (non modifiés, signalé). La suite basse fidélité (Éq. 31-32 : u₀ = −5,
décalage δ/k) est conforme (écart 4·10⁻¹⁶). Corrélations mesurées entre niveaux (3 000 points LHS) : k = 1 : 0,90 / 0,83 / 0,66
pour δ = 0 / 0,05 / 0,1 ; k = 3 : 0,95 / 0,94 / 0,92 (cohérent avec la Fig. 3 de [S]). Autres défauts : la fonction était évaluée
**deux fois** par appel ; `L = 2` codé en dur dans le simulateur ; message d'erreur « 2 or 6 » alors que seul 6 est accepté.
Le `ego_backup.json` de Hartmann sur `main` était **contaminé** (d = 3, tous les y = 1e6, effectifs 10/12 incompatibles avec le
script [20, 10]) : il provenait d'un autre problème ; il a été régénéré.

**Hydrofoil (`example/hydrofoil_optim`)** — E2. Variables normalisées dans [0,1]², converties dans le simulateur en cambrure
2-9 % (position 30 %) et épaisseur 8-17 %. Mesures (`check_simulators.py`, grille 21×21 avec NeuralFoil) :

* **bracket [−5° ; 15°] pour C_l = 1 valide sur tout le domaine** (aucun changement de signe manquant, aucun échec) ;
* coût par évaluation : 0,063 s (niveau 1) contre 0,067 s (niveau 2) → la recherche de α(C_l = 1) avec `xxxlarge`, **commune aux
  deux niveaux** (choix voulu, qui permet aussi le test mono-fidélité L = 1), domine le coût : les coûts `[1, 1]` sont **cohérents** ;
* corrélation C_d niveau 1 / niveau 2 : 0,94 (écart moyen 7,6·10⁻⁴) ;
* **minimum de référence** de l'objectif HF sur les bornes réelles : **C_d = 0,0138978** (cambrure 4,51 %, épaisseur 8,0 % = borne
  basse). L'ancienne cible 0,0139 était une estimation arrondie (calculée à α = 5° avec `xxlarge`) ; elle est remplacée.
* La grille de référence `find_optimal_foil` couvre cambrure 4-9 % alors que le simulateur va de 2 % à 9 % : écart signalé, sans
  conséquence (l'optimum est à 4,5 %).

Défauts corrigés : mapping niveau → modèle `modelclasses[int(level/L)*len]` toujours égal à `xxsmall` pour tout niveau < L (le legacy
`custom_fluid_functions.py` faisait `modelclass[level-1]`, restauré) ; échec renvoyé comme **pénalité 1e6** (une seule valeur 1e6
au milieu de C_d ≈ 0,014 détruit le GP) → `NaN`, traité comme évaluation en échec (R4) ; `L` lu dans `self.num_levels`.

---

## 7. Axe 4 — Reconstruction des surfaces de réponse depuis le JSON

**Défaut X2 (critique).** `ask()` ajuste le modèle puis sauvegarde ; `tell()` ajoute le point évalué et **sauvegarde à nouveau
avec les hyperparamètres de l'ajustement précédent**. Le `ModelVisualizer` initial reconstruisait donc un modèle avec n+1 points
et les hyperparamètres de n : un modèle qui n'a jamais été entraîné. Le modèle en mémoire à la fin de `run()` était lui aussi
« en retard » d'un point.

Mesure (`check_reconstruction.py`, Forrester, 4 itérations) :

| Code | Points dans le JSON | Points du modèle entraîné | max |μ_reconstruit − μ_entraîné| / std(f) |
|---|---|---|---|
| initial | 12 BF / 6 HF | 12 / **5** | **1,89** |
| corrigé | 12 / 6 | 12 / 6 | **0** (identiques au bit près) |

**Correctifs** : ajustement et sauvegarde finale en fin de `run()` ; le JSON contient un instantané exact du modèle entraîné
(`"surrogate"` : données réellement utilisées, `fit_sizes`, hyperparamètres, normalisation, ρ) ; le visualiseur et
`load_surrogate` reconstruisent via `MultifidelityModel.from_dict` puis `GaussianProcess.condition`, **le même code que
l'ajustement** (une seule source de vérité). Les anciens JSON restent lisibles (reconstruction d'origine conservée en repli).

**Défauts de tracé (X3)** : `hf_points[param_x_idx]` sélectionnait la **ligne** 0 et la ligne 1 (3 valeurs tracées au lieu de 9
points, mesuré) → `[:, idx]` ; grille `linspace(0,1)` codée en dur → bornes du problème (sauvegardées dans le JSON) ; dimensions
libres fixées à 0,5 → meilleur point HF observé ; seuls les points HF étaient affichés → points de tous les niveaux et courbes de
chaque niveau en 1D ; `print` de débogage supprimés ; libellés cohérents en mono-fidélité (« Single fidelity »).

---

## 8. Axe 5 — README vs versions de Python et des bibliothèques

Mesures (`check_env.py`) :

| Interpréteur | Code initial | Code corrigé |
|---|---|---|
| Python 3.13 (traitlets 5.14.3) | **TypeError** : `traitlets.Tuple is not a generic class` | OK |
| Python 3.14 (sans traitlets) | **ModuleNotFoundError** : `traitlets` | OK |

Le code initial ne fonctionnait qu'avec **Python ≥ 3.14 et traitlets installé** (annotations évaluées paresseusement, PEP 649 —
les `__pycache__` en `cpython-314` le confirment), alors que le README annonçait « Python 3.8+ ».

| Élément du README initial | Réalité | Correction |
|---|---|---|
| Python 3.8+ | `tuple[...]` (PEP 585) impose ≥ 3.9 ; scipy 1.15, neuralfoil 0.3, aerosandbox 4.2 imposent ≥ 3.10 ; numpy 2.4 et matplotlib 3.11 installés imposent ≥ 3.11 | « Python 3.10+, testé en 3.13 » |
| numpy ≥ 1.20, scipy ≥ 1.7, matplotlib ≥ 3.4 | OK pour l'API utilisée (qmc depuis 1.7), DE vectorisée depuis scipy 1.9 | numpy ≥ 1.24, scipy ≥ 1.9, matplotlib ≥ 3.5 |
| neuralfoil ≥ 0.1.0 | `n_crit`, `xtr_upper`, `xtr_lower` et `xxxlarge` **absents de 0.1.10**, présents depuis **0.2.0** (vérifié dans les wheels) | neuralfoil ≥ 0.2.0 |
| aerosandbox ≥ 4.0.0 | neuralfoil 0.2.0 requiert aerosandbox ≥ 4.2.3 | aerosandbox ≥ 4.2.3 |
| (absent) traitlets | dépendance cachée et mal utilisée | supprimée (R6) |
| (absent) pytest, plotly | nécessaires aux tests / graphiques interactifs | ajoutés, `requirements.txt` créé |
| `matrices.generate_initial_design`, `get_cross_covariance_vector` | noms inexistants | corrigés |
| « fit optimise θ, ρ, σ_ε » | ρ n'était pas optimisé | maintenant vrai (et options documentées) |
| `x_next, l_next = ego.ask()` | `ask()` renvoie 3 valeurs → `ValueError` | corrigé |
| `from visualization import ModelVisualizer` | module `src.visualization` | corrigé |
| `def evaluate(...) -> float, dict:` | syntaxe invalide | corrigé |
| `charts/relations.png` | inexistant | diagrammes réels |
| instructions de lancement | absentes (`main.py` depuis `mfego/`, exemples en `python -m` depuis la racine) | ajoutées |

Remarque scipy : depuis 1.15, `LatinHypercube(seed=…)` est remplacé par `rng=` (SPEC 7) ; `seed=` reste accepté sans
avertissement en 1.15.3 et 1.18.1 (testé). À surveiller lors des prochaines versions.

---

## 9. Axes 6-7 — Optimisation numérique, performances, mémoire

Mesures (`profile_perf.py`, d = 6) :

| Opération | Initial | Corrigé | Gain |
|---|---|---|---|
| Matrice de covariance n = 100 | 30,9 ms | 0,32 ms | ×96 |
| Matrice de covariance n = 400 | 493 ms | 9,0 ms | ×55 |
| Ajustement d'un GP n = 20 | 4,07 s (2 781 matrices construites) | 0,03 s (96 matrices) | ×136 |
| Ajustement d'un GP n = 40 | 18,9 s (3 561 matrices) | 0,05 s (103 matrices) | ×377 |
| Ajustement d'un GP n = 80 | 39,7 s (1 941 matrices) | 0,10 s (103 matrices) | ×397 |
| Itération EGO Hartmann 6D (fit 2 niveaux + recherche DE), 6 premières itérations | 4,12 s | 0,18 s | ×22 |

(Mesures sur machine non chargée ; une première série, faite pendant que l'étude de ρ occupait tous les cœurs, donnait des gains
plus faibles mais du même ordre.) Avec le code corrigé, une itération EGO reste à 0,25 s en moyenne sur 40 itérations
(n passe de 30 à 70 points).

Causes et correctifs : double boucle Python (N1 → vectorisation) ; gradient par différences finies, d + 3 évaluations de NLL
supplémentaires par itération L-BFGS-B et paramètres sur 7 ordres de grandeur (N2 → log, N3 → gradient analytique
[RW] Éq. 5.9, vérifié par `check_grad`, dérivées du noyau SE reprises du sandbox, cellule 17) ; `scipy.linalg.solve` générique sur
un facteur triangulaire et inverse explicite (N4 → `cho_solve`, α précalculé, moyenne en O(n)) ; L² prédictions par évaluation
du mérite et DE évaluant un point à la fois (N6 → une prédiction vectorisée pour tous les niveaux et toute la population DE) ;
pénalité 1e10 qui rend l'objectif discontinu (N5 → jitter progressif) ; redémarrages toujours aléatoires (N8 → le premier repart
des hyperparamètres précédents).

**Mémoire.** Aucune fuite : la mémoire tracée (`tracemalloc`) oscille entre 0,05 et 0,43 Mo sur 40 itérations sans tendance
croissante (pic 0,82 Mo) ; les figures matplotlib sont fermées (`plt.close`), aucun cache global. La croissance attendue en O(n²)
(facteur de Cholesky par niveau) est négligeable aux tailles visées. Effet de bord corrigé : `logging.basicConfig` à l'import
(R3) — un script utilisateur appelant `basicConfig(filename=…)` sans `force=True` voyait sa configuration ignorée.

---

## 10. Étude de la démarche « ρ fixé »

### 10.1 Raisonnement

1. **Le niveau 1 n'a pas de ρ** : Y⁽¹⁾ = ρ₍₀₎·Y⁽⁰⁾ + δ⁽¹⁾ avec Y⁽⁰⁾ = 0 ([S] Éq. 9). Fixer/initialiser `rhos = [1.0, …]` est
   donc légitime et a été **conservé**.
2. Pour l ≥ 2, fixer ρ = 1 est le **modèle de correction additive** (cas particulier de Kennedy-O'Hagan) :
   * la moyenne est moins bonne si les niveaux n'ont pas la même échelle (Forrester : δ doit modéliser f₁ − 20(x−1) au lieu de
     −20(x−1)) ;
   * le mérite perd l'information R²ₗ = Πρ², qui sert dans [S] à ne pas choisir un niveau mal corrélé ;
   * **mais** fixer ρ **régularise** : avec peu de points du niveau l, ρ et les hyperparamètres de δ ne sont pas identifiables
     séparément (sandbox, cellule 48 : ρ̂ = 1,78 / 3,85 avec une vraisemblance figée ; mesuré ici : 4 points HF Forrester →
     ρ̂ = 0,60 au lieu de 2).
3. **Démarche implémentée** (T1/T1b) : ρ est **profilé** — pour θ fixé, ρ̂(θ) = FᵀK⁻¹y / FᵀK⁻¹F ([LG] Éq. 4.10, cas non imbriqué
   B.1.2, H = f̂⁽ˡ⁻¹⁾(X⁽ˡ⁾)) est injecté dans la vraisemblance de [S] Éq. 15 : même optimum qu'une estimation jointe, une dimension
   de moins ; le gradient reste exact (théorème de l'enveloppe). Le **mode hybride** garde ρ = `rho_init` tant que le niveau a
   moins de `min_points_rho` points (défaut d + 4 = nombre d'hyperparamètres de δ + 1). L'option `estimate_rho=False` reproduit
   exactement l'ancien comportement.

### 10.2 Protocole (`analysis/scripts/rho_study.py`, 5 graines, calcul parallèle)

* **Partie A — précision du surrogate** (sans optimisation) en fonction du nombre de points HF : Forrester (10 BF, 3 à 10 HF) et
  Hartmann 6D à 2 niveaux (k = 1 / ∞, δ ∈ {0 ; 0,05 ; 0,1}, 20 BF, 6 à 30 HF). Mesures : RMSE/std(f), log-densité prédictive,
  couverture à 95 %.
* **Partie B — convergence NN-MF-EGO** : Forrester (DOE 10/4, coûts 1/10, 10 itérations) et **le protocole de [S] §4.1** sur
  Hartmann 6D : L = 3 (k = 1, 3, ∞), coûts 1 / 100 / 1 000, DOE initial **imbriqué** 20 / 15 / 10, δ ∈ {0 ; 0,05 ; 0,1},
  100 itérations (coût final ≈ 1,2·10⁴). Mesure : erreur de l'optimum du surrogate |f(x̂) − f*| (métrique de [S]).
* Trois variantes : ρ = 1 fixe, ρ profilé toujours estimé, mode hybride.

### 10.3 Résultats

**Partie A — Forrester (RMSE/std(f), moyenne sur 5 graines ; ρ̂ moyen)** :

| n_HF | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| ρ = 1 | 0,607 | 0,663 | **0,313** | 0,260 | **0,079** | 0,055 |
| hybride | 0,607 | 0,663 | 0,384 | **0,133** | 0,144 | **0,041** |
| ρ̂ moyen | — | — | 1,22 | 1,75 | 1,78 | 1,98 |

ρ̂ ne devient fiable (≈ 2) qu'à partir d'environ 10 points HF ; en dessous il est instable d'une graine à l'autre.
**Hartmann (2 niveaux)** : neutre (rapport moyen des RMSE 0,998-0,999) — la suite Uₖ est une transformation non linéaire de f
pour laquelle ρ̂ ≈ 1 (0,97 à 1,04). Le surrogate global reste médiocre (RMSE relatif ≈ 1,1 à 30 points) pour toutes les
variantes : limite des données en 6D, pas du traitement de ρ.

**Partie B — erreur médiane de l'optimum du surrogate après 100 itérations (protocole [S])** :

| Cas | ρ = 1 fixe | hybride (= profilé ici) | Rapport | ρ̂ final (niv. 2 ; niv. 3) |
|---|---|---|---|---|
| Forrester (10 it.) | 1,1·10⁻⁵ | 3,6·10⁻⁵ (profilé 4,3·10⁻⁶) | tous convergés | 2,00 |
| Hartmann δ = 0 | 4,0·10⁻² | **1,2·10⁻²** | **÷ 3,2** | 0,99 ; 1,77 |
| Hartmann δ = 0,05 | 0,47 | **0,24** | **÷ 1,9** | 0,95 ; 1,26 |
| Hartmann δ = 0,1 | **0,60** | 1,19 | **× 2,0** | 1,05 ; 1,15 |

Sur Forrester, ρ = 1 a demandé une évaluation HF de plus (coût médian 123 contre 114).

### 10.4 Décision

Règle fixée **avant** l'étude (plan) : l'estimation devient le défaut si elle améliore au moins la moitié des cas sans en dégrader
aucun (seuil 10 %, deux erreurs toutes deux sous 10⁻³·|f*| étant considérées égales). Bilan : 2 cas améliorés (Hartmann δ = 0 et
0,05), **1 dégradé** (Hartmann δ = 0,1), 5 neutres → **`estimate_rho=False` reste le défaut** (comportement initial), et
`estimate_rho=True` est une option documentée, **recommandée quand les niveaux n'ont pas la même échelle et que le niveau
estimé dispose d'au moins ~10 points** (c'est le cas de `main.py`, Forrester, où ρ vrai = 2). Interprétation du cas dégradé : à
δ = 0,1 le niveau 1 est le plus mal corrélé (corrélation 0,66) et ρ̂ est estimé sur 10-15 points HF/intermédiaires : il est
bruité, et le ρ = 1 fixe agit comme une régularisation — c'est exactement l'argument de la démarche « ρ fixé ».

### 10.5 Point d'attention découvert : choix des niveaux

Dans les trois variantes, sur le protocole de [S], le mérite choisit le **niveau 1 dans ≈ 95 % des itérations** (ex. ρ = 1,
δ = 0 : 460 fois le niveau 1, 37 le niveau 2, 3 le niveau 3 sur 5 × 100 itérations), alors que [S] (Fig. 4) rapporte une
préférence pour le niveau 2 et des évaluations du niveau 1 seulement au-delà d'un coût de 4·10⁴. Diagnostic sur le DOE initial
(300 points aléatoires) : le niveau 1 maximise le mérite sur **90 % du domaine avec le code initial**, contre **16 % avec le code
corrigé** (le niveau 2 domine sur 84 %) ; la DE cherchant le maximum global, les zones où le GP du niveau 1 est encore incertain
l'emportent souvent grâce au rapport de coûts W₃/W₁ = 1 000. Ce comportement **n'est pas introduit par les corrections** (il était
plus marqué avant) ; il dépend fortement des hyperparamètres du GP du niveau 1 (que [S] estime par CMA-ES). C'est la principale
question ouverte pour reproduire quantitativement la Fig. 4 de [S] ; pistes : vérifier l'échelle relative des variances
s²₁/s²₂ dans le mérite, tester CMA-ES, ou un critère de réduction de variance intégré (évoqué en [S] §3.2).

*Note* : l'étude a été exécutée avant le correctif R7 (point déjà évalué → point aléatoire au lieu d'itération sautée). Les
runs Hartmann n'ont eu **aucune** itération sautée, les runs Forrester au plus une : R7 ne modifie pas ces conclusions.

---

## 11. Remarques : Leave-One-Out et choix de l'optimiseur

* **Le code ne contient aucun Leave-One-Out** (ni la version initiale, ni le legacy, ni le sandbox) : les hyperparamètres sont
  estimés par **maximum de vraisemblance**, ce qui est **conforme à [S]** (Éq. 15-16). Conformément à la décision prise, le MLE
  est conservé. Pour mémoire, [LG] fournit des formules LOO **rapides** (sans ré-inversion) : erreurs ε_CV,i = [K⁻¹(z − Fβ)]ᵢ /
  [K⁻¹]ᵢᵢ et variances σ²/[K⁻¹]ᵢᵢ (Éq. 1.35-1.41), un critère d'estimation LOO (Éq. 1.40-1.43, Bachoc 2013) et leur extension au
  co-krigeage récursif non imbriqué (annexe B.2). Elles pourraient servir plus tard à **valider** le surrogate (diagnostic) sans
  changer l'estimation.
* [S] estime les hyperparamètres avec CMA-ES (stratégie d'évolution globale). Le code utilise L-BFGS-B multi-départs : avec le
  gradient exact, la paramétrisation log et le démarrage à chaud, c'est un choix raisonnable (et beaucoup plus rapide), mais ce
  n'est pas l'algorithme de l'article.

---

## 12. Journal exhaustif des modifications (par catégorie)

Format : **ID — quoi** · *où* · **origine** (défaut constaté ou référence) · **effet sur le résultat final**.

### 12.1 Théorie (T)

* **T1 — ρ profilé en forme fermée.** · *`surrogate_models.py` : `GaussianProcess.fit`, `_profile_rho`,
  `negative_log_likelihood` ; `MultifidelityModel.__init__/fit`* · **Origine** : `rho = 1.0` réécrit à chaque ajustement
  (initial `:130`) alors que [S] Éq. 15 / Algo 1 estime ρ₍ₗ₋₁₎ ; forme fermée ρ̂(θ) = FᵀK⁻¹y / FᵀK⁻¹F de [LG] Éq. 4.10 (GLS) avec
  H = F = f̂⁽ˡ⁻¹⁾(X⁽ˡ⁾) (cas non imbriqué [LG] B.1.2 = [S] Éq. 18), bornée par `rho_bounds = (-5, 5)` comme le legacy. Profiler ρ
  donne le même optimum qu'une estimation jointe sans ajouter de dimension. Disponible par `estimate_rho=True` (défaut `False`
  après l'étude du §10 ; activé dans `main.py`). · **Effet** : Forrester ρ̂ = 1,96-2,01 (vrai 2), RMSE du surrogate HF
  4,36 (initial) → 1,81 (ρ = 1) → 0,19 (ρ estimé) ; R² du mérite informatif ; sur le protocole de [S] (Hartmann L = 3) : erreur de l'optimum divisée par 3,2 (δ = 0) et 1,9 (δ = 0,05), multipliée par 2 (δ = 0,1) → option, défaut inchangé (§10). `estimate_rho=False` reproduit exactement
  l'ancien comportement.
* **T1b — Mode hybride.** · *`MultifidelityModel(min_points_rho=None)` (défaut d + 4)* · **Origine** : remarque de
  l'utilisateur (ρ peut être fixé au départ) ; non-identifiabilité de ρ avec peu de points HF (sandbox cellule 48 : ρ̂ = 1,78 /
  3,85, vraisemblance figée ; mesuré : 4 points HF Forrester → ρ̂ = 0,60). · **Effet** : ρ = `rho_init` tant que n⁽ˡ⁾ < d + 4,
  estimé ensuite ; évite les ρ̂ aberrants de début de run.
* **T2 — Facteur AEI (Éq. 20).** · *`acquisition.py` : `evaluate_merits_batch`* · **Origine** : absent (régression du legacy
  `aei_multi_fidelity`). · **Effet** : facteur ≈ 1 sur données déterministes (aucun changement mesurable), pénalisation correcte
  des zones déjà connues quand un niveau est bruité (0,71 pour un bruit relatif de 0,1).
* **T3 — Effective best solution (Éq. 19).** · *`AcquisitionFunction.update()` appelé par `ask()`* · **Origine** : f_best =
  min(y_HF) au lieu de f̂⁽ᴸ⁾(x_best) ([S] Éq. 19, Huang et al.). · **Effet** : la référence de l'EI intègre les points BF
  (Forrester : −0,905 au lieu de −0,458) ; sortie `x_best` exploitée par T3b et par `summary()`.
* **T3b — Repli « confirmation de l'optimum » et `stop_on_convergence`.** · *`EGOOptimizer.run`* · **Origine** : effet de bord de
  T3 observé (mérite nul, trois évaluations HF aléatoires gaspillées sur Forrester). · **Effet** : run `main.py` : meilleure
  observation HF −5,936 → **−6,02066** (optimum −6,02074) au même coût.
* **T4 — Réduction de variance latente.** · *`evaluate_merits_batch`* · **Origine** : [S] Éq. 22, 28-29. · **Effet** : exacte quel
  que soit le bruit (erreur initiale +1 % à bruit 10⁻², +167 % à bruit = variance) ; plus de valeurs négatives/infinies.

### 12.2 Numérique (N)

* **N1 — Covariance vectorisée.** · *`kernels.py` : `base_covariance_matrix`, `pairwise_sq_diff`* · **Origine** : double boucle
  Python. · **Effet** : valeurs identiques (écart 0), ×55 à ×96 plus rapide.
* **N2 — Paramétrisation log + bornes normalisées.** · *`surrogate_models.py` : constantes `*_BOUNDS`, `fit`* · **Origine** : bornes
  absolues actives, différences finies sans sens pour τ² ([RW] §5.4.1). · **Effet** : bornes actives supprimées (§5), modèle
  indépendant des unités.
* **N3 — Gradient analytique.** · *`Kernel/SquaredExponentialKernel.get_log_params_gradients`, `negative_log_likelihood`* ·
  **Origine** : [RW] Éq. 5.9, dérivées du sandbox cellule 17 (jamais vérifiées ni reprises) ; théorème de l'enveloppe pour ρ
  profilé. · **Effet** : 96-103 matrices construites au lieu de 1 941-3 561 par ajustement ; gradient validé par `check_grad`
  (erreur relative < 10⁻⁴).
* **N4 — Cholesky partagée (`condition`) et prédiction sans inverse.** · *`GaussianProcess.condition/predict_batch`* · **Origine** :
  `solve` générique sur un facteur triangulaire, inverse explicite `k_inv`. · **Effet** : stabilité, moyenne en O(n), même code pour
  l'ajustement et la reconstruction (prérequis de X2/X3).
* **N5 — Jitter progressif.** · *`safe_cholesky`* · **Origine** : retour 1e10 discontinu ; matrices non finies non détectées
  (test). · **Effet** : moins d'échecs, erreur claire sinon.
* **N6 — Vectorisation des prédictions et du mérite.** · *`cross_covariance_matrix`, `predict_batch`, `evaluate_merits_batch`,
  DE `vectorized=True`* · **Origine** : L² prédictions point par point. · **Effet** : une seule prédiction vectorisée par population DE au lieu de L² prédictions par point ; avec N1-N5, itération EGO ×22 plus rapide (4,12 s → 0,18 s sur Hartmann 6D).
* **N7 — Normalisation des sorties.** · *`GaussianProcess._normalize`, `y_mean`, `y_std`* · **Origine** : choix utilisateur,
  bornes saturées, z-score déjà pratiqué dans le sandbox. · **Effet** : équivariance d'échelle (erreur 1-4 → ≤ 2·10⁻⁵).
* **N8 — Démarrage à chaud.** · *`_initial_guesses`* · **Origine** : hyperparamètres précédents (et ceux chargés par `load_state`)
  inutilisés. · **Effet** : convergence plus rapide d'une itération EGO à la suivante.

### 12.3 Robustesse et reproductibilité (R)

* **R1 — Graines.** · *`GaussianProcess(seed)`, `MultifidelityModel(seed)`, `EGOOptimizer(seed)`, `generate_initial_design(seed)`*
  · **Origine** : RNG global, DE non graine. · **Effet** : runs identiques (test `test_seeded_runs_are_reproducible`), artefacts
  régénérables.
* **R2 — Échec explicite.** · *`GaussianProcess.fit`* · **Origine** : `ValueError` inaccessible. · **Effet** : message (n, d,
  bornes) au lieu d'une `LinAlgError` sans contexte.
* **R3 — Pas de `basicConfig` à l'import.** · *`surrogate_models.py`* · **Origine** : *Python Logging HOWTO* (bibliothèques). ·
  **Effet** : la configuration des logs du script appelant est respectée.
* **R4 — Évaluations en échec.** · *`ExperimentData.add_observation/get_training_data/best_observation/n_failed`,
  `MultifidelityModel.fit`, `EGOOptimizer.tell`* · **Origine** : NaN propagés à `np.min`, l'EI et la NLL ; pénalité 1e6. ·
  **Effet** : un échec de simulation est mémorisé (coût compté, point non reproposé) mais exclu du GP.
* **R5 — Coût du plan d'expériences.** · *`EGOOptimizer._init_history`* · **Origine** : coût compté dans `__init__`, avant le DOE.
  · **Effet** : historique de coût correct (Forrester : coût final 114 au lieu de 64 ; hydrofoil 48 au lieu de 40) ; axe des
  abscisses des tracés de convergence correct.
* **R7 — Point déjà évalué.** · *`EGOOptimizer.run`* · **Origine** : découvert par le test
  `test_failed_simulations_do_not_break_the_run` : un point HF en échec est exclu du GP, le modèle ne change pas, la DE (graine
  fixe, R1) repropose exactement le même point, qui était « sauté » à chaque itération (blocage ; avant R1 le hasard de la DE le
  masquait). · **Effet** : un point aléatoire (graine) du même niveau est évalué à la place ; plus d'itération perdue.
* **R6 — Import `traitlets`.** · *`data_management.py`* · **Origine** : `TypeError`/`ModuleNotFoundError` mesurés. · **Effet** :
  le paquet s'importe en Python 3.13 et 3.14.

### 12.4 Données, export, visualisation (X)

* **X1 — Export du surrogate.** · *`MultifidelityModel.to_dict/from_dict/predict_batch`, `load_surrogate`,
  `EGOOptimizer.export_surrogate/summary`* · **Origine** : demande utilisateur, pas d'export exploitable. · **Effet** : `model =
  load_surrogate("surrogate.json")` redonne exactement le modèle entraîné (écart < 10⁻⁹) ; `summary()` en fin de run.
* **X2 — État JSON cohérent.** · *`EGOOptimizer.run/save_state`* · **Origine** : §7. · **Effet** : écart reconstruction / modèle
  1,89 std → 0.
* **X3 — Visualiseur.** · *`visualization.py`* · **Origine** : §7. · **Effet** : tracés fidèles, en 1 ou plusieurs niveaux,
  versions plotly interactives (`plot_*_interactive`).

### 12.5 Exemples (E) et documentation (D)

* **E1 — Hartmann.** Constantes [S] Éq. 30, évaluation unique, `self.num_levels`. Effet : minimum −3,32237 (au lieu de −3,3975),
  cible correcte, coût de simulation divisé par 2.
* **E2 — Hydrofoil.** Mapping `modelclasses[level-1]`, `NaN` au lieu de 1e6, `L` du simulateur, configuration L = 1 explicite,
  cible 0,0138978 recalculée. Effet : niveaux corrects pour L > 2, GP protégé des pénalités, tracé de convergence juste.
* **E3 — `main.py`.** Validation du niveau, log réinitialisé à chaque run (`filemode='w'` dans les trois scripts), export et
  graphiques interactifs.
* **D1 — README** réécrit (versions, dépendances, lancement, API, export, tests). **D2 — Diagrammes** : `charts/mfego_structure.mmd`
  (classes réelles) et `charts/mfego_ego_loop.mmd` (boucle), rendus en SVG/PNG avec mermaid-cli.

### 12.6 Tests et analyse

* **`tests/`** (63 tests, 10 s) : noyau, GP (NLL vs `multivariate_normal`, `check_grad`, équivariance, bornes, graines, R2),
  modèle MF (Éq. 11-12, 18, ρ̂ ≈ 2, modes de ρ, 3 niveaux, échecs, export), mérite (EI vs Monte-Carlo, AEI, Δσ² vs mise à jour
  par blocs, R² sur 3 niveaux, x_best), données, optimiseur (coût DOE, JSON, sauvegarde/chargement, graines, convergence Forrester,
  échecs, arrêt), **mono-fidélité de bout en bout** (modèle = GP simple, mérite = AEI, run, export, tracés PNG/HTML 1D et 2D),
  export/visualisation, simulateurs (Hartmann, mapping NeuralFoil, échec, évaluation réelle `slow`).
* **`analysis/`** : 7 scripts reproductibles, résultats JSON (`analysis/results/`), figures plotly (`analysis/figures/`).

---

## 13. Ce qui a été conservé tel quel (correct) et pourquoi

* `cov_fct`, `k_l_vector` et l'API `Kernel` / `SquaredExponentialKernel` : noyau conforme à l'Éq. 2, `k_l_vector` déjà vectorisé.
* `BaseSimulator` (interface abstraite) et l'architecture ask / tell / run : saine et adaptée aux calculs longs.
* La génération LHS indépendante par niveau avec graines 42 + l (conforme à l'esprit non imbriqué).
* Les formules récursives de prédiction (Éq. 11-12) et les résidus non imbriqués (Éq. 18) : conformes.
* Le ratio de coûts W_L/W_l et le produit R² de l'Éq. 24 (indices vérifiés sur 3 niveaux).
* L'EI calculé avec la variance prédictive incluant le bruit (conforme aux Éq. 5 et 12 de [S]).
* L'initialisation `rhos = [1.0, …]` (le niveau 1 n'a pas de ρ).
* `NumpyEncoder`, `generate_continuous_naca4`, la recherche de racine `xxxlarge` commune aux niveaux (choix voulu, documenté).
* Les méthodes matplotlib (PNG) du visualiseur : conservées, corrigées, et complétées par plotly.
* Les noms et signatures publics (compatibilité : `fit`, `predict`, `_predict_up_to`, `evaluate_merit`, JSON rétro-compatible).

---

## 14. Revue de `legacy/` et `sandbox/` (non modifiés)

* `legacy/legacy_nested` : **le noyau n'utilise que x₁** (`nested_mf_covariance.py:32-43` : boucle sur `x.shape[0]` = 1 après
  `reshape(1, d)`) → toutes les longueurs de corrélation sauf l₁ restent à leur valeur initiale et les points choisis ont x₁ en
  butée (`testlogfile.log`). Le Clean avait corrigé ce bug. AEI codé mais commenté (`nested_mf_optimizer.py:248-250`) ; bruit
  initial 10⁻⁴ hors de ses bornes (10⁻⁸, 10⁻⁵) ; même bruit utilisé pour tous les niveaux dans la réduction de variance ; `LogTee`
  jamais fermé ; graines absentes.
* `legacy/legacy_non_nested` : estimait ρ ∈ [−5, 5] (variable L-BFGS-B), contenait l'AEI et θ₂ ∈ [10⁻³ ; 50] — la version Clean a
  régressé sur ces points (corrigés ici). `custom_fluid_functions.py` avait le bon mapping niveau → modèle.
* `sandbox/*.ipynb` : contient les bons éléments repris ici (NLL en `cho_solve` cellule 15, gradient analytique cellule 17, z-score
  cellules 19-27, AEI cellules 23 et 42). Défauts : bruit ajouté en σ dans une cellule et en σ² ailleurs (13), fidélités mélangées
  dans un même jeu (27), `costs = [1, 10, 1]` (48), multistart DE sans effet (47), code mort (41).
* Constantes de Hartmann fausses dans toutes les copies legacy et dans le notebook.

---

## 15. Résultats des exemples régénérés (graines fixes)

Les trois exemples ont été relancés avec le code corrigé et `seed = 0` (deux exécutions successives donnent des résultats
identiques au bit près). Les fichiers `ego_backup.json` contiennent maintenant l'instantané exact du modèle entraîné, et chaque
exemple écrit aussi `surrogate.json` et des figures interactives (`*.html`).

| Exemple | Avant (fichiers de `main`) | Après |
|---|---|---|
| Forrester `mfego/main.py` (10 BF / 4 HF, 10 it., coûts 1/10, `estimate_rho=True`) | meilleure obs. HF −6,0207 (2ᵉ run du log ; 1ᵉʳ run −6,0155), coût affiché **64** (DOE non compté) | meilleure obs. HF **−6,02066** en x = 0,7569 (optimum −6,02074, écart 7,5·10⁻⁵) ; optimum du surrogate f̂ = −6,0262 en x = 0,7569 ; **coût 114** (DOE 50 + 4 BF + 6 HF) ; ρ̂ = 2,008 |
| Hartmann 6D (20 BF / 10 HF, 30 it., coûts 1/10) | JSON **contaminé** (autre problème, d = 3, y = 1e6) ; le log montrait −3,1961 sur la fonction **fausse** (minimum −3,3975) | meilleure obs. HF **−3,1209** (fonction corrigée, minimum global −3,32237) ; coût 303 ; 33 BF / 27 HF. Le run est dans le bassin du deuxième minimum local de Hartmann (≈ −3,20 au voisinage de (0,40 ; 0,88 ; 0,85 ; 0,57 ; …)) : 30 itérations ne suffisent pas à trouver le minimum global sur ce cas (cf. [S] : 400 itérations) |
| Hydrofoil (6 BF / 2 HF, 40 it., coûts 1/1) | meilleur C_d **0,014019**, coût affiché 40 | meilleur C_d **0,0139078** en (cambrure 4,63 % ; épaisseur 8 %), à **0,07 %** de la référence 0,0138978 (avant : 0,87 %) ; coût 48 ; 14 BF / 34 HF ; aucun échec |

Remarque Forrester : après convergence, le mérite s'annule ; l'algorithme a d'abord évalué la HF à l'optimum du surrogate (T3b),
puis effectué 3 évaluations HF aléatoires (repli d'origine conservé). `run(n_iterations, stop_on_convergence=True)` les évite
(économie de 30 unités de coût ici).

---

## 16. Limites restantes et recommandations

* **Calibration à très petit n** : avec n⁽ˡ⁾ ≤ d + 3, le MLE sur-ajuste (intervalles trop étroits). Pistes : priors faibles sur les
  hyperparamètres (MAP), ou LOO de [LG] comme diagnostic.
* **Hartmann 6D** : avec 10-30 points HF le surrogate global reste médiocre (RMSE relatif ≈ 0,8-1,1 pour toutes les variantes) :
  c'est une limite des données (fonction presque plate avec des puits étroits en 6D), pas du code ; l'EGO se concentre de toute
  façon sur la zone de l'optimum.
* **Formulation exacte non imbriquée** ([LG] B.1) : propager la covariance du niveau inférieur aux points X⁽ˡ⁾ est possible avec la
  structure actuelle (`condition`) si l'on veut aller au-delà de [S].
* **Optimiseur d'hyperparamètres** : CMA-ES (comme [S]) pourrait être ajouté en option pour les grandes dimensions.
* **Plan d'expériences initial imbriqué** (protocole de [S] §4.1) : pris en charge dans `rho_study.py` ; pourrait devenir une
  option de `generate_initial_design`.
* **Choix des niveaux sur le protocole de [S]** (§10.5) : le niveau 1 est choisi dans ≈ 95 % des itérations (la Fig. 4 de [S]
  montre une préférence pour le niveau 2) ; à investiguer (hyperparamètres du niveau 1 / CMA-ES, critère intégré de [S] §3.2).
* **Sécurité** : révoquer le PAT (§0).

---

## 17. Figures interactives

Toutes dans `analysis/figures/` (ouvrir dans un navigateur ; plotly.js chargé depuis le CDN) :

| Fichier | Contenu |
|---|---|
| `theory_forrester_before_after.html` | Surrogate Forrester initial vs corrigé, bandes à 95 %, observations |
| `theory_rho_profile_likelihood.html` | NLL(ρ) de Forrester : minimum en ρ ≈ 2, ρ = 1 du code initial |
| `theory_variance_reduction_aei.html` | Erreur de l'ancienne Δσ², poids du facteur AEI |
| `hyperparams_active_bounds.html` | Bornes actives et erreur d'échelle, avant/après |
| `hartmann_mf_correlations.html` | f_k vs f pour la suite multi-fidélité de Hartmann |
| `hydrofoil_feasibility_map.html` | C_d des deux niveaux, validité du bracket, minimum de référence |
| `reconstruction_trained_vs_rebuilt.html` | Modèle entraîné vs reconstruit depuis le JSON, avant/après |
| `performance_memory.html` | Temps (covariance, ajustement) et mémoire tracée |
| `rho_study_accuracy.html` | Précision du surrogate selon le traitement de ρ |
| `rho_study_ego_convergence.html` | Convergence NN-MF-EGO selon le traitement de ρ (protocole [S] §4.1) |
| `env_import_checks.html` | Imports par interpréteur et version du code |

Les exemples produisent aussi leurs figures interactives (`mfego/*.html`, `example/*/*.html`).
