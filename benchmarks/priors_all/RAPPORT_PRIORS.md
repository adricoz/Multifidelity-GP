# Priors sur tous les hyperparamètres — benchmark Hartmann 6D

> **Note v2.** Ce rapport décrit **mfego v1** (16/10/2026, matin). Les défauts ont changé depuis : noyau Matérn 5/2 pour
> `CoKriging` et le pipeline, ρ centré, prior N(1, 2²) sur ρ, entrées normalisées. Voir
> [`benchmarks/surrogate_v2/RAPPORT_V2.md`](../surrogate_v2/RAPPORT_V2.md).
>
> Relancer ces scripts aujourd'hui utilise les défauts v2. Pour retrouver les chiffres ci-dessous, il faut les options v1 :
> `priors=V1_PRIORS, rho_centering=False` et `SquaredExponentialKernel`.

16/10/2026 · `mfego` avec `use_map=True` et des priors sur les longueurs de corrélation, la variance du signal t1, la
variance de biais t2 et le bruit, comparé au MLE, à l'ancien MAP (longueurs seules) et aux autres bibliothèques de GP ·
dossier `benchmarks/priors_all/`

Les figures interactives sont dans [`figures/`](figures/) (à ouvrir dans un navigateur ; plotly.js vient du CDN, thème
sombre automatique). Les résultats bruts sont dans [`results/`](results/).

Notations : **MLE** = maximum de vraisemblance, sans prior ; **MAP-l** = l'ancien `use_map=True`, avec une prior sur les
longueurs seules ; **MAP-all** = le nouveau `use_map=True`, avec des priors sur les longueurs, t1, t2 et le bruit.

---

## 1. Résumé

* **Implémenté.** Chaque hyperparamètre a maintenant sa prior, avec un gradient toujours analytique. Ce sont des
  triplets `(famille, a, b)` de type InvGamma, Gamma ou LogNormale. Une valeur `None` retire la prior d'un groupe. Le
  comportement précédent reste disponible (`LENGTHSCALE_ONLY_PRIORS`), et un ancien JSON est relu à l'identique. Les tests
  passent : 125, dont 26 nouveaux (priors, co-krigeage, bench_lib).
* **Précision :** MAP-all ≈ MAP-l, et les deux font nettement mieux que le MLE. MAP-all bat le MLE sur 9 à 10 graines sur 10
  en RMSE comme en NLPD, à toutes les tailles. Contre MAP-l, le bilan est partagé (3 à 8 graines sur 10 selon la taille) :
  **aucun gain net de précision sur Hartmann**.
* **Robustesse :** avec MAP-all, plus aucun hyperparamètre ne s'arrête sur une borne de recherche, à aucun niveau. Avec
  MAP-l, t2 touche encore sa borne basse (7,5 % des ajustements du niveau HF) et le bruit sa borne haute (1 graine sur 10 à
  n = 300). Avec le MLE, 45 à 51 % des longueurs et 22 à 40 % des t2 sont en butée.
* **Ce que font les nouvelles priors.** Sur des données déterministes, la vraisemblance ne contraint ni le bruit ni t2. Le
  MAP les place au mode de leur prior (bruit 1,8·10⁻⁶, t2 1,8·10⁻⁴). Sans prior, ils restent à leur **valeur initiale**
  de l'optimiseur (10⁻⁶ et 10⁻⁴). La prior remplace donc une valeur arbitraire par une valeur choisie et documentée.
* **Beaucoup de points :** le MLE dégénère (RMSE 1,00 à n = 160 et 300) ; tous les MAP l'évitent, MAP-l comme MAP-all
  (0,36–0,37 à n = 300, au niveau de BoTorch et de scikit-learn). La prior sur les longueurs suffit pour cela.
* **Optimisation (budget 250, 10 graines) :** NN-MF-EGO MAP-all ≈ MAP-l (erreur de recommandation médiane 0,25 contre 0,21 ;
  MAP-all meilleur sur 5 graines sur 10) et meilleur que le MLE (8/10) et BoTorch qLogEI (0,71). En monofidélité,
  SF-EGO MAP-all fait moins bien que MAP-l (0,59 contre 0,37 ; 2 graines sur 10). Avec 10 graines, l'écart reste dans le
  bruit des trajectoires EGO : les variantes de sensibilité, presque identiques en précision, donnent des médianes de 0,20
  à 0,30.
* **Le réglage qui compte reste le mode des longueurs.** IG(3, 1) est encore la meilleure variante : NLPD 0,32 contre 0,53 à
  n_HF = 40, recommandation 0,16 contre 0,25. C'est la même conclusion que dans RAPPORT_MAP.

**Recommandation.** Garder MAP-all par défaut avec `use_map=True` : il est sans perte de précision mesurable et plus robuste,
et il ne laisse plus le bruit et t2 à des valeurs initiales arbitraires. Ne pas en attendre de gain de précision. Le levier
principal reste le mode de la prior des longueurs : passer `LENGTHSCALE_PRIOR` à (3, 1) après validation sur un cas réel
(suite déjà proposée par RAPPORT_MAP).

---

## 2. Ce qui a été implémenté

### 2.1 `mfego/src/surrogate_models.py`

* `neg_log_prior(prior, log_value)` : −log p(θ), constante omise, et sa dérivée par rapport à ln θ, pour trois familles.
  Le code vient de `benchmarks/map_hartmann/priors.py`, qui importe maintenant cette fonction du cœur.

  | Famille | −log p(θ) | Dérivée par rapport à ln θ | Mode (cible du MAP) |
  |---|---|---|---|
  | `("invgamma", α, β)` | (α + 1) ln θ + β / θ | (α + 1) − β / θ | β / (α + 1) |
  | `("gamma", k, taux)` | −(k − 1) ln θ + taux θ | −(k − 1) + taux θ | (k − 1) / taux |
  | `("lognormal", μ, σ)` | ln θ + (ln θ − μ)² / (2σ²) | 1 + (ln θ − μ) / σ² | e^(μ − σ²) |

* `DEFAULT_PRIORS`, pour des sorties normalisées (moyenne 0, variance 1) et des entrées dans [0, 1] :

  | Groupe | Prior | Mode | Raison |
  |---|---|---|---|
  | longueurs l₁..l_d | InvGamma(3, 2) | 0,5 | inchangée (`LENGTHSCALE_PRIOR`) |
  | variance du signal t1 | LogNormale(1, 1) | 1 | sorties de variance 1 ; 95 % de la masse dans [0,38 ; 19] |
  | variance de biais t2 | LogNormale(ln 10⁻², 2) | 1,8·10⁻⁴ | terme constant mal identifié (la moyenne est déjà retirée) |
  | bruit | LogNormale(ln 10⁻⁴, 2) | 1,8·10⁻⁶ | pépite d'un code déterministe |
  | ρ | aucune | — | reste profilé en forme fermée (`_profile_rho`), comme convenu |

  Le MAP est pris **en espace θ**, sans jacobien. C'est la convention existante, la même que GPyTorch. La cible est donc
  le **mode** de p(θ). Pour une LogNormale large, il est loin de la médiane : avec σ = 2, mode = médiane / 55. La figure
  `prior_shapes.html` trace p(θ) / max pour cette raison : son pic est la cible du MAP.
* `GaussianProcess(..., use_map, lengthscale_prior=None, priors=None)` : `priors` est un dictionnaire partiel qui surcharge
  `DEFAULT_PRIORS` ; `None` retire la prior d'un groupe. Une seule méthode, `neg_log_priors(log_params)`, parcourt les
  quatre groupes (indices `[:d]`, `d`, `d+1`, `d+2`) et remplace l'ancien bloc InvGamma de `negative_log_likelihood`.
* `MultifidelityModel(..., priors=None)` : les mêmes priors pour tous les niveaux. `lengthscale_prior` reste par niveau.
* **Compatibilité :**
  * `lengthscale_prior=(α, β)` donne toujours IG(α, β) sur les longueurs ;
  * `priors=LENGTHSCALE_ONLY_PRIORS` reproduit exactement l'ancien `use_map=True` (testé) ;
  * `to_dict`/`from_dict` sauvegardent les priors **réellement utilisées**, pas seulement les surcharges. Le fichier se
    décrit donc seul, et un changement futur de `DEFAULT_PRIORS` ne modifie pas un modèle rechargé puis réajusté ;
  * un JSON écrit avant cette version est relu avec les longueurs seules
    (testé, même modèle) ;
  * `diagnostics()` et `estimator_label()` affichent toutes les priors actives.
* **Effet sur les études existantes :** le pipeline bdFoil (`use_map: true` par défaut) et les exemples utilisent désormais
  MAP-all. En Python, l'ancien comportement s'obtient avec `priors=LENGTHSCALE_ONLY_PRIORS`. **La configuration JSON du
  pipeline n'a pas de clé `priors`** (ses clés sont vérifiées et `run.py` ne transmet pas ce paramètre) : depuis le
  pipeline, on ne peut pas revenir à l'ancien comportement sans ajouter cette clé. Ce n'est pas fait, le pipeline n'a pas
  été modifié. Une reprise d'EGO (`load_state`) réajuste aussi le modèle avec les nouvelles priors.

### 2.2 Tests (`tests/test_gaussian_process.py`, `tests/test_multifidelity.py`)

* Valeur de chaque famille comparée à `scipy.stats` (invgamma, gamma, lognorm) et dérivée comparée aux différences finies.
* Terme de prior de chaque groupe, avec les priors par défaut, comparé à `scipy.stats`.
* Ancien comportement reproduit avec `LENGTHSCALE_ONLY_PRIORS`.
* Validation : groupe inconnu, famille inconnue, paramètres invalides.
* `check_grad` avec et sans MAP, avec et sans ρ profilé, avec une prior Gamma sur t1.
* Priors partagées par tous les niveaux et sauvegardées ; relecture d'un ancien JSON.

---

## 3. Protocole

Même protocole que `benchmarks/map_hartmann/` : les chiffres MAP-l reproduisent exactement ceux de RAPPORT_MAP. C'est un
contrôle de non-régression.

* **Problème :** Hartmann 6D multifidélité de Sacher et al. (2021, éq. 30–32), `bench_lib.HartmannMF`. BF = U₁ décalé de
  δ = 0,05, HF = Hartmann, coûts 1 et 10, entrées dans [0, 1]⁶.
* **Précision :** n_HF ∈ {5, 10, 20, 40}, n_BF = 2 n_HF, LHS indépendants par niveau. 10 graines, le même plan pour tous les
  modèles d'une graine (comparaison appariée), 2000 points de test. Mesures : RMSE relative (RMSE / écart-type de f ;
  1 = la moyenne), NLPD (log-vraisemblance prédictive négative, plus bas = mieux) et couverture de l'intervalle à 95 %.
* **Beaucoup de points :** HF seule, n ∈ {80, 160, 300}.
* **Optimisation :** plan initial de coût 120 (MF : 20 BF + 10 HF ; SF : 12 HF), budget 250, 10 graines.
  Recommandation = minimum de la moyenne prédite (même évolution différentielle pour toutes les méthodes).
* **Bibliothèques :** BoTorch SingleTaskMultiFidelityGP et SingleTaskGP (MAP avec leurs priors par défaut), scikit-learn
  (MLE), SMT MFK (co-krigeage récursif de Le Gratiet) ; en optimisation, BoTorch qLogEI.

---

## 4. Précision du modèle de substitution

Figure : [`accuracy_vs_n.html`](figures/accuracy_vs_n.html). Médianes sur 10 graines, dans chaque case :
**RMSE relative · NLPD · couverture 95 %**.

| Multifidélité | n_HF = 5 | n_HF = 10 | n_HF = 20 | n_HF = 40 |
|---|---|---|---|---|
| mfego MLE | 1,08 · 33,22 · 0,44 | 1,10 · 5,45 · 0,66 | 1,03 · 12,75 · 0,62 | 0,93 · 1,24 · 0,83 |
| mfego MAP-l | 1,01 · 2,31 · 0,84 | 0,95 · 0,73 · 0,90 | 0,92 · 1,06 · 0,87 | 0,78 · 0,53 · 0,89 |
| **mfego MAP-all** | 1,01 · 2,36 · 0,84 | 0,95 · 0,74 · 0,89 | 0,92 · 1,08 · 0,87 | 0,78 · 0,53 · 0,89 |
| mfego MAP-all, IG(3, 1) | 1,02 · 1,50 · 0,87 | 0,97 · 0,62 · 0,92 | 0,93 · 0,78 · 0,89 | 0,74 · 0,32 · 0,93 |
| BoTorch MF-GP | 1,24 · 0,73 · 0,99 | 0,97 · 0,50 · 0,95 | 0,91 · 1,24 · 0,86 | 0,82 · 0,81 · 0,87 |
| SMT MFK | 1,17 · 175,86 · 0,23 | 1,24 · 18,05 · 0,48 | 1,04 · 7,60 · 0,66 | 1,02 · 1,27 · 0,81 |

| HF seule | n_HF = 5 | n_HF = 10 | n_HF = 20 | n_HF = 40 |
|---|---|---|---|---|
| mfego MLE | 1,09 · 20,55 · 0,57 | 1,04 · 7,37 · 0,63 | 1,01 · 5,27 · 0,69 | 1,03 · 1,55 · 0,80 |
| mfego MAP-l | 1,02 · 2,44 · 0,84 | 0,96 · 0,70 · 0,90 | 0,94 · 1,10 · 0,86 | 0,81 · 0,58 · 0,88 |
| **mfego MAP-all** | 1,02 · 2,49 · 0,83 | 0,96 · 0,73 · 0,90 | 0,94 · 1,10 · 0,86 | 0,81 · 0,58 · 0,88 |
| BoTorch SingleTaskGP | 1,02 · 1,97 · 0,86 | 0,99 · 0,94 · 0,85 | 0,99 · 3,16 · 0,76 | 0,90 · 1,15 · 0,85 |
| scikit-learn | 1,10 · 99,85 · 0,44 | 1,06 · 15,62 · 0,67 | 1,05 · 3,85 · 0,77 | 1,02 · 1,46 · 0,81 |

**Graines où MAP-all fait mieux** (même plan) :

| n_HF | MF : contre MLE (RMSE / NLPD) | MF : contre MAP-l (RMSE / NLPD) | SF : contre MLE | SF : contre MAP-l |
|---|---|---|---|---|
| 5 | 9/10 · 10/10 | 3/10 · 4/10 | 9/10 · 10/10 | 4/10 · 4/10 |
| 10 | 9/10 · 10/10 | 6/10 · 3/10 | 8/10 · 10/10 | 8/10 · 3/10 |
| 20 | 10/10 · 10/10 | 4/10 · 3/10 | 10/10 · 9/10 | 6/10 · 3/10 |
| 40 | 10/10 · 10/10 | 8/10 · 7/10 | 10/10 · 9/10 | 5/10 · 7/10 |

**Hyperparamètres sur une borne de recherche** (toutes tailles et graines ; figure
[`hyperparameters.html`](figures/hyperparameters.html)) :

| Modèle MF, niveau | longueurs | t1 | t2 | bruit |
|---|---|---|---|---|
| MLE, BF / HF | 45 % / 51 % | 0 % / 0 % | 28 % / 40 % | 8 % / 18 % |
| MAP-l, BF / HF | 0 % / 0 % | 0 % / 0 % | 0 % / 7,5 % | 0 % / 0 % |
| **MAP-all**, BF / HF | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |

**Lecture.**

* L'écart MAP-all / MAP-l est minuscule (± 0,05 en NLPD). Les priors sur t1, t2 et le bruit ne changent presque pas
  la moyenne prédite : t1 est bien identifié par les données (valeurs ajustées de 0,77 à 1,78 selon le niveau, médianes 1,08 à 1,29), et t2 et le
  bruit sont petits devant la variance du signal.
* La figure des hyperparamètres montre l'effet réel. Sans prior, t2 et le bruit ne bougent pas de leur point de départ :
  la vraisemblance est plate dans ces directions et L-BFGS-B s'arrête aussitôt. Avec MAP-all, ils sont au mode de leur
  prior. Le résultat ne dépend plus de l'initialisation de l'optimiseur.
* Temps d'ajustement médian en multifidélité : 0,11 s pour MAP-all, 0,085 s pour MAP-l, 0,17 s pour le MLE et 3,8 s pour
  BoTorch.

---

## 5. Beaucoup de points (HF seule)

Figure : [`many_points.html`](figures/many_points.html). Médianes : **RMSE relative · NLPD · couverture 95 %**.

| HF seule | n = 80 | n = 160 | n = 300 |
|---|---|---|---|
| mfego MLE | 0,96 · 0,49 · 0,92 | 1,00 · 0,49 · 0,94 | 1,00 · 0,50 · 0,93 |
| mfego MAP-l | 0,66 · 0,21 · 0,92 | 0,50 · −0,22 · 0,93 | 0,36 · −0,51 · 0,93 |
| **mfego MAP-all** | 0,66 · 0,21 · 0,92 | 0,51 · −0,20 · 0,93 | 0,37 · −0,50 · 0,93 |
| BoTorch SingleTaskGP | 0,65 · 0,23 · 0,91 | 0,52 · −0,16 · 0,93 | 0,37 · −0,49 · 0,94 |
| scikit-learn | 0,96 · 0,70 · 0,90 | 0,55 · −0,09 · 0,92 | 0,37 · −0,49 · 0,93 |

* Le MLE de mfego s'arrête sur des longueurs ≈ 0,01 (17 à 32 % des longueurs à la borne basse) : c'est le modèle « bruit
  blanc » de RAPPORT_MAP § 6. Les deux MAP l'évitent et rejoignent BoTorch et scikit-learn.
* MAP-all est très légèrement moins bon que MAP-l à grand n (MAP-all meilleur sur 2 à 3 graines sur 10 en RMSE, 1 à 3 en NLPD ; 0,37 contre 0,36 à
  n = 300). Le t1
  ajusté est le même (0,81 contre 0,78). En revanche, avec beaucoup de points, les données commencent à identifier t2 :
  sans prior, t2 monte à 0,035 (n = 160) puis 0,17 (n = 300), alors que la prior de MAP-all le maintient à 2·10⁻⁴. C'est
  la cause probable de ce petit écart, non isolée par une variante. L'effet est faible mais systématique.
* Temps à n = 300 : 25 s pour les MAP, 18 s pour le MLE, 17 s pour scikit-learn et 2,7 s pour BoTorch. mfego refait une
  Cholesky de taille n à chaque évaluation de la vraisemblance, sans parallélisme.

---

## 6. Convergence de l'optimisation

Figure : [`optimization.html`](figures/optimization.html). Erreur f − f* ; 10 graines.

| Méthode | Meilleure obs. HF, médiane (moyenne) | Recommandation, médiane [quartiles] | Itérations | Part BF | Temps / itération |
|---|---|---|---|---|---|
| NN-MF-EGO · MLE | 0,33 (0,82) | 0,30 [0,30–1,72] | 36 | 70 % | 0,38 s |
| NN-MF-EGO · MAP-l | 0,22 (0,50) | 0,21 [0,16–0,50] | 14 | 7 % | 0,21 s |
| **NN-MF-EGO · MAP-all** | 0,26 (0,59) | 0,25 [0,11–0,55] | 14 | 7 % | 0,21 s |
| NN-MF-EGO · MAP-all, IG(3, 1) | 0,16 (0,29) | 0,16 [0,02–0,41] | 18,5 | 27 % | 0,23 s |
| SF-EGO · MLE | 1,53 (1,33) | 1,65 [0,90–1,84] | 13 | 0 % | 0,18 s |
| SF-EGO · MAP-l | 0,36 (0,51) | 0,37 [0,25–0,55] | 13 | 0 % | 0,16 s |
| **SF-EGO · MAP-all** | 0,43 (0,63) | 0,59 [0,28–0,95] | 13 | 0 % | 0,15 s |
| BoTorch qLogEI (SF) | 0,62 (0,98) | 0,71 [0,48–1,66] | 13 | 0 % | 0,63 s |

Graines où MAP-all fait mieux (recommandation) :

| Comparaison | Graines |
|---|---|
| NN-MF-EGO, contre MLE | 8/10 |
| NN-MF-EGO, contre MAP-l | 5/10 |
| SF-EGO, contre MLE | 10/10 |
| SF-EGO, contre MAP-l | 2/10 |

Les trajectoires EGO sont chaotiques. Un changement minime du modèle, comme le bruit à 1,8·10⁻⁶ au lieu de 10⁻⁶ (il entre
dans l'AEI), suffit à envoyer une itération ailleurs. L'écart SF-EGO MAP-all / MAP-l (2 graines sur 10) n'est pas
significatif : un test du signe donne p ≈ 0,11. À confirmer avec plus de graines avant de conclure.

---

## 7. Sensibilité des nouvelles priors (un facteur à la fois autour de MAP-all)

Figure : [`sensitivity.html`](figures/sensitivity.html). Les priors de t1 gardent leur mode à 1 (μ = σ²). Les variantes de
bruit changent la médiane de la prior.

| Variante | NLPD n_HF = 10 | NLPD n_HF = 40 | RMSE n_HF = 40 | Recommandation NN-MF-EGO |
|---|---|---|---|---|
| MLE | 5,45 | 1,24 | 0,93 | 0,30 [0,30–1,72] |
| MAP-l | 0,73 | 0,53 | 0,78 | 0,21 [0,16–0,50] |
| **MAP-all** | 0,74 | 0,53 | 0,78 | 0,25 [0,11–0,55] |
| MAP-all, longueurs IG(3, 1) | **0,62** | **0,32** | **0,74** | **0,16** [0,02–0,41] |
| MAP-all, t1 σ = 0,5 | 0,76 | 0,54 | 0,78 | 0,28 [0,12–0,52] |
| MAP-all, t1 σ = 2 | 0,73 | 0,53 | 0,78 | 0,28 [0,12–0,78] |
| MAP-all, bruit médiane 10⁻⁶ | 0,74 | 0,53 | 0,78 | 0,30 [0,08–0,54] |
| MAP-all, bruit médiane 10⁻³ | 0,74 | 0,53 | 0,78 | 0,27 [0,16–0,88] |
| MAP-all sans prior sur t1 | 0,73 | 0,53 | 0,78 | 0,20 [0,16–0,30] |
| MAP-all sans prior sur t2 | 0,74 | 0,53 | 0,78 | 0,23 [0,10–0,32] |
| MAP-all sans prior sur le bruit | 0,74 | 0,53 | 0,78 | 0,28 [0,19–1,07] |

* En précision, **toutes les variantes de t1, t2 et du bruit sont équivalentes** (NLPD 0,73–0,76 et 0,53–0,54). Seul le mode
  des longueurs change le résultat.
* En optimisation, les médianes varient de 0,20 à 0,30 sans ordre lisible. C'est l'ordre de grandeur du bruit entre
  trajectoires. Aucune de ces variantes ne justifie de changer les valeurs par défaut.

---

## 8. Recommandations et limites

* **Valeurs par défaut :** garder `DEFAULT_PRIORS` tel quel. Elles sont sans effet mesurable sur la précision et rendent le
  résultat indépendant de l'initialisation de l'optimiseur. Si un problème a un vrai bruit, ou une sortie très éloignée
  d'une variance 1 après normalisation, régler `priors={"noise": ...}` ou `priors={"signal_variance": ...}`.
* **Le levier principal reste la prior des longueurs** : IG(3, 1) bat IG(3, 2) partout ici, comme dans RAPPORT_MAP.
  À valider sur un cas réel (hydrofoil) avant de changer `LENGTHSCALE_PRIOR`.
* **t2 à grand n :** la LogNormale(ln 10⁻², 2) empêche t2 de monter quand les données l'identifient (300 points : perte de
  1 à 3 % en RMSE). Une prior plus large sur t2 serait plus neutre. Non testé à grand n.
* **Limites :**
  * un seul problème (fonction lisse, déterministe, d = 6) et 10 graines ;
  * ρ n'a pas de prior (choix validé). Ce benchmark ne mesure pas le biais de ρ décrit dans
    `benchmarks/cokriging/RAPPORT_COKRIGING.md` § 7 ;
  * les entrées doivent être dans [0, 1] et les sorties sont normalisées par mfego : les priors en dépendent.

---

## Reproduction

Depuis la racine du dépôt, avec l'environnement `.venv-benchmark` (environ 2 min 30 sur 32 cœurs) :

```
.venv-benchmark\Scripts\python benchmarks\priors_all\run_priors_benchmark.py     # --quick : 3 graines
.venv-benchmark\Scripts\python benchmarks\priors_all\make_figures.py
```

| Fichier | Rôle |
|---|---|
| `run_priors_benchmark.py` | Précision, beaucoup de points, optimisation, sensibilité (`ESTIMATORS`, `SENSITIVITY`) |
| `make_figures.py` | Les 6 figures de `figures/` |
| `results/` | `accuracy.csv` (avec les hyperparamètres ajustés), `large_n.csv`, `optimization.csv`, `optimization_histories.json`, `summary.txt` |

## Figures

| Figure | Contenu |
|---|---|
| [`accuracy_vs_n.html`](figures/accuracy_vs_n.html) | RMSE, NLPD et couverture contre n_HF, multifidélité et HF seule |
| [`hyperparameters.html`](figures/hyperparameters.html) | Valeurs ajustées de l, t1, t2 et du bruit, bornes de recherche, mode des priors |
| [`many_points.html`](figures/many_points.html) | Régime à beaucoup de points (dégénérescence du MLE) |
| [`optimization.html`](figures/optimization.html) | Convergence à budget égal et recommandation finale par graine |
| [`sensitivity.html`](figures/sensitivity.html) | Sensibilité, un facteur à la fois |
| [`prior_shapes.html`](figures/prior_shapes.html) | Forme des priors par défaut (p(θ) / max) et valeurs ajustées |

## Références

* **[S]** M. Sacher et al., *A Non-Nested Infilling Strategy for Multi-Fidelity based Efficient Global Optimization*,
  Int. J. Uncertainty Quantification (2021) — problème Hartmann multifidélité (éq. 30–32).
* **[RW]** C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press (2006), chap. 5.
* `benchmarks/map_hartmann/RAPPORT_MAP.md` — étude de la prior sur les longueurs (même protocole).
