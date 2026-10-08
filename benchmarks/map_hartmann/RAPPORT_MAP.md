# Estimation MAP des longueurs de corrélation — benchmark Hartmann 6D

08/10/2026 · `mfego` avec l'option `use_map` comparé au maximum de vraisemblance (MLE) et à BoTorch ·
dossier `benchmarks/map_hartmann/`

Les figures interactives sont dans [`figures/`](figures/) (à ouvrir dans un navigateur ; plotly.js est chargé depuis le CDN,
thème sombre automatique). Les résultats bruts sont dans [`results/`](results/).

---

## 1. Résumé

* **Le MAP corrige le principal défaut de mfego quand il y a peu de points.** Sans prior, 42 à 70 % des longueurs de
  corrélation ajustées partent sur la borne haute (l = 10) dès que n_HF ≤ 20 : le modèle devient presque plat et trop sûr
  de lui (couverture médiane de l'intervalle à 95 % entre 0,44 et 0,69). Avec la prior InvGamma IG(3, 2), aucune
  longueur n'atteint une borne et la couverture remonte à 0,84–0,90.
* **Précision :** le MAP fait mieux que le MLE sur 8 à 10 graines sur 10, pour chaque taille d'échantillon et pour la RMSE
  comme pour la NLPD. À partir de 10 points HF, mfego MAP égale ou dépasse BoTorch en RMSE ; BoTorch reste mieux calibré
  avec 5 et 10 points HF.
* **Optimisation (budget 250, 10 graines) :** erreur médiane de la recommandation finale 0,21 pour NN-MF-EGO MAP, contre
  0,30 sans prior et 0,71 pour BoTorch qLogEI. Le MAP fait mieux que le MLE sur 9 graines sur 10.
* **Avec peu de points, la longueur ajustée est en pratique le mode de la prior.** Le fit MLE n'atteint qu'une
  log-vraisemblance supérieure de 1 à 2,5 nats à celle du point MAP, alors que la prior pénalise de plusieurs nats *par
  dimension* tout écart au mode. C'est donc le **mode β/(α+1)** qui fixe l'estimation ; la force α compte beaucoup moins.
* **Le mode par défaut (0,5) est un peu trop grand pour Hartmann**, dont les longueurs de référence valent 0,2 à 0,45.
  IG(3, 1), de mode 0,25, fait mieux en NLPD sur 33 cas sur 40 et en optimisation (erreur médiane 0,14 contre 0,21,
  différence non significative avec 10 graines). La règle des queues de Betancourt — 1 % de masse sous 0,1 et 1 % au-dessus
  de 2 — donne **IG(2,9 ; 0,83)**, soit presque IG(3, 1), **sans regarder les résultats**.
* **L'erreur de réglage est asymétrique.** Un mode trop petit (0,1) donne un modèle peu informatif mais honnête ; un mode
  trop grand (1 ou 2) donne un modèle trop sûr de lui (couverture 0,58–0,83) et dégrade fortement l'optimisation (erreur
  médiane 0,67 et 2,17). **En cas de doute, viser petit.**
* **À mode égal, la loi** (InvGamma, Gamma, LogNormale) **change peu les résultats**. Elle compte par ses queues, ce qui
  oriente le choix selon le cas (§ 8).
* **Point ouvert, indépendant du MAP :** à partir de 80 points, l'optimiseur MLE de mfego s'arrête sur une solution dégénérée
  (longueurs ≈ 0,01, modèle « bruit blanc ») alors qu'un bien meilleur optimum existe (§ 6).

**Recommandation.** Utiliser `use_map=True`. Passer `LENGTHSCALE_PRIOR` à (3, 1), ou au réglage donné par la règle des
queues, après vérification sur un second problème (hydrofoil). Normaliser les entrées dans [0, 1] avant tout usage hors de
ce benchmark : la prior, comme les bornes existantes, suppose des entrées dans [0, 1].

---

## 2. Ce qui a été implémenté

### 2.1 Dans `mfego/src/surrogate_models.py`

* Une seule fonction `GaussianProcess.negative_log_likelihood(..., with_grad=False, use_map=False)`. Avec `use_map=True`,
  elle ajoute pour chaque longueur de corrélation l_m (m = 1..d) le terme −log p(l_m) d'une loi inverse-gamma
  InvGamma(α, β), constante omise :

  > −log p(l) = (α + 1) ln l + β / l  ·  dérivée par rapport à ln l : (α + 1) − β / l

  Le gradient reste analytique (vérifié par `check_grad`). La prior ne porte que sur l_1..l_d : la variance du signal t1, la
  variance de biais t2 et le bruit restent libres entre leurs bornes. Elle ne dépend pas de ρ : le ρ profilé et
  l'argument d'enveloppe du gradient restent valables.
* Le MAP est pris **en espace l** (densité sur l, sans jacobien de la paramétrisation log), comme dans GPyTorch et BoTorch.
  Le point visé est donc le mode de p(l), β/(α+1) ; en espace ln l, ce serait β/α.
* Constante `LENGTHSCALE_PRIOR = (3.0, 2.0)` ; option `use_map` sur `GaussianProcess` et `MultifidelityModel`. Elle vaut
  `False` par défaut (comportement inchangé) et elle est sauvegardée dans le JSON (`to_dict` / `from_dict`).
* Tests ajoutés ou étendus : gradient avec et sans MAP, terme de prior comparé à `scipy.stats.invgamma`, aller-retour JSON,
  précision sur Forrester. La suite complète passe (95 tests).

### 2.2 Dans ce dossier

| Fichier | Rôle |
|---|---|
| `run_map_benchmark.py` | Précision et optimisation : mfego MLE contre MAP contre BoTorch |
| `run_prior_sensitivity.py` | Sensibilité à α, β et à la loi ; longueurs de référence ; diagnostic du MLE |
| `priors.py` | Lois InvGamma, Gamma, LogNormale (valeur, gradient, quantiles) et `PriorGP` |
| `make_figures.py` | Les 7 figures interactives de `figures/` |
| `results/` | CSV et JSON bruts, `summary.txt` |

Les lois autres que l'InvGamma ne sont testées que dans ce dossier, par la sous-classe `priors.PriorGP` : le cœur de mfego
n'est pas modifié. Contrôle : `PriorGP` avec IG(3, 2) reproduit exactement `use_map=True` (écart de prédiction nul).

Reproduction, depuis la racine du dépôt avec l'environnement `.venv-benchmark` (environ 2 min sur 32 cœurs) :

```
.venv-benchmark\Scripts\python benchmarks\map_hartmann\run_map_benchmark.py
.venv-benchmark\Scripts\python benchmarks\map_hartmann\run_prior_sensitivity.py
.venv-benchmark\Scripts\python benchmarks\map_hartmann\make_figures.py
```

L'option `--mfkg` de `run_map_benchmark.py` ajoute BoTorch MF-KG (environ 25 min par run).

---

## 3. Protocole

* **Problème :** Hartmann 6D multifidélité de Sacher et al. (2021, Éq. 30–32), `bench_lib.HartmannMF`. Niveau basse fidélité
  (BF) = U₁ décalé de δ = 0,05 ; niveau haute fidélité (HF) = Hartmann. Coûts 1 et 10, entrées dans [0, 1]⁶.
* **Précision :** n_HF ∈ {5, 10, 20, 40} et n_BF = 2 n_HF, plans LHS indépendants par niveau ; 10 graines ; le même plan
  pour tous les modèles d'une graine, ce qui permet une comparaison appariée ; 2000 points de test LHS. Métriques :
  * RMSE relative : RMSE divisée par l'écart-type de f (1 = pas mieux que la moyenne) ;
  * NLPD : log-vraisemblance négative prédictive moyenne. Plus bas = mieux ; elle pénalise une incertitude trop faible ;
  * couverture : part des points de test dans l'intervalle prédit à 95 % (idéal 0,95).
* **Optimisation :** plan initial de coût 120 (MF : 20 BF + 10 HF ; SF : 12 HF), budget total 250, 10 graines. Erreur = f − f*
  de la meilleure observation HF et de la recommandation finale. La recommandation est le minimum de la moyenne prédite,
  trouvé par la même évolution différentielle pour toutes les méthodes.
* **Modèles :** mfego MF et SF sans prior (MLE) et avec prior (MAP, IG(3, 2)) ; BoTorch SingleTaskMultiFidelityGP (MF) et
  SingleTaskGP (SF) ; en optimisation, BoTorch qLogEI (SF). Les deux modèles BoTorch font déjà du MAP avec leurs priors par
  défaut (§ 8.2). MF-KG n'a pas été relancé ; ses résultats du notebook sont rappelés au § 5.

---

## 4. Précision du modèle de substitution

Figures : [`precision_vs_n.html`](figures/precision_vs_n.html) · [`map_vs_mle_par_graine.html`](figures/map_vs_mle_par_graine.html) ·
[`longueurs_correlation.html`](figures/longueurs_correlation.html)

Médianes sur 10 graines, dans chaque case : **RMSE relative · NLPD · couverture 95 %**.

| Multifidélité | n_HF = 5 | n_HF = 10 | n_HF = 20 | n_HF = 40 |
|---|---|---|---|---|
| mfego MLE (sans prior) | 1,08 · 33,2 · 0,44 | 1,10 · 5,45 · 0,66 | 1,03 · 12,7 · 0,62 | 0,93 · 1,24 · 0,83 |
| **mfego MAP** IG(3, 2) | 1,01 · 2,31 · 0,84 | **0,95** · 0,73 · 0,90 | 0,92 · **1,06** · 0,87 | **0,78** · **0,53** · 0,89 |
| BoTorch MF-GP | 1,24 · **0,73** · 0,99 | 0,97 · **0,50** · 0,95 | **0,91** · 1,24 · 0,86 | 0,82 · 0,81 · 0,87 |

| HF seule | n_HF = 5 | n_HF = 10 | n_HF = 20 | n_HF = 40 |
|---|---|---|---|---|
| mfego MLE (sans prior) | 1,09 · 20,5 · 0,57 | 1,05 · 7,37 · 0,63 | 1,01 · 5,27 · 0,69 | 1,03 · 1,55 · 0,80 |
| **mfego MAP** IG(3, 2) | 1,02 · 2,44 · 0,84 | **0,96** · **0,70** · 0,90 | **0,94** · **1,10** · 0,86 | **0,81** · **0,58** · 0,88 |
| BoTorch SingleTaskGP | 1,02 · **1,97** · 0,86 | 0,99 · 0,95 · 0,85 | 0,99 · 3,16 · 0,76 | 0,90 · 1,15 · 0,85 |

Temps d'ajustement médian en multifidélité : 0,04 à 0,21 s pour mfego MAP, 0,14 à 0,31 s pour mfego MLE et 1,8 à 4,0 s
pour BoTorch.

**Graines où le MAP fait mieux que le MLE** (même plan d'expériences) :

| n_HF | MF · RMSE | MF · NLPD | SF · RMSE | SF · NLPD |
|---|---|---|---|---|
| 5 | 9/10 | 10/10 | 9/10 | 10/10 |
| 10 | 9/10 | 10/10 | 8/10 | 10/10 |
| 20 | 10/10 | 10/10 | 10/10 | 9/10 |
| 40 | 10/10 | 10/10 | 10/10 | 9/10 |

**Longueurs de corrélation sur une borne** (niveau HF, toutes dimensions et graines) :

| n_HF | MF MLE : borne haute / basse | SF MLE : borne haute / basse | MAP (MF et SF) |
|---|---|---|---|
| 5 | 70 % / 0 % | 65 % / 0 % | 0 % |
| 10 | 60 % / 0 % | 45 % / 17 % | 0 % |
| 20 | 50 % / 0 % | 42 % / 10 % | 0 % |
| 40 | 18 % / 5 % | 22 % / 5 % | 0 % |

**Lecture.**

* Avec peu de points, le maximum de vraisemblance préfère des longueurs très grandes. Le modèle devient presque
  linéaire, sa variance prédite est faible, et la NLPD explose (33 à 5 points HF). Ce n'est pas un défaut de l'optimiseur :
  ce maximum est réel (§ 6). La prior pénalise l = 10 de 8,2 nats par dimension alors que la vraisemblance n'y gagne que
  1 à 2,5 nats au total : les longueurs restent près du mode (médianes 0,44 à 0,49).
* À 5 points HF, aucun modèle ne prédit Hartmann mieux que sa moyenne (RMSE relative ≈ 1). Les écarts portent alors sur la
  calibration : c'est la NLPD et la couverture qu'il faut regarder.
* BoTorch MF-GP est le mieux calibré à 5 et 10 points HF. Sa RMSE est la plus mauvaise à 5 points (1,24), mais son
  incertitude large le couvre (couverture 0,99).

---

## 5. Optimisation à budget égal

Figure : [`optimisation.html`](figures/optimisation.html)

| Méthode | Meilleure obs. HF, médiane (moyenne) | Recommandation, médiane (moyenne) | Itérations (médiane) | Part BF (médiane) | Temps / itération |
|---|---|---|---|---|---|
| **NN-MF-EGO · MAP** | **0,22** (0,50) | **0,21** (0,46) | 14 | 7 % | 0,21 s |
| NN-MF-EGO · MLE | 0,33 (0,82) | 0,30 (0,92) | 36 | 70 % | 0,31 s |
| SF-EGO · MAP | 0,36 (0,51) | 0,37 (0,49) | 13 | — | 0,17 s |
| SF-EGO · MLE | 1,53 (1,33) | 1,65 (1,48) | 13 | — | 0,19 s |
| BoTorch qLogEI (SF) | 0,62 (0,98) | 0,71 (1,00) | 13 | — | 0,73 s |
| *BoTorch MF-KG (notebook, graines 0–4)* | *2,45* | *1,23* | *75* | *91 %* | *18,6 s* |

Le MAP fait mieux que le MLE sur 9 graines sur 10, pour NN-MF-EGO comme pour SF-EGO et sur les deux critères. Les valeurs de
MF-KG viennent de `benchmarks/hartmann_benchmark.ipynb`, avec le même protocole mais sur 5 graines seulement.

**Effet sur l'usage de la basse fidélité.** Avec IG(3, 2), NN-MF-EGO n'utilise presque plus la BF (7 % contre 70 %).
L'avantage du multifidélité sur le monofidélité se réduit alors : erreur médiane 0,21 contre 0,37, mais moyennes presque
égales (0,46 contre 0,49). Cette part dépend du mode de la prior : elle remonte à 31 % avec IG(3, 1) et Gamma(3, 6),
de modes 0,25 et 0,33 (§ 7). Le mécanisme exact (rapport des réductions de variance dans le mérite AEI) reste à analyser.

---

## 6. Pourquoi le MLE de mfego échoue : deux régimes

Figure : [`diagnostic_mle.html`](figures/diagnostic_mle.html)

Test : on ajuste un GP seul sur n points de Hartmann, avec et sans prior, puis on évalue la NLL *sans prior* aux deux
points obtenus. Si la NLL au point MAP est plus basse que celle atteinte par le fit MLE, l'optimiseur MLE a raté un meilleur
optimum.

| n (points HF) | 5 | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|---|
| Graines où l'optimiseur MLE rate un meilleur optimum | 0/10 | 0/10 | 1/10 | 2/10 | 5/10 | 10/10 |
| NLL(fit MLE) − NLL(point MAP), médiane | −1,05 | −2,06 | −2,45 | −1,71 | +6,4 | +74,0 |
| RMSE relative MLE → MAP (moyenne) | 1,23 → 1,08 | 1,17 → 0,98 | 1,02 → 0,92 | 1,06 → 0,83 | 0,86 → 0,67 | 1,00 → 0,51 |
| Longueur médiane MLE → MAP (moyenne) | 8,75 → 0,49 | 5,92 → 0,47 | 4,82 → 0,48 | 1,06 → 0,44 | 0,20 → 0,37 | 0,01 → 0,35 |

* **n ≤ 40 :** l'optimum MLE est réel. Il est meilleur en vraisemblance que le point MAP, mais seulement de 1 à 2,5 nats :
  c'est un problème d'identifiabilité et de surajustement, et la prior est la bonne correction.
* **n ≥ 80 :** l'optimiseur MLE de mfego s'arrête sur une solution dégénérée (l ≈ 0,01 : la covariance devient diagonale et
  le modèle se réduit à du bruit blanc, RMSE ≈ 1). Avec 300 points, sa NLL vaut 425,7 contre 220,8 à l'optimum trouvé par
  scikit-learn (RMSE 1,00 contre 0,40). Causes probables, non vérifiées : bruit initial 10⁻⁶ et matrice très mal
  conditionnée avec beaucoup de points ; gradient nul renvoyé en cas d'échec de Cholesky ; borne haute du bruit (10⁻²)
  sous la valeur estimée par scikit-learn (2,7·10⁻²). C'est à corriger séparément. Le MAP évite ce piège ici.
* Conséquence pour le multifidélité : le niveau BF a 2 n_HF points, soit 80 à n_HF = 40. Le MLE multifidélité à
  n_HF = 40 est donc en partie touché par ce second régime.

Les longueurs de référence utilisées ailleurs dans ce rapport viennent pour cette raison de scikit-learn (300 points, ARD RBF,
redémarrages) :

| Niveau | l₁ | l₂ | l₃ | l₄ | l₅ | l₆ |
|---|---|---|---|---|---|---|
| HF (Hartmann) | 0,27 | 0,45 | 4,72 | 0,35 | 0,31 | 0,32 |
| BF (U₁) | 0,21 | 0,37 | 0,87 | 0,21 | 0,27 | 0,23 |
| Résidu δ = y_HF − ρ y_BF | 0,26 | 0,35 | 0,63 | 0,20 | 0,29 | 0,23 |

La dimension 3 est presque inactive au niveau HF. Les autres longueurs valent 0,2 à 0,45.

---

## 7. Choix de α et β

Figures : [`sensibilite_prior.html`](figures/sensibilite_prior.html) · [`formes_des_priors.html`](figures/formes_des_priors.html)

### 7.1 Ce que règlent α et β

* **Le mode m = β/(α + 1)** est la valeur vers laquelle la prior tire l'estimation quand les données sont faibles.
* **α règle la concentration.** Rapport des quantiles 95 % et 5 % : 22 pour α = 1,5 ; 7,7 pour α = 3 ; 2,9 pour α = 10.
* **Les queues sont asymétriques.** À gauche, la queue est très légère (e^(−β/l)) : avec IG(3, 2), descendre à l = 0,01
  coûte 180 nats. À droite, elle est lourde (l^−(α+1)) : monter à l = 10 ne coûte que 8,2 nats. Ces deux propriétés font
  de l'InvGamma un bon choix par défaut : elle interdit les pics d'interpolation et laisse une dimension inactive s'éteindre.

### 7.2 Résultats de sensibilité (NN-MF-EGO, 10 graines)

| Prior | Mode | NLPD médiane (n_HF = 10 / 40) | RMSE (n_HF = 40) | Longueur HF ajustée (n_HF = 10 → 40) | Recommandation, médiane [quartiles] | Part BF |
|---|---|---|---|---|---|---|
| MLE (sans prior) | — | 5,45 / 1,24 | 0,93 | sur les bornes | 0,30 [0,30–1,72] | 70 % |
| IG(3, 0.4) | 0,10 | 0,67 / 0,51 | 1,00 | 0,10 → 0,10 | 0,55 [0,25–1,01] | 28 % |
| **IG(3, 1)** | **0,25** | **0,62 / 0,32** | **0,74** | 0,25 → 0,34 | **0,14** [0,05–0,41] | 31 % |
| IG(3, 2) *(défaut)* | 0,50 | 0,73 / 0,53 | 0,78 | 0,49 → 0,45 | 0,21 [0,16–0,51] | 7 % |
| IG(3, 4) | 1,00 | 1,44 / 1,12 | 0,86 | 0,92 → 0,63 | 0,67 [0,50–0,89] | 0 % |
| IG(3, 8) | 2,00 | 3,19 / 2,38 | 1,15 | 1,91 → 1,17 | 2,17 [1,38–2,77] | 10 % |
| IG(1.5, 1.25) *(faible)* | 0,50 | 0,75 / 0,52 | 0,79 | 0,49 → 0,45 | 0,30 [0,17–0,46] | 17 % |
| IG(10, 5.5) *(forte)* | 0,50 | 0,69 / 0,41 | 0,77 | 0,49 → 0,45 | 0,44 [0,30–0,56] | 4 % |
| Gamma(3, 4) | 0,50 | 0,76 / 0,52 | 0,81 | 0,50 → 0,45 | 0,40 [0,15–0,71] | 9 % |
| LogN(−0.13, 0.75) | 0,50 | 0,72 / 0,57 | 0,82 | 0,50 → 0,45 | 0,23 [0,17–1,05] | 0 % |
| Gamma(3, 6), défaut BoTorch MF | 0,33 | **0,61 / 0,41** | 0,77 | 0,33 → 0,41 | 0,17 [0,13–0,58] | 31 % |
| LogN(2.31, 1.73), défaut BoTorch | 0,50 | 1,08 / 0,89 | 0,87 | 0,76 → 0,44 | 0,42 [0,25–1,24] | 4 % |

Couverture moyenne : 0,86–0,93 pour IG(3, 1) et 0,86–0,94 pour IG(3, 0.4), contre 0,84–0,90 pour IG(3, 2), 0,71–0,83 pour
IG(3, 4) et 0,58–0,69 pour IG(3, 8).

### 7.3 Lecture

1. **L'estimation est le mode.** Pour toutes les priors, les longueurs ajustées restent au mode : 0,10 → 0,10 ;
   0,25 → 0,25–0,34 ; 2 → 1,2–2,0. Les données ne commencent à tirer qu'à n_HF = 40 (IG(3, 4) : 0,92 → 0,63). Avec les
   tailles d'échantillon courantes en optimisation multifidélité, **choisir α et β revient à choisir l'échelle de
   corrélation**.
2. **Le meilleur mode est proche de l'échelle réelle.** IG(3, 1) (mode 0,25), dans la plage de référence 0,2–0,45, donne la
   meilleure NLPD (mieux qu'IG(3, 2) sur 33 cas sur 40) et la meilleure optimisation (0,14). Son gain porte surtout sur la
   calibration : sa RMSE n'est meilleure qu'IG(3, 2) que sur 15 cas sur 40, et seulement nettement à n_HF = 40. En
   optimisation, l'écart avec IG(3, 2) (6 graines sur 10) n'est pas significatif.
3. **L'erreur est asymétrique.** Un mode trop petit (0,1) donne une RMSE ≈ 1 : le modèle n'apprend presque rien mais reste
   honnête (couverture 0,86–0,94). Un mode trop grand (1 ou 2) rend le modèle trop sûr de lui, et l'EI exploite alors de
   mauvaises régions : erreur de recommandation 0,67 et 2,17. **Mieux vaut sous-estimer l'échelle que la surestimer.**
4. **La force α compte peu pour la précision.** À mode 0,5, α = 1,5, 3 ou 10 donnent des NLPD proches. Une prior forte
   (α = 10) dégrade en revanche l'optimisation (0,44 ; jamais meilleure qu'IG(3, 2)) : elle empêche chaque dimension de
   s'adapter, comme la dimension 3, presque inactive. Un α de 2 à 3 est un bon compromis.

### 7.4 Comment régler α et β en pratique

**Règle des queues** (Betancourt, *Robust Gaussian Processes in Stan*). On choisit α et β pour que
P(l < l_min) = P(l > l_max) = 1 %, avec :

* l_min = plus petite échelle qu'on accepte de modéliser (de l'ordre de l'espacement des points, ou 0,1 du domaine) ;
* l_max = un peu plus que la largeur du domaine (au-delà, la dimension est en pratique inactive).

| (l_min ; l_max) | α | β | Mode |
|---|---|---|---|
| (0,1 ; 1) | 4,63 | 1,10 | 0,20 |
| **(0,1 ; 2)** | **2,94** | **0,83** | **0,21** |
| (0,1 ; 5) | 1,91 | 0,65 | 0,22 |
| (0,05 ; 2) | 2,10 | 0,34 | 0,11 |

Cette règle ne dépend pas des résultats du benchmark. Avec (0,1 ; 2), elle donne IG(2,9 ; 0,83), très proche du meilleur
réglage mesuré, IG(3, 1). Le réglage actuel IG(3, 2) a son quantile à 1 % à 0,24 : il interdit presque les longueurs
inférieures à 0,24, alors que celles de Hartmann valent 0,2 à 0,45. Sa queue gauche mord sur les vraies échelles. IG(3, 1) a
ses quantiles à 1 % et 99 % à 0,12 et 2,3.

**Réglages conseillés** (entrées dans [0, 1]) :

* **Par défaut :** IG(3, 1), ou la règle des queues avec (0,1 ; 2).
* **Fonction réputée lisse** (réponse quasi quadratique, peu d'oscillations) : mode 0,5 à 1, soit IG(3, 2) à IG(3, 4).
* **Fonction à variations rapides :** mode 0,1 à 0,2, à condition d'avoir assez de points pour résoudre cette échelle.
* **À valider sur le cas hydrofoil** avant de changer la valeur par défaut du code.

---

## 8. Choix de la loi selon les cas

### 8.1 À mode égal, la loi compte par ses queues

À mode 0,5, IG(3, 2), Gamma(3, 4) et LogN(−0,13 ; 0,75) donnent des longueurs ajustées identiques (0,45–0,50) et des NLPD
proches (0,72–0,76 à n_HF = 10). Dans ce régime, la loi compte beaucoup moins que le mode. Elle se distingue par le
comportement de ses queues :

| Loi | Petites longueurs (queue gauche) | Grandes longueurs (queue droite) | Conséquence |
|---|---|---|---|
| InvGamma(α, β) | très légère : l ≪ β presque interdit | lourde, en l^−(α+1) | empêche le surajustement ; laisse une dimension inactive s'éteindre |
| Gamma(k, taux) | lourde, en l^(k−1) | légère, en e^(−taux·l) | autorise les petites longueurs, interdit les grandes : une dimension inactive est ramenée vers le mode (Gamma(3, 6) : 20,9 nats pour l = 4,7) |
| LogNormale(μ, σ) | quadratique en ln l | quadratique en ln l | symétrique en échelle log ; σ se lit comme « l connu à un facteur e^σ près » |
| Log-uniforme (bornes seules) | — | — | aucune régularisation : c'est le MLE |

### 8.2 Les priors par défaut de BoTorch

Valeurs vérifiées dans les versions installées (botorch 0.18.1, gpytorch 1.15.2) :

* **SingleTaskGP :** LogNormale(√2 + ln(d)/2 ; √3) sur chaque longueur (Hvarfner et al., 2024), soit LogN(2,31 ; 1,73) pour
  d = 6 : médiane 10, mode 0,50 en espace l, très large (quantiles 5 % et 95 % : 0,58 et 174). S'y ajoutent une prior
  LogNormale(−4, 1) sur le bruit et une variance du signal fixée (sorties standardisées, sans variance de signal libre).
* **SingleTaskMultiFidelityGP :** Gamma(3, 6) sur les longueurs (noyau de Matérn 5/2), de mode 0,33, et Gamma(2 ; 0,15) sur
  la variance du signal.

Transposées dans mfego :

* **Gamma(3, 6)** est parmi les meilleures : NLPD 0,41 à n_HF = 40, recommandation 0,17.
* **La LogNormale de SingleTaskGP fonctionne mal** : NLPD 4,5 à n_HF = 20, couverture 0,72–0,85, recommandation 0,42. Sa
  queue droite très lourde laisse les longueurs dériver (médiane 1,05 à n_HF = 20) et le modèle devient trop sûr de lui.

Une prior ne se transpose pas seule : celle de SingleTaskGP est conçue avec une variance de signal fixe et pour la grande
dimension, où de grandes longueurs sont nécessaires.

### 8.3 Recommandations par cas

| Cas | Loi conseillée | Réglage indicatif (entrées dans [0, 1]) | Pourquoi |
|---|---|---|---|
| Peu de points (n ≲ 10 d), échelle inconnue, dimensions peut-être inactives (cas courant de mfego) | InvGamma | IG(3, 1), ou règle des queues (0,1 ; 2) → IG(2,9 ; 0,83) | La queue gauche interdit le surajustement ; la queue droite laisse une dimension inactive s'éteindre |
| Échelle connue a priori (campagne précédente, physique du problème) | LogNormale | médiane = échelle connue, σ ≈ 0,5 à 1 | Centre la prior sur l'information disponible, avec une incertitude symétrique en facteur |
| Toutes les dimensions actives et MLE qui part vers la borne haute (modèle trop plat) | Gamma | Gamma(3, 6), mode 0,33 | La queue droite légère interdit les modèles plats ; à éviter si une dimension peut être inactive |
| Grande dimension (d ≳ 20) | LogNormale dépendant de d, ou prior parcimonieuse | LogN(√2 + ln(d)/2 ; √3) ; SAAS : demi-Cauchy sur 1/l² | Les longueurs utiles croissent avec d ; SAAS si seules quelques dimensions comptent |
| Beaucoup de points (n ≫ 10 d) | InvGamma faible | α ≈ 1,5 à 2 | La vraisemblance domine ; la prior sert de garde-fou et évite l'échec de l'optimiseur MLE (§ 6) |
| Fonction réputée lisse | InvGamma | mode 0,5 à 1 : IG(3, 2) à IG(3, 4) | Connaissance a priori de la régularité |
| Entrées non normalisées | — | normaliser dans [0, 1], ou β_j = β · (b_j − a_j) par dimension | La prior et les bornes supposent [0, 1] ; le GP de mfego reçoit aujourd'hui les entrées brutes |
| Plusieurs niveaux de fidélité | même loi, réglage par niveau possible | résidu δ souvent plus lisse que la BF : mode égal ou plus grand | Ici, δ et BF ont des échelles proches (0,2 à 0,6) : une prior commune suffit |
| Loi normale sur l (première idée) | à éviter | — | Densité non nulle en l = 0 et pour l < 0, donc pas de protection contre les petites longueurs. Une gaussienne se met sur ln l : c'est la LogNormale |

---

## 9. Limites et suites

* **Un seul problème** (Hartmann 6D : d = 6, fonction lisse, échelles ≈ 0,3) et 10 graines. Les différences d'optimisation
  entre priors voisines ne sont pas significatives.
* **La prior ne porte que sur les longueurs.** t1, t2 et le bruit restent libres entre leurs bornes, alors que BoTorch met
  aussi une prior sur le bruit. Pistes : LogNormale sur le bruit, prior centrée sur 1 pour t1 (sorties normalisées) et prior
  sur t2, mal identifiée.
* **Les entrées ne sont pas normalisées** dans mfego. C'est indispensable avant d'utiliser le MAP sur le cas hydrofoil.
* **Part de la basse fidélité :** elle dépend du mode de la prior (7 % à 31 %), et l'avantage du multifidélité avec elle.
  Le mécanisme reste à comprendre.
* **L'optimiseur MLE échoue à partir de 80 points** (§ 6) : à corriger séparément.
* **MF-KG n'a pas été relancé** (environ 25 min par run).

Suites proposées, par ordre de priorité :

1. Vérifier IG(3, 1) sur le cas hydrofoil, puis changer `LENGTHSCALE_PRIOR`.
2. Rendre (α, β) configurables par modèle et par niveau.
3. Normaliser les entrées dans le GP.
4. Ajouter des priors sur le bruit, t1 et t2.
5. Corriger l'optimiseur MLE.

---

## Annexe A — Paramètres des priors testées (l en unités du domaine [0, 1])

| Prior | Mode | Quantile 5 % | Médiane | Quantile 95 % | P(l < 0,1) | P(l > 10) |
|---|---|---|---|---|---|---|
| IG(3, 0.4) | 0,10 | 0,064 | 0,15 | 0,49 | 23,8 % | 0 % |
| IG(3, 1) | 0,25 | 0,16 | 0,37 | 1,22 | 0,3 % | 0 % |
| IG(3, 2) | 0,50 | 0,32 | 0,75 | 2,45 | 0 % | 0,1 % |
| IG(3, 4) | 1,00 | 0,64 | 1,50 | 4,89 | 0 % | 0,8 % |
| IG(3, 8) | 2,00 | 1,27 | 2,99 | 9,78 | 0 % | 4,7 % |
| IG(1.5, 1.25) | 0,50 | 0,32 | 1,06 | 7,11 | 0 % | 3,1 % |
| IG(10, 5.5) | 0,50 | 0,35 | 0,57 | 1,01 | 0 % | 0 % |
| Gamma(3, 4) | 0,50 | 0,20 | 0,67 | 1,57 | 0,8 % | 0 % |
| LogN(−0.13, 0.75) | 0,50 | 0,26 | 0,88 | 3,01 | 0,2 % | 0,1 % |
| Gamma(3, 6), BoTorch MF | 0,33 | 0,14 | 0,45 | 1,05 | 2,3 % | 0 % |
| LogN(2.31, 1.73), BoTorch | 0,50 | 0,58 | 10,1 | 174 | 0,4 % | 50,2 % |

## Annexe B — Formules (MAP en espace l, constantes omises)

| Loi | −log p(l) | Dérivée par rapport à ln l | Mode |
|---|---|---|---|
| InvGamma(α, β) | (α + 1) ln l + β / l | (α + 1) − β / l | β / (α + 1) |
| Gamma(k, taux r) | −(k − 1) ln l + r l | −(k − 1) + r l | (k − 1) / r |
| LogNormale(μ, σ) | ln l + (ln l − μ)² / (2σ²) | 1 + (ln l − μ) / σ² | e^(μ − σ²) |

## Figures

| Figure | Contenu |
|---|---|
| [`precision_vs_n.html`](figures/precision_vs_n.html) | RMSE, NLPD et couverture en fonction de n_HF : mfego MAP, mfego MLE, BoTorch (MF et HF seule) |
| [`map_vs_mle_par_graine.html`](figures/map_vs_mle_par_graine.html) | MLE contre MAP, graine par graine (sous la diagonale, le MAP fait mieux) |
| [`longueurs_correlation.html`](figures/longueurs_correlation.html) | Longueurs ajustées, bornes, mode de la prior et références |
| [`optimisation.html`](figures/optimisation.html) | Convergence, erreur de recommandation, part de la basse fidélité |
| [`sensibilite_prior.html`](figures/sensibilite_prior.html) | Sensibilité à α, β et à la loi (précision et optimisation) |
| [`formes_des_priors.html`](figures/formes_des_priors.html) | Densités des priors testées et longueurs de référence |
| [`diagnostic_mle.html`](figures/diagnostic_mle.html) | Les deux régimes d'échec du MLE de mfego |

## Références

* **[S]** M. Sacher et al., *A Non-Nested Infilling Strategy for Multi-Fidelity based Efficient Global Optimization*,
  Int. J. Uncertainty Quantification (2021) — problème Hartmann multifidélité (Éq. 30–32).
* **[RW]** C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press (2006), chap. 5.
* M. Betancourt, *Robust Gaussian Processes in Stan*, étude de cas Stan (2017) — règle des queues pour la prior inverse-gamma.
* C. Hvarfner, E. O. Hellsten, L. Nardi, *Vanilla Bayesian Optimization Performs Great in High Dimensions*, ICML (2024) —
  prior LogNormale dépendant de la dimension (défaut de BoTorch).
* D. Eriksson, M. Jankowiak, *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*, UAI (2021) — prior
  SAAS.
