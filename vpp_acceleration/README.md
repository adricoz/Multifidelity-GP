# vpp_acceleration — plan d'expériences séquentiel pour le co-krigeage

19/10/2026 · analyse des deux références de [`references/`](references/) au regard de la classe `CoKriging` de mfego
([`mfego/src/cokriging.py`](../mfego/src/cokriging.py)) · 21/10/2026 : **plan séquentiel implémenté et testé sur
Forrester** ([`src/sequential_design.py`](src/sequential_design.py), voir le § 0 ci-dessous)

---

## 0. Le code (21/10/2026)

> **Nouveau (21/10/2026, soir) : couplage avec la VPP MDS (branche `new_master`) et raffinement ciblé VPP**, testés sur un
> méta-surrogate 16D (12 modes PCA + cant, rake, yaw, Hvol) : voir [`RAPPORT_VPP_CIBLE.md`](RAPPORT_VPP_CIBLE.md).
> Le ciblé divise par 5 l'erreur de vitesse face à un LHS de même coût (3 graines sur 3). Le rapport chiffre aussi
> N_sobol (16 points par état), le plan de départ (128 BF + 32 HF) et la CFD de validation (20 à 40 calculs aux
> états VPP).

### Ce qui est implémenté

[`src/sequential_design.py`](src/sequential_design.py) construit le plan d'expériences séquentiel du co-krigeage. Le
surrogate reste la classe `CoKriging` de mfego, utilisée en lecture seule. Le module décide **où** calculer et **avec
quels codes**. Il suit l'algorithme de la note ([`doc/build/sequential_cokriging.pdf`](doc/build/sequential_cokriging.pdf)),
en trois parties :

| Partie | Fonctions | Équation de la note |
| --- | --- | --- |
| 1. Grandeurs lues dans le co-krigeage | `level_weights` (P_j), `level_variances` (σ_j²), `loo_errors`, `loo_squared_errors` (z²), `most_correlated_point`, `adjustment_factors` (a_j), `adjusted_variance` (k_adj), `joint_posterior_covariance`, `imse_gains` (G_l), `choose_level` (l*) | N2 à N6 |
| 2. Où calculer | `find_next_point` : maximum de k_adj par évolution différentielle, avec contraintes connues et exclusion des points en échec | N4 |
| 3. La boucle | `SequentialCoKriging` : `ask` (ajuster, LOO, point, niveau), `evaluate` (lancer les niveaux manquants, du moins cher au plus cher), `run` (répéter jusqu'au budget) ; `evaluate_initial_design`, `stored_value` | — |

**Un écart avec la note, voulu.** Le gain G_l (N5) est calculé **exactement avec la covariance jointe du
co-krigeage**, et non avec la formule par niveau de la note. Les deux coïncident quand la basse fidélité est sans
bruit, ce que vérifie un test. Mais le bruit appris par mfego, pourtant minuscule (2·10⁻⁶ en unités normalisées),
suffit à écarter la formule par niveau de 26 à 70 % sur Forrester avec le noyau exponentiel carré. Ce noyau est très
mal conditionné, et l'écart grandit avec le bruit : 10⁻⁴ pour un bruit de 10⁻⁸, 10⁻⁶ pour 10⁻¹⁰. La décomposition par
niveau ne sert plus qu'à la variance ajustée du choix du point, qui est de toute façon une heuristique.

**Deux comportements voulus :**

* l'algorithme peut choisir un point déjà calculé en basse fidélité. Il n'y lance alors que la haute fidélité, et ne
  compte que son coût ;
* après un calcul raté (valeur non finie), les niveaux supérieurs ne sont pas lancés en ce point. Le plan reste
  imbriqué, et le point est exclu des recherches suivantes.

### Lancer les tests et l'étude

Depuis la racine `Multifidelity-GP-Clean` :

```powershell
.venv-benchmark\Scripts\python -m pytest vpp_acceleration\tests          # 39 tests, environ 10 s
.venv-benchmark\Scripts\python vpp_acceleration\examples\forrester_study.py   # quelques minutes
```

Les tests ([`tests/test_sequential_design.py`](tests/test_sequential_design.py)) vérifient :

* **les formules contre des calculs indépendants** :
  * la décomposition de la variance, exacte sans bruit (écart inférieur à 10⁻⁴) et approchée avec le bruit appris
    (1,5 %) ;
  * le LOO d'un GP contre la suppression explicite d'un point ;
  * la proposition 2 sans bruit ;
  * le gain contre un reconditionnement complet, avec le bruit réel (10⁻⁶), y compris quand la basse fidélité est
    déjà connue au point ;
  * le gain sans bruit contre la formule par niveau N5 ;
* **les décisions** : la règle gain / coût cumulé, le maximum de k_adj trouvé par l'évolution différentielle,
  les contraintes et les exclusions ;
* **la boucle** : le plan reste imbriqué, le budget est respecté, un calcul raté bloque les niveaux supérieurs, et
  le critère d'arrêt Q² fonctionne.

### Résultats sur Forrester, à coût égal (10 graines)

Les deux stratégies séquentielles partent de 6 BF + 3 HF (coût 36), puis l'algorithme dépense un budget de 60. La
référence est un LHS imbriqué 16 BF + 8 HF, de coût 96 lui aussi. L'erreur porte sur la HF, sur 501 points.
Résultats : [`examples/results/`](examples/results/) ; figure :
[`runs/2026-10-21_cokriging_benchmarks/figures/forrester_study.html`](runs/2026-10-21_cokriging_benchmarks/figures/forrester_study.html).

| Stratégie | RMSE / std(f2), médiane [p10 ; p90] | Points BF / HF (médiane) | Couverture à 95 % |
| --- | --- | --- | --- |
| Séquentiel, variance ajustée | 2,5·10⁻⁴ [1,2·10⁻⁴ ; 4,8·10⁻⁴] | 23,5 / 7 | 1,00 |
| Séquentiel, variance seule | 2,3·10⁻⁴ [1,4·10⁻⁴ ; 4,3·10⁻⁴] | 25,5 / 7 | 1,00 |
| LHS imbriqué, coût égal | 2,7·10⁻³ [1,1·10⁻³ ; 4,7·10⁻³] | 16 / 8 | 0,97 |

* **Le séquentiel bat le LHS sur les 10 graines**, avec une erreur environ 10 fois plus faible pour le même coût. Il
  dépense surtout en BF : il ajoute environ 18 BF pour 4 HF.
* **L'ajustement par LOO n'apporte rien ici.** C'est attendu : Forrester est lisse et homogène, alors que [LC]
  annoncent un gain pour les fonctions hétérogènes.
* **Limites de ce test.** Le problème est facile : la correction HF est presque linéaire, ρ = 2 exactement, et tout le
  monde finit sous 1 % d'erreur. Il valide la mécanique, pas encore la performance sur un cas réaliste. Le maximum de
  variance place des points HF aux bords (x = 0 et x = 1), comme prévu (§ 3 de la note). La suite logique : les cas
  Hartmann et profil de `benchmarks/cokriging`, traités ci-dessous.

### Résultats sur deux problèmes plus durs (10 graines) : l'algorithme ne bat pas le LHS

Script : [`examples/benchmark_study.py`](examples/benchmark_study.py), avec les problèmes et les jeux de test fixes de
`benchmarks/cokriging` :

* **Hartmann 6D** : la BF est une transformation non linéaire de la HF, donc le modèle AR(1) est mal spécifié ;
* **profil NACA 4D** : NeuralFoil xxsmall pour la BF, xxxlarge pour la HF.

Les coûts sont de 1 et 10. Le départ est imbriqué, 40 BF + 10 HF (coût 140), et le budget choisi par l'algorithme est
de 140. La référence est un LHS imbriqué 80 BF + 20 HF, de coût 280. Le modèle est un `CoKriging` Matérn 5/2 MAP, avec
deux estimations de ρ (option `--rho`). Résultats : `examples/results/benchmark_*` ; figures :
`runs/2026-10-21_cokriging_benchmarks/figures/benchmark_study_rho_v3.html` et `_rho_v2.html`.

RMSE / std(HF), médiane sur 10 graines. Entre parenthèses : nombre de graines où la stratégie bat le LHS, et points
BF / HF.

| Problème, estimation de ρ | Séquentiel ajusté | Séquentiel, variance seule | LHS imbriqué |
| --- | --- | --- | --- |
| Hartmann, v3 (ρ appris, défaut) | 1,27 (5/10 ; 180/10) | 1,30 (2/10 ; 170/11) | 0,89 (80/20) |
| Hartmann, v2 (ρ borné à ±5) | **0,80** (7/10 ; 145/14) | 0,94 (3/10 ; 105/18) | 0,82 |
| Profil, v3 | 0,26 (1/10 ; 180/10) | 0,26 (3/10 ; 170/11) | **0,21** |
| Profil, v2 | 0,27 (1/10 ; 180/10) | 0,25 (2/10 ; 170/11) | **0,21** |
| **Hartmann, ALC + v2** (21/10, 13 h) | **0,71** (9/10 ; 145/14) | **0,69** (8/10 ; 135/14) | 0,82 |
| **Profil, ALC + v2** | 0,24 (3/10 ; 170/11) | 0,21 (4/10 ; 150/13) | 0,21 |

**Avec l'ALC et ρ borné** (`--point alc --rho v2`) :

* **Hartmann bat nettement le LHS.** L'erreur passe de 0,82 à 0,69–0,71, et le séquentiel gagne sur 8 à 9 graines sur
  10. Il est aussi mieux calibré : NLPD 0,17–0,32 contre 0,59.
* **Le profil fait au mieux jeu égal.** Avec la variance seule, il fait 0,21 comme le LHS ; avec la variance ajustée,
  il fait moins bien.
* **L'effet de bord a disparu.** Sur une graine du profil, les nouveaux points sont à 0,066 du bord en médiane, contre
  0,001 avec le maximum de variance, et l'écart-type BF reste à 31 au lieu de 47.
* **La HF reste affamée sur le profil** : 90 itérations sur 90 en BF sur cette graine. La correction HF, apprise sur
  10 points en 4D, reste 3 fois trop sûre d'elle (variance prédite 5,6, erreur réelle 18).
* **Coût du calcul** : environ 10 s par itération et 15 à 23 min par run, avec 30 runs en parallèle sur la machine.

**Départ plus grand, profil seulement** (5 graines, ALC + v2, 1 024 candidats). Départ 80 BF + 20 HF, budget de 280,
contre un LHS 160 BF + 40 HF, coût total 560 :

| Stratégie | RMSE / std(HF), médiane [p10 ; p90] | Graines gagnées contre le LHS | Points BF / HF |
| --- | --- | --- | --- |
| Séquentiel ajusté | 0,206 [0,195 ; 0,218] | 1 / 5 | 330 / 23 |
| Séquentiel, variance seule | 0,181 [0,154 ; 0,184] | 3 / 5 | 250 / 31 |
| LHS imbriqué | 0,168 [0,146 ; 0,251] | — | 160 / 40 |

* **L'hypothèse du nombre de points se vérifie en partie.** Partie de 20 HF au lieu de 10, la variance seule achète
  maintenant 11 HF au lieu de 3, et elle rejoint le LHS : 3 graines sur 5, une médiane voisine et une **dispersion
  bien plus faible**. Son pire cas est 0,184, contre 0,251 pour le LHS. Avec 5 graines, l'écart de médiane n'est pas
  significatif.
* **L'ajustement par LOO reste nuisible sur le profil**, qui est une fonction lisse : il pousse vers la BF (3 HF
  ajoutés seulement).
* Sur une fonction lisse et pour une précision globale, le LHS reste donc un adversaire très difficile à battre. C'est
  attendu : à hyperparamètres fixés, la variance ne dépend pas des valeurs observées. Le séquentiel apporte surtout de
  la robustesse, et il trouve seul le bon dosage BF / HF.

### Test orienté région, à la manière de la VPP (protocole validé le 21/10/2026, 20 graines) : SUCCÈS

Script : [`examples/region_study.py`](examples/region_study.py). Résultats : `examples/results/region_*` ; figure :
[`runs/2026-10-21_cokriging_benchmarks/figures/region_study.html`](runs/2026-10-21_cokriging_benchmarks/figures/region_study.html).

**Le test.**

* **Problème :** le profil NACA 4D, avec les coûts 1 et 10.
* **Région cible :** les états de trim, c'est-à-dire les points où la portance BF est comprise entre 0,4 et 0,6. C'est
  l'analogue des états d'équilibre de la VPP, et la région couvre 19 % du domaine.
* **Mesure principale :** l'erreur sur 1 000 points de test pris dans la région, évalués en HF.
* **Stratégies :** toutes coûtent 560. S1 et S2 partent du même plan, 80 BF + 20 HF sur tout le domaine, puis disposent
  de 280 de budget.
* **Graines :** 20, appariées entre stratégies.

| Stratégie | Erreur dans la région, médiane [p10 ; p90] | Couverture à 95 % | Erreur globale | BF / HF | Nouveaux points dans la région |
| --- | --- | --- | --- | --- | --- |
| **S1** ALC orienté région (ρ borné, variance seule) | **0,149** [0,115 ; 0,182] | 0,94 | 0,180 | 160 / 40 | 93 % |
| S2 ALC global (même chose, intégration sur tout le cube) | 0,199 [0,166 ; 0,298] | 0,98 | 0,170 | 235 / 32 | 17 % |
| B1 LHS imbriqué, tout le domaine (160 / 40) | 0,202 [0,154 ; 0,249] | 0,98 | 0,160 | 160 / 40 | 19 % |
| B2 LHS imbriqué, tout le domaine (240 / 32) | 0,225 [0,180 ; 0,320] | 0,99 | 0,178 | 240 / 32 | 20 % |
| B3 plan dans la région (oracle, 160 / 40) | 0,134 [0,120 ; 0,163] | 0,94 | 0,736 | 160 / 40 | 100 % |

**Critère de succès, fixé avant le calcul : rempli sur les quatre points.**

| Condition | Résultat |
| --- | --- |
| Erreur médiane au moins 20 % plus faible que B1 | **−27 %** |
| Au moins 16 graines gagnées sur 20 | **17** |
| Test de Wilcoxon apparié p < 0,05 | **p = 1,3·10⁻⁵** |
| Couverture à 95 % d'au moins 0,90 | **0,94** |

* **Contre le LHS de référence (B1).** S1 atteint dès le coût 352 l'erreur finale que B1 obtient avec 560, soit
  **37 % de budget économisé**. Contre B2, l'économie est de 43 % (−34 % d'erreur, 19 graines sur 20).
* **Le gain vient du ciblage.** S1 bat S2 (−25 %, 19 graines sur 20), alors que S2 ne fait que jeu égal avec le LHS,
  comme dans les tests précédents.
* **S1 s'approche de l'oracle sans connaître la région au départ.** Il fait 0,149 contre 0,134 pour B3 (10 graines sur
  20), alors que la moitié de son budget est passée dans un plan de départ sur tout le domaine. Il garde en plus une
  précision globale correcte : 0,180, contre 0,736 pour l'oracle.
* **La HF n'est plus affamée.** S1 a choisi seul le même dosage que le LHS, 80 BF + 20 HF ajoutés.

**Conséquence pour le projet.** Le séquentiel vaut l'effort dans le cas d'usage visé : un raffinement orienté vers la
région où la VPP travaille, avec l'ALC, ρ borné et un plan de départ suffisant. Pour la précision globale seule, un
bon LHS imbriqué reste la base.

**Diagnostic**, mesuré itération par itération sur une graine (détails dans l'historique de la session) :

1. **ρ décroche avec la v3.** Sur Hartmann, ρ̂ passe de 1,1 à 31,85 après l'ajout de deux points HF. La part BF de la
   variance HF (ρ²σ₁²) écrase alors tout, et la règle n'achète plus que de la BF. Or ρ n'est estimé que sur les points
   HF : il reste bloqué à 31,85, et l'erreur monte de 1,0 à 3,9. Borner ρ (v2) supprime ce décrochage.
2. **Le maximum de variance colle les points au bord.** Les nouveaux points sont à 0,001 de la frontière en médiane.
   Sur le profil, ces points de bord ont des traînées extrêmes (347 coups, contre au plus 247 au départ) et doublent
   l'écart-type des données BF (25 → 47). Comme mfego normalise par cet écart-type, la **variance BF prédite est
   multipliée par 7 alors que l'erreur BF réelle est divisée par 2**.
3. **La correction HF est trop sûre d'elle.** Sa variance prédite est de 5,6 pour une erreur réelle de 18, avec 10
   points HF en 4D. Le LOO ne le voit pas, car l'erreur est en extrapolation.
4. Au total, la règle gain / coût suit des variances mal calibrées, et en sens opposés selon le niveau. Elle affame la
   HF (180 BF / 10 HF), et l'erreur du profil ne baisse pas du tout pendant la phase séquentielle.

**Ce qui marche quand même.** Sur Hartmann avec ρ borné, le séquentiel ajusté est légèrement meilleur que le LHS (7
graines sur 10) et mieux calibré : couverture 0,94 contre 0,89, NLPD 0,33 contre 0,59. **L'ajustement par LOO y est
utile** : 0,80 contre 0,94 avec la variance seule. Hartmann est hétérogène, ce qui est le cas annoncé par [LC].

**Correctifs à tester, non faits :**

* **choisir le point par le gain intégré** (variante ALC, § 7 de la note) plutôt que par le maximum de variance. Un
  point de bord ne réduit la variance que localement, donc il serait pénalisé. Cela couperait les boucles 2 et 4 ;
* **borner ρ** (v2) pour le plan séquentiel ;
* **un garde-fou sur la part de budget HF** ;
* **pour la VPP, intégrer sur les états de fonctionnement**, à l'intérieur du domaine, plutôt que sur tout le cube.

### Benchmark contre emukit et BoTorch (21/10/2026, 10 graines) : critère NON atteint, l'écart vient du modèle

[`examples/library_study.py`](examples/library_study.py) reprend le test région ci-dessus : mêmes graines (0 à 9), même
départ 80 BF + 20 HF, même budget (coût total 560), mêmes points d'intégration et mêmes jeux de test. Il oppose S1 au
critère séquentiel de chaque bibliothèque :

* **E1 emukit** : `GPyLinearMultiFidelityModel` (le même AR(1) que le nôtre, maximum de vraisemblance sans prior) et
  `IntegratedVarianceReduction` sur la région, divisé par le coût ;
* **T1 BoTorch** : `SingleTaskMultiFidelityGP` (la fidélité est une entrée du noyau) et
  `qNegIntegratedPosteriorVariance` sur la région, divisé par le coût.

Chaque bibliothèque ajoute un seul niveau par point : ses plans ne sont pas imbriqués, et son coût n'est pas cumulé.
Pour la vitesse, les deux critères sont calculés par leur forme fermée, cov(x', x)² / (var(x) + bruit). BoTorch, lui,
construit un modèle « fantôme » par candidat, ce qui coûtait 160 s par itération. Le script vérifie à chaque run que
la forme fermée redonne le critère de la bibliothèque : écart relatif ≤ 10⁻¹¹ pour emukit et ≤ 1,5·10⁻³ pour BoTorch.
Chaque plan final est noté deux fois : avec le modèle de sa méthode, puis avec notre CoKriging réajusté dessus. Pour ce
second ajustement, environ 20 valeurs BF manquantes sont calculées pour rendre le plan imbriqué ; elles ne sont pas
comptées dans le coût.

| Méthode | Erreur région, modèle propre (médiane [10 % ; 90 %]) | Erreur région, notre CoKriging sur le plan | Couverture 95 % | BF / HF | Nouveaux points dans la région |
| --- | --- | --- | --- | --- | --- |
| S1 (nous) | 0,135 [0,120 ; 0,187] | 0,135 | 0,94 | 170 / 39 | 93 % |
| B1 LHS imbriqué | 0,200 [0,159 ; 0,248] | 0,200 | 0,99 | 160 / 40 | 19 % |
| E1 emukit | 0,130 [0,107 ; 0,207] | 0,140 | **0,77** | 140 / 42 | 61 % |
| **T1 BoTorch** | **0,120 [0,095 ; 0,179]** | 0,141 | 0,97 | 155 / 40 | 79 % |

**Critère fixé avant le calcul : non atteint.** La meilleure bibliothèque est BoTorch. L'erreur médiane de S1 est 13 %
plus haute, donc au-delà des 5 % tolérés, et S1 ne bat BoTorch que sur 2 graines sur 10. Le test de Wilcoxon « S1 pire »
donne p = 0,053, à la limite du seuil.

**Lecture.**

1. **Le plan séquentiel n'est pas en cause.** Notés avec le même modèle (notre CoKriging), les trois plans séquentiels
   se valent : S1 0,135, E1 0,140, T1 0,141. S1 est 4 % meilleur, mais sur 5 graines sur 10 seulement, ce qui n'est pas
   significatif. Tous battent nettement le LHS (0,200). Nos corrections du critère (plan imbriqué, coût cumulé, ρ
   borné) n'apportent donc pas de gain mesurable ici face au critère standard « réduction de variance intégrée / coût ».
   Ce qui compte est de **cibler la région**.
2. **L'écart vient du modèle.** Sur les plans de T1, le modèle de BoTorch fait 0,120, contre 0,141 pour notre CoKriging,
   soit 15 % de mieux sur les mêmes données, et il reste bien calibré (0,97). Mon hypothèse : son noyau de fidélité laisse
   la corrélation BF/HF varier avec x, alors que l'AR(1) impose un ρ constant. Je ne l'ai pas vérifiée.
3. emukit, avec le même AR(1) mais sans prior, est un peu plus précis que nous sur son plan, mais **trop sûr de lui**
   (couverture 0,77 au lieu de 0,95).

**Limites.** 10 graines, un seul problème (4D). Les runs S1 et B1 sont repris du test région. Les temps ont été
mesurés avec 20 processus en parallèle : S1 556 s, E1 165 s, T1 941 s par graine.

```powershell
.venv-benchmark\Scripts\python vpp_acceleration\examples\library_study.py               # environ 20 min, 20 processus
.venv-benchmark\Scripts\python vpp_acceleration\examples\library_study.py --seeds 1 --quick   # vérification, 1 min
```

### Test rapide : un ρ constant est-il trop rigide ? (21/10/2026, 10 graines) — pas dans notre cas d'usage

Prototype ([`examples/rho_study.py`](examples/rho_study.py), mfego non modifié) : un co-krigeage récursif de Le Gratiet
& Garnier (2014), où ρ(x) = β₀ + βᵀx est estimé en forme fermée, comme notre ρ actuel.

* P0 et P1 ne diffèrent que par ρ : constant pour P0, linéaire en x pour P1.
* P0 redonne bien notre CoKriging : sur le LHS de la graine 0, 0,2253 contre 0,2255.
* Tous les modèles sont ajustés sur les mêmes plans.

| Erreur région (médiane) | CK (nous) | P0, ρ constant | P1, ρ(x) linéaire | BoTorch |
| --- | --- | --- | --- | --- |
| B1 : LHS sur tout le domaine | 0,200 | 0,191 | **0,149** (ρ de 0,22 à 0,71) | 0,173 |
| B3 : plan dans la région | 0,130 | 0,129 | 0,152 | **0,114** |

* **Sur tout le domaine, ρ varie vraiment.** P1 fait −22 % face à P0 (8 graines sur 10, p = 0,053). Mais sa queue
  est lourde (90e centile 0,28) et sa couverture plus basse.
* **Dans la région, ρ est presque constant.** Les 4 paramètres de plus coûtent alors de la variance avec 40 points HF,
  et P1 fait pire que P0.
* **BoTorch reste devant dans la région** : 9 graines sur 10 face à CK (p = 0,002), et 10 sur 10 face à P1. Or nos
  plans séquentiels y concentrent 79 à 93 % de leurs points. Le ρ constant n'explique donc pas l'avance de BoTorch dans
  notre cas d'usage. Une autre piste, non testée : BoTorch estime tous ses hyperparamètres en même temps sur toutes les
  données, alors que nous les estimons niveau par niveau ; la correction HF n'apprend ainsi ses longueurs de
  corrélation que sur les 40 points HF.

**Ce que ρ(x) changerait dans le code** (non fait, mfego est en lecture seule) :

1. `mfego/src/surrogate_models.py`, dans `GaussianProcess` :
   * le régresseur f_prev (vecteur) devient une matrice H = f_prev·g(x), avec g(x) = (1, x₁…x_d) ;
   * `_profile_rho` passe de la formule scalaire à la forme vectorielle β̂ = (HᵀK⁻¹H + Σ⁻¹)⁻¹(HᵀK⁻¹y + Σ⁻¹m), avec un
     prior N(1, 2²) sur β₀ et N(0, s²) sur les pentes ;
   * `negative_log_likelihood` prend la cible y − Hβ ;
   * le centrage se fait par colonne ;
   * `rho` devient un vecteur. Seul le ρ profilé (v2) s'étend simplement ; le ρ appris sous prior lognormale (v3) ne
     s'étend pas tel quel.
2. `mfego/src/cokriging.py`, dans `CoKriging` :
   * `rhos` devient une liste de vecteurs, avec une méthode ρ(niveau, x) ;
   * `_factors(levels)` devient `_factors(levels, x)`, avec un facteur par point au lieu d'un facteur par niveau ;
   * à reprendre aussi : `_covariance`, `predict_batch`, `leave_one_out`, `to_dict` / `from_dict` et `diagnostics`.
3. `vpp_acceleration/src/sequential_design.py` :
   * `level_weights(model)` devient `level_weights(model, x)`, car P₁ = ρ(x) dépend du point ;
   * P²σ² devient ρ(x)²σ₁²(x) dans `level_variances` et `adjusted_variance` ;
   * `joint_posterior_covariance` et `batched_imse_gains` passent aux facteurs par point ;
   * les tests de décomposition sont à refaire.
4. Les scripts d'étude : la colonne `rho` devient la plage de ρ(x).

À la louche : 150 à 250 lignes dans mfego et une trentaine dans `sequential_design.py`. Vu le résultat, **je ne le
recommande pas pour l'instant**.

```powershell
.venv-benchmark\Scripts\python vpp_acceleration\examples\rho_study.py     # 30 s avec 20 processus
```

### Test rapide : estimation conjointe ou structure du modèle ? (21/10/2026, 10 graines) — la structure

**La théorie d'abord.** Avec un plan imbriqué et une BF sans bruit, la vraisemblance de l'AR(1) se factorise exactement :
p(y_BF, y_HF) = p(y_BF) · p(y_HF | y_BF). C'est la proposition 3.1 de [LG], et les deux facteurs n'ont aucun
hyperparamètre commun. Estimer tout en même temps ou niveau par niveau donne donc le même optimum.

Ce qui distingue BoTorch, c'est la **structure** de son modèle :

* son noyau a une composante commune aux deux fidélités, qui est la HF elle-même, plus un biais porté par la seule BF :
  Z_BF = Z_HF + biais(x) ;
* les longueurs de corrélation de la HF y sont donc apprises sur les 200 points ;
* dans notre AR(1), Z_HF = ρ·Z_BF + δ, et δ n'est vu qu'aux 40 points HF.

[`examples/joint_study.py`](examples/joint_study.py) ajuste six modèles sur les plans de `rho_study.py` :

* EJ : emukit, le même AR(1) mais estimé conjointement ;
* PS : notre AR(1) où δ reprend les longueurs de corrélation du niveau BF ;
* BN : BoTorch sans ses priors Gamma sur les longueurs de corrélation ;
* plus CK, P0 et BT, comme dans `rho_study.py`.

| Erreur région (médiane) | CK | P0 | EJ conjoint | PS longueurs BF | BoTorch | BoTorch sans prior |
| --- | --- | --- | --- | --- | --- | --- |
| B1 : LHS sur tout le domaine | 0,200 | 0,191 | 0,177 | 0,180 | 0,173 | **0,154** |
| B3 : plan dans la région | 0,130 | 0,129 | 0,125 | 0,120 | 0,114 | **0,105** |

Dans la région (B3), sur 10 graines :

* **EJ contre P0 : −3 %**, EJ meilleur sur 6, p = 0,11. L'estimation conjointe seule ne change rien, comme le prévoit
  la théorie ;
* **PS contre P0 : −7 %**, PS meilleur sur 8, p = 0,03, et la couverture passe de 0,88 à 0,92. Donner à δ les
  longueurs de corrélation de la BF aide : le mécanisme « 40 points » est réel ;
* **BoTorch contre PS : −6 %**, BoTorch meilleur sur 9, p = 0,03. PS comble donc environ la moitié de l'écart ;
  le reste vient de la structure du modèle ;
* **BoTorch sans prior contre BoTorch : −8 %**, meilleur sur 8, p = 0,02. Ses priors ne sont pas la cause de son
  avance ; ils le freinent même ici. Sans eux, il fait −19 % face à notre CoKriging.

Les tests de Wilcoxon sont bilatéraux. Limites : 10 graines, un seul problème (4D), et P0, PS et EJ sont estimés par
maximum de vraisemblance, sans prior.

**Ce que chaque option changerait dans le code** (rien n'est fait) :

* **longueurs de δ partagées (≈ −7 %)** :
  * `mfego/src/surrogate_models.py` : `GaussianProcess.fit` apprend à fixer les longueurs de corrélation et à
    n'estimer que t1, t2, le bruit et ρ, avec un masque sur les paramètres de `negative_log_likelihood` ;
  * `mfego/src/cokriging.py` : `CoKriging.fit` passe les longueurs du niveau l − 1 au niveau l, derrière une option
    `share_lengthscales` ;
  * `sequential_design.py` est inchangé. À la louche, 40 à 60 lignes dans mfego, qui est en lecture seule ;
* **modèle BoTorch sans prior dans notre boucle séquentielle (≈ −19 %)** :
  * `sequential_design.py` : les gains intégrés ne lisent plus la structure AR(1) (P_j, covariance jointe de mfego)
    mais la covariance a posteriori du modèle, en forme fermée, comme dans `library_study.py` ;
  * plan imbriqué, coût cumulé et ciblage de la région restent tels quels ;
  * la décomposition par niveau (P_j, variance ajustée, LOO de chaque niveau) n'a plus de sens ;
  * cela ajoute une dépendance à torch et BoTorch.

### Option 1 implémentée : `CoKriging(share_lengthscales=True)` (21/10/2026) — gain faible, critère NON atteint

**Ce qui a changé dans mfego** (25 lignes, `git diff mfego/src`). L'option est désactivée par défaut : les autres études
ne voient aucun changement.

* `mfego/src/surrogate_models.py` : `GaussianProcess.fit(..., fixed_lengthscale=None)`. Si l'argument est donné, les
  longueurs de corrélation reçoivent une borne basse égale à la borne haute dans L-BFGS-B. Elles restent donc fixes, et
  seuls t1, t2, le bruit et ρ sont estimés ;
* `mfego/src/cokriging.py` : `CoKriging(share_lengthscales=False)`. Si l'option est active, `fit` passe au niveau l les
  longueurs du niveau l − 1. L'option est enregistrée dans `to_dict` / `from_dict` ; un ancien fichier se relit avec
  False ;
* tests mfego (`tests/test_cokriging.py`, 4 de plus, 172 au total) :
  * les longueurs sont partagées sur 3 niveaux ;
  * figer les longueurs à l'optimum libre redonne le même ajustement ;
  * les arguments invalides sont refusés ;
  * l'option survit au JSON ;
* vpp_acceleration :
  * `benchmark_study.make_model(..., share_lengthscales)` ;
  * la stratégie S3 de `region_study.py` ;
  * le modèle CS de `joint_study.py` ;
  * un test du plan séquentiel avec l'option (40 tests au total). Les gains intégrés restent exacts ;
  * `sequential_design.py` est inchangé.

**Validation A, à plan égal** (`joint_study.py`, modèle CS, 10 graines) :

| Erreur région (médiane) | CK | CS (partage) | BoTorch |
| --- | --- | --- | --- |
| B1 : LHS sur tout le domaine | 0,200 | 0,179 (−10 %, 7/10, p = 0,06) | 0,173 |
| B3 : plan dans la région | 0,130 | 0,123 (−5 %, 7/10, p = 0,11) | 0,114 |

**Validation B, en séquentiel** ([`examples/share_study.py`](examples/share_study.py), test région, graines 0 à 9) :

| | Erreur région (médiane [10 % ; 90 %]) | Couverture | Erreur globale |
| --- | --- | --- | --- |
| S3 (S1 + partage) | 0,142 [0,109 ; 0,186] | 0,95 | 0,159 |
| S1 | 0,135 [0,120 ; 0,187] | 0,94 | 0,183 |
| T1 BoTorch | 0,120 [0,095 ; 0,179] | 0,97 | 0,156 |

**Critère fixé avant le calcul : ÉCHEC.**

* S3 gagne bien 7 graines sur 10 face à S1, et sa couverture vaut 0,95 ;
* mais son erreur médiane est 5 % **plus haute** que celle de S1, au lieu de 5 % plus basse ;
* Wilcoxon : p = 0,23.

La médiane des écarts graine à graine est favorable à S3 (−0,008). Mais les trois graines perdues le sont nettement
(graines 3, 7 et 9), et les médianes des deux séries s'inversent. L'erreur globale baisse de 13 %, mais ce point n'avait
pas de critère.

**Lecture.** Partager les longueurs de corrélation n'apporte que quelques pourcents. C'est du même ordre que la
dispersion entre graines avec 10 graines, et ce n'est pas démontré en séquentiel. L'option reste dans mfego,
désactivée par défaut et testée, mais je ne la recommande pas comme réglage par défaut. L'écart restant avec BoTorch
(environ 10 %) tient à la structure de son modèle, Z_BF = Z_HF + biais.

### Section Kulfan : réduction par PCA (21/10/2026) — 8 paramètres : aucune réduction ; 16 poids lisses : k = 12

[`src/section_pca.py`](src/section_pca.py) reprend la PCA de `Dim-Reduction/experiments.ipynb` avec les choix validés :
famille autour de la référence C-foil, métrique géométrique centrée, nombre de modes k fixé par une tolérance aéro,
poids de bord d'attaque 0 et bord de fuite 0,00197.

**Le point clé : un profil Kulfan est linéaire en ses poids.** À abscisses fixées, les coordonnées valent y = A w + b
(Eq. 1 du module). Trois conséquences :

* générer 10 000 profils revient à un seul produit matriciel ;
* chaque mode de la PCA correspond à un vecteur de poids, donc le décodage scores → poids Kulfan est exact (Eq. 3) ;
* une famille décrite par p paramètres linéaires a au plus p modes.

**La famille.** C'est celle de la dernière étude (`studies/cfoil_intens_kulfan_trim_2`), sous la forme
épaisseur/cambrure à 4 poids par côté : t_0..t_3, s_1..s_3, plus le décalage de cambrure δ. L'étude résolvait δ par
le modèle de lacet. Ici, le lacet devient une entrée du surrogate du torseur, donc δ redevient un paramètre de la
section. Je l'ai borné à [−0,01 ; 0,20], ce qui couvre tous les δ du dernier run (de −0,006 à 0,193). Seule
contrainte appliquée : épaisseur maximale ≥ 12,25 %. Les autres contraintes connues de l'étude (Ixx, épaisseur au bord
de fuite, poids de bord d'attaque, ondulation, confiance NeuralFoil) ne le sont pas ; il faudra les ajouter dans
`CFoilFamily.contains` si on les garde.

**Résultat : la famille a exactement 8 modes, et il les faut tous.** Il s'agit des écarts entre chaque section de test
et sa projection sur k modes ([`examples/section_pca_study.py`](examples/section_pca_study.py)) :

| k | écart géométrique max | ΔCL (p95) | ΔCD / CD (p95) | ΔCm (p95) | ΔCpmin (p95) |
| --- | --- | --- | --- | --- | --- |
| 5 | 1,2 % c | 0,057 | 4,0 % | 0,017 | 0,19 |
| 6 | 0,63 % c | 0,048 | 2,4 % | 0,015 | 0,11 |
| 7 | 0,24 % c | 0,015 | 1,2 % | 0,0032 | 0,082 |
| **8** | **2·10⁻¹⁴ % c** | **10⁻¹⁵** | **2·10⁻¹³ %** | **2·10⁻¹⁶** | **5·10⁻¹⁵** |
| tolérance | 0,02 % c (0,14 mm) | 0,002 | 0,2 % | 0,001 | 0,01 |

Conditions : NeuralFoil xxxlarge, Re 5·10⁶, n_crit 1, transition à 10 %, α = 0, 2, 4 et 6°. On garde 1 698 sections de
test sur 2 000, celles où NeuralFoil a une confiance ≥ 0,9 à tous les angles. Cpmin est pris sur les stations de
couche limite de NeuralFoil (x/c ≥ 1,6 %), donc sans le pic de bord d'attaque. Les tolérances sont celles que je
propose, à valider. Même en les multipliant par 5, aucun k < 8 ne passe.

**Pourquoi.** La part de variance mesure la géométrie, pas l'aérodynamique. Le mode 1 porte 87 % de la variance : c'est
δ, le niveau de cambrure, dont la plage est grande. Les modes suivants sont des formes indépendantes et toutes utiles
([`figures/section_pca_modes.html`](runs/2026-10-21_section_pca/figures/section_pca_modes.html)) :

* mode 2 : l'épaisseur globale ;
* mode 3 : la cambrure arrière ;
* mode 4 : l'épaisseur avancée ou reculée ;
* modes 5 et 7 : des formes de cambrure ;
* modes 6 et 8 : des ondulations d'épaisseur.

Le mode 7 ne porte que 0,15 % de la variance, mais il déplace encore CL de 0,015 et CD de 1,2 %. C'est une cambrure
arrière, et elle pèse lourd sur CL et Cm. La figure [`figures/section_pca_errors.html`](runs/2026-10-21_section_pca/figures/section_pca_errors.html)
trace ces écarts en fonction de k.

**Conséquence pour cette famille : les scores n'apportent rien.** (Le choix validé ensuite est la famille à 16 poids
lisses, plus bas.) Avec k = 8, la PCA n'est
qu'une rotation des 8 paramètres, et cette rotation abîme le domaine. Le pavé des scores ne contient que **1,6 %** de
sections de la famille, parce qu'un pavé de dimension 8 tourné n'occupe qu'une petite partie de sa boîte englobante. Le
pavé des paramètres en contient **69 %** : seule l'épaisseur minimale élimine des sections. Or le GP et le plan
séquentiel travaillent dans un pavé, donc 98 % de leurs candidats seraient perdus.

Sur cette famille, il vaut donc mieux prendre comme entrées de section (t_0..t_3, δ, s_1..s_3). Avec les 4 grandeurs d'attitude, cela
fait 12 dimensions. La PCA reste utile, et le module est prêt, pour une famille décrite par plus de paramètres que sa
dimension effective : 16 poids libres (le notebook atteint 0,07 % c avec k = 12), une base de sections existantes, ou
plusieurs familles mélangées. Une nouvelle famille n'a besoin que de deux méthodes, `sample(n, seed)` et
`contains(weights)`.

#### La famille retenue : 16 poids lisses autour du C-foil d'origine — k = 12

Choix validé ensuite : les 16 poids Kulfan, autour du C-foil d'origine, avec assez de marge pour changer par exemple la
position de cambrure, et une ondulation inférieure à 2 fois celle de la référence (contrainte de l'étude).

Une boîte sur les 16 poids ne marche pas : sur 65 536 tirages, **aucun** n'a une ondulation inférieure à 2 fois la
référence. Même la famille à 8 paramètres n'en a que 20 %. `SmoothKulfanFamily` sépare donc ce que l'ondulation ne
voit pas de ce qu'elle limite. Chaque section est écrite en épaisseur t = u − l et cambrure c = (u + l)/2. Chacun des
deux vecteurs de 8 poids se décompose exactement en v = Q g + D₂⁺ d :

* **le niveau et la pente des poids** (g, 2 nombres) ne coûtent aucune ondulation. Ce sont les grands mouvements :
  épaisseur globale, épaisseur avancée ou reculée, cambrure globale, **position de cambrure**. Ils varient dans la plage
  qu'ils prennent dans la famille à 8 paramètres, élargie de 25 % de chaque côté ;
* **les 6 différences secondes** (d) portent les formes fines. Elles sont tirées uniformément dans la boule
  « ondulation ≤ 2 fois la référence ».

Les sections de la famille à 8 paramètres qui sont assez lisses en font partie (un test le vérifie).

**Résultat : rang 16, et 12 modes suffisent** ([`figures/section_pca_errors_smooth16.html`](runs/2026-10-21_section_pca/figures/section_pca_errors_smooth16.html)) :

| k | écart géométrique max | ΔCL (p95) | ΔCD / CD (p95) | ΔCm (p95) | ΔCpmin (p95) |
| --- | --- | --- | --- | --- | --- |
| 4 | 1,1 % c | 0,065 | 4,9 % | 0,014 | 0,25 |
| 8 | 0,099 % c | 0,0072 | 0,66 % | 0,0016 | 0,044 |
| 10 | 0,031 % c | 0,0028 | 0,24 % | 0,00066 | 0,015 |
| 11 | 0,028 % c | 0,0013 | 0,19 % | 0,00031 | 0,015 |
| **12** | **0,009 % c** | **0,00085** | **0,084 %** | **0,00020** | **0,0054** |
| tolérance | 0,02 % c | 0,002 | 0,2 % | 0,001 | 0,01 |

k = 11 échoue sur Cpmin et sur la géométrie ; k = 12 passe tout, avec une marge d'un facteur 2 au moins. Les
modes se lisent bien ([`figures/section_pca_modes_smooth16.html`](runs/2026-10-21_section_pca/figures/section_pca_modes_smooth16.html)) :

* mode 1 : le niveau de cambrure ;
* mode 2 : l'épaisseur globale ;
* mode 3 : **la position de cambrure** ;
* mode 4 : l'épaisseur avancée ou reculée ;
* modes 5 à 12 : des formes fines, par paires.

Les 4 premiers modes portent 99,8 % de la variance, mais les modes fins restent nécessaires pour Cpmin et CD.
L'entrée du surrogate fait donc **12 + 4 (attitude) = 16 dimensions**.

Deux points à traiter avant l'intégration VPP :

* **seuls 2,7 % du pavé des scores sont des sections de la famille.** Les différences secondes vivent dans une boule,
  et une boule de dimension 8 n'occupe qu'une petite partie de son cube englobant. Le plan initial et les candidats du
  plan séquentiel devront donc être tirés **dans la famille** (`sample`, puis `encode`), et non dans le pavé ;
* **NeuralFoil n'est fiable (confiance ≥ 0,9 de 0 à 6°) que sur 60 % des sections de la famille.** L'étude gardait
  cette contrainte : il faudra décider si on l'ajoute à la famille.

```powershell
C:\ProgramData\Anaconda3\python.exe vpp_acceleration\examples\section_pca_study.py --family smooth16
```

**Le GPU.** Dans le notebook, seule la PCA tournait sur le GPU. Le vrai coût était la boucle de génération, qui appelle
aerosandbox profil après profil. Temps mesurés pour 10 000 sections, carte RTX 3090 :

| Étape | Temps |
| --- | --- |
| boucle du notebook (aerosandbox puis interpolation) | 7,1 s |
| produit matriciel numpy (CPU) | 0,010 s |
| produit matriciel torch (GPU, transferts compris) | 0,004 s |
| SVD complète numpy (CPU) | 0,18 s |
| `torch.pca_lowrank` (GPU, comme le notebook) | 0,026 s |

Supprimer la boucle fait gagner un facteur 700. Une fois la boucle supprimée, le GPU ne gagne plus que quelques
millisecondes : il n'est pas utile ici. La boucle ajoutait en plus une erreur d'interpolation de 3·10⁻⁵ c, que la
formule exacte supprime.

**Tests** ([`tests/test_section_pca.py`](tests/test_section_pca.py), 21 tests). Ils vérifient :

* les coordonnées contre aerosandbox (écart inférieur à 10⁻¹²) ;
* le caractère affine de y et l'élévation de degré ;
* l'aller-retour paramètres ↔ poids et l'appartenance à la famille ;
* le rang 8 et le décodage exact ;
* les scores blanchis ;
* la décroissance de l'erreur avec k, et la projection orthogonale pour k < 8 ;
* la sauvegarde JSON et l'espace des scores ;
* pour la famille à 16 poids : la base niveau / pente, l'aller-retour exact, le tirage dans la famille, l'inclusion
  des sections lisses de la famille à 8 paramètres et le rang 16.

```powershell
C:\ProgramData\Anaconda3\python.exe vpp_acceleration\examples\section_pca_study.py   # environ 1 min (torch CUDA)
```

---

## 1. Objet

`CoKriging` est un modèle de substitution seulement : il ne choisit pas les points suivants. L'acquisition actuelle de
mfego ([`mfego/src/acquisition.py`](../mfego/src/acquisition.py), AEI de Sacher et al. 2021, éq. 20–24) cherche le
**minimum** de la sortie haute fidélité (HF). Elle ne cherche pas un modèle précis partout.

Ce dossier prépare une stratégie de plan séquentiel dont le but est la **précision globale** du co-krigeage. À chaque
itération, elle doit choisir :

* le point x à calculer ;
* le niveau de fidélité l à lancer en ce point, compte tenu du coût de chaque niveau.

Hypothèse de travail, à confirmer : le surrogate sert à remplacer un solveur coûteux dans une VPP (voir § 7).

## 2. Contenu du dossier

| Chemin | Contenu |
| --- | --- |
| [`references/Cokriging-Based Sequential Design…pdf`](references/Cokriging-Based%20Sequential%20Design%20Strategies%20Using%20Fast%20Cross-Validation%20Techniques%20for%20Multi-Fidelity%20Computer%20Codes.pdf) | **[LC]** Le Gratiet & Cannamela, Technometrics 57(3), 2015 |
| [`references/IEEE Xplore Full-Text PDF_.pdf`](references/IEEE%20Xplore%20Full-Text%20PDF_.pdf) | **[W]** Westermann, Zea & Hanebeck, ECC 2021, *Adaptive Sampling for Global Meta Modeling Using a Gaussian Process Variance Measure* (nom de fichier à changer) |
| [`doc/sequential_cokriging.tex`](doc/sequential_cokriging.tex) | Note d'implémentation (5 pages) de la méthode [LC] corrigée : théorie, algorithme, noyau de calcul testé, intégration à mfego · PDF : [`doc/build/sequential_cokriging.pdf`](doc/build/sequential_cokriging.pdf) · compilation : `tectonic -X compile sequential_cokriging.tex --outdir build` |
| [`doc/complements_mathematiques.tex`](doc/complements_mathematiques.tex) | Compléments (5 pages) : dictionnaire notes ↔ notations de [LC], démonstrations de la décomposition, de la proposition 2, du gain (N5), de la correction en ρ² et de la suppression du premier test · PDF : [`doc/build/complements_mathematiques.pdf`](doc/build/complements_mathematiques.pdf) |
| [`doc/mindmap_algo.tex`](doc/mindmap_algo.tex) | Mind map (2 pages paysage) : une itération de l'algorithme, puis les étapes de construction du code · PDF : [`doc/build/mindmap_algo.pdf`](doc/build/mindmap_algo.pdf) |
| [`src/sequential_design.py`](src/sequential_design.py) | Le plan séquentiel (§ 0) |
| [`tests/`](tests/) | 14 tests pytest sur Forrester (`conftest.py` : chemins d'import, problème, simulateur) |
| [`examples/forrester_study.py`](examples/forrester_study.py) | Étude à coût égal contre un LHS imbriqué, 10 graines, avec ses résultats (`examples/results/`) et sa figure (`examples/figures/`) |
| [`examples/benchmark_study.py`](examples/benchmark_study.py) | La même étude sur Hartmann 6D et sur le profil NACA 4D, avec deux estimations de ρ (`--rho v3` ou `v2`, `--quick`, `--figure-only`) |

Le matériel supplémentaire de [LC] n'est pas dans le dossier : démonstrations des éq. 6–7, comparaison des critères pour
le krigeage simple (s = 1), code R.

## 3. Résumé

* **[LC] est la bonne base.** L'article vise la précision globale (IMSE, erreur quadratique intégrée sur le domaine) et
  choisit à la fois le point et le niveau. Il travaille sur plans imbriqués, comme `CoKriging`. La plupart des briques
  existent déjà dans mfego (§ 4.6).
* **Deux corrections avant de l'utiliser** (§ 4.7) :
  * la règle de choix du niveau compare des réductions de variance mesurées sur des sorties différentes. Il manque un
    facteur ρ², ce qui fausse la répartition BF/HF dès que ρ s'éloigne de 1 ;
  * le proxy de réduction de l'IMSE (« volume d'influence » Π θ) n'est pas borné.
* **Notre modèle diffère du leur sur un point** : ρ est une estimation ponctuelle chez nous et intégré a priori chez eux.
  Notre variance est donc plus optimiste au début.
* **La version par lots (MCMC + k-means) est trop lourde.** Une sélection gloutonne est exacte pour la variance et bien
  plus simple.
* **[W] a un intérêt marginal.** C'est un critère mono-fidélité heuristique, évalué sur un seul cas 2D. Deux idées sont à
  garder : pondérer la variance par une fonction d'importance w(x), et arrêter sur une RMSE LOO pondérée.
* **Proposition** (§ 6) : une implémentation minimale en cinq étapes. Elle serait validée contre un LHS imbriqué à coût
  égal, sur les problèmes de `benchmarks/cokriging`.

## 4. [LC] Le Gratiet & Cannamela (2015)

### 4.1 Modèle

Le modèle est le co-krigeage auto-régressif récursif (Kennedy & O'Hagan ; Le Gratiet & Garnier 2014) :

* s niveaux, sur des plans imbriqués D_s ⊂ … ⊂ D_1 ;
* Z_l(x) = ρ_(l−1) Z̃_(l−1)(x) + δ_l(x), où Z̃_(l−1) est le niveau l−1 conditionné à ses données ;
* priors de Jeffreys sur (ρ_(l−1), β_l) : ρ et la tendance sont intégrés (krigeage universel) ;
* σ²_l et les longueurs de corrélation θ_l estimés par maximum de vraisemblance restreint (REML).

La variance au niveau haut se décompose par niveau (éq. 4) :

```text
k_s(x) = Σ_{j=1..s}  σ²_j(x) · Π_{k=j..s−1} ρ̂²_k
```

* σ²_j(x) est la variance a posteriori du biais δ_j, conditionné à ses données sur D_j ;
* ρ̂²_k est le moment d'ordre 2 de la loi a posteriori de ρ_k, soit ρ̂² + var(ρ).

L'imbrication est **nécessaire** pour extraire ces contributions.

### 4.2 Choix du point et du niveau (§ 3.1, algorithme 1)

1. x_new = argmax_x k_s(x).
2. Proxy de la réduction d'IMSE du niveau l quand on ajoute x_new aux niveaux 1..l (éq. 5) :

   ```text
   IMSE_red^l(x) = Σ_{j=1..l}  σ²_j(x) · Π_{k=j..l−1} ρ̂²_k · Π_{m=1..d} θ_{j,m}
   ```

   Π_m θ_{j,m} est le « volume d'influence » d'un point au niveau j.
3. Pour l = 2..s, avec B_(l/l−1) le rapport des temps de calcul des niveaux l et l−1 :
   * si σ²_l(x_new) est inférieur à la moyenne de σ²_l sur le domaine, le niveau l ne vaut pas la peine : on calcule
     les niveaux 1..l−1 en x_new et on s'arrête ;
   * sinon, si B_(l/l−1) · IMSE_red^(l−1) > IMSE_red^l, on calcule les niveaux 1..l−1 et on s'arrête ;
   * sinon on passe au niveau suivant.

   Si la boucle atteint s, on calcule tous les niveaux 1..s en x_new. L'imbrication est donc conservée.
4. Les hyperparamètres sont ré-estimés après chaque ajout, pas pendant la boucle.

L'algorithme suppose que l'efficacité est monotone en l. La remarque du § 3.1 donne la variante sans cette hypothèse :
comparer tous les niveaux à budget égal à une simulation HF.

### 4.3 Variance ajustée par validation croisée (§ 3.2)

**Problème visé :** la variance du krigeage dépend de la position des points, pas des erreurs du modèle.

**Erreurs LOO par niveau.** Pour chaque niveau j, l'article calcule en forme close (éq. 6–7, coût O(n_l Σ_j n_j²)) :

* l'erreur LOO e_j(x_i) ;
* la variance LOO s²_j(x_i).

Le point x_i est retiré à **tous** les niveaux ≤ j.

**Variance ajustée (éq. 8).** V_{j,i} est la cellule de Voronoï de x_i dans D_j :

```text
k_s,adj(x) = Σ_j  σ²_j(x) · Π_k ρ̂²_k · ( 1 + Σ_i z²_{j,i} · 1{x ∈ V_{j,i}} )

z²_{j,i} = (e_j − ρ̂_{j−1,−i} e_{j−1})² / (s²_j − ρ̂²_{j−1,−i} s²_{j−1})
```

Le numérateur est la part de l'erreur LOO due au biais δ_j ; le dénominateur est sa variance.

**Utilisation :**

* le point devient x_new = argmax k_s,adj (éq. 9) ;
* IMSE_red,adj remplace IMSE_red dans l'algorithme 1 (éq. 10) ;
* le critère est continu dans chaque cellule. L'article propose un algorithme génétique ou un tirage Monte-Carlo pour
  parcourir les cellules, puis une descente locale dans chacune ;
* calculer la tessellation n'est pas nécessaire : trouver le plus proche voisin suffit.

### 4.4 Version par lots (§ 3.3, algorithme 2)

On place q_l points par niveau, du niveau haut vers le bas :

1. tirages de Metropolis-Hastings selon une densité ∝ k_l(x) (50 000 tirages, 5 000 de chauffe, taux d'acceptation
   d'environ 30 %) ;
2. regroupement k-means des tirages en N_l centres ;
3. choix des q_l centres de plus grande variance ajustée ;
4. aux niveaux inférieurs, même procédure, avec la variance mise à jour par les points des niveaux supérieurs. Cette
   mise à jour est exacte, car la variance ne dépend pas des valeurs observées.

La répartition {q_l} est cherchée exhaustivement sous un budget T.

### 4.5 Exemple de l'article

Réservoir sphérique sous pression, 8 entrées :

* **HF :** éléments finis Code_Aster (environ 1 min) ;
* **BF :** formule analytique 1D ;
* **ratio de coût :** B = 10, imposé (le vrai rapport est bien plus grand) ;
* **protocole :** plan initial de 10 HF + 20 BF imbriqués, 40 répétitions, 1 000 points de test, Matérn 5/2.

Trois réponses sont étudiées, avec une corrélation BF–HF d'environ 99 %, 80 % et 45 %.

* **Résultat :** le co-krigeage séquentiel bat le krigeage séquentiel HF seul quand la BF est informative. Il fait
  aussi bien quand elle ne l'est pas (45 %), car l'algorithme cesse alors de lui-même de lancer la BF.
* **Version par lots :** testée avec T = 120.

### 4.6 Correspondance avec mfego

| Brique de [LC] | Dans mfego | État |
| --- | --- | --- |
| Plans imbriqués | `CoKriging.fit` refuse un plan non imbriqué ; `missing_points_for_nesting` | existe |
| σ²_j(x) | variance latente de `gps[j-1]` (`GaussianProcess.predict_batch` moins le bruit) | existe |
| Π ρ²_k | `CoKriging._factors` | existe |
| LOO qui retire le point à tous les niveaux | `CoKriging.leave_one_out(by_location=True)` | existe (niveau demandé, pas la décomposition) |
| Erreurs et variances LOO par niveau (éq. 6–7) | — | à écrire (voir ci-dessous) |
| ρ intégré (prior de Jeffreys) | ρ ponctuel (MAP, prior lognormale en v3) | différence de modèle |
| Optimisation du critère | `scipy.optimize.differential_evolution`, comme `EGOOptimizer._find_next_point` | réutilisable |
| Coûts par niveau, contraintes connues | `ExperimentData.costs`, masque `feasibility` de `AcquisitionFunction` | réutilisable |
| Scores de qualification | `validation.prediction_scores` (Q², couverture, NLPD, z_rms) | existe |

**Simplification de l'éq. 8 chez nous.** On se place à ρ fixé : mfego n'a pas de ρ_{−i} en forme close, car ρ est appris
avec les hyperparamètres du noyau. On suppose aussi un plan imbriqué et peu bruité. Alors :

```text
y_j(x_i)      = ρ y_{j−1}(x_i) + d_j(x_i)            (d_j : résidu observé, données de gps[j-1])
μ_{j,−i}(x_i) = ρ μ_{j−1,−i}(x_i) + μ_{δj,−i}(x_i)
```

donc e_j − ρ e_{j−1} = d_j(x_i) − μ_{δj,−i}(x_i). C'est l'erreur LOO classique (Dubrule 1983) du **seul** GP δ_j sur
ses propres résidus. De même, s²_j − ρ² s²_{j−1} est sa variance LOO.

z²_{j,i} est donc le résidu LOO normalisé, au carré, de `gps[j-1]`. Il se calcule avec son facteur de Cholesky, sans
réajustement, en une dizaine de lignes. Avec le bruit appris par mfego, l'égalité n'est qu'approchée : il faut la
vérifier numériquement contre `CoKriging.leave_one_out`.

### 4.7 Réserves

**1. La règle de choix du niveau oublie un facteur ρ².** On prend deux niveaux, avec R₁ = IMSE_red¹ = σ²₁V₁ et
R₂ = IMSE_red² = ρ²σ²₁V₁ + σ²₂V₂, où V_j est le volume d'influence.

R₁ est une réduction mesurée sur la sortie **BF** ; R₂ est mesurée sur la sortie **HF**. Une simulation BF réduit l'IMSE
de la HF de ρ²R₁, pas de R₁. D'où :

```text
règle de l'article :  B · R₁ > R₂        ⇔  (B − ρ²) · R₁ > σ²₂V₂
règle cohérente    :  B · ρ² · R₁ > R₂   ⇔  (B − 1) · ρ² · R₁ > σ²₂V₂
```

Les deux règles coïncident pour ρ ≈ 1, ce qui est probablement le cas de l'exemple de l'article (une contrainte comparée
à la même contrainte).

* Avec ρ ≈ 2,4 (ρ² ≈ 5,8), valeur imposée par les données dans l'EGO sur Hartmann
  ([`RAPPORT_V3.md`](../benchmarks/surrogate_v3/RAPPORT_V3.md)), la règle de l'article sous-pondère la BF d'un facteur
  d'environ ρ².
* Dès que ρ² ≥ B, elle ne choisit plus jamais la BF. Pour ρ < 1, elle la sur-pondère au contraire.
* L'AEI actuel de mfego propage déjà ρ² (`r2_lp` dans `acquisition.py`). Reprendre la règle telle quelle serait un recul.

**Forme générale proposée.** On note G_l(x) la réduction de l'IMSE du niveau haut quand x est calculé aux niveaux 1..l :

```text
G_l(x) = Σ_{j=1..l}  σ²_j(x) · V_j · Π_{k=j..s−1} ρ²_k
```

On choisit le niveau l qui maximise G_l(x) / C_l, avec C_l = Σ_{j≤l} c_j. Le coût est cumulé, car un point au niveau l
impose les niveaux inférieurs. L'article n'utilise que le rapport t_l / t_(l−1) de deux niveaux successifs. Le critère
« par unité de coût » garde l'hypothèse de l'article : B simulations BF réduisent B fois plus qu'une seule.

**2. Le volume d'influence Π_m θ_{j,m} n'est pas borné.** Prenons un niveau qui dépend peu d'une entrée : sa longueur de
corrélation dépasse 1 dans le cube unité. Le cas est fréquent ; dans l'exemple de l'article, la BF ne dépend que de 3 des
8 entrées. Le volume, et donc la réduction attribuée à ce niveau, explose alors.

* Correction minimale : V_j = Π_m min(θ_{j,m}, 1).
* Mieux : la réduction exacte de l'IMSE par Monte-Carlo (points de Sobol). La covariance jointe de `CoKriging` la donne
  directement, sans proxy, avec ρ et le bruit cohérents.

**3. ρ ponctuel contre ρ intégré.** Notre variance ne contient pas l'incertitude sur ρ. Elle est trop optimiste quand il
y a peu de points HF, et la répartition BF/HF devient très sensible à la valeur estimée de ρ. Cela rejoint une décision
ouverte de mfego : ρ borné (v2) ou ρ libre sous prior lognormale (v3). Sur Hartmann, le ρ̂ de la v3 vaut 5,8–8,5 en
médiane et monte jusqu'à 25. La variance ajustée compense en partie.

**4. Limites de la variance ajustée :**

* elle est discontinue aux frontières des cellules. Ce n'est pas gênant pour l'évolution différentielle ;
* elle ne fait qu'augmenter la variance, jamais la réduire là où le modèle est trop prudent ;
* elle suppose que l'erreur LOO en x_i vaut pour toute sa cellule. Or les cellules du bord du domaine sont très grandes
  dès d ≥ 5 ;
* avec une dizaine de points HF, les z² sont très bruités.

À faire : mesurer la distance du plus proche voisin dans le cube unité, ou en coordonnées divisées par les longueurs de
corrélation du niveau.

**5. La version par lots est lourde.** Elle combine MCMC, k-means et recherche exhaustive de la répartition, avec
beaucoup de réglages. Les auteurs écrivent eux-mêmes qu'elle demande une mise en œuvre et des diagnostics soignés.

Alternative : la variance ne dépend pas des valeurs observées. Une sélection gloutonne est donc exacte pour la variance :

1. choisir un point ;
2. l'ajouter avec une valeur fictive via `CoKriging.condition()` ;
3. recalculer le critère et recommencer.

**6. Les preuves sont minces :**

* une seule application (8D, réponse lisse, 2 niveaux, B imposé) ;
* la seule comparaison est un krigeage HF seul. Il n'y a de comparaison ni avec un LHS imbriqué à ratio fixe, ni avec
  une autre stratégie multi-fidélité ;
* l'apport de la variance ajustée n'est montré que pour s = 1, dans le supplément.

## 5. [W] Westermann, Zea & Hanebeck (2021)

### 5.1 Ce que propose l'article

Il s'agit d'un GP mono-fidélité (noyau exponentiel carré) et d'une fonction f > 0 qu'on veut approcher plus finement là
où elle est grande. Le critère (éq. 10–11) est :

```text
J(x) = f̂_clamp(x) · σ²(x),   f̂_clamp(x) = max(α · f̂_max, f̂(x)),   α ∈ [0, 1]
```

* **Version continue :** x_new = argmax J.
* **Version « Voronoï » :** les candidats sont les sommets des cellules de Voronoï des points existants, bornées au
  domaine.
* **Critère d'arrêt :** RMSE LOO pondérée par |y_i| (éq. 13).
* **Test :** 2D, somme de 5 gaussiennes, sans bruit, 16 points initiaux + 33 points adaptatifs, 100 répétitions.
  α = 0,010 (version continue) et 0,025 (version Voronoï).

### 5.2 Réserves

* **Pas de multi-fidélité.** L'article ne dit rien du choix du niveau.
* **Hypothèses fragiles :**
  * il faut f > 0, et ajouter une constante à f change la stratégie ;
  * l'écrêtage par α vient d'une moyenne a priori nulle, alors que nos GP normalisés reviennent vers `y_mean` ;
  * des efforts ou des moments négatifs rendent le critère inapplicable tel quel.
* **Évaluation faible :**
  * α est réglé différemment pour chaque variante, sans étude de sensibilité ;
  * un seul cas test 2D sans bruit ;
  * comparaison seulement avec une grille et CV-Voronoï, pas avec le simple maximum de variance ou l'IMSE.
* **Voronoï en dimension :** le nombre de sommets de Voronoï croît très vite avec la dimension. Cela contredit l'argument
  « haute dimension » de la version Voronoï.

### 5.3 À garder

* **L'idée d'une pondération w(x) de la variance**, pour viser une région d'intérêt. L'IMSE pondéré de Picheny et al.
  (2010) est une forme mieux fondée de cette idée. Pour une VPP, w(x) devrait venir du **domaine de fonctionnement**
  (par exemple la densité des états visités par le solveur VPP), pas de la valeur de f̂.
* **Le critère d'arrêt par RMSE LOO pondérée.** Il se calcule directement avec `CoKriging.leave_one_out`.

## 6. Proposition d'implémentation (non faite, à valider)

1. **Contributions par niveau.** c_j(x) = P(j, s)² · σ²_j(x), avec σ²_j la variance latente de `gps[j-1]` et P les
   facteurs de `_factors`.
   * Test : sur plan imbriqué peu bruité, Σ_j c_j(x) doit égaler la variance de
     `CoKriging.predict_batch(x, include_noise=False)` ([LG] Prop. 3.1).
2. **LOO par niveau.** z²_{j,i}, résidu LOO normalisé de `gps[j-1]` (formule de Dubrule, sans réajustement).
   * Test : comparaison avec une suppression explicite du point et avec `CoKriging.leave_one_out`.
3. **Choix du point.**
   * Critère : k_adj(x) = Σ_j c_j(x) · (1 + z²_{j, ppv_j(x)}), où ppv_j(x) est le plus proche voisin de x dans D_j,
     dans le cube unité.
   * Recherche : maximisation par `differential_evolution`, comme dans `EGOOptimizer`, avec le masque `feasibility`.
4. **Choix du niveau.** On prend l qui maximise G_l / C_l (§ 4.7), avec V_j = Π min(θ_{j,m}, 1) et les variances
   ajustées.
   * Contrôle : on compare ensuite le proxy à la réduction exacte de l'IMSE par Monte-Carlo.
5. **Boucle demander / évaluer.**
   * Évaluation : on calcule les niveaux 1..l en x_new, ce qui garde le plan imbriqué, puis on réajuste.
   * Arrêt : quand le Q² LOO (`validation.prediction_scores`) dépasse un seuil, ou quand le budget est épuisé.

**Validation.**

* Problèmes : ceux de `benchmarks/cokriging` (Forrester 1D, `HartmannMF`, `AirfoilMF`).
* Référence : LHS imbriqué à coût égal, 10 graines.
* Métriques : RMSE relative, Q², couverture à 95 %, NLPD, part de BF.
* Variantes comparées :
  * variance seule contre variance ajustée ;
  * règle de l'article contre règle corrigée par ρ².

## 7. Questions ouvertes

* **But du surrogate.** Est-ce bien un modèle global pour la VPP, ou faut-il aussi une variante orientée optimum ?
* **Sorties multiples.** Une VPP a besoin de plusieurs coefficients (portance, traînée, effort latéral, moments). Une
  simulation les donne toutes à la fois, donc le critère doit les agréger : somme ou maximum des variances normalisées
  par sortie. Ni [LC] ni [W] ne traitent ce cas.
* **Évaluations en parallèle.** Si les calculs tournent en parallèle, il faut une version par lots (sélection gloutonne,
  § 4.7 point 5).
* **Pondération w(x).** Faut-il viser le domaine de fonctionnement de la VPP, et d'où viendrait w(x) ?
* **Emplacement du code.** Le module va-t-il dans `vpp_acceleration/src/`, qui importe mfego, ou dans `mfego/src/`, pour
  être réutilisé ailleurs ?
* **Choix de ρ.** v2 (borné) ou v3 (prior lognormale) pour l'infill ? La répartition BF/HF y est sensible (§ 4.7
  point 3).

## Références

* **[LC]** L. Le Gratiet, C. Cannamela, *Cokriging-Based Sequential Design Strategies Using Fast Cross-Validation
  Techniques for Multi-Fidelity Computer Codes*, Technometrics 57(3), 418–427 (2015),
  doi:10.1080/00401706.2014.928233.
* **[W]** J. Westermann, A. Zea, U. D. Hanebeck, *Adaptive Sampling for Global Meta Modeling Using a Gaussian Process
  Variance Measure*, European Control Conference (ECC), 573–579 (2021).
* **[LG]** L. Le Gratiet, *Multi-fidelity Gaussian process regression for computer experiments*, thèse, Université Paris
  Diderot (2013) — [`../references/Multi_fidelity_Gaussian_process_regressi.pdf`](../references/Multi_fidelity_Gaussian_process_regressi.pdf).
* **[S]** M. Sacher et al., *A Non-Nested Infilling Strategy for Multi-Fidelity based Efficient Global Optimization*,
  Int. J. Uncertainty Quantification (2021) — [`../references/multifidelity_opt.pdf`](../references/multifidelity_opt.pdf).
* L. Le Gratiet, J. Garnier, *Recursive Co-kriging Model for Design of Computer Experiments With Multiple Levels of
  Fidelity*, Int. J. Uncertainty Quantification 4, 365–386 (2014).
* O. Dubrule, *Cross Validation of Kriging in a Unique Neighborhood*, Math. Geology 15, 687–699 (1983).
* V. Picheny, D. Ginsbourger, O. Roustant, R. T. Haftka, N.-H. Kim, *Adaptive Designs of Experiments for Accurate
  Approximation of a Target Region*, J. Mechanical Design 132(7) (2010).
