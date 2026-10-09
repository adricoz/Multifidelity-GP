# Rapport : première calibration du GP multifidélité et des solveurs NPLLT × NeuralFoil — section du C-foil d'Intens SY

Date : 09/10/2026 · branche `feat/cfoil-kulfan-3d` de `Multifidelity-GP-Clean` (non commitée) · environnement conda
`bdToolbox` (Python 3.10.14, NeuralFoil 0.2.3, aerosandbox 4.2.4, numba 0.56.4) · bdFoil `fix/npllt-review`.

Le guide pas à pas du pipeline est dans [GUIDE_PIPELINE.md](GUIDE_PIPELINE.md) ; toutes les figures interactives sont
accessibles depuis [figures/index.html](figures/index.html).

---

## 1. Résumé

| Question | Réponse |
|---|---|
| Le MAP était-il bien intégré ? | **Non** dans le pipeline : il tournait toujours en MLE, et une clé `use_map` dans un JSON aurait été ignorée sans erreur. C'est corrigé : MAP par défaut, prior réglable et sauvegardé, diagnostics, clés inconnues refusées. |
| Le chemin 3D du pipeline marchait-il ? | Non : les deux backends 3D renvoyaient NaN à chaque évaluation (2 bugs de clés). Ils sont réécrits, avec le planform du C-foil, l'attitude, l'image « mur », la contrainte CL2d(0°) et des contrôles de cohérence. |
| La géométrie suit-elle les conventions bdFoil ? | Oui : arc identique à `CFoil_Arc_T1` (écart 0), Sref/Cref/Bref du core, fichier AVL relu (5·10⁻⁵ près), signe de l'incidence vérifié. 17 contrôles sur 17 au step 0. |
| NPLLT × NeuralFoil est-il un bon niveau basse fidélité ? | **Avec xxlarge (L2), oui pour les tendances globales** : traînée de profil proche de XFOIL (Spearman 0,955), classement à portance égale proche d'AVL × XFOIL (Spearman 0,925, ρ ≈ 1,00), 45 fois moins cher. Moins sûr pour départager les toutes meilleures sections (la moitié seulement des 10 % meilleures sont communes). **Avec xxsmall (L1), peu utile** : même coût que xxlarge (0,18 s contre 0,21 s) et classement moins fiable (Spearman 0,56 avec L3). |
| Découverte principale | À attitude fixe, la portance AVL change d'une section à l'autre (±3,2 %), car AVL ne voit que la ligne de cambrure : le CD brut d'AVL compare des sections à des portances différentes (Spearman L2–L3 = −0,21). L'objectif retenu avec toi est la **traînée à portance égale** CD*. |
| MAP ou MLE ? | Sur les 49 sections du step 1, le MLE est légèrement meilleur (p = 0,005). Mais **avec 6 à 20 points L3, le régime de l'optimisation, le MAP est nettement meilleur et bien calibré** (couverture 0,88–0,96 contre 0,35–0,61 pour le MLE). **Le MAP IG(3, 2) a été retenu.** Le run fait d'abord avec le MLE le confirme : ses prédictions à l'optimum étaient fausses d'environ 5σ, celles du run MAP sont dans leur incertitude. |
| Le GP multifidélité apporte-t-il quelque chose ? | Oui. À données L3 égales (43 points, et L1/L2 connus partout), l'erreur de prédiction de L3 passe de 41–43 % à 22–24 % de la dispersion. Avec 10 points L3 : 35 % contre 89 % (MAP). C'est le cas favorable ; le contrôle hors échantillon du modèle final donne 33 %. |
| Résultat de l'optimisation | Meilleure section mesurée (run MAP) : CD* = 0,02432 contre 0,02454 pour la référence, **−0,9 % à portance égale** (traînée de profil −2,9 %, portance +0,8 %). Le paysage est plat : toutes les sections candidates vérifiées sont entre −0,4 et −1,2 %. L'optimum avance l'épaisseur maximale (18 % de corde contre 26 %) et amincit l'arrière jusqu'aux bornes : **c'est une calibration de la chaîne, pas une section à fabriquer** (§ 7). |

**Attention, la référence n'est pas la section actuelle telle quelle** : MC2_60_6101_1 est symétrique, et la référence
est son épaisseur plus la cambrure δ·C(x) nécessaire pour CL2d(0°) = 0,45 (cambrure max 4,2 % à 33 % de corde).

---

## 2. Vérification et intégration du MAP (sans lancer les tests)

### 2.1 Ce qui était en place (commit 6c36944)

`mfego/src/surrogate_models.py` implémentait le MAP (a priori InvGamma(3, 2) sur chaque longueur de corrélation,
gradient analytique, `use_map` sauvegardé), testé dans `tests/` et évalué par `benchmarks/map_hartmann/`.

### 2.2 Ce qui manquait ou était incohérent

* `pipelines/bdtoolbox_foil/run.py` construisait le modèle **sans** `use_map` : tout le pipeline tournait en MLE ;
* `config.py` acceptait n'importe quelle clé dans `optimization` : `"use_map": true` dans un JSON était ignoré ;
* le prior était une constante de module (pas réglable par modèle ni par niveau, recommandation n° 2 de RAPPORT_MAP) ;
* `mfego/main.py`, les deux exemples et les README ne parlaient que du MLE ; docstring et messages en français dans
  `visualization.py`, `Hartmann6d.py`, `optim_neuralfoil.py` ; lignes mortes dans `acquisition.py`.

### 2.3 Ce qui a été modifié (tous les nouveaux arguments ont une valeur par défaut : appels existants inchangés)

| Fichier | Modification |
|---|---|
| `mfego/src/surrogate_models.py` | `lengthscale_prior` réglable (une paire, ou une par niveau), validé, sauvegardé dans `to_dict` / relu par `from_dict` (les anciens fichiers relisent (3, 2)) ; `diagnostics()` (longueurs, t1, t2, bruit, ρ, valeurs collées à une borne) ; avertissement si un hyperparamètre touche une borne ou si les entrées sortent de [0, 1] avec le MAP ; `estimator_label()` |
| `mfego/src/acquisition.py` | crochet `feasibility` (contraintes connues : mérite nul hors domaine faisable) ; lignes mortes supprimées |
| `mfego/src/optimizer.py` | `_random_point()` : les replis aléatoires tirent des points faisables |
| `mfego/src/data_management.py` | plan initial faisable (LHS sur-échantillonné + sélection maximin `greedy_maximin`) |
| `pipelines/bdtoolbox_foil/config.py` | toutes les clés sont vérifiées ; `use_map` (défaut **true**), `lengthscale_prior` (défaut [3, 2]), `n_restarts`, `known_constraints` |
| `pipelines/bdtoolbox_foil/run.py` | transmet MAP / prior / restarts / contraintes connues ; écrit les diagnostics du GP et les versions de l'environnement dans `results.json` |
| `mfego/main.py`, `example/*` | `USE_MAP = True` explicite ; texte et identifiants en anglais |
| `README.md`, `pipelines/README.md` | documentation du MAP, des contraintes connues, du chemin 3D |
| `benchmarks/map_hartmann/priors.py` | libellé correct de l'estimateur des modèles à prior du benchmark |

### 2.4 Contrôles statiques effectués (pas de pytest, comme demandé)

* compilation de tous les fichiers modifiés avec le Python de `bdToolbox`, pyflakes sans remarque ;
* recensement de tous les appels de `GaussianProcess(`, `MultifidelityModel(`, `negative_log_likelihood(`,
  `AcquisitionFunction(`, `EGOOptimizer(`, `generate_initial_design(`, `build_section(`, `make_backend(` :
  compatibles ;
* les 7 `surrogate.json` existants se rechargent et donnent les mêmes prédictions ;
* toutes les configurations se chargent ; une faute de frappe (`use_mpa`) est refusée avec un message clair ;
* un agent de relecture indépendant a revu tout le diff : aucun problème critique, aucun test existant cassé (analyse
  test par test), 4 points majeurs et ~20 mineurs, **tous corrigés**.

**À faire de ton côté** : lancer la suite `pytest` (non lancée ici, à ta demande).

---

## 3. Le pipeline 3D complété (`pipelines/bdtoolbox_foil`)

### 3.1 Bugs trouvés dans les gabarits 3D existants (jamais exécutés)

1. `NpLltBackend` : `dict(zip(noms, solver.get_bdfoil_coefficients()))` itère sur les **clés** du dict renvoyé →
   `"Cx"` valait la chaîne `"Cant [°]"` → NaN systématique ;
2. `AvlCoreBackend` : l'objectif cherchait `"Cx"`, la clé du core est `"Cx [-]"` → NaN systématique ;
3. NPLLT ignorait le Reynolds, n_crit et xtr de la configuration ; CSV supposé en mm ; `PolarSpec` sans n_crit/xtr
   côté AVL ; modèle visqueux du core `"legacy"` par défaut (≠ définition NPLLT) ;
4. erreurs du core bdFoil non attrapées (une seule section ratée arrêtait tout), dossiers temporaires jamais nettoyés ;
5. points infaisables (épaisseur) proposés indéfiniment : un échec (NaN) n'apprend rien au GP (run
   `pipelines/runs/1008_150237` : 34 échecs sur 80 points L1).

### 3.2 Ce qui a été ajouté

| Module | Rôle |
|---|---|
| `planform.py` (nouveau) | arc du C-foil (`CFoil_Arc_T1` réécrit sans `foil.py`), CSV planform en mètres, attitude `cant = gîte + cant de puits` |
| `geometry.py` | Kulfan forme épaisseur/cambrure, élévation de degré exacte 4 → 8 poids (entrée NeuralFoil), épaisseur analytique vectorisée, fit d'une section existante, lecteur `.xf` (BOM, sens de parcours) |
| `section_constraint.py` (nouveau) | projection CL2d(α) = cible : décalage de cambrure δ résolu (NeuralFoil vectorisé + Brent), cache |
| `constraints.py` (nouveau) | contraintes géométriques connues (épaisseur), vectorisées |
| `solvers.py` | `NpLltBackend` et `AvlCoreBackend` réécrits : même Reynolds, transition, traînée de profil, image du plan z = 0 (`wall` / `free_surface` / `none`), contrôle du décrochage (zone de pied exclue), CL2d(α) vu à chaque section, correctif de la station de pied AVL |
| `simulator.py` | chemin « section sur planform fixe », objectif `cd3d` brut ou **à portance égale**, compteur d'évaluations, erreurs du core → NaN (sauf exécutable manquant) |
| `configs/foil3d_cfoil_kulfan_3levels.json` | exemple 3D fonctionnel (remplace le gabarit non exécutable) |

---

## 4. Le problème étudié

| Élément | Valeur |
|---|---|
| Géométrie (fixe, surface 2,275 m²) | arc unique R = 9,8 m, longueur d'arc immergée 3,25 m, corde 0,70 m, une seule section sur toute l'envergure |
| Attitude | gîte 5° + cant de puits 3,28° = cant 8,28°, rake 0,5°, yaw 4°, sink 0 (foil sous le vent, voir § 7) |
| Coque | plan de symétrie (mur) en z = 0 |
| Écoulement | Re = 5·10⁶, n_crit 1, transition forcée à 10 % sur les deux faces (identique aux 3 niveaux) |
| Section | Kulfan 4+4, 7 variables (t_0..t_3, s_1..s_3) ; δ résolu pour CL2d(0°) = 0,45 (NeuralFoil xxlarge) |
| Contrainte | t/c max ≥ 12,25 % (section actuelle MC2_60_6101_1) |
| Référence | épaisseur de MC2_60_6101_1 ajustée en Kulfan 4+4 (erreur max 0,44 % de corde, mise à l'échelle à 12,25 %, épaisseur max à 26 % de corde) + cambrure δ·C(x), δ = 0,108 (MC2 est symétrique) |
| Niveaux | L1 NPLLT × NeuralFoil xxsmall · L2 NPLLT × NeuralFoil xxlarge · L3 AVL × XFOIL (core bdFoil) |

Bornes des variables : t_i à ±50 % de la référence, sauf t_0 ≥ 0,26 ; s_i ∈ [−0,10 ; 0,10]. La borne basse de t_0
vient de la calibration : avec t_0 < ~0,23 (bord d'attaque très fin), XFOIL ne converge sur **aucune** incidence
(décollement au bord d'attaque de l'intrados), quels que soient les réglages (panneaux, transition libre, n_crit 9,
densité de points). Taux d'échec XFOIL sur des sections tirées au hasard (tests exploratoires de 16 et 24 sections,
hors scripts de l'étude) : 25 % avec les bornes initiales, 8 % avec les bornes retenues ; 6/49 (12 %) au step 1 ; 0
dans les runs d'optimisation. Les échecs restants sont groupés vers t_0 ∈ [0,26 ; 0,33] avec t_1 ≥ 0,43.

---

## 5. Calibration des solveurs (step 1 : 49 sections × 3 niveaux)

### 5.1 Contrôles de base

* **Coût** par évaluation, mesuré au premier passage du step 1 (sans cache des polaires) : L1 0,18 s, L2 0,21 s, L3
  9,5 s (polaire XFOIL 4,1 s + AVL 5 s). Coûts utilisés pour l'optimisation : **1 : 1,16 : 52,6**. La figure
  `step1_costs.html` montre le second passage, où les polaires étaient en cache (L3 ≈ 6 s) ; dans le run MLE, les
  temps moyens ont été 0,17 / 0,19 / 7,2 s. Le temps de L1/L2 est dominé par la projection CL2d (0,13 s) et la mise en
  place de NPLLT, pas par le réseau NeuralFoil : passer de xxsmall à xxlarge ne coûte presque rien.
* **Bruit numérique** : perturbations de 10⁻³ du domaine → rapport de variance ≤ 4·10⁻⁵ à tous les niveaux (la borne
  du GP est 10⁻²) : les réponses sont lisses.
* **Maillage** : NPLLT de 25 à 400 nœuds → CD brut à ±0,2 % ; AVL nspan 100 / 205 / 300 → CD identique. Le rayon de
  cœur de NPLLT est **nécessaire** avec le mur (sans lui, le CD augmente de 6,2 % à cause de la singularité de pied)
  et le CD varie peu entre 0,002 et 0,02 (±0,3 %). L'objectif CD* d'un niveau NPLLT varie un peu plus avec ces
  réglages (−0,1 / +0,5 % avec les nœuds, jusqu'à −1,3 % avec un rayon de 0,02) : c'est un décalage commun, appris par
  le modèle multifidélité, mais du même ordre que les gains recherchés.
* **Échecs** : 1 non-convergence de NPLLT (niveau L2, section 34, qui échoue aussi dans XFOIL), 6 échecs XFOIL.

### 5.2 La découverte : le CD brut d'AVL compare des sections à des portances différentes

Premier passage, objectif = CD brut à attitude fixe :

| Spearman (classement) | L1–L3 | L2–L3 | L1–L2 |
|---|---|---|---|
| CD brut | 0,11 | **−0,21** | 0,50 |
| traînée de profil seule | 0,46 | **0,955** | 0,44 |

La traînée de profil est très bien prédite par NeuralFoil xxlarge. Le désaccord vient de la traînée induite (2/3 du
CD) :

* portance 3D, écart-type d'une section à l'autre : L1 1,6 %, L2 0,7 %, **L3 3,2 %** (figure
  `step1_lift_per_level.html`) ;
* à L3, la traînée induite suit le carré de la portance (r = 0,9998) ;
* AVL calcule la portance avec la seule ligne de cambrure (pente 2π, sans épaisseur ni décambrure visqueuse) ; la
  contrainte CL2d(0°) = 0,45 est visqueuse (NeuralFoil ; XFOIL la retrouve : 0,4515 ± 0,0024). La référence a un cl0
  « mince » de 0,433 : d'où une portance AVL 6 % plus faible que NPLLT et une traînée induite 12 % plus faible
  (0,0160 contre 0,0181). À portance égale, NPLLT et AVL donnent la même efficacité d'envergure (e = 1,936 / 1,934),
  et la traînée induite d'AVL en champ proche et dans le plan de Trefftz diffère de 0,02 % : les méthodes de traînée
  induite sont d'accord, c'est la portance qui diffère.

Conséquence : à L3, un optimiseur aurait choisi des sections qui « portent moins » dans AVL, un artefact du modèle.

### 5.3 L'objectif retenu : la traînée à portance égale

    CD* = Cd_profil + Cdi · (CL_réf / CL)²,   CL = √(Cy² + Cz²),   CL_réf = portance de la référence au même niveau

| Spearman (classement) | L1–L3 | L2–L3 | L1–L2 |
|---|---|---|---|
| **CD*** | 0,556 | **0,925** (Pearson 0,926, pente 0,90) | 0,378 |

Avec CD*, les trois niveaux ont la même dispersion (écart-type ≈ 3·10⁻⁴, 1,2 % du CD*) et L2 classe les sections
presque comme L3 ; 50 % des 10 % meilleures sections sont communes à L2 et L3. **Limite** : CD* ne corrige que la
traînée induite ; la traînée de profil reste calculée au cl local d'AVL, donc à la portance de la section. C'est une
correction au premier ordre (même forme de répartition de portance), d'autant plus sûre que la section a une portance
proche de la référence.

Correctif associé : la station de pied d'AVL (cl ≈ 8, singularité du mur) recevait le cd de bout de polaire, un
artefact de ~0,25 % du CD dépendant de la section ; elle reçoit maintenant le cd de la première station saine.

### 5.4 Le CL2d(0°) vu par chaque niveau

| Niveau | n sections | CL2d(0°) vu : moyenne ± écart-type (min – max) |
|---|---|---|
| L1 (NeuralFoil xxsmall) | 49 | 0,4477 ± 0,0081 (0,433 – 0,470) |
| L2 (NeuralFoil xxlarge, référence de la projection) | 48 | 0,4500 ± 0,0000 |
| L3 (XFOIL) | 43 | 0,4515 ± 0,0024 (0,446 – 0,456) |

La projection est exacte pour le modèle de référence (NeuralFoil refait son propre ajustement Kulfan dans NPLLT et
retrouve 0,45000) ; XFOIL confirme la contrainte à ±1,2 % au plus.

### 5.5 Points d'attention : confiance de NeuralFoil

NeuralFoil fournit une « confiance » d'analyse. Au step 1, elle descend sous 0,5 pour 10 % (L1) et 15 % (L2) des
sections, parfois sur toute l'envergure : ce sont les sections hors du domaine d'entraînement de NeuralFoil, celles où
XFOIL échoue aussi, et où L1/L2 donnent des valeurs extrêmes (L1 jusqu'à 0,0300 contre 0,0268 ± 0,0003). Elles sont
exclues des statistiques d'accord (pas de valeur L3) mais restent dans les données d'entraînement L1/L2. Dans les runs
d'optimisation : 2 % (L1) et 7 % (L2).

### 5.6 Conclusion de la calibration des niveaux

* **L2 = NPLLT × NeuralFoil xxlarge est un bon niveau basse fidélité pour les tendances** (Spearman 0,925, ρ ≈ 1,00,
  45 fois moins cher que L3) ; pour départager les toutes meilleures sections, L3 reste nécessaire.
* **L1 = NPLLT × NeuralFoil xxsmall coûte autant que L2** (1,16×) et classe moins bien (Spearman 0,56 avec L3) :
  dans les runs, le mérite a alterné L1 et L2 (29–32 choix L1, 47–50 choix L2).

---

## 6. Calibration du GP (step 2) et optimisation (steps 3 et 4)

### 6.1 Validation croisée « leave-one-out » sur les 43 sections L3 valides (L1/L2 connus en chaque section)

| Modèle | Estimateur | RMSE / dispersion | NLPD | Couverture 95 % |
|---|---|---|---|---|
| L3 seul | MLE | 0,43 | −6,51 | **0,65** |
| L3 seul | MAP IG(3, 2) | 0,41 | −7,46 | 0,95 |
| L1 + L3 | MLE | 0,22 | −8,53 | 0,86 |
| L1 + L3 | MAP IG(3, 2) | 0,34 | −8,00 | 0,91 |
| L2 + L3 | MLE | 0,22 | −8,42 | 0,95 |
| L2 + L3 | MAP IG(3, 2) | 0,24 | −8,22 | 0,93 |
| L1 + L2 + L3 | MLE | 0,22 | −8,42 | 0,95 |
| L1 + L2 + L3 | MAP IG(3, 2) | 0,24 | −8,22 | 0,93 |

(IG(3, 1) et IG(3, 4) : proches d'IG(3, 2) ; détails dans `results/step2_gp_validation.json`. L2 + L3 et L1 + L2 + L3
sont identiques : L2 étant connu au point testé, L1 ne peut rien ajouter dans ce test.)

**Courbe d'apprentissage** (sections L3 tirées au hasard pour l'entraînement, les autres pour le test ; médianes) :

| Points L3 | 3 niveaux, MAP : RMSE / couverture | 3 niveaux, MLE | L3 seul, MAP | L3 seul, MLE |
|---|---|---|---|---|
| 6 | 0,39 / 0,92 | 0,48 / 0,35 | 1,08 / 0,84 | 1,07 / 0,70 |
| 10 | 0,35 / 0,88 | 0,38 / 0,49 | 0,89 / 0,91 | 0,78 / 0,67 |
| 15 | 0,32 / 0,93 | 0,35 / 0,61 | 0,71 / 0,96 | 0,73 / 0,57 |
| 20 | 0,32 / 0,96 | 0,43 / 0,61 | 0,65 / 0,96 | 0,66 / 0,70 |
| 30 | 0,20 / 1,00 | 0,19 / 0,85 | 0,45 / 1,00 | 0,28 / 0,92 |

* Le multifidélité réduit nettement l'erreur à nombre de points L3 égal (dans le cas favorable où L1/L2 sont connus
  aux points testés).
* **Choix de l'estimateur** (`results/step2_choice.json`) : MAP, sauf si le MLE est significativement meilleur, ne
  dégénère pas à grand n **et** n'est pas moins bon avec peu de points L3. Le MLE est meilleur sur les 43 points
  (Wilcoxon p = 0,005) et sain à 150 points L1 (aucune dégénérescence, RMSE 7 % pour les 4 estimateurs), mais avec
  ≤ 15 points L3 sa NLPD médiane est −0,8 contre −7,7 pour le MAP (intervalles beaucoup trop étroits). **Le MAP
  IG(3, 2) est retenu.** Le MAP reste aussi le défaut du pipeline.
* Hyperparamètres de l'ajustement complet du step 2 : ρ(L1→L2) = 0,93, ρ(L2→L3) = 1,00 ; t_0 est la variable la plus
  influente à tous les niveaux (longueur ≈ 0,45) ; t_2 est quasi inactive à L3 (longueur 10, de même dans l'ajustement
  à 150 points L1), t_3 à L1. Dans les runs, avec 15 points L3, les longueurs du GP L3 restent proches du mode du prior
  (0,5–0,7) : à ce niveau, c'est le prior qui parle.

### 6.2 Optimisation (EGO multifidélité, deux runs)

Plan initial faisable 70 / 35 / 14 points, 80 itérations, coûts 1 : 1,16 : 52,6. Le premier run a été fait avec le
MLE (choix initial du step 2), le second avec le MAP après l'ajout du test « petit n » ; le step 4 vérifie ensuite à
L3 les sections prometteuses qu'aucun run n'avait évaluées à ce niveau.

| | Run MLE (`runs/1009_161751`) | **Run MAP (`runs/1009_164012`, retenu)** |
|---|---|---|
| Durée | 4 min 37 s | 3 min 01 s |
| Évaluations L1 / L2 / L3 | 102 / 82 / 16 | 99 / 85 / 15 |
| Choix de L3 par le mérite | 1 sur 80 itérations | 1 sur 80 itérations |
| Meilleure section mesurée à L3, CD* | 0,02425 (−1,2 %) | 0,02432 (−0,9 %) |
| … sa portance / la référence | −3,3 % | +0,8 % |
| … sa traînée de profil / la référence | −2,6 % | −2,9 % |
| Prédictions vérifiées à L3 (step 4) | 0,02422 ± 0,00003 prédit, 0,02437 mesuré : **≈ 5σ d'écart** | 0,02435 ± 0,00009 prédit, 0,02439 mesuré : dans l'incertitude |
| Contrôle hors échantillon (43 sections du step 1) | RMSE 40 % de la dispersion, couverture 0,79 | **RMSE 33 %, couverture 0,98** |

Toutes les sections candidates vérifiées à L3 (optimum effectif du modèle, minimum de la moyenne L3, meilleures
sections L1 et L2, pour les deux runs) sont entre 0,02425 et 0,02444, soit **−0,4 à −1,2 %**. Sur les 43 sections
aléatoires du step 1, la meilleure était déjà à −0,55 % et la référence était 5ᵉ : **le paysage est plat**, et les
écarts entre les meilleures sections sont du même ordre que les incertitudes de modèle (correction de portance,
réglages NPLLT).

Le résultat du run MAP est le plus fiable : sa portance est presque celle de la référence (+0,8 %), donc le gain ne
dépend presque pas de la correction de portance ; celui du run MLE s'appuie sur une portance 3,3 % plus faible,
corrigée au premier ordre (correction ≈ 4 % du CD*, plus grande que le gain).

**Section optimale du run MAP** (`figures/step4_sections.html`) :

| Variable | Référence | Optimum MAP | Bornes |
|---|---|---|---|
| t_0 | 0,332 | 0,448 | [0,26 ; 0,50] |
| t_1 | 0,376 | 0,266 | [0,19 ; 0,56] |
| t_2 | 0,161 | 0,089 | [0,08 ; 0,24] (près de la borne basse) |
| t_3 | 0,209 | **0,100** | [0,10 ; 0,31] (borne basse) |
| s_1, s_2, s_3 | 0, 0, 0 | 0,010 ; −0,053 ; 0,046 | [−0,10 ; 0,10] |
| δ (résolu) | 0,108 | 0,099 | — |
| t/c max (position) | 12,25 % (26 % de corde) | 12,25 % (**18 % de corde**) | ≥ 12,25 % (active) |
| t/c à 40 % / 50 % de corde | 11,2 % / 9,6 % | 9,3 % / 7,2 % | — |

**Lecture** : avec pour seule contrainte l'épaisseur maximale, l'optimiseur avance l'épaisseur maximale et amincit
l'arrière de la section jusqu'aux bornes (le run MLE va plus loin : épaisseur max à 15 % de corde, t/c 8,4 % à 40 %).
Le gain sur la traînée de profil (−3 %) est cohérent entre les deux runs, mais la forme est structurellement
discutable (épaisseur au longeron, bord de fuite plus fin). **Ce résultat calibre la chaîne ; ce n'est pas une
section à fabriquer.**

**Convergence** : avec un rapport de coût de 53, le mérite n'a choisi L3 qu'une fois par run ; l'optimum effectif de
chaque modèle n'a été vérifié qu'au step 4. Les runs ne sont donc pas convergés au sens de L3.

---

## 7. Limites et recommandations

1. **Contraintes structurelles** : ajouter des épaisseurs minimales locales (par ex. t/c au longeron, à 30–40 % de
   corde), un angle de bord de fuite et/ou un rayon de bord d'attaque minimal. L'infrastructure est prête
   (`constraints.py` : contraintes connues, vectorisées, utilisées par le plan initial et l'acquisition). Élargir
   ensuite les bornes actives (t_0, t_2, t_3).
2. **Budget L3** : imposer quelques évaluations L3 près de l'optimum (vérification des meilleurs candidats à chaque
   dizaine d'itérations, ou un rapport de coût plus faible), puis comparer les deux ou trois meilleures sections avec
   un trim à portance égale.
3. **Niveaux** : passer à 2 niveaux (L2 + L3), ou remplacer L1 par un niveau réellement moins cher (le coût est dominé
   par la projection CL2d et la mise en place de NPLLT).
4. **Modèle de portance d'AVL** : CD* corrige au premier ordre. Pour aller plus loin : L3 = NPLLT × XFOIL (option
   `polar_source="xfoil"` de NPLLT, portance visqueuse à tous les niveaux), ou correction de l'incidence AVL par
   l'angle de portance nulle de XFOIL, ou un trim en yaw à portance égale pour la validation finale.
5. **Un seul point de fonctionnement, transition forcée** : la transition forcée à 10 % favorise une épaisseur
   maximale avancée ; la tendance pourrait s'inverser en transition naturelle. Pas encore de vérification hors point
   de conception (yaw, gîte, vitesse), ni de cavitation (Cp min) : à ajouter avant de retenir une section.
6. **Image « mur »** : c'est la borne haute de l'effet de plaque d'extrémité de la coque (e ≈ 1,93). Elle diffère du
   défaut bdFoil (surface libre, `zsym = −1`) : les résultats ne sont pas comparables aux surfaces de réponse maison.
7. **Signe de la gîte** : avec la convention bdFoil (bout de l'arc vers −y, intérieur), une gîte positive correspond au
   foil **sous le vent** (force latérale vers −y, Cz > 0). Pour le foil au vent, la gîte doit entrer avec un signe
   moins.
8. **MAP** : le prior IG(3, 2) est bon avec peu de points mais pilote les longueurs L3 dans les runs ; avec plus de
   points L3, réévaluer le choix (le MLE devient meilleur au-delà de ~30 points L3 sur ce problème lisse).
9. **Configuration NACA du pipeline** : `section2d_naca_3levels.json` avait `cl_target` passé de 0,5 à 1,0 dans le
   commit 6c36944 sans mise à jour du nom ni de la description ; j'ai aligné nom et description sur 1,0 — à confirmer.
10. **Comportement modifié du pipeline 2D Kulfan** : NeuralFoil reçoit désormais les poids Kulfan exacts (élévation de
    degré) au lieu de coordonnées réajustées : les résultats des anciens runs Kulfan 2D peuvent changer légèrement.

---

## 8. Vérifications indépendantes

* **Revue de code** (agent séparé, lecture seule) : aucun problème critique ; compatibilité de tous les tests
  existants analysée ; 4 points majeurs et ~20 mineurs, tous corrigés.
* **Revue physique et conventions bdFoil** (agent séparé) : géométrie, unités, type FOIL, convention de cant, modèle
  d'image, Reynolds / transition / traînée de profil cohérents ; a confirmé indépendamment le problème de portance
  d'AVL (§ 5.2) et proposé le correctif de la station de pied (appliqué).
* **Revue des résultats** (agent séparé) : a recalculé les chiffres du rapport à partir des fichiers de résultats
  (quasi tous confirmés) et relevé des erreurs et des interprétations trop fortes. Toutes ont été corrigées ; elles
  ont aussi conduit au test « petit n » du step 2 (changement d'estimateur MLE → MAP) et à la vérification à L3 des
  candidats au step 4.

## 9. Fichiers produits

| Dossier / fichier | Contenu |
|---|---|
| `step0_check_setup.py` … `step4_make_figures.py` | les 5 étapes (docstrings explicatives) |
| `config_cfoil.json` | la configuration de l'étude |
| `results/step0_checks.json` | 17 contrôles du step 0 |
| `results/step1_calibration.csv`, `step1_summary.json` | calibration (objectif CD*) ; `*_rawcd_run.*` : premier passage (CD brut) |
| `results/step2_gp_validation.json`, `step2_choice.json`, `step2_large_n_l1.csv` | validation du GP et choix de l'estimateur (`step2_choice_first.json` : premier choix, MLE) |
| `runs/1009_164012/` (MAP, retenu), `runs/1009_161751/` (MLE) | journal, `ego_backup.json`, `surrogate.json` (surface de réponse rechargeable), `results.json` |
| `results/step3_summary.json`, `step3_summary_mle_run.json` | résumés des deux runs |
| `results/step4_l3_checks.json`, `step4_l3_checks_mle_run.json` | vérification à L3 des sections candidates |
| `results/response_surface_sobol.csv`, `response_surface_slices.npz` | surface de réponse du run MAP (4096 points de Sobol × 3 niveaux, coupes 1D/2D) |
| `figures/index.html` | tableau de bord de toutes les figures |
