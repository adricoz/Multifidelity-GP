# Guide du pipeline : optimisation multifidélité de la section du C-foil d'Intens SY

Ce guide explique, pas à pas, ce que fait chaque brique du pipeline et comment s'en servir. Il est volontairement
scolaire : on part du problème physique, on descend jusqu'au processus gaussien, puis on suit l'exécution script par
script. Le code est en anglais ; les noms de fichiers et de fonctions sont donnés tels quels pour pouvoir les retrouver.

---

## 1. Le problème

On cherche **la section 2D (une seule, constante sur toute l'envergure)** qui minimise la **traînée 3D** du C-foil
d'Intens SY, la géométrie 3D étant figée :

| Élément | Valeur (fichier `config_cfoil.json`) |
|---|---|
| Plan de forme | un seul arc de cercle (`CFoil_Arc_T1` de bdFoil) : rayon 9,8 m, longueur d'arc immergée 3,25 m, corde 0,70 m (surface 2,275 m², arc de 19°) |
| Attitude | gîte 5° + cant de puits 3,28° → **cant 8,28°**, rake 0,5°, yaw 4°, sink 0 |
| Coque | **plan de symétrie (mur) en z = 0** : le foil est attaché à la coque |
| Écoulement | Re = 5·10⁶ (sur la corde), turbulent : n_crit = 1, transition forcée à 10 % de corde sur les deux faces |
| Objectif | minimiser la traînée 3D **à portance égale** CD* = Cd_profil + Cdi·(CL_réf/CL)² (§ 4.3) |
| Contrainte 2D | **CL2d(α = 0°) = 0,45** pour la section (NeuralFoil) |
| Contrainte géométrique | épaisseur relative max ≥ 12,25 % (celle de la section actuelle MC2_60_6101_1) |

Toutes ces valeurs sont des entrées de la configuration : on peut les changer sans toucher au code (voir § 8).

Pourquoi la contrainte CL2d(0°) = 0,45 ? À attitude fixe, si l'on laissait la cambrure libre, l'optimiseur pourrait
réduire la traînée en réduisant la portance. Fixer le CL2d à incidence nulle fixe l'angle de portance nulle de la
section : dans NPLLT (polaires visqueuses), la portance 3D varie alors très peu d'une section à l'autre. Dans AVL,
non : la calibration (step 1) a montré que la portance AVL ne dépend que de la ligne de cambrure. C'est pourquoi
l'objectif est la traînée **ramenée à portance égale** (§ 4.3).

---

## 2. La section : Kulfan (CST) sous forme épaisseur / cambrure

Une section Kulfan (CST) décrit chaque face par

    y(x) = C(x) · Σ w_i B_i(x),   C(x) = √x (1 − x),   B_i = polynômes de Bernstein de degré n − 1

avec n = 4 poids par face ici (4 + 4). On écrit les poids sous une forme **épaisseur / cambrure**
(`geometry.py`, `form: "thickness_camber"`) :

    poids extrados  u_i = c_i + t_i / 2
    poids intrados  l_i = c_i − t_i / 2
    épaisseur(x)  = C(x) Σ t_i B_i(x) + x · TE
    cambrure(x)   = C(x) Σ c_i B_i(x)

* les **t_0..t_3** pilotent l'épaisseur (et seulement elle) : la contrainte d'épaisseur est une fonction simple et
  rapide des t_i, et t_i > 0 garantit que les faces ne se croisent pas ;
* la cambrure est **c_i = δ + s_i** avec s_0 = 0 : les **s_1..s_3** sont la *forme* de la cambrure, et **δ** un
  décalage uniforme, qui ajoute le mode de cambrure de base δ·C(x).

**Les 7 variables de conception sont t_0..t_3 et s_1..s_3.** δ n'est pas une variable : il est **calculé** pour
chaque section (§ 3). Le bord d'attaque (`leading_edge_weight`) est fixé à 0 et l'épaisseur de bord de fuite à
0,197 % (celle de MC2).

L'optimiseur travaille dans [0, 1]^7 ; `geometry.to_physical` convertit en valeurs physiques (`lower + u·(upper −
lower)`). Les bornes sont centrées sur la section de référence.

**Section de référence** (`study_lib.baseline_parameters`) : la section actuelle `MC2_60_6101_1.xf` (symétrique,
t/c = 12,25 %) est ajustée par moindres carrés sur 4 poids d'épaisseur (`geometry.fit_kulfan_thickness_camber`,
erreur max 0,44 % de corde), mise à l'échelle pour retrouver exactement 12,25 %, avec s_i = 0. Sa cambrure est
donc le seul mode δ·C(x) nécessaire pour atteindre CL2d(0°) = 0,45.

NeuralFoil prend en entrée 8 poids par face : `geometry.neuralfoil_kulfan` élève **exactement** le degré des
polynômes de Bernstein (4 → 8 poids, même forme au 1e-16 près, vérifié au step 0), ce qui évite un réajustement.

---

## 3. La contrainte d'égalité CL2d(0°) = 0,45 : projection

Une contrainte d'égalité ne peut pas être traitée par rejet (l'ensemble des sections qui la satisfont exactement a
un volume nul). On **l'élimine** (`section_constraint.py`, `CamberProjector`) : pour chaque point (t, s), on cherche
δ tel que

    CL2d(α = 0°; t, s, δ) = 0,45     (NeuralFoil xxlarge, Re 5·10⁶, n_crit 1, xtr 0,1)

CL2d croît presque linéairement avec δ : une évaluation vectorisée de NeuralFoil sur 25 valeurs de δ encadre la
racine, puis la méthode de Brent l'affine (7 appels à NeuralFoil, ~0,1 s). Le résultat est mis en cache par point.

Conséquence importante : **la géométrie évaluée est exactement la même aux trois niveaux**. Chaque niveau rapporte
ensuite le CL2d(0°) qu'il voit lui-même à chaque station (`cl2d_check_min/max`) : NeuralFoil xxsmall et xxlarge
dans NPLLT, XFOIL dans AVL. Ces écarts font partie de la calibration (`step1_cl2d_per_level.html`).

---

## 4. La géométrie 3D et les trois niveaux de fidélité

### 4.1 Planform (`planform.py`)

L'arc est recalculé exactement comme `CFoil_Arc_T1.arc_geometry` de bdFoil (vérifié au step 0 : écart 0), **sans
importer foil.py**. Repère bdFoil : bord d'attaque du palier inférieur à l'origine, x vers l'avant, y vers bâbord,
z vers le haut (négatif vers le bas). À chaque évaluation, on écrit dans un dossier temporaire :

* la section `sec_<hash>.xf` (121 points par face : AVL lit au plus 300 points) ;
* le planform `planform.csv` au format bdGeometry, **en mètres** : `Xle, Yle, Zle, Chord, Ainc, Section, MS`.

Les deux solveurs 3D lisent **le même fichier**, donc la même géométrie. L'attitude est appliquée par le core bdFoil
(rotation R = Rz(−yaw) Ry(−rake) Rx(−cant) autour de l'origine, puis z += sink).

### 4.2 Les niveaux (`solvers.py`)

| Niveau | Solveur | Polaires 2D | Traînée induite | Coût mesuré |
|---|---|---|---|---|
| L1 | NPLLT (ligne portante non plane, Phillips & Snyder) | NeuralFoil **xxsmall** | champ proche | voir step 1 |
| L2 | NPLLT | NeuralFoil **xxlarge** | champ proche | voir step 1 |
| L3 | AVL (bdFoil core) | XFOIL (bdFoil core, cache des polaires) | plan de Trefftz | voir step 1 |

Pour que l'écart entre niveaux ne mesure que la physique (et pas des conventions) :

* même Reynolds unique pour les polaires (`local_reynolds = False` dans NPLLT) ;
* même transition (n_crit 1, xtr 0,1) ;
* même définition de la traînée de profil : CD de la polaire au cl local de la bande (modèle `"profile"` du core) ;
* même image du plan z = 0 : **mur** (AVL `iZsym = +1`, NPLLT `surface_type = 0`) ;
* mêmes valeurs de référence (Sref, Cref, Bref du core bdFoil).

Deux détails numériques :

* un mur rencontré en biais (cant 8,28°) crée une **singularité au pied** de la ligne portante. NPLLT utilise un
  rayon de cœur `core_radius = 0.005` (fraction de corde) et le premier point (2 % de l'envergure) est exclu du
  contrôle de décrochage. Le pic de cl à la racine visible sur `step1_spanwise_loading.html` est cet artefact, sans
  effet sur les forces intégrées ;
* **décrochage** : si une bande (AVL) ou un point de contrôle (NPLLT) sort de la polaire, la valeur serait bornée
  (« clampée ») et l'optimiseur pourrait l'exploiter. L'évaluation est alors déclarée en échec (NaN).

### 4.3 Pourquoi la traînée « à portance égale » (résultat de la première calibration)

Premier passage du step 1 avec le CD brut à attitude fixe : les niveaux ne classaient pas les sections de la même
façon (Spearman L2–L3 = −0,21). La décomposition de la traînée a montré pourquoi :

* la **traînée de profil** est très bien prédite par NPLLT × NeuralFoil xxlarge (Spearman 0,955 avec XFOIL) ;
* la **portance 3D** varie de ±3,2 % d'une section à l'autre dans AVL, contre ±0,7 % (L2) et ±1,6 % (L1) dans
  NPLLT : AVL calcule la
  portance avec la seule ligne de cambrure (théorie mince, pente 2π), sans l'épaisseur ni la décambrure visqueuse,
  alors que la contrainte CL2d(0°) = 0,45 est visqueuse (NeuralFoil ; XFOIL la retrouve à 0,4515 ± 0,0024) ;
* la traînée induite (2/3 du CD) suit exactement le carré de la portance (r = 1,00) : à L3, le CD brut comparait des
  sections à des portances différentes, et un optimiseur aurait choisi des sections qui « portent moins » dans AVL.

L'objectif retenu (avec ton accord) compare les sections **à la portance de la section de référence** :

    CD* = Cd_profil + Cdi · (CL_réf / CL)²,   CL = √(Cy² + Cz²)

où CL_réf est la portance de la section de référence (MC2) au même niveau et à la même attitude (calculée une fois
par niveau). C'est une correction au premier ordre : à planform et attitude fixés, la forme de la répartition de
portance change peu, donc Cdi ∝ CL². Avec CD*, Spearman L2–L3 = 0,925 et les trois niveaux ont la même dispersion.
Le CD brut, CL, Cy et Cz restent enregistrés pour chaque évaluation.

Dans le code : `objective: {"type": "cd3d", "equal_lift_reference": {...}}` (`config.py`), calcul dans
`simulator.py` (`lift_reference`, `_evaluate_section_on_planform`).

Autre correctif lié au mur : la station de pied d'AVL reçoit le cl (≈ 8) de la première bande, singularité du mur
rencontré en biais, et le core lui applique le cd de bout de polaire. Le Cd de profil d'AVL est recalculé en donnant
à la zone de pied (2 % de l'envergure) le cd de la première station saine (`solvers.py`, `Cdprofile_core` conserve la
valeur d'origine).

---

## 5. Le processus gaussien multifidélité (mfego, `mfego/src/surrogate_models.py`)

### 5.1 Un GP par niveau, en cascade

Le modèle est **récursif** (Le Gratiet, Sacher et al. 2021, NN-MF-EGO) :

    ŷ_1(x) = GP_1(x)
    ŷ_l(x) = ρ_(l−1) · ŷ_(l−1)(x) + δ_l(x)     pour l = 2, 3

* GP_1 apprend L1 ;
* pour l ≥ 2, on n'apprend pas y_l directement mais l'**écart** δ_l(x) = y_l(x) − ρ·ŷ_(l−1)(x), avec un nouveau GP ;
* ρ (corrélation entre niveaux) est estimé à chaque ajustement par sa forme fermée (moindres carrés généralisés) ;
* les points des niveaux n'ont pas besoin d'être les mêmes (« non emboîtés ») : on prédit ŷ_(l−1) aux points du
  niveau l.

La variance se propage de la même façon : σ²_l = ρ²·σ²_(l−1) + σ²_δ,l. C'est cette incertitude qui guide
l'optimisation.

Ce qu'il faut retenir : le multifidélité **ne demande pas que L1 donne la même valeur que L3**, il demande que L3 soit
« L1 multiplié par ρ, plus une correction lisse ». C'est pourquoi on regarde au step 1 la corrélation et le
classement (Spearman), pas l'écart absolu.

### 5.2 Le noyau et ses hyperparamètres

Chaque GP utilise un noyau exponentiel carré avec une longueur de corrélation par variable (ARD) :

    k(x, x') = t1 · exp(−Σ_m (x_m − x'_m)² / (2 l_m²)) + t2,   + bruit σ_ε²

* l_m petit → la traînée varie vite avec la variable m ; l_m grand → variable peu influente ;
* les sorties sont normalisées (moyenne nulle, variance unité), les entrées sont dans [0, 1].

### 5.3 MLE ou MAP ?

Les hyperparamètres (l_1..l_7, t1, t2, σ_ε) sont ajustés en maximisant :

* **MLE** : la vraisemblance seule. Avec peu de points elle surajuste (l très grands ou très petits) et, au-delà
  d'environ 80 points, l'optimiseur de mfego est tombé sur une solution dégénérée « bruit blanc » sur le cas Hartmann
  (`benchmarks/map_hartmann/RAPPORT_MAP.md`, § 6). Le niveau L1 de cette étude dépasse 80 points ; le step 2 l'a
  vérifié sur 150 sections L1 : pas de dégénérescence ici (réponse lisse), mais un MLE trop confiant avec peu de
  points L3.
* **MAP** : la vraisemblance plus un a priori InvGamma(α, β) sur chaque longueur de corrélation :

      −log p(l) = (α + 1)·ln l + β / l   (à une constante près)

  De mode β/(α + 1) : IG(3, 2) → 0,5 ; IG(3, 1) → 0,25. La queue gauche, très légère, interdit les longueurs trop
  petites (surajustement) ; la queue droite, lourde, laisse une variable inutile partir vers les grandes longueurs.

Dans le pipeline, le MAP est **activé par défaut** (`"use_map": true`, `"lengthscale_prior": [3, 2]`). Le step 2
compare MLE, IG(3, 2), IG(3, 1) et IG(3, 4) sur les données du step 1 et **choisit** l'estimateur pour
l'optimisation (`results/step2_choice.json`) : MAP, sauf si le MLE est significativement meilleur (test de Wilcoxon
apparié sur la NLPD de chaque point), ne dégénère pas avec beaucoup de points (test sur 150 sections L1) **et** n'est
pas moins bon avec peu de points L3 (courbe d'apprentissage, ≤ 15 points L3 : le régime de l'optimisation, qui part de
14 points L3). Sur ce problème, la réponse est très lisse : le MLE est meilleur avec les 43 points L3 du step 1
(p = 0,005) et sain à 150 points, mais avec peu de points L3 ses intervalles sont beaucoup trop étroits (couverture
0,35 à 0,61 au lieu de 0,95). **Le MAP IG(3, 2) est donc retenu.** Après chaque ajustement, le journal signale les
hyperparamètres collés à une borne (`MultifidelityModel.diagnostics()`).

---

## 6. L'optimisation (EGO multifidélité, `optimizer.py`, `acquisition.py`)

Boucle « ask / tell » :

1. **Plan initial** : pour chaque niveau, un hypercube latin sur-échantillonné, réduit aux sections **faisables**
   (contrainte d'épaisseur), puis sélection des points les plus espacés (maximin) ;
2. **ajustement** du modèle à 3 niveaux (estimateur choisi au step 2 : ici MAP IG(3, 2)) ;
3. **choix du prochain point et du niveau** : on maximise, par évolution différentielle, la fonction de mérite

       M(x, l) = AEI(x) × (coût_L3 / coût_l) × (réduction de variance au niveau L3 apportée par un point au niveau l)

   * AEI = amélioration espérée augmentée (Huang) : récompense un point prometteur (moyenne basse) ou incertain ;
   * le rapport des coûts favorise les niveaux bon marché ;
   * le rapport d'information pénalise un niveau trop éloigné de L3 (ρ, écart propre) ;
   * le mérite est **nul hors du domaine faisable** (contraintes connues) : une section trop mince n'est jamais
     proposée (avant ce travail, ces sections étaient proposées sans fin, car un échec n'apprend rien au GP) ;
4. **évaluation** au niveau choisi, ajout aux données, retour en 2 ;
5. en fin de course, **vérification** de l'optimum du modèle à L3 (AVL × XFOIL) et **sauvegarde de la surface de
   réponse** (`surrogate.json`).

Les coûts relatifs des niveaux viennent des mesures du step 1 : 0,18 s (L1), 0,21 s (L2), 9,5 s (L3 avec calcul de
la polaire XFOIL), soit 1 : 1,16 : 52,6. L1 et L2 coûtent presque autant (le temps est dominé par la projection
CL2d et la mise en place de NPLLT, pas par la taille du réseau NeuralFoil) : dans les runs, le mérite a alterné L1
(29–32 fois) et L2 (47–50 fois) et n'a choisi L3 qu'une fois sur 80 itérations. C'est l'un des résultats de cette
première calibration ; le step 4 vérifie donc à L3 les sections prometteuses (optimum du modèle, meilleures
sections L1 et L2) que le run n'a pas évaluées à ce niveau.

---

## 7. Lancer l'étude, script par script

Environnement : `C:\Users\SIM\.conda\envs\bdToolbox\python.exe` (Python 3.10, NeuralFoil 0.2.3, aerosandbox,
numba, plotly). Depuis le dossier `studies/cfoil_intens_kulfan/` :

| Étape | Commande | Durée | Ce qu'il faut regarder |
|---|---|---|---|
| 0 | `python step0_check_setup.py` | ~1 min | `results/step0_checks.json` : tous PASS ; `step0_geometry_3d.html`, `step0_baseline_section.html` |
| 1 | `python step1_calibrate_solvers.py --n 48` | ~8–15 min | accord entre niveaux, coûts, CL2d vu par niveau, maillage |
| 2 | `python step2_validate_gp.py` | ~2 min | MLE vs MAP, mono vs multifidélité, petit n / grand n, choix de l'estimateur |
| 3 | `python step3_run_optimization.py --costs 1.0 1.16 52.6` | ~3–5 min | `runs/<date>/` : journal, `results.json`, `surrogate.json` |
| 4 | `python step4_make_figures.py` | ~1–2 min | vérification à L3 des candidats, `figures/index.html` (tableau de bord) |

Chaque script commence par une longue docstring qui décrit ce qu'il fait et pourquoi. Les journaux sont dans
`results/<step>.log` (et `runs/<date>/*.log` pour l'optimisation). `step1_calibrate_solvers.py --figures-only`
redessine les figures du step 1 sans relancer les solveurs. Les polaires XFOIL sont mises en cache dans
`results/polar_cache/` (une section déjà calculée ne repasse pas dans XFOIL : pour mesurer le vrai coût de L3, vider
ce dossier).

### Lire les figures

* `step1_level_agreement.html` : chaque point est une section évaluée à deux niveaux (objectif CD*). Un nuage aligné
  (Spearman proche de 1) = le niveau bon marché classe les sections comme le niveau cher : il est utile, même décalé.
  `step1_level_agreement_raw_cd.html` montre le même graphique avec le CD brut (désaccord) et
  `step1_lift_per_level.html` la cause : la portance AVL qui varie d'une section à l'autre.
* `step1_drag_breakdown.html` : traînée induite et de profil de chaque section, par niveau. Montre d'où viennent les
  écarts entre niveaux (ici, la traînée induite champ proche vs Trefftz).
* `step2_gp_scores.html` : erreur (RMSE / dispersion), NLPD (pénalise aussi une incertitude mal calibrée) et taux de
  couverture de l'intervalle à 95 % (doit être proche de 0,95).
* `step4_convergence.html` : en haut, meilleure traînée L3 en fonction du coût cumulé ; en bas, chaque évaluation dans
  l'ordre, colorée par niveau : on voit quand l'algorithme a choisi le niveau cher.
* `step4_gp_slices_1d.html` / `_2d.html` : la surface de réponse autour de l'optimum. Bande = ±2σ ; zones vides =
  sections trop minces.
* `step4_designs_parallel.html` : faire glisser sur un axe pour filtrer (par ex. level = L3 et les plus faibles CD).

### La surface de réponse sauvegardée

* `runs/<date>/surrogate.json` : le modèle exact, rechargeable :

      from src.surrogate_models import load_surrogate
      model = load_surrogate("runs/<date>/surrogate.json")
      mean, var, _ = model.predict_batch(x)            # x dans [0, 1]^7, niveau L3
      mean1, var1, _ = model.predict_batch(x, level=1) # surface du niveau L1

* `results/response_surface_sobol.csv` : moyenne et écart-type des 3 niveaux sur 4096 points de Sobol (avec les
  variables physiques et un drapeau de faisabilité) ;
* `results/response_surface_slices.npz` : les coupes 1D et 2D des figures.

---

## 8. Changer le problème

Tout se règle dans `config_cfoil.json` (les clés inconnues sont refusées, pour éviter une faute de frappe silencieuse) :

| Je veux changer… | Clé |
|---|---|
| rayon, longueur d'arc, corde, effilement, vrillage, cant géométrique, côté du bout | `planform` |
| gîte, cant de puits, rake, yaw, sink | `attitude` |
| coque = mur / surface libre / rien | `image` : `"wall"`, `"free_surface"`, `"none"` |
| Reynolds, transition, plage d'incidence XFOIL | `flow` |
| CL2d cible, incidence de la contrainte, modèle NeuralFoil de référence | `section_constraint` |
| épaisseur min / max | `constraints` |
| bornes des variables | `variables` |
| solveurs, modèles NeuralFoil, maillage NPLLT (`nodes_count`), réglages AVL | `levels[].options` |
| taille du plan initial, itérations, MAP/MLE, prior | `optimization` |

Après un changement : relancer le step 0 (contrôles), puis les steps 1 à 4.

---

## 9. Où est le code

| Fichier | Rôle |
|---|---|
| `pipelines/bdtoolbox_foil/config.py` | lecture et validation de la configuration |
| `pipelines/bdtoolbox_foil/geometry.py` | sections (Kulfan épaisseur/cambrure, élévation de degré, fit, épaisseur analytique) |
| `pipelines/bdtoolbox_foil/section_constraint.py` | projection CL2d(0°) = cible |
| `pipelines/bdtoolbox_foil/constraints.py` | contraintes géométriques connues (épaisseur) |
| `pipelines/bdtoolbox_foil/planform.py` | arc du C-foil (conventions bdFoil), CSV planform, attitude |
| `pipelines/bdtoolbox_foil/solvers.py` | niveaux NPLLT × NeuralFoil et AVL × XFOIL |
| `pipelines/bdtoolbox_foil/simulator.py` | point de conception → section → solveur → CD |
| `pipelines/bdtoolbox_foil/run.py` | plan initial, boucle EGO, vérification, export |
| `mfego/src/surrogate_models.py` | GP, modèle multifidélité, MAP |
| `mfego/src/acquisition.py`, `optimizer.py`, `data_management.py` | mérite, boucle ask/tell, plan initial faisable |
| `studies/cfoil_intens_kulfan/step*.py` | les étapes de l'étude |
