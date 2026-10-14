# Rapport : optimisation de la section du C-foil d'Intens SY à la portance du foil de base (trim en yaw)

Date : 09/10/2026 · run `runs/1009_175827` (le run précédent `runs/1009_172637` est gardé pour comparaison) · branche
`feat/cfoil-kulfan-3d` (non commitée) · environnement conda `bdToolbox` (NeuralFoil 0.2.3).

* Figures interactives : [figures/index.html](figures/index.html).
* Le guide du pipeline et la calibration qui a fixé les réglages de ce test sont dans le premier dossier :
  [../cfoil_intens_kulfan/GUIDE_PIPELINE.md](../cfoil_intens_kulfan/GUIDE_PIPELINE.md) et
  [../cfoil_intens_kulfan/RAPPORT.md](../cfoil_intens_kulfan/RAPPORT.md).

**Cette version remplace la précédente.** Une relecture a trouvé des défauts dans le code du trim (§ 3.2) et des
chiffres mal comptés dans le rapport. Le code a été corrigé, puis l'optimisation et l'analyse ont été relancées. Tous les
chiffres ci-dessous viennent du nouveau run et de `results/analysis.json`.

---

## 1. Résumé

| Question | Réponse |
|---|---|
| Ce qui change par rapport au premier test | La portance 3D est **tenue exactement**. Pour chaque section et à chaque niveau, le yaw est ajusté jusqu'à retrouver la portance du foil de base, calculée au même niveau. Le petit écart de portance restant (≤ 10⁻⁴) est retiré de la traînée induite. Il n'y a plus de correction au premier ordre (CD*). |
| Meilleure section mesurée | CD = **0,024265** contre **0,024536** pour le foil de base avec AVL × XFOIL, soit **−1,10 % à portance égale**. Avec NPLLT × NeuralFoil xxlarge : −0,33 %. Cette section a été trouvée par le run précédent et **re-mesurée avec le code corrigé** : le chiffre n'a pas bougé. |
| Meilleure section du nouveau run | 0,024288, soit **−1,01 %**. Elle a la même famille d'épaisseur mais une autre cambrure. Deux runs identiques à quelques détails près finissent donc à 0,09 % l'un de l'autre, sur deux sections différentes (§ 5.3). |
| D'où vient le gain | De la **traînée de profil** : −3,1 % avec XFOIL, −0,8 % avec NeuralFoil. La traînée induite ne bouge presque pas (−0,06 % dans AVL), puisque la planform et la portance sont les mêmes. |
| Prix à payer | Dans AVL, la nouvelle section demande un peu plus de yaw pour porter autant (4,06° contre 4,00°). Sa forme est discutable : épaisseur maximale avancée à 16 % de corde, arrière très aminci (t/c 6,5 % à mi-corde contre 9,6 %), et trois variables d'épaisseur sur ou près de leurs bornes. |
| La correction CD* du premier test était-elle juste ? | **En partie.** Elle est juste à 0,03 % près quand la portance de la section reste à moins de 1 % de la référence. Elle se trompe de 0,13 % quand l'écart atteint −3,3 %. C'est plus que les écarts entre les meilleures sections : CD* suffit pour dégrossir, pas pour départager (§ 5.4). |
| Le GP multifidélité a-t-il bien marché ? | **Bien pour l'ensemble, insuffisant près de l'optimum.** Leave-one-out sur les 19 points AVL : RMSE de 22 % de la dispersion, 18 points sur 19 dans l'intervalle à 95 %. Les 5 sections hors données sont prédites à moins de 1,4σ, mais les 3 venues d'autres runs sont toutes meilleures que prévu (−0,11 à −0,24 %) : le modèle ne connaît pas leur région. Le mérite n'a choisi AVL que 2 fois en 60 itérations. |
| Ce que NPLLT sait faire | Il classe bien l'ensemble des sections (Spearman 0,96 avec AVL sur les 19 points AVL). Il **ne classe pas** les meilleures entre elles : sa meilleure section est la dernière des 6 candidates selon AVL. |
| À retenir | Plusieurs sections différentes donnent −0,8 à −1,1 %. Les écarts entre les meilleures (≤ 0,1 %) sont du même ordre que la variabilité d'un run à l'autre et que certains choix de modélisation (correction au pied de 0,2 %, § 6). Le gain est réel dans les modèles mais petit, et la forme optimale n'est pas encore une section à fabriquer (§ 7). |

---

## 2. Le problème

| Élément | Valeur (`config_cfoil_trim.json`) |
|---|---|
| Géométrie (fixe) | C-foil d'Intens SY : arc unique R = 9,8 m, arc immergé 3,25 m, corde 0,70 m, une seule section sur toute l'envergure |
| Attitude nominale | gîte 5° + cant de puits 3,28°, soit cant 8,28° ; rake 0,5° ; **yaw 4° (point de départ du trim)** ; sink 0 |
| Coque | plan de symétrie (mur) en z = 0 |
| Écoulement | Re 5·10⁶, n_crit 1, transition forcée à 10 % de corde (identique aux 2 niveaux) |
| Section | Kulfan 4+4 en forme épaisseur/cambrure, 7 variables (t_0..t_3, s_1..s_3). Contrainte CL2d(0°) = 0,45, tenue par un décalage de cambrure δ résolu avec NeuralFoil xxlarge. Contrainte t/c max ≥ 12,25 % |
| **Contrainte de portance 3D** | **CL = √(Cy² + Cz²) = CL du foil de base à l'attitude nominale, au même niveau** : 0,71526 (NPLLT), 0,67219 (AVL) |
| **Objectif** | **CD à la portance cible** = Cdprofile + Cdi·(CL_ref/CL)², au yaw trimmé. Le facteur (CL_ref/CL)² ne fait que retirer le résidu du trim (\|CL/CL_ref − 1\| ≤ 10⁻⁴) |
| Foil de base | épaisseur de MC2_60_6101_1 (ajustée en Kulfan 4+4) + cambrure δ·C(x) pour CL2d(0°) = 0,45 |

**Pourquoi un CL par niveau ?** AVL et NPLLT ne donnent pas la même portance pour la même géométrie : AVL ne voit que la
ligne de cambrure (§ 5 du premier rapport). « Même portance que le foil de base » est donc appliqué dans chaque modèle :
chaque solveur compare les sections à la portance qu'il calcule lui-même pour le foil de base.

---

## 3. Mise en place

### 3.1 Réglages repris de la calibration (pas de nouvelle calibration)

| Réglage | Valeur | Justification (premier test) |
|---|---|---|
| Niveaux | NPLLT × NeuralFoil xxlarge, AVL × XFOIL | xxsmall coûtait autant que xxlarge et classait moins bien |
| Hyperparamètres du GP | MAP, prior InvGamma(3, 2) | bien calibré avec peu de points AVL ; le MLE était trop confiant (≈ 5σ d'erreur) |
| Plan initial | 50 NPLLT + 16 AVL, faisable et bien réparti | l'erreur multifidélité tombait à ~30 % de la dispersion avec 15–20 points AVL |
| Itérations | 60 ; l'optimum du modèle est vérifié par AVL toutes les 10 itérations | au premier test, le mérite n'avait choisi AVL qu'une fois en 80 itérations |
| Coûts relatifs | **1 : 48** | voir ci-dessous |

**Mesure des coûts.** Au début du run, trois sections tirées au hasard ont été trimmées aux deux niveaux. La résolution
de la section de référence en est exclue.

| Niveau | Coût par évaluation | Détail |
|---|---|---|
| NPLLT | 0,29 s | trim 0,12–0,21 s + projection CL2d 0,12–0,15 s |
| AVL | 14,0 s | 11,9–17,5 s, polaire XFOIL comprise |

Pendant le run, les moyennes ont été de 0,26 s et 10,8 s, soit un rapport réel d'environ 42.

### 3.2 Ce qui a été ajouté ou corrigé dans le pipeline (générique)

* **Trim** (`"trim"` dans la configuration ; `simulator.py`, `_trim`)
  * Principe :
    * sécante sur le yaw à partir de 4° ;
    * tolérance relative 10⁻⁴ sur la portance ;
    * bornes de yaw [−2°, 12°].
  * Corrections après relecture :
    * une portance non finie (NaN) fait échouer l'évaluation au lieu d'être acceptée ;
    * la pente apprise sur les trims précédents n'est mise à jour qu'à partir de deux calculs distants d'au moins 0,02°. Elle est bornée à [0,25 ; 4] × la pente initiale, pour qu'une évaluation bruitée ne dérègle pas les suivantes ;
    * l'interpolation entre les deux derniers calculs n'est faite que si la cible est **encadrée** (pas d'extrapolation) et à moins de 0,5 % (au lieu de 2 %). Elle porte seulement sur les coefficients d'effort (« [-] ») ;
    * **le résidu de portance est retiré de la traînée** : objectif = Cdprofile + Cdi·(CL_ref/CL)², avec le CD brut gardé dans les métriques. Avant, un résidu de 10⁻⁴ sur la portance laissait jusqu'à ~1,3·10⁻⁴ de bruit relatif sur CD. Maintenant, la même section évaluée deux fois par deux chemins de trim différents donne 0,0242875 et 0,0242877 (10⁻⁵) ;
    * la résolution de la section de référence n'est plus comptée dans le temps de la première évaluation.
* **Validation de la configuration** (`config.py`) :
  * bloc `trim` : bornes ordonnées et contenant l'angle nominal, tolérance > 0, `max_iterations` ≥ 1, bande ≥ 0, pente ≠ 0 ;
  * `verify_every` : null ou entier ≥ 1.
* **Boucle** (`run.py`) : si `stop_on_convergence` arrête un bloc, toute la boucle s'arrête (avant, seul le bloc
  s'arrêtait).
* **Attitude par appel** dans les deux backends (`solvers.py`). NPLLT est reconstruit à chaque attitude. AVL tourne dans
  un dossier de travail par appel ; la polaire XFOIL de la section est calculée une fois, puis relue dans le cache.
* **Vérification périodique** (`optimization.verify_every`, `run.py`) : la boucle tourne par blocs de 10 itérations.
  Après chaque bloc, l'optimum du modèle est évalué avec AVL s'il ne l'a pas déjà été.

Vérifications faites sur le code : compilation de tous les modules et chargement de toutes les configurations. La suite
pytest n'a pas été lancée.

### 3.3 Fichiers du dossier

| Fichier | Rôle |
|---|---|
| `config_cfoil_trim.json` | le problème |
| `study_trim.py` | outils communs (chemins, niveaux, lecture d'un run) ; reprend la section de base et le style des figures du premier dossier |
| `step1_run_optimization.py` | mesure des coûts (≈ 1 min) puis optimisation (≈ 4,5 min) |
| `step2_analysis.py` | vérification des candidats avec AVL, contrôle du CD*, statistiques du run, leave-one-out, figures, export de la surface de réponse (≈ 2,5 min) |

---

## 4. Déroulement du run

**Durée : 4 min 32 s, plan initial compris** (plus ~1 min de mesure des coûts).

| Poste | Temps |
|---|---|
| Plan initial | 2 min 51 s (NPLLT 14 s, AVL 2 min 37 s) |
| Simulations de la boucle | 51 s (NPLLT 14 s, AVL 37 s) |
| Vérification périodique | 12 s |
| Ajustements du GP | 15 s |
| Recherche du point suivant | 17 s |

**Évaluations : 127**, aucun échec.

| Niveau | Nombre | Détail |
|---|---|---|
| NPLLT | 108 | 50 du plan initial + 58 choisies par le mérite |
| AVL | 19 | 16 du plan initial + 2 choisies par le mérite + 1 vérification |

Les indices ci-dessous comptent les évaluations dans l'ordre du run, de 1 à 127.

* Le mérite n'a choisi AVL que **2 fois sur 60 itérations**, aux évaluations 112 et 120.
* La meilleure section du run est la dernière évaluée par AVL (évaluation 120) : **rien ne montre que la recherche a
  convergé**.

**Vérification périodique : une seule sur 6 blocs**, à l'évaluation 77 : 0,0243046 mesuré pour 0,0243048 prédit. Aux
blocs 2 à 5, l'optimum du modèle était déjà un point AVL. Au bloc 6, c'était le point qui venait d'être évalué.

**Trim** (toutes les évaluations ont convergé) :

| Niveau | Appels au solveur | dont interpolées | Résidu de portance brut | Yaw trimmé |
|---|---|---|---|---|
| NPLLT | 1 appel : 5 évaluations ; 2 appels : 103 | 23 | ≤ 8,8·10⁻⁵ | 3,83° à 4,06° |
| AVL | 2 appels : 19 évaluations | 3 | ≤ 5,2·10⁻⁵ | 3,52° à 4,28° |

* Dans AVL, d'une section à l'autre, il faut de −0,48° à +0,28° de yaw pour porter autant. C'est cohérent avec les ±3 % de
  portance à yaw fixe trouvés au premier test.
* L'interpolation a évité 26 appels, soit environ 15 s de passages AVL (≈ 5 s chacun) : un gain modeste.

**Polaires XFOIL.** Un point manquant a été comblé par interpolation, à α = −1,5° pendant le run et à α = 5° pendant
l'analyse. Aucune section n'a fait échouer XFOIL.

Figures :

* `step2_convergence.html` : meilleure traînée AVL en fonction du coût, ordre des évaluations, vérification cerclée ;
* `step2_trim.html` : yaw trimmé de chaque section, nombre d'appels et interpolations.

---

## 5. Résultats

### 5.1 Foil de base contre optimum, à la portance du foil de base

L'optimum est la meilleure section mesurée : celle du run `1009_172637`, ré-évaluée avec le code corrigé.

| | NPLLT : base | NPLLT : optimum | AVL : base | AVL : optimum |
|---|---|---|---|---|
| CL cible | 0,71526 | 0,71526 | 0,67219 | 0,67219 |
| yaw trimmé | 4,000° | 3,980° | 4,000° | **4,060°** |
| **CD à la portance cible** | 0,026660 | 0,026573 (**−0,33 %**) | 0,024536 | 0,024265 (**−1,10 %**) |
| traînée induite Cdi | 0,018122 | 0,018105 (−0,09 %) | 0,016022 | 0,016013 (−0,06 %) |
| traînée de profil | 0,008538 | 0,008469 (−0,8 %) | 0,008514 | 0,008252 (**−3,1 %**) |
| CL2d(0°) vu par le niveau | 0,4500 | 0,4500 | 0,4533 | 0,4499 |

* Chaque solveur a son propre décalage de traînée : on compare les sections à l'intérieur d'un même niveau.
* Les deux modèles ne s'accordent pas sur le yaw : AVL en demande un peu plus (+0,06°), NPLLT un peu moins (−0,02°).
* Figure : `step2_comparison.html`.

### 5.2 La section optimale

| Variable | Foil de base | Optimum | Meilleure du nouveau run | Bornes |
|---|---|---|---|---|
| t_0 | 0,332 | **0,487** (95 % de sa plage) | 0,458 (83 %) | [0,26 ; 0,50] |
| t_1 | 0,376 | 0,204 (4 %) | 0,251 (17 %) | [0,19 ; 0,56] |
| t_2 | 0,161 | **0,080** (borne basse) | 0,082 (1 %) | [0,08 ; 0,24] |
| t_3 | 0,209 | **0,100** (borne basse) | 0,100 (borne basse) | [0,10 ; 0,31] |
| s_1 ; s_2 ; s_3 | 0 ; 0 ; 0 | −0,070 ; −0,044 ; −0,032 | −0,061 ; **+0,023** ; −0,067 | [−0,10 ; 0,10] |
| δ (résolu) | 0,108 | 0,143 | 0,139 | — |
| t/c max (position) | 12,25 % (26 % de corde) | 12,25 % (**16 %**), contrainte active | 12,25 % (17,5 %), contrainte active | ≥ 12,25 % |
| t/c à 40 % / 50 % / 70 % de corde | 11,2 / 9,6 / 5,7 % | **8,6 / 6,5 / 3,2 %** | 9,1 / 7,0 / 3,4 % | — |
| cambrure max (position) | 4,2 % (33 %) | 4,0 % (**23 %**) | 4,4 % (33 %) | — |

* **L'épaisseur** est de la même famille dans les deux runs et au premier test : nez épais, épaisseur maximale avancée,
  arrière aminci jusqu'aux bornes de t_2 et t_3. La traînée de profil baisse, car il reste moins de surface « épaisse » à
  l'arrière avec une transition forcée à 10 % de corde.
* **La cambrure, elle, n'est pas partagée.** L'optimum a une cambrure avancée (max à 23 %). La meilleure section du
  nouveau run garde une cambrure maximale à 33 %, plus forte, comme le foil de base.
* Les deux sections ont presque la même traînée (0,09 % d'écart), mais pas pour les mêmes raisons :

  | Section | Traînée induite (Cdi) | Traînée de profil |
  |---|---|---|
  | optimum | 0,016013 | 0,008252 |
  | meilleure du nouveau run | 0,015980 (plus faible) | 0,008307 (plus forte) |

  La surface de réponse a donc au moins deux creux de niveau presque égal.
* Figure : `step2_sections.html`, où l'optimum, la meilleure section du nouveau run et l'optimum du premier test sont
  superposés au foil de base.

### 5.3 Toutes les sections candidates, vérifiées avec AVL au trim

Chaque candidate est évaluée aux deux niveaux avec le code corrigé. La « prédiction » est celle du modèle final du
nouveau run. z = (mesure − prédiction)/σ.

| Section | AVL : CD | écart / base | prédiction ± σ | z | NPLLT : CD |
|---|---|---|---|---|---|
| **optimum = meilleure section du run précédent** | **0,024265** | **−1,10 %** | 0,024323 ± 0,000047 | −1,24 | 0,026573 |
| optimum du premier test, run MLE | 0,024282 | −1,04 % | 0,024325 ± 0,000031 | −1,40 | 0,026604 |
| meilleure section du nouveau run (= optimum effectif du modèle) | 0,024288 | −1,01 % | 0,024287 ± 0,000005 | (point du run) | 0,026585 |
| optimum du premier test, run MAP | 0,024312 | −0,91 % | 0,024339 ± 0,000037 | −0,73 | 0,026561 |
| minimum de la moyenne AVL du modèle | 0,024341 | −0,79 % | 0,024344 ± 0,000023 | −0,13 | 0,026643 |
| meilleure section NPLLT du run | 0,024385 | −0,62 % | 0,024383 ± 0,000056 | +0,03 | **0,026512** |
| foil de base | 0,024536 | — | — | — | 0,026660 |

**Prédictions hors données.** Cinq candidates n'avaient pas été évaluées par AVL dans le run.

* Toutes tombent dans l'intervalle à 95 % (\|z\| ≤ 1,4).
* Les deux sections proches des points du run sont très bien prédites (−0,01 % et +0,01 %).
* Les trois sections venues d'**autres runs** sont toutes **meilleures que prévu**, de 0,11 à 0,24 %. Le modèle est
  pessimiste dans les régions que ce run n'a pas explorées avec AVL. En particulier, il prédisait −0,87 % pour l'optimum,
  qui fait −1,10 %.

**Le nouveau run n'a pas retrouvé l'optimum du précédent.** Les deux runs ont le même plan initial et le même problème. Ils
ne diffèrent que par les coûts relatifs (48 contre 51) et par les corrections du trim, et ils finissent sur deux sections
à 0,09 % l'une de l'autre. Avec 2 évaluations AVL choisies par le mérite en 60 itérations, l'exploration au niveau AVL est
trop mince pour départager des creux aussi proches.

**Classement par NPLLT.** Sur les 6 candidates distinctes, le rang de Spearman entre NPLLT et AVL est **−0,26** (0,21 si
l'on ajoute le foil de base). La meilleure section selon NPLLT est la dernière des 6 selon AVL. Avec 6 sections, ce chiffre
n'est pas significatif. Il montre seulement que NPLLT ne départage pas des sections qui diffèrent de moins de 0,5 %.

### 5.4 La correction CD* du premier test, contrôlée par le trim exact

Écart = (CD* − CD au trim exact) / CD au trim exact.

| Section | Niveau | CD* (yaw 4°, correction au 1ᵉʳ ordre) | CD au trim exact | écart | portance à yaw 4° / référence |
|---|---|---|---|---|---|
| optimum premier test, run MAP | NPLLT | 0,026565 | 0,026561 | +0,02 % | +0,25 % |
| | AVL | 0,024319 | 0,024312 | +0,03 % | +0,82 % |
| optimum premier test, run MLE | NPLLT | 0,026606 | 0,026604 | +0,01 % | +0,10 % |
| | AVL | 0,024249 | 0,024282 | **−0,13 %** | **−3,30 %** |

* La correction au premier ordre est très bonne quand la portance de la section reste proche de la référence (à moins de
  1 %).
* Quand l'écart de portance atteint −3,3 %, CD* sous-estime la traînée de 0,13 %. C'est plus que l'écart entre les deux
  meilleures sections (0,07 %).
* Conséquence : CD* convient pour dégrossir une optimisation, mais pas pour départager les meilleures sections. Le trim
  exact est nécessaire pour cela (`step2_cdstar_check.html`).

---

## 6. Qualité du modèle multifidélité

* **Leave-one-out sur les 19 points AVL** (les 108 points NPLLT gardés) : RMSE **22 %** de la dispersion des traînées AVL,
  NLPD −8,30, **18 points sur 19** dans l'intervalle à 95 % (`step2_leave_one_out.html`). Le leave-one-out garde les points
  postérieurs au point retiré. C'est donc une mesure de la précision de la surface finale, pas de ce que le modèle savait
  au moment de choisir.
* **Prédictions vraiment hors données** : la vérification du run (−0,001 %) et les 5 candidates du § 5.3. Elles sont
  bonnes près des données, pessimistes de 0,1 à 0,24 % loin d'elles.
* **Accord NPLLT–AVL** : la surface NPLLT du modèle classe les 19 sections AVL avec un Spearman de **0,96**, et
  ρ (NPLLT → AVL) = 1,0008. Ce bon classement vaut pour l'ensemble de l'espace (traînées réparties sur ±2 %), pas pour les
  meilleures sections entre elles (§ 5.3).
* **Hyperparamètres**
  * Niveau NPLLT : t_0 est la variable la plus influente (longueur 0,52), puis s_3, s_2 et t_1 (0,86 à 1,09). t_2 et t_3
    pèsent peu (3,5 et 2,7).
  * Niveau AVL (écart à NPLLT) : toutes les longueurs sont entre 0,53 et 1,02, autour du mode du prior (0,5). Avec 19
    points, c'est le prior qui pilote ce niveau, comme prévu par la calibration.
* **Correction de traînée de profil au pied (AVL)** : la station du pied, singulière avec l'image « mur », est corrigée.
  La correction vaut 0,19 à 0,28 % de CD selon la section (0,24 % en moyenne), soit 0,09 % d'écart d'une section à
  l'autre. C'est du même ordre que les écarts entre les meilleures sections : ce choix de modélisation peut changer leur
  classement.
* **Surface de réponse**
  * `runs/1009_175827/surrogate.json` : modèle rechargeable avec `load_surrogate`.
  * `results/response_surface_sobol.csv` : 4096 points de Sobol × 2 niveaux, avec moyenne, écart-type et drapeau de
    faisabilité.
  * `results/response_surface_slices.npz` : coupes 1D et 2D passant par l'optimum.
  * Figures : `step2_gp_slices_1d.html` et `step2_gp_slices_2d.html`.

---

## 7. Limites et recommandations

1. **Contraintes structurelles d'abord.** L'optimum est encore sur les bornes : t_0 en haut, t_1, t_2 et t_3 en bas. Il
   amincit beaucoup l'arrière (t/c 6,5 % à mi-corde, 3,2 % à 70 %). Avant toute nouvelle optimisation, ajouter :
   * une épaisseur minimale au longeron ;
   * un angle de bord de fuite minimal ;
   * un rayon de bord d'attaque.

   L'infrastructure de contraintes connues est prête (`constraints.py`).
2. **Le gain est petit et le paysage est plat.** On trouve −0,8 à −1,1 % pour des sections assez différentes. Les écarts
   entre les meilleures (≤ 0,1 %) sont du même ordre que :
   * la variabilité d'un run à l'autre (0,09 %) ;
   * l'erreur de CD* (0,13 %) ;
   * la correction au pied (écart de 0,09 %) ;
   * les incertitudes de modèle (portance AVL par la seule ligne de cambrure, réglages NPLLT, transition forcée).

   Le classement final entre les 2–3 meilleures sections doit se faire au trim exact, avec le niveau le plus fiable,
   idéalement un modèle plus fidèle (NPLLT × XFOIL ou CFD).
3. **Plus de budget AVL pour la prochaine optimisation.** Le mérite choisit rarement AVL (rapport de coût ~45). Pour
   mieux explorer au niveau AVL :
   * vérifier l'optimum plus souvent (`verify_every` 5) ;
   * ou partir de plusieurs graines et garder la meilleure section mesurée ;
   * ou ajouter les meilleures sections connues au plan initial AVL.
4. **NPLLT** guide bien la recherche globale mais ne départage pas les meilleures sections. Il faut garder AVL (ou mieux)
   pour la décision finale, ainsi que la vérification périodique de l'optimum.
5. **Un seul point de fonctionnement.** Il reste à vérifier :
   * l'optimum hors point de conception (autres yaw, gîte, vitesse) ;
   * la cavitation (Cp min) ;
   * l'effet d'une transition naturelle : la transition forcée à 10 % favorise une épaisseur avancée.
6. **Rappels du premier test**
   * Les échecs XFOIL coûtent ~2 min par section et se concentrent vers t_0 faible ; aucun ici.
   * L'image « mur » donne la borne haute de l'effet de plaque d'extrémité de la coque. Elle diffère de la convention
     bdFoil par défaut.
   * Une gîte positive met le foil sous le vent.
   * La suite pytest n'a pas été lancée.

---

## 8. Relancer

```
cd studies/cfoil_intens_kulfan_trim
C:\Users\SIM\.conda\envs\bdToolbox\python.exe step1_run_optimization.py      # mesure des coûts + optimisation
C:\Users\SIM\.conda\envs\bdToolbox\python.exe step2_analysis.py              # vérifications, figures, export
```

**Options :**

* `step1` : `--iterations 80`, `--doe 60 20`, `--cost-sections 3`, `--keep-costs` (garde les coûts de la configuration) ;
* `step2` : `--run runs/<id>`. Step 2 ajoute automatiquement aux candidates les meilleures sections AVL des autres runs du
  dossier.

Tout le problème se règle dans `config_cfoil_trim.json` :

* bloc `trim` : variable, bornes, tolérance, `interpolation_band` ;
* section de référence : `objective.equal_lift_reference`.
