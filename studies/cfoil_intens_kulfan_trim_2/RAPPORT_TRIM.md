# Rapport : section du C-foil d'Intens SY à portance égale, contraintes de réalisabilité, 3 niveaux

Date : 15/10/2026 · run `runs/1015_120910` · branche `feat/cfoil-kulfan-3d` (non commitée) · environnement conda
`bdToolbox` (NeuralFoil 0.2.3, AeroSandbox 4.2.4).

* Figures interactives : [figures/index.html](figures/index.html) (step 0 et step 2).
* Le précédent rapport du dossier était une copie de `../cfoil_intens_kulfan_trim/RAPPORT_TRIM.md` ; il reste dans git
  et dans ce dossier voisin.

---

## 1. Résumé

| Question | Réponse |
|---|---|
| Ce qui a été optimisé | La section Kulfan unique du C-foil, pour la **traînée 3D minimale à la portance 3D de la section de référence**. Le yaw est trimé pour retrouver cette portance, et il doit rester dans **±5 % de 4°, soit [3,8 ; 4,2]°**. |
| Section de référence | La même que dans `../cfoil_intens_kulfan` : l'épaisseur de **MC2_60_6101_1** (t/c 12,25 %) plus la cambrure δ·C(x) avec δ = 0,108139. Cette cambrure est maintenant figée. |
| Contrainte CL2d(0°) = 0,45 | **Supprimée**, comme demandé. Sans elle, la marge de yaw devient une couche très mince : sur 300 sections faisables à cambrure libre, 8 seulement (2,7 %) la tiennent. La cambrure moyenne δ est donc **résolue pour viser un yaw** donné (§ 3.2). |
| Réalisabilité de la section | Sept contraintes connues, la plupart reprises du code du collègue : t/c, Ixx, épaisseur au BF, poids du BA, wiggliness, confiance NeuralFoil et existence de δ (§ 3.3). Elles sont toutes respectées par l'optimum. Ixx est sur sa limite (+0,2 %). |
| Niveaux | NPLLT × NeuralFoil **xxxlarge**, **NPLLT × XFOIL** (nouveau, sans toucher bdFoil) et AVL × XFOIL. Ils ont la **même turbulence** (n_crit 1, transition forcée à 10 %), ce qui est vérifié automatiquement. |
| XFOIL | **0 polaire en échec sur 161** pendant le run (88 NPLLT × XFOIL, 73 AVL). Les runs précédents en avaient : 3 au run du 12/10, 1 au 13/10 (ses 32 autres échecs AVL venaient du trim borné à [3,9 ; 4,1]°), et une section au nez pointu au 14/10. Avec les contraintes de BA et de confiance NeuralFoil, les 30 sections du test du step 0 passent sous les trois réglages essayés ; le réglage d'XFOIL ne change rien de mesurable (§ 4.3). |
| Meilleure section | Selon AVL × XFOIL : CD **0,024489** contre **0,024535**, soit **−0,19 %**. Selon NPLLT × XFOIL : **−0,18 %**. Selon NPLLT × NeuralFoil : **+0,22 %**. Le yaw trimé vaut 4,15 / 4,13 / 4,12°, **dans la marge aux 3 niveaux**. |
| Peut-on conclure ? | **Pas à un gain réel.** Sur les meilleures candidates, les deux modèles XFOIL voient −0,12 à −0,26 % et NeuralFoil +0,1 à +0,4 %. Les trois modèles ne classent pas les candidates de la même façon (Spearman NeuralFoil/AVL −0,74). Sous ces contraintes, **la section de référence est pratiquement optimale**. La section trouvée est une variante de même traînée, plus lisse et au nez plus rond (§ 6). |
| Ce qui limite | **Ixx ≥ référence** (actif), **l'épaisseur au BF** (+1 %), et **la borne haute de la marge de yaw**. L'optimum prend `yaw_target` = 4,14° pour une borne à 4,15° : il veut moins de cambrure et plus de yaw. |
| Changements par rapport au plan validé | (1) δ est résolu par un modèle du yaw, et `yaw_target` devient une variable. (2) La marge de yaw est **imposée aux deux niveaux visqueux**, pas à AVL. AVL demande souvent moins de yaw (jusqu'à 0,55° de moins sur les sections de contrôle) car sa portance ne voit que la ligne de cambrure. Son yaw est rapporté : 66 % de ses évaluations sont dans la marge, et l'optimum l'est. Un réglage permet de revenir en arrière (`BAND_LEVELS`). |

---

## 2. Le problème

| Élément | Valeur (`config_cfoil_trim.json`) |
|---|---|
| Géométrie (fixe) | C-foil d'Intens SY : arc unique R = 9,8 m, arc immergé 3,25 m, corde 0,70 m, une seule section sur toute l'envergure. Coque = plan de symétrie (mur). |
| Attitude | gîte 5° + cant de puits 3,28° ; rake 0,5° ; **yaw trimé** ; sink 0. Seul le yaw varie ; gîte, cant et rake sont fixes. |
| Portance | Pour chaque section et à chaque niveau, le yaw est ajusté jusqu'à ce que CL = √(Cy² + Cz²) égale la portance de la référence au yaw de 4°, calculée **au même niveau**. Valeurs : 0,71828 (NPLLT × NF), 0,71644 (NPLLT × XFOIL), 0,67219 (AVL). |
| Marge | Yaw trimé dans **[3,8 ; 4,2]°** (±5 %). Hors marge, l'objectif reçoit CD + 0,01 × dépassement (°). |
| Objectif | CD = Cdprofile + Cdi·(CL_ref/CL)² au yaw trimé. Le facteur ne fait que retirer le résidu du trim, ≤ 10⁻⁴. |
| Section | Kulfan 4 + 4 poids, forme épaisseur/cambrure. **8 variables** : t₀..t₃ (épaisseur), s₁..s₃ (forme de cambrure), `yaw_target` ∈ [3,85 ; 4,15]°. La cambrure vaut c_i = δ + s_i, avec s₀ = 0 et **δ résolu** (§ 3.2). TE 0,00197, comme MC2. |
| Écoulement | Re 5·10⁶, **n_crit 1, transition forcée à 10 % de corde sur les deux faces**, à tous les niveaux et dans la contrainte de confiance. Polaires de −6° à 12° par pas de 0,5°, npane 250, niter 300. |

---

## 3. Ce qui a été fait

### 3.1 NPLLT × XFOIL, sans toucher bdFoil

bdFoil contient le NPLLT utilisé et il est actif côté collègues : fetch le 15/10, branche locale `fix/npllt-review` (2 commits de retard et 2 d'avance sur `origin/master`). **Rien n'y a
été modifié**, et rien n'y a été écrit : `git status` propre, pas de `bdFoil/tmp/`.

* NPLLT savait déjà lire des polaires XFOIL (`polar_source="xfoil"`). Mais il construisait sa spec XFOIL avec Re et α
  seulement. n_crit, xtr, npane et niter prenaient alors les défauts du core. n_crit, xtr et npane coïncidaient par hasard avec la config ; niter non (100 contre 300).
  Et les polaires partaient dans le cache par défaut, à l'intérieur du clone bdFoil.
* **Correction dans Multifidelity-GP-Clean** (`pipelines/bdtoolbox_foil/solvers.py`) :
  * une sous-classe de `NonPlanarLiftingLine` remplace sa méthode privée de polaire XFOIL ;
  * elle utilise **le même `PolarSpec` qu'AVL** (helper `xfoil_spec`) et le cache de l'étude ;
  * NPLLT × XFOIL et AVL partagent donc les polaires : la deuxième lecture d'une même section prend ~0 s ;
  * un garde-fou s'arrête avec un message clair si bdFoil renomme cette méthode.
* Test ajouté : `tests/test_pipeline_bdtoolbox.py::test_npllt_xfoil_polars_share_the_avl_spec_and_cache`. Il passe dans
  l'env `bdToolbox`. Le reste de la suite donne 95 réussis et 3 sautés (modules absents du venv de test).

### 3.2 Sans contrainte CL2d : δ résolu par un modèle du yaw

Avec la cambrure moyenne libre, le yaw nécessaire à la portance de référence varie énormément. Premier échantillon du
step 0 : 300 sections faisables, δ tiré dans [−0,1 ; 0,4].

* 135 de ces sections ne trouvent pas de trim dans [1 ; 8]° (129 hors d'atteinte, 6 non convergées) ;
* 8 seulement (2,7 %) tombent dans [3,8 ; 4,2]°, et 85 % sortent même de [3 ; 5]° ;
* la marge est donc une couche oblique très mince de l'espace de conception ;
* l'optimiseur de mfego (évolution différentielle sur le mérite) n'y tire presque jamais de point.

**Solution : la cambrure moyenne δ n'est plus une variable, elle est résolue.**

1. **Modèle du yaw** (`study_trim.YawModel`). C'est une régression quadratique du yaw trimé de NPLLT × NeuralFoil en
   fonction de (t₀..t₃, s₁..s₃, δ).
   * Données : 500 sections tirées au step 0, dont 365 trimées. D'abord 300 réparties dans l'espace (165 trimées), puis
     200 proches de la marge (toutes trimées). Le trim est élargi à [1 ; 8]° pour mesurer le yaw même hors marge.
   * Le modèle est recalé pour être **exact sur la référence** : δ = 0,108139 redonne bien 4,000°.
   * Erreur hors échantillon : 0,041°. Validation croisée près de la marge : 0,041°, et 95 % des erreurs < 0,052°
     (`step0_yaw_model.html`).
2. **Projection** (`YawProjector`). Pour chaque section, δ est la cambrure pour laquelle le modèle donne `yaw_target`.
   `yaw_target` est une variable bornée à [3,85 ; 4,15]° : la marge réduite de l'erreur du modèle.
3. Toute section proposée tient donc la marge du niveau 1 **par construction**, à l'erreur du modèle près : 258 sur
   258 pendant le run. Au niveau NPLLT × XFOIL, 81 sur 88 la tiennent ; les 7 autres (3,75 à 4,31°) ont été pénalisées.
   Le trim de chaque niveau reste exact, et le yaw mesuré est contrôlé (§ 3.4).

Ce n'est pas la contrainte CL2d qui revient. La cible porte sur le **yaw 3D**, c'est-à-dire ton exigence réelle, et elle
laisse libre toute la marge.

### 3.3 Contraintes de réalisabilité (connues, géométriques sauf la confiance)

Toutes sont calculées sur la section réelle, δ résolu compris, et rapportées à la référence. Elles servent au plan initial
et annulent le mérite en dehors du domaine faisable.

| Contrainte | Condition | Origine (code du collègue) | Part de l'espace qui la respecte |
|---|---|---|---|
| δ existe | cible de yaw atteignable avec δ ∈ [−0,1 ; 0,4] | (nouvelle) | 100 % |
| Épaisseur | t/c max ≥ 12,25 %, surfaces sans croisement | `local_thickness() > 0` | 69 % |
| **Flexion** | **Ixx ≥ Ixx de la référence**, cambrure comprise | `Ixx() >= mc2_init.Ixx()` | 79 % |
| Bord de fuite | épaisseur locale à x/c 0,90 et 0,95 ≥ référence | `local_thickness(0.9 / 0.95)` | 92 % / 90 % |
| **Bord d'attaque** | **upper₀ > 0,05 et lower₀ < −0,05** | `upper_weights[0] > 0.05`, `lower_weights[0] < -0.05` | 100 % / 81 % |
| Régularité | wiggliness < 2 × référence | `get_wiggliness(...) < 2 * ...` | 35 % |
| Validité NeuralFoil | confiance xxxlarge > 0,90 à α = 0, 2, 4 et 6° | `analysis_confidence > 0.90` | 90 % |
| **Toutes** | | | **14,7 %** |

Ces parts ont été mesurées au step 0 avec `yaw_target` ∈ [3,8 ; 4,2]°, avant son resserrement à [3,85 ; 4,15]°. Avec
les bornes finales, la part totale vaut 14,68 % (`run_summary.json`).

* **Ce que veulent dire les poids du BA.** Près du nez, chaque face suit y ≈ w₀·√x : son rayon vaut w₀²/2. Exiger
  lower₀ < −0,05 impose un rayon minimal à l'intrados. Le run du 14/10, au nez pointu, donnait une confiance NeuralFoil
  de 0 et 2 min d'XFOIL sans convergence (deux essais de 60 s).
* **Contrôle d'Ixx.** Pour la référence, Ixx = 6,7175·10⁻⁵ contre 6,7167·10⁻⁵ avec AeroSandbox (`KulfanAirfoil.Ixx()`),
  soit 0,01 % d'écart. La cambrure compte pour +8 % : c'est pourquoi l'Ixx est maintenant calculé avec δ.
* **Contraintes non reprises.**
  * J = Ixx + Iyy (torsion) : la condition structurelle demandée porte sur Ixx. J est calculé et l'optimum fait +3 %. On
    peut le réimposer dans `STRUCTURAL`.
  * CL(0) ≥ CL_ref et les cibles de CL multipoint : elles sont remplacées par le trim 3D et la marge de yaw.

### 3.4 La marge de yaw et AVL

Sur les mêmes sections, AVL demande **souvent moins de yaw** que les deux modèles visqueux. Au step 0, l'écart va
de −0,55° à +0,04°. Pendant le run, le yaw AVL va de 2,96° à 4,41°, contre 3,75–4,31° pour NPLLT × XFOIL.

La raison est connue des tests précédents : AVL ne voit que la ligne de cambrure, sans décambrure visqueuse. Une section
qui compense une autre répartition d'épaisseur par plus de cambrure paraît donc trop portante dans AVL.

**Choix fait :**

* la marge est **imposée aux deux niveaux visqueux** (NPLLT × NeuralFoil et NPLLT × XFOIL, `BAND_LEVELS = (1, 2)`) ;
* AVL n'est pas pénalisé, mais son yaw est mesuré et rapporté ;
* une section n'est retenue que si elle est dans la marge aux niveaux visqueux.

**L'optimum retenu est aussi dans la marge pour AVL (4,12°).** Pour imposer la marge à AVL aussi, il suffit de mettre
`BAND_LEVELS = (1, 2, 3)`.

### 3.5 Les autres réglages

* **Plan initial et itérations.** Plan initial plus dense, 150 / 40 / 25 points (L1 / L2 / L3), contre 100 / 30 en
  2 niveaux au run du 13/10. **200 itérations** au lieu de 100. L'optimum du modèle est évalué en L3 toutes les 10 itérations.
* **Turbulence.** n_crit 1 et transition forcée à 10 % partout. `study_trim.level_settings` relit les réglages
  effectifs de chaque niveau et s'arrête s'ils diffèrent. Avant ce travail, le niveau NPLLT × XFOIL aurait ignoré la
  config.
* **NeuralFoil xxxlarge** au niveau 1 et dans la contrainte de confiance.

---

## 4. Mise en place (step 0, `step0_check_setup.py`)

### 4.1 Référence

* L'ajustement Kulfan de MC2_60_6101_1 est retrouvé à 5·10⁻⁷ près sur les poids d'épaisseur.
* δ résolu par le modèle = 0,108139, la valeur figée.
* Toutes les contraintes sont respectées. Confiance NeuralFoil 0,977, upper₀ 0,274, lower₀ −0,058.

### 4.2 Référence aux 3 niveaux, au yaw de 4°

| Niveau | CL_ref | CD_ref | Cdi | Cdprofile |
|---|---|---|---|---|
| NPLLT × NeuralFoil xxxlarge | 0,71828 | 0,026769 | 0,018270 | 0,008499 |
| NPLLT × XFOIL | 0,71644 | 0,026760 | 0,018190 | 0,008570 |
| AVL × XFOIL | 0,67219 | 0,024535 | 0,016022 | 0,008514 |

* NeuralFoil xxxlarge et XFOIL donnent **la même portance à 0,3 % près** dans NPLLT. AVL porte 6 % de moins, avec son
  propre modèle 3D.
* Le CL2d(0°) de la référence est vu à 0,453 par NeuralFoil comme par XFOIL.

### 4.3 Robustesse XFOIL (30 sections faisables, sans cache)

| Réglage | Polaires en échec | Sections avec points comblés | Temps par polaire |
|---|---|---|---|
| pas 0,5°, niter 100 (runs précédents) | 0 / 30 | 3 | 6,4 s |
| **pas 0,5°, niter 300 (retenu)** | **0 / 30** | 3 | 9,3 s |
| pas 0,25°, niter 300 | 0 / 30 | 14 | 23,8 s |

* **Avec les contraintes de forme, les 30 sections passent sous les trois réglages.** Le driver bdFoil balaie de −6° à
  12° en partant à froid de −6°, et une extrémité manquante fait échouer toute la polaire : les sections au nez trop
  fin, écartées ici par les contraintes de BA et de confiance NeuralFoil, étaient celles qui échouaient au 14/10.
* **Un pas plus fin ou plus d'itérations n'y changent rien**, et le pas fin coûte 2,5 fois plus cher. niter 300 a été
  gardé, puisqu'il coûte peu.
* **Pendant le run : 0 échec XFOIL** sur 161 polaires. 24 évaluations L2 et 17 évaluations L3 ont eu au moins un angle
  comblé par interpolation, 0,3 point par polaire en moyenne (`step2_xfoil.html`).

---

## 5. Déroulement du run

**Durée : 2 h 04.**

| Poste | Temps |
|---|---|
| Recherche du point suivant (évolution différentielle + contraintes) | **75 min** |
| Simulations | 26 min |
| Ajustements du GP | 5 min |

| Niveau | Évaluations | dont plan initial / choisies par le mérite / vérifications | Échecs | Yaw trimé | Dans la marge |
|---|---|---|---|---|---|
| NPLLT × NF xxxlarge | 258 | 150 / 108 / 0 | 0 | 3,81–4,19° | 258 (100 %) |
| NPLLT × XFOIL | 88 | 40 / 48 / 0 | 0 | 3,75–4,31° | 81 (92 %), 7 pénalisées |
| AVL × XFOIL | 73 | 25 / 44 / 4 | 3 (yaw AVL hors de [3 ; 5]°) | 2,96–4,41° | 46 / 70 (66 %), non imposée |

* **Coûts mesurés** : 0,43 / 13,0 / 15,0 s par évaluation, soit un rapport 1 : 30,5 : 35,2. Pendant le run, les moyennes
  ont été de 0,32 / 12,8 / 20,4 s, soit environ 1 : 40 : 64. La mesure était biaisée vers le bas : une des deux sections
  L2 a trouvé sa polaire dans le cache (2,65 s contre 23,3 s). Le step 1 mesure désormais les coûts avec un cache
  vide.
* **Le mérite a choisi les niveaux chers bien plus souvent qu'avant** : 48 fois L2 et 44 fois L3 en 200 itérations,
  contre 8 fois AVL en 100 itérations au run du 13/10, et 0 en 50 au 12/10.
* **Le meilleur point a été trouvé à l'évaluation 277**, lors d'une vérification de l'optimum du modèle. Plus rien de
  meilleur ensuite (`step2_convergence.html`).
* **Vérifications de l'optimum du modèle** : 4. L'écart entre mesure et prédiction va de −0,06 % à +0,23 %.
* **Trim** : 2 appels au solveur dans 98 % des cas, et interpolation de la dernière étape dans la moitié des cas.
  Résidu de portance ≤ 10⁻⁴.

---

## 6. Résultats

### 6.1 Référence contre optimum, à la portance de la référence

L'optimum est la meilleure section AVL du run. Elle est aussi l'optimum effectif du modèle.

| | NPLLT × NF : réf. | NPLLT × NF : opt. | NPLLT × XFOIL : réf. | NPLLT × XFOIL : opt. | AVL : réf. | AVL : opt. |
|---|---|---|---|---|---|---|
| **CD** | 0,026769 | 0,026829 (**+0,22 %**) | 0,026760 | 0,026711 (**−0,18 %**) | 0,024535 | 0,024489 (**−0,19 %**) |
| Cdi | 0,018270 | 0,018281 | 0,018190 | 0,018186 | 0,016022 | 0,016028 |
| Cdprofile | 0,008499 | 0,008549 (+0,6 %) | 0,008570 | 0,008522 (−0,6 %) | 0,008514 | 0,008459 (−0,6 %) |
| yaw trimé | 4,000° | 4,150° | 4,000° | 4,132° | 4,000° | 4,122° |
| CL2d(0°) vu | 0,453 | 0,438 | 0,453 | 0,438 | 0,453 | 0,438 |

* **Tout se joue sur la traînée de profil.** La traînée induite ne bouge pas : même planform, même portance.
* **Les deux modèles XFOIL disent −0,6 % de traînée de profil, NeuralFoil dit +0,6 %.** C'est le désaccord central de
  ce run.
* La section porte un peu moins en 2D à 0° (CL2d 0,438 contre 0,453). Elle compense par 0,12 à 0,15° de yaw en plus,
  ce qui la place dans le haut de la marge.

### 6.2 Les sections candidates aux 3 niveaux

Écart de CD à la référence, au trim exact de chaque niveau. L'astérisque signale un yaw hors marge.

| Section | NPLLT × NF | NPLLT × XFOIL | AVL × XFOIL |
|---|---|---|---|
| **optimum** (meilleure AVL du run = optimum du modèle) | +0,22 % (4,15°) | −0,18 % (4,13°) | **−0,19 %** (4,12°) |
| 2ᵉ section AVL du run | +0,19 % (4,17°) | −0,22 % (4,14°) | −0,17 % (4,06°) |
| 3ᵉ section AVL du run | +0,39 % (4,07°) | −0,12 % (4,04°) | −0,17 % (4,16°) |
| minimum de la moyenne AVL du modèle | +0,37 % (4,03°) | −0,21 % (3,98°) | −0,14 % (3,98°) |
| meilleure NPLLT × XFOIL du run | +0,21 % (4,05°) | **−0,26 %** (4,02°) | −0,12 % (3,91°) |
| meilleure NPLLT × NF du run | **−0,05 %** (3,88°) | −0,12 % (3,85°) | +0,07 % (3,73°*) |
| meilleure AVL du run de fumée `1015_120921` | +0,10 % (4,03°) | −0,22 % (4,01°) | −0,01 % (3,84°) |

* **NPLLT × XFOIL voit un petit gain sur toutes les candidates**, de −0,12 à −0,26 %. **AVL aussi sur les
  meilleures**, de −0,12 à −0,19 %, mais pas sur deux d'entre elles (+0,07 % et −0,01 %). **NeuralFoil voit une petite
  perte presque partout**, de +0,1 à +0,4 %.
* **Classement des 8 sections, Spearman** : NeuralFoil/AVL −0,74 ; NeuralFoil/NPLLT × XFOIL −0,26 ; NPLLT × XFOIL/AVL
  +0,38. À l'échelle de 0,1 à 0,3 %, les trois modèles **ne s'accordent pas sur l'ordre**.
* Ces écarts sont plus petits que les différences entre modèles. Aucune section ne bat la référence de façon
  convaincante (`step2_levels.html`, `step2_comparison.html`).

### 6.3 La section optimale

| Grandeur | Référence | Optimum |
|---|---|---|
| t₀ ; t₁ ; t₂ ; t₃ | 0,332 ; 0,376 ; 0,161 ; 0,209 | 0,394 ; 0,313 ; 0,177 ; 0,210 |
| δ ; s₁ ; s₂ ; s₃ | 0,108 ; 0 ; 0 ; 0 | 0,085 ; 0,008 ; 0,043 ; 0,019 |
| yaw_target | 4,00° | 4,14° (borne haute 4,15°) |
| t/c max (position) | 12,25 % (26 %) | 12,30 % (24 %) |
| t/c à 40 % / 70 % | 11,2 / 5,7 % | 10,9 / 5,7 % |
| cambrure max (position) | 4,16 % (33 %) | 3,87 % (41 %) |
| rayon de BA extrados / intrados (w₀²/2) | 0,038 / 0,0017 | 0,040 / **0,0063** |
| Ixx / référence | 1 | **1,002 (contrainte active)** |
| J / référence (non imposé) | 1 | 1,03 |
| épaisseur à x/c 0,90 / 0,95 | réf. | +1,3 % / +1,0 % |
| wiggliness | 0,068 | 0,024 (2,8 fois plus lisse) |
| confiance NeuralFoil | 0,977 | 0,93 |

* **Épaisseur** : maximum légèrement avancé, un peu plus mince entre 30 et 50 % de corde.
* **Cambrure** : plus faible et plus reculée, ce qui explique que la section demande plus de yaw.
* **Nez** : **intrados nettement plus rond**, avec un rayon multiplié par 3,7.
* **Réalisabilité** : l'arrière garde l'épaisseur de la référence. Ce n'est plus la section au bord de fuite aminci des
  tests précédents.
* Figures : `step2_sections.html` (formes, zooms BA et BF, épaisseur et cambrure) et `step2_constraints.html` (marge à
  chaque contrainte).

---

## 7. Qualité du modèle multifidélité

* **Leave-one-out sur les 70 points AVL**, avec les points NPLLT gardés :
  * RMSE de **25 %** de la dispersion ;
  * **96 %** des points dans l'intervalle à 95 %.
* **Prédictions hors données** (4 candidates jamais évaluées par AVL dans le run) :
  * écart de −0,04 à −0,12 %, et |z| ≤ 1,3 ;
  * le modèle est légèrement pessimiste.
* **Corrélations entre niveaux**
  * ρ vaut **1,010** (NF → XFOIL) et **0,995** (XFOIL → AVL).
  * Les surfaces des niveaux 1 et 2 classent les 70 points AVL avec un Spearman de 0,87. C'est un bon accord sur
    l'ensemble de l'espace : les traînées AVL du run vont de −0,19 % à +5,1 % autour de la référence. Il ne vaut pas
    entre les meilleures sections (§ 6.2).
* **Longueurs de corrélation du terme d'écart AVL − ρ·(NPLLT × XFOIL)** : t₃ (0,35), s₃ (0,46), `yaw_target` (0,57)
  et s₁ (0,59) sont les plus courtes, t₂ (1,5) la plus longue. Elles décrivent où AVL s'écarte de NPLLT × XFOIL, pas les
  variables qui pilotent la traînée AVL elle-même.
* Surface de réponse :
  * modèle rechargeable : `runs/1015_120910/surrogate.json` ;
  * échantillon de Sobol aux 3 niveaux : `results/response_surface_sobol.csv` ;
  * coupes : `results/response_surface_slices.npz`, `step2_gp_slices_1d.html`, `step2_gp_slices_2d.html`.

---

## 8. Limites et recommandations

1. **Pas de gain démontré sous ces contraintes.**
   * Le meilleur résultat est −0,19 % (AVL) et −0,18 % (NPLLT × XFOIL), alors que NeuralFoil xxxlarge donne +0,22 %.
   * Ces écarts sont plus petits que le désaccord entre modèles.
   * Avec Ixx, l'épaisseur au bord de fuite et la marge de yaw imposés, **la section de référence MC2 + cambrure est déjà
     quasi optimale**.
   * Les −1 % des tests précédents venaient d'un arrière aminci (t/c 6,5 % à mi-corde, `../cfoil_intens_kulfan_trim/RAPPORT_TRIM.md` § 5.2), que les nouvelles contraintes interdisent.
2. **Choisir l'arbitre.**
   * NeuralFoil et XFOIL divergent sur la traînée de profil à l'échelle de 0,5 %.
   * Pour départager, il faudrait un calcul plus fidèle (CFD 2D sur 2 ou 3 sections) ou un essai.
   * Sinon, retenir les deux modèles XFOIL, qui s'accordent sur le signe.
3. **La marge de yaw est active.** L'optimum vise 4,14° pour une borne à 4,15° : moins de cambrure et plus d'incidence
   réduisent la traînée de profil. Élargir la marge déplacerait l'optimum, mais c'est une exigence de conception.
4. **Définition de la marge.**
   * Elle dépend du modèle de portance, puisqu'AVL demande souvent moins de yaw que les niveaux visqueux.
   * Le choix fait ici est la portance visqueuse.
   * Pour imposer aussi AVL : `BAND_LEVELS = (1, 2, 3)` dans `study_trim.py`.
5. **Le modèle du yaw est propre à cette planform et à cette attitude.** Si la config change (géométrie, attitude, Re,
   référence), relancer `step0_check_setup.py`, qui le réajuste en ~3 min.
6. **Temps de calcul.** La recherche du point suivant prend 60 % du run. Pour aller plus vite :
   * réduire la population de l'évolution différentielle ;
   * ou ne calculer la confiance NeuralFoil que sur les meilleurs candidats.
7. **Restent à vérifier** :
   * hors point de conception (autres gîte, vitesse) ;
   * cavitation (Cp min) ;
   * transition naturelle (n_crit 9, xtr libre).

---

## 9. Fichiers et relance

| Fichier | Rôle |
|---|---|
| `config_cfoil_trim.json` | le problème |
| `study_trim.py` | référence, contraintes connues, modèle du yaw et projection de δ, simulateur avec marge, lecture d'un run |
| `step0_check_setup.py` | ajustement du modèle du yaw (`results/yaw_model_level1.json`), contrôles, robustesse XFOIL (~10 min) |
| `step1_run_optimization.py` | mesure des coûts puis optimisation 3 niveaux (~2 h) |
| `step2_analysis.py` | re-mesure des candidates aux 3 niveaux, leave-one-out, figures, export (~6 min) |
| `../../pipelines/bdtoolbox_foil/solvers.py` | NPLLT × XFOIL (générique) |

```
cd studies/cfoil_intens_kulfan_trim_2
C:\Users\SIM\.conda\envs\bdToolbox\python.exe step0_check_setup.py        # modèle du yaw + contrôles
C:\Users\SIM\.conda\envs\bdToolbox\python.exe step1_run_optimization.py   # coûts + optimisation
C:\Users\SIM\.conda\envs\bdToolbox\python.exe step2_analysis.py           # vérifications, figures
```

**Options :**

* `step0` : `--fit-sections 300 200`, `--skip-xfoil`, `--keep-yaw-model` (garde le modèle déjà ajusté) ;
* `step1` : `--doe 150 40 25`, `--iterations 200`, `--cost-sections 2`, `--keep-costs` ;
* `step2` : `--run runs/<id>`.

Les réglages de l'étude (marge, pénalité, seuils des contraintes, niveaux où la marge est imposée) sont des constantes
en tête de `study_trim.py`.
