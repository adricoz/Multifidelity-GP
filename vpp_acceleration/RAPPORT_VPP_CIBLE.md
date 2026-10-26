# Raffinement du surrogate du C-foil ciblé par la VPP — premier essai sur méta-surrogate

21/10/2026 · branche `new_master` de MDS · code dans [`src/`](src/), études dans [`examples/`](examples/)

Ce rapport répond à trois questions, sur un méta-surrogate léger (les codes NPLLT, AVL et CFD ne sont pas lancés dans
ce premier essai) :

1. combien de points d'intégration (N_sobol) le critère séquentiel demande ;
2. quel plan LHS de départ prendre en dimension 16 (12 modes PCA de la section + cant, rake, yaw, Hvol) ;
3. combien de calculs CFD il faudra pour valider le surrogate.

## Les réponses en bref

| Question | Réponse sur le méta-problème | Étude |
| --- | --- | --- |
| **N_sobol pour l'intégration** | **8 à 16 points par état VPP**, soit 160 à 320 pour 20 états. Dès 4 par état, le critère choisit le même point qu'avec 10 240 points (erreur 0,6 %) ; au-delà de 16, c'est du temps perdu. **512 à 1 024 candidats** donnent 99 % du meilleur rendement. | 1 |
| **Plan LHS de départ en 16D** | **8 BF par HF.** Un LHS seul plafonne à 0,15 nd d'erreur de vitesse dès 256 BF + 32 HF (coût 576) ; tripler le coût ne fait presque rien. Il vaut mieux **démarrer à 128 BF + 32 HF (≈ 8d BF, 2d HF, d = 16) puis raffiner** en ciblé. | 2, 3 |
| **Le raffinement ciblé VPP** | À coût égal (598), l'erreur de vitesse vraie vaut **0,015 nd en moyenne (0,062 nd au pire) contre 0,081 nd (0,28 nd) pour le LHS**, soit 5 fois moins, sur 3 graines sur 3. L'ALC global fait moins bien que le LHS. | 3 |
| **Calculs CFD de validation** | **20 calculs aux états VPP** donnent l'erreur RMS de vitesse à ±30 % près (90 %), et **40 à ±20 %**. **5 à 10 suffisent** à voir une incertitude mal calibrée (ici surestimée d'un facteur 2,5). Avec un 3e niveau CFD, compter **30 à 50 calculs**, d'après le nombre de HF utilisés ici. | 4, 3 |

Ces chiffres valent pour le méta-problème : RS réelle du foil, VPP réelle, mais niveaux BF/HF fabriqués. Les vrais
NPLLT et AVL peuvent différer plus, ou moins régulièrement, et demander plus de points HF. Les ordres de grandeur et les
règles (points par état, rapport BF/HF, gain du ciblage) sont ce qui se transpose.

---

## 1. La VPP qu'on utilise : MDS, branche `new_master`

**Ce que c'est.** MDS est un VPP écrit en Python : `Code/` contient le cadre, `Systems/` les bateaux. Le bateau
**IntensSY** (`Systems/Ships/Sail/IntensSY/`) y a été transféré depuis vpp36. Il reproduit vpp36 à 2·10⁻⁶ nd près à
TWA 45 / TWS 16 (`MDS/CLAUDE.md`). Le bateau est une somme de composants qui rendent chacun un torseur à 6 composantes :
poids, coque scindée (Archi + Dyn), foil (Dyn + Archi), deux safrans, voile, fardage. La VPP cherche l'état où le torseur
total s'annule, en maximisant la vitesse. J'ai fait passer le clone local `Code-Adri/MDS` de `master` à `new_master` ;
c'est une avance rapide de 107 commits, sans perte, car seules les cartes Qwen n'étaient pas suivies.

**Comment l'utiliser, vérifié dans le code :**

| Besoin | Où / comment |
| --- | --- |
| Construire le bateau | `build_IntensSY(foil=..., sail=..., hull='split_v1', rudder='v2', windage='rs_v1', twin_rudders=True)` (`IntensSY_config.py`) ; renvoie `(system, origin)` |
| Variantes de foil | `'Dagg'` (dérive, défaut), `'CFoil'` (C-foil R 7,8 m, entrées Cant/Rake/Yaw), `'CFoilFSI'` (R 9,8 m, ajoute `Speed_kts`, sorties Bending/Twisting) ; ou un dict `{'dyn', 'archi', 'surface', 'origin'}` |
| Un point de VPP | `IntensSY_flying_NCpre.py` : SLSQP maximise `Speed` sous les 6 équations `Eq.Fx…Eq.Mz`, inconnues Speed, Leeway, Heel, Trim, CC_Dspl, Rudder_Angle, Flat |
| Un balayage TWA × TWS | `Scripts/VPP/vpp_sweep.py` (618 calculs en ~3 min, d'après son README) |
| Interface graphique | `python -m Cockpit` (NiceGUI ; lire `Cockpit/CLAUDE.md` avant d'y toucher) |

**Ce que fait MDS du foil** (`intenssy_subsystems.py`, classe `IntensSY_foil_dyn`, lu et vérifié) :

* `Cant_Eff = Heel + Foil_Cant`, `Rake_Eff = Trim + Foil_Rake`, `Yaw_Eff = Leeway + Foil_Yaw` ;
* le vecteur d'entrée de la RS est assemblé **par nom** (`foil_rs_vector`), d'après ses `input_names` ;
* `surrogate.regression(x)` remplit `prediction` = (Fx, Fy, Fz, Mx, My, Mz, Cpmin) ;
* les 6 premières sorties sont multipliées par qS = ½ ρ S V², avec S = 2,275 m² pour le C-foil ; `Cpmin` est lu par
  son nom.

**Conséquence pour nous : on branche le surrogate sans toucher MDS.** Il suffit qu'un objet ait les quelques
attributs de `VppChebyRS` : `SurrogateFoilRS` dans [`src/vpp_coupling.py`](src/vpp_coupling.py). Il ajoute ce que la
RS ne voit pas, les 12 scores de la section et le Hvol, lu dans les variables globales du bateau (les coques sont
évaluées avant le foil). Il n'y a plus de contrainte Python 2 : celle-ci venait de vpp36. MDS prévoit aussi une entrée
pour la variance d'un surrogate (`sub_system.py`, l. 1129) ; on ne l'utilise pas encore.

**Mesures sur cette machine :**

* construction du bateau : 7 s ;
* un point de VPP : 1,3 à 2 s depuis le point de départ fixe, 0,07 s depuis un point voisin ;
* environ 240 évaluations du foil par point.

Avec notre adaptateur branché sur la RS de production, la vitesse est **identique à 0,0 près** (test
`test_adapter_reproduces_the_production_foil`).

**Pièges rencontrés :**

* **Environnement.** MDS demande pymoo, numba, gekko, pydmd… L'environnement `.venv` de bdToolbox n'a pas pymoo, et
  conda `bdToolbox` a un pymoo cassé (`future` manquant). Je n'ai touché à aucun environnement partagé. J'ai créé
  `Multifidelity-GP-Clean/.venv-mds` : un venv du conda `bdToolbox` qui voit ses paquets, avec en plus `future`,
  `transforms3d`, `PyWavelets`, `pydmd` (sans dépendances) et `pytest`.
* **Le résultat d'un calcul dépend du calcul précédent.** L'état interne de MDS, par exemple la fermeture de
  l'enfoncement par la coque, se reporte d'un calcul à l'autre. SLSQP peut alors s'arrêter, en déclarant le succès,
  sur un équilibre plus lent : 14,83 nd une fois, 17,58 nd depuis le même point de départ après d'autres calculs.
  `IntensSYVPP.solve(..., extra_guesses=...)` relance depuis plusieurs départs et garde le plus rapide ; la vérité de
  l'étude est calculée ainsi.
* **Les boîtes de la RS du C-foil.** Cant [0, 8]°, Rake [−1,5, 1,5]°, Yaw [−6, 6]°. MDS évalue le foil **sans
  écrêtage** : hors boîte, la RS extrapole.

Qwen a cartographié IntensSY et Cockpit (`REPO_MAP.md`, à vérifier avant de citer) et extrait les lois du foil, que
j'ai vérifiées ligne à ligne.

## 2. Le méta-surrogate (premier essai, sans NPLLT, AVL ni CFD)

[`src/meta_foil.py`](src/meta_foil.py). Il prend les **16 entrées** z₁…z₁₂ (les scores PCA de la section, famille à
16 poids lisses, k = 12), Cant_Eff, Rake_Eff, Yaw_Eff et Hvol, et rend les 7 coefficients de la RS du C-foil.

* **L'ancre, c'est la RS C-foil du bateau.** Au profil de référence et à Hvol = 0,20 m, le niveau HF **est** le foil
  de production (test). La VPP du méta-problème est donc la vraie VPP d'Intens SY.
* **Effet de la section (ligne portante).** NeuralFoil donne, pour chaque profil, la pente de portance, l'angle de
  portance nulle α₀, CD(α), CM(α) et Cpmin(α) (Re 6·10⁶, n_crit 1, transition à 10 %). La portance de l'ancre est lue
  comme a₃D (α_eff − α₀,réf) ; le profil la change en a₃D′ (α_eff − α₀′). La traînée prend la différence de CD de profil
  plus la variation de traînée induite L²/(π AR e). Les moments transportent toute variation de force (r × ΔF, avec r
  ajusté sur l'ancre), plus le moment de profil.
* **Effet de l'immersion.** Toutes les forces sont multipliées par 1 + (Hvol − 0,20)/envergure.
* **Deux niveaux.**
  * **« AVL » (HF, coût 10)** : NeuralFoil xxxlarge, e = 0,85, perte de portance près de la surface libre.
  * **« NPLLT » (BF, coût 1)** : NeuralFoil xxsmall, ligne portante elliptique avec un allongement trop grand de 10 %,
    pas de surface libre, portance de l'ancre +5 % et traînée −10 %.

  Corrélation BF/HF > 0,99 à attitude fixe, avec des écarts systématiques : la structure que suppose l'AR(1).
* **Domaine du test.** Les profils de la famille sur lesquels NeuralFoil est fiable (confiance ≥ 0,9 de 0 à 6°), soit
  60 % de la famille. C'est un choix de ce test, pas une décision sur la famille.
* **Coût.** Une polaire NeuralFoil par nouveau profil (1,2 s pour 200 profils en xxxlarge, puis en cache).

## 3. L'algorithme : co-krigeage séquentiel ciblé par la VPP

[`src/vpp_targeted.py`](src/vpp_targeted.py). C'est **notre co-krigeage séquentiel** : mêmes gains ALC exacts
(`sequential_design.batched_imse_gains`), même règle gain / coût cumulé, même plan imbriqué. Trois ajouts :

1. **Un co-krigeage par coefficient** (6 modèles, ρ borné v2), tous sur les mêmes points : un calcul de code donne tout
   le torseur.
2. **On intègre là où la VPP travaille.** On place les moyennes du surrogate dans MDS, puis on résout l'équilibre de
   chaque (section de conception, condition de navigation). Les points d'intégration forment un nuage de Sobol autour
   de chaque état : section fixe, attitude ± (1° de cant, 0,3° de rake, 1° de yaw, 5 cm de Hvol).
3. **On pondère par ce qui compte pour la vitesse.** Au premier ordre, Var V = Σ_k (∂V/∂C_k)² Var C_k. Le critère est
   donc Σ_k w_k G_k / coût, avec w_k = (∂V/∂C_k)², calculé par différences finies dans la VPP. C'est la baisse de la
   variance de la vitesse prédite, en nd², par unité de coût.

**Une itération :**

* les hyperparamètres sont réajustés tous les 5 pas ; entre deux, le modèle est seulement reconditionné ;
* les états VPP sont mis à jour tous les 10 pas, et les poids une mise à jour sur trois ;
* les candidats sont pris autour des états, partout ailleurs dans le domaine, et aux points BF sans HF ;
* le couple (point, niveau) au meilleur rendement est lancé.

Un pas coûte environ 1,7 s, et une mise à jour des états avec leurs sensibilités environ 25 s.

**Les sensibilités mesurées sur Intens SY** (méta-HF, profil de référence) :

| Condition | ∂V/∂C_Fx | ∂V/∂C_Fy | ∂V/∂C_Fz | ∂V/∂C_Mx |
| --- | --- | --- | --- | --- |
| TWA 45 / TWS 16 | 108 | −1,9 | 4,8 | 2,5 |
| TWA 75 / TWS 16 | 91 | 0,9 | 5,9 | 5,1 |
| TWA 90 / TWS 14 | 281 | 14,7 | 14,7 | −0,5 |

Les sensibilités sont en nœuds par unité de coefficient. Une erreur de 10⁻⁴ sur le coefficient de traînée coûte
environ 0,01 nd : **c'est la traînée qu'il faut apprendre**, et le critère pondéré le sait.

### 3 bis. Où le ciblage VPP intervient dans le code, et ce qu'il change par rapport à l'ALC simple

**Le cœur est le même : une moyenne de réductions de variance.** Les deux critères calculent, pour chaque couple
(point candidat x, niveau l), le gain

  G(x, l) = (1/N) Σ_j [ σ²_avant(u_j) − σ²_après(u_j | on observe x au niveau l) ]

dans [`batched_imse_gains`](src/sequential_design.py#L284). C'est la [ligne 335](src/sequential_design.py#L335), où la
moyenne porte sur les points d'intégration u_j. Le critère lance ensuite le couple qui a le meilleur G / coût. Pour un
GP, σ²_après ne dépend pas de la valeur qu'on observera, donc G se calcule avant de lancer le code.

**Trois ingrédients changent :**

| | ALC simple ([`sequential_design.py`](src/sequential_design.py#L605-L609)) | Ciblé VPP ([`vpp_targeted.py`](src/vpp_targeted.py#L265-L297), [`vpp_optimize.py`](src/vpp_optimize.py)) |
| --- | --- | --- |
| **Où**, les points u_j | 1 024 points de Sobol répartis sur tout le domaine | des nuages de Sobol autour des **états d'équilibre de la VPP** ([`integration_cloud`](src/vpp_targeted.py#L204)) |
| **Quoi**, la variance réduite | celle d'**une** sortie du code | celle de la **vitesse** : Σ_k w_k σ²_k, avec w_k = (∂V/∂C_k)², calculé dans MDS par différences finies ([`speed_weights`](src/vpp_targeted.py#L232), [ligne 238](src/vpp_targeted.py#L238)) |
| **Quand**, le lieu | fixé une fois pour toutes | recalculé tous les 10 pas : le surrogate du moment va dans la VPP, on résout, les états bougent ([`update_states`](src/vpp_targeted.py#L327)) |

En maths, l'ALC simple intègre la variance contre la mesure uniforme du domaine. Le ciblé l'intègre contre la mesure
empirique des états VPP, chaque sortie pesant ce qu'elle coûte en vitesse. C'est la réduction intégrée de la variance de
la vitesse prédite, en nd².

**Une image.** L'ALC simple révise tout le livre uniformément. Le ciblé révise les chapitres qui tomberont à l'examen
(les états où la VPP interroge le surrogate), au prorata de leur barème (l'effet de chaque coefficient sur la vitesse),
et regarde à nouveau, à chaque révision, quels chapitres tomberont.

**Ce que ça change, mesuré sur le méta-problème** (étude 3) :

* **les points tombent près des états :** à 0,8 demi-largeur de nuage, contre 4,7 pour l'ALC global ;
* **le choix du niveau s'inverse :** sur tout le domaine, la BF est le moyen le moins cher de baisser la variance, et
  l'ALC global n'a acheté que de la BF (150 sur 150). Près des états, c'est la traînée HF qui compte pour la vitesse :
  le ciblé a acheté 12 HF ;
* **l'erreur de vitesse est 5 fois plus faible**, à coût égal.

**Pour l'optimisation de la section** ([`vpp_optimize.py`](src/vpp_optimize.py)), les états sont ceux des profils que
l'optimiseur juge les meilleurs à l'instant t :

* le meilleur profil du moment ;
* les 3 profils les mieux classés par un tri au premier ordre (gain de vitesse prédit plus son incertitude).

Le surrogate devient donc précis **là où se prend la décision « quel profil est le meilleur »**, et non sur tout
l'espace des profils.

**Depuis le 23/10** (4e run, [`CHANGEMENTS_2026-10-23.md`](CHANGEMENTS_2026-10-23.md)), deux choses changent dans
la version réelle :

* **chaque point de contrôle a son poids**, ω_ik = w_c (∂V/∂C_k)²_c / n_s, au lieu d'un poids par coefficient moyenné
  sur les conditions. w_c est la probabilité de la condition c de son état, et n_s le nombre de points de cet état
  ([`point_weights`](src/vpp_optimize.py#L577)) ;
* **la recherche est locale :** la moitié des sections criblées sont tirées dans une région de confiance autour du
  meilleur profil, et un point de contrôle sur deux déplace aussi la section de ce rayon.

Le détail et les formules sont dans [`doc/algorithme_vpp_cible.tex`](doc/algorithme_vpp_cible.tex) (§ 7 et 9) et
[`doc/selection_points_vpp.tex`](doc/selection_points_vpp.tex).

## 4. Le cadre commun des études

[`examples/vpp_study_common.py`](examples/vpp_study_common.py) fixe le cadre des quatre études.

* **Navigation.** 5 conditions (TWA / TWS à 10 m) : 45/12, 45/16, 60/16, 75/16, 90/14.
* **Sections de conception.** 4 : le profil de référence et 3 profils de la famille, à moins de 2,5° de l'angle de
  portance nulle de la référence. Ce sont les profils qu'un optimiseur explorerait d'abord.
* **La « vérité ».** C'est la VPP MDS avec le niveau HF comme foil, résolue depuis deux départs, avec ses sensibilités
  ∂V/∂C_k. Elle donne 20 états, tous convergés, entre 11,5 et 22,8 nd (`results/vpp_truth.npz`, 4 min une fois pour
  toutes). Les sensibilités utilisées sont la médiane par condition sur les 4 sections : elles varient d'environ 10 %
  d'une section à l'autre, et une différence finie qui saute d'un équilibre à l'autre donnerait une valeur aberrante.
* **La mesure principale.** e_V = RMS sur les 20 états de Σ_k (∂V/∂C_k)(Ĉ_k − C_k) : l'erreur de vitesse, au premier
  ordre, que causeraient les erreurs du torseur. On la calcule sans relancer la VPP. Pour l'étude 3, on la vérifie
  contre la vraie erreur, celle de la VPP résolue avec le surrogate.

La figure [`figures/vpp_meta_overview.html`](runs/2026-10-21_meta_vpp_cible/figures/vpp_meta_overview.html) montre ce que voit
l'algorithme : les coefficients BF/HF, les vitesses, les états et les sensibilités. Les états n'occupent qu'une petite
partie de la boîte d'attitude : cant de 2 à 5,8°, yaw de −3 à +1,6°, Hvol de 0 à 0,34 m. C'est pour cela que le ciblage
paie.

## 5. Étude 1 — combien de points d'intégration et de candidats ?

[`examples/vpp_integration_study.py`](examples/vpp_integration_study.py) ·
[`figures/vpp_integration_study.html`](runs/2026-10-21_meta_vpp_cible/figures/vpp_integration_study.html).

* **Cadre.** Départ 128 BF + 32 HF, 20 états VPP, 1 120 candidats. Le critère est calculé avec n points de Sobol par
  état, 5 tirages pour chaque n. La référence en compte 512 par état, soit 10 240 points.

| Points par état (total) | Erreur relative du critère | Même choix que la référence | Efficacité du choix (pire) | Temps |
| --- | --- | --- | --- | --- |
| 2 (40) | 4,3 % | 40 % | 99,4 % (97,3 %) | 4,0 s |
| 4 (80) | 0,6 % | 100 % | 100 % | 4,5 s |
| **8 (160)** | **0,4 %** | **100 %** | **100 %** | **5,3 s** |
| **16 (320)** | **0,2 %** | **100 %** | **100 %** | **7,2 s** |
| 64 (1 280) | < 0,1 % | 100 % | 100 % | 18,8 s |
| 256 (5 120) | < 0,1 % | 100 % | 100 % | 34,5 s |

**Pourquoi si peu ?** L'intégrale porte sur un petit nuage autour de chaque état : ±1° de cant, ±0,3° de rake, ±1° de
yaw, ±5 cm de Hvol. À cette échelle, la variance du co-krigeage est très régulière. Sobol l'intègre bien dès quelques
points, et c'est le nombre d'états qui fait la taille utile. Le temps restant (4 s) vient de la partie fixe du calcul,
la covariance des candidats avec les données. **Recommandation : 16 points par état.** Si le nuage s'élargit, il faut
refaire cette mesure.

**Candidats.** On tire des sous-ensembles de M candidats dans un réservoir de 8 192, puis on compare leur meilleur
rendement au meilleur du réservoir :

| M | 64 | 128 | 256 | **512** | **1 024** | 2 048 | 4 096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| médiane | 95,7 % | 97,5 % | 97,5 % | **99,4 %** | **98,9 %** | 99,5 % | 99,5 % |
| pire tirage | 92,9 % | 93,3 % | 95,5 % | **96,7 %** | **97,5 %** | 98,2 % | 98,8 % |

**Recommandation : 512 à 1 024 candidats**, dont trois quarts autour des états.

## 6. Étude 2 — quel LHS de départ en dimension 16 ?

[`examples/vpp_lhs_study.py`](examples/vpp_lhs_study.py) ·
[`figures/vpp_lhs_study.html`](runs/2026-10-21_meta_vpp_cible/figures/vpp_lhs_study.html). Plans imbriqués du méta-surrogate : sections tirées
dans la famille, attitude par hypercube latin, HF choisis par maximin. Médiane de 3 graines.

| Plan | Coût | e_V [nd] | Erreur relative globale de Fx | Ajustement des 6 modèles |
| --- | --- | --- | --- | --- |
| HF seul, 32 | 320 | 1,09 | 0,59 | 3 s |
| HF seul, 64 | 640 | 0,36 | 0,31 | 4 s |
| HF seul, 128 | 1 280 | 0,18 | 0,10 | 15 s |
| co-krigeage 64 BF + 16 HF | 224 | 0,36 | 0,34 | 3 s |
| co-krigeage 128 BF + 16 HF | 288 | 0,30 | 0,12 | 14 s |
| co-krigeage 128 BF + 32 HF | 448 | 0,27 | 0,11 | 15 s |
| **co-krigeage 256 BF + 32 HF** | **576** | **0,15** | 0,09 | 44 s |
| co-krigeage 256 BF + 64 HF | 896 | 0,15 | 0,08 | 45 s |
| co-krigeage 512 BF + 64 HF | 1 152 | 0,15 | 0,07 | 4 min |
| co-krigeage 512 BF + 128 HF | 1 792 | 0,14 | 0,07 | 4 min |

* **La BF paie.** Le co-krigeage 64 BF + 16 HF (coût 224) égale le krigeage HF seul à 64 HF (coût 640), pour trois
  fois moins cher.
* **Mettre beaucoup de BF.** À 32 HF, passer de 128 à 256 BF divise e_V par 1,8. À 64 HF, le rapport 4 ou 8 ne change
  plus rien.
* **Un plateau.** Dès le coût 576, un LHS plus gros n'améliore presque plus l'erreur aux états VPP. En dimension 16,
  remplir l'espace progresse lentement : l'erreur décroît comme n^(−ν/d). Or les états n'occupent qu'une petite partie
  du domaine. C'est le signal qu'il faut cibler.
* **L'ajustement devient lourd au-delà de 500 BF** : 4 min pour les 6 modèles.

**Recommandation :** un départ de **128 BF + 32 HF** (8d et 2d), assez grand pour que les hyperparamètres soient
crédibles, puis le raffinement ciblé. Pour un plan en un coup, 256 BF + 32 HF atteint déjà le plafond.

## 7. Étude 3 — le raffinement ciblé VPP contre l'ALC global et le LHS

[`examples/vpp_targeted_study.py`](examples/vpp_targeted_study.py) ·
[`figures/vpp_targeted_study.html`](runs/2026-10-21_meta_vpp_cible/figures/vpp_targeted_study.html).

* **Cadre.** Départ 128 BF + 32 HF (coût 448), budget 150, 3 graines.
* **Réglages.** 16 points par état, 512 candidats locaux et 256 globaux. Réajustement et mise à jour des états tous les
  10 pas, poids une mise à jour sur trois.
* **Témoin.** Un LHS de même coût au meilleur rapport de l'étude 2 : 264 BF + 33 HF.

| Stratégie | e_V | Erreur vraie, moyenne | Erreur vraie, max | BF / HF à la fin | Temps par run |
| --- | --- | --- | --- | --- | --- |
| **ciblé VPP** | **0,030 nd** | **0,015 nd** | **0,062 nd** | 158 / 44 | 7,5 min |
| LHS imbriqué | 0,141 nd | 0,081 nd | 0,28 nd | 264 / 33 | 2 min |
| ALC global | 0,157 nd | 0,094 nd | 0,55 nd | 278 / 32 | 45 min |

Par graine, l'erreur vraie moyenne du ciblé vaut 0,017 / 0,015 / 0,015 nd, contre 0,079 / 0,129 / 0,081 nd pour le
LHS : **3 graines sur 3**, avec un facteur 5 à 8.

**Ce qui se passe :**

* **Le ciblé place ses points autour des états.** La distance médiane au plus proche vaut 0,8 demi-largeur de nuage,
  contre 4,7 pour l'ALC global.
* **Il achète de la HF.** Sur 150 de budget, il a fait 12 HF et 30 BF. Près des états, la traînée HF est ce qui coûte
  en vitesse, et le critère le sait par ses poids.
* **L'ALC global dépense tout en BF.** Il a fait 150 BF, une à la fois, car sur tout le domaine la BF est le moins cher
  pour baisser une variance. Ses 150 itérations, avec des réajustements sur un plan BF qui grossit, expliquent ses
  45 min par run.
* **Sa courbe stagne.** Celle du ciblé descend régulièrement, de 0,2 à 0,03 nd.

**Couverture.** Les trois stratégies ont 100 % des erreurs dans leur intervalle à 95 % : l'incertitude du surrogate est
trop prudente (voir l'étude 4).

## 8. Étude 4 — combien de calculs CFD pour valider ?

[`examples/vpp_validation_study.py`](examples/vpp_validation_study.py) ·
[`figures/vpp_validation_study.html`](runs/2026-10-21_meta_vpp_cible/figures/vpp_validation_study.html).

* **Cadre.** Co-krigeage 256 BF + 64 HF, avec une région de fonctionnement de 4 000 points autour des 20 états.
* **Mesures.** L'erreur de vitesse vraie y vaut 0,093 nd RMS, avec des queues un peu lourdes (kurtosis 3,8). On tire
  4 000 jeux de n points de validation pour chaque n.

| n calculs | 5 | 10 | **20** | **40** | 80 | 160 |
| --- | --- | --- | --- | --- | --- | --- |
| RMS estimée / vraie (90 %) | 0,51 – 1,61 | 0,62 – 1,43 | **0,71 – 1,30** | **0,79 – 1,20** | 0,84 – 1,16 | 0,89 – 1,11 |
| RMS de z = erreur/σ (vrai 0,40) | 0,25 – 0,52 | 0,29 – 0,48 | 0,33 – 0,46 | 0,35 – 0,44 | 0,36 – 0,43 | 0,37 – 0,42 |

La théorie gaussienne donne ±1,645/√(2n) : ±37 % à n = 10, ±26 % à n = 20. Le méta-problème est un peu moins favorable,
à cause des queues lourdes.

**Recommandations pour la CFD de validation :**

* **20 calculs CFD, un par (section, condition)**, donnent l'erreur RMS de vitesse du surrogate à ±30 % près. 40
  calculs la donnent à ±20 % près. Il faut les placer **aux états d'équilibre de la VPP**, là où le surrogate sert.
* **5 à 10 calculs suffisent pour juger la calibration.** Ici, l'incertitude annoncée est 2,5 fois trop large (RMS de
  z = 0,40). C'est sans danger pour la sécurité, mais cela rend les intervalles inutiles et dilue le critère. Un
  recalibrage (σ × 0,4) se ferait sur ces calculs.
* **Si la CFD devenait un 3e niveau** au-dessus d'AVL, il faudrait l'ordre de grandeur des HF de l'étude 3 : 32 au
  départ et 12 ciblés. Je compte donc **30 à 50 calculs CFD** pour l'apprendre, en espérant que l'écart CFD − AVL soit
  aussi régulier que l'écart AVL − NPLLT. Ce point reste à vérifier sur de vrais calculs.

## 9. Limites et suites

* **Le méta-problème n'est pas validé.** Son ancre et sa VPP sont réelles, mais l'écart BF/HF est construit. Il faudra
  refaire les études 1 à 3 avec NPLLT et AVL, en version courte : les scripts acceptent n'importe quelle fonction
  `x (m, 16) → (m, 7)`.
* **Les sections de conception sont fixes (4).** Dans la boucle finale, l'optimiseur de section changera de profils. Il
  faudra alors intégrer autour des états des profils qu'il explore, par exemple ses meilleurs candidats du moment. Le
  code accepte déjà n'importe quelle liste de sections.
* **Hypothèses du critère.** Il néglige les corrélations entre coefficients, puisqu'il y a un modèle par coefficient, et
  reste au premier ordre en vitesse. La surestimation de l'incertitude (×2,5) en vient peut-être en partie.
* **`Cpmin` n'est pas modélisé.** L'adaptateur renvoie 0 à MDS, qui ne fait que le reporter. Pour une contrainte de
  cavitation, il faudra un septième modèle.
* **Le domaine du test** se limite aux profils fiables pour NeuralFoil (60 % de la famille). La décision sur la famille
  reste à prendre.
* **Les temps de calcul.** Le ciblé coûte 7,5 min par run sur le méta. Avec les vrais codes, le coût des calculs dominera. 3 graines seulement, mais avec un écart de 5 à 8.

## 10 bis. Les vrais codes et le premier vrai run (22/10/2026)

**Étape 0, le recalage des codes contre la RS de production.** Sorties dans `runs/2026-10-22_*_step0_calibration*/`,
script [`scripts/real_step0_calibration.py`](scripts/real_step0_calibration.py).

* **La RS du C-foil de MDS** est faite par bdFoil (`bdFoil/Scripts/IntensSY/CFoil_geom/CFoil_generation.ipynb`), avec
  AVL × XFOIL :
  * arc de R 9,8 m, 3,25 m d'envergure, corde 0,70 m, **cant_geom = 20,168°** ;
  * image paroi, Re 5·10⁶ ;
  * moments réduits au bord d'attaque de la cale basse, entrées passées telles quelles ;
  * Cpmin = 20e percentile des Cpmin de bande.
* **Notre pipeline** (`src/foil_codes.py`), avec cette forme en plan et le profil cambré de référence, retrouve la RS
  **sans aucune conversion.** Pour AVL, les pentes valent 0,86 à 0,96 et R² ≥ 0,996. La RS a 1,29 fois plus de
  traînée, sans doute par la FSI.
* **Deux pièges, documentés dans le code :**
  * le fichier `MC2_60_6101_1.xf` du dossier de la dérive est **symétrique** et donne une portance nulle à attitude
    nulle, au lieu de −0,29 ;
  * AVL ne lit pas un profil de 161 points par côté et échoue sans rien écrire : on en met 121.
* **Cpmin reste éloigné de la RS** (forme du bord d'attaque).

**Le run** : [`runs/2026-10-22_1021_real_1h/README.md`](runs/2026-10-22_1021_real_1h/README.md).

* **Le gain :** +0,05 nd de vitesse moyenne prédite par rapport à la section moyenne.
* **La précision :** 0,003 nd d'erreur de vitesse contre AVL aux états de la meilleure section, là où le raffinement a
  porté ses points.
* **La limite :** la meilleure section est trop mince (6,2 %) faute de contrainte d'épaisseur, et la boîte
  Dim-Reduction ne contient pas le profil actuel du foil.

**Le 2e run** : [`runs/2026-10-22_1131_real_1h30_current_box/README.md`](runs/2026-10-22_1131_real_1h30_current_box/README.md).

* **Le cadre :** une boîte de demi-largeur centrée sur le profil actuel, avec Ixx ≥ Ixx actuel, et une recherche qui
  part du profil actuel.
* **Le gain :** un profil moins cambré, de 12,5 % d'épaisseur et Ixx +2,4 %, qui gagne **+0,18 nd de vitesse moyenne
  (+0,95 %)**. La mesure est cohérente : surrogate final, puis correction contre AVL au premier ordre. Le gain tient dans
  les 5 conditions.
* **La précision :** 0,005 nd d'erreur de vitesse contre AVL au meilleur profil.
* **À confirmer** avec NPLLT × XFOIL et la CFD, car AVL ne voit que la ligne de cambrure.

**Le 3e run** : [`runs/2026-10-22_1519_real_npllt_xfoil_fsi/README.md`](runs/2026-10-22_1519_real_npllt_xfoil_fsi/README.md).

* **Le cadre :** celui du 2e run, avec trois changements.
  * Le niveau HF est **NPLLT × XFOIL**, et AVL × XFOIL ne sert plus qu'à la validation.
  * Le foil est **flexible** : la correction de déformée de la RS FSI de production est mise à l'échelle de l'Ixx
    (`vpp_coupling.FSICorrection`). Elle redonne la VPP de production à 0,0005 nd près.
  * Les calculs tournent **en parallèle** : lots de 16 points choisis par « kriging believer », 28 processus de calcul,
    30 processus VPP, co-krigeages en fils.
* **Le résultat :** le même profil que le 2e run. Le gain sur le profil actuel vaut **+0,24 nd vu par NPLLT × XFOIL et
  +0,22 nd vu par AVL**, et il tient dans les 5 conditions. Le Cpmin passe de −2,57 à −0,87.
* **Pourquoi le même profil :** les deux runs criblent les mêmes réserves de sections, tirées avec la même graine. Le
  choix tient au changement de modèle et à la déformée. Il reste le meilleur de tirages aléatoires : un raffinement
  local autour de lui est la suite naturelle.
* **Le débit :** 816 choix en 65 min, contre 373 au 2e run ; 313 calculs HF dans le plan final, contre 121. Le choix
  des lots, en n², est devenu le goulot.

**Le 4e run** (23/10, 8 h 10) :
[`runs/2026-10-23_1702_real_search_profile_local/README.md`](runs/2026-10-23_1702_real_search_profile_local/README.md).

* **Le cadre :** celui du 3e run, avec les changements du 23/10
  ([`CHANGEMENTS_2026-10-23.md`](CHANGEMENTS_2026-10-23.md)).
  * La recherche vise la **distribution d'usage supposée** (`profiles/guess_bimodal.json`) : un mode près, un mode
    portant, soit 18 nœuds pondérés par leur probabilité.
  * Une **région de confiance** autour de la titulaire : la moitié des sections criblées et des nuages y sont tirés.
  * Une graine de criblage propre, et des hyperparamètres figés au-delà de 2 500 points.
* **Le résultat :** un profil **encore moins cambré**, de 12,54 % d'épaisseur, avec un **Ixx exactement à la
  contrainte**. Les vitesses sont prises en moyenne pondérée sur les 13 conditions résolues pour toutes les sections, soit
  72 % de la probabilité.
  * Le gain sur le profil actuel vaut **+0,16 nd**, vu par NPLLT × XFOIL comme par AVL ; il est de +0,056 nd sur le
    profil des runs 2 et 3.
  * Il est positif dans les 13 conditions : faible au près par petit temps (+0,01 à +0,03 nd), et de +0,17 à +0,42 nd
    au portant et à 16 nd au près.
* **La convergence :** la titulaire ne bouge plus de plus de 0,006 nd après 1 h 15. Le rayon de la région de confiance
  atteint son minimum à 3 h 43. Le plan final compte 2 029 BF et 1 408 HF.
* **La leçon :** l'optimum est sur la contrainte de raideur. Le niveau de cette contrainte est donc un choix de
  conception qui fixe le résultat.
* **Un piège corrigé le 26/10.** Résoudre la VPP depuis un seul point de départ peut aboutir à un équilibre lent mais
  « réussi ». L'analyse sur la distribution donnait ainsi −0,23 nd au lieu de +0,15. Elle utilise maintenant des départs
  multiples.

## 10. Fichiers

| Fichier | Rôle |
| --- | --- |
| [`src/meta_foil.py`](src/meta_foil.py) | méta-surrogate 16D à deux niveaux |
| [`src/vpp_coupling.py`](src/vpp_coupling.py) | adaptateur MDS (`SurrogateFoilRS`), VPP Intens SY (`IntensSYVPP`), sensibilités |
| [`src/vpp_targeted.py`](src/vpp_targeted.py) | co-krigeage du torseur, nuages d'intégration, critère pondéré, boucle |
| [`tests/test_vpp_coupling.py`](tests/test_vpp_coupling.py) | 7 tests dans l'environnement MDS : RS = MDS, ancre exacte, moments cohérents, adaptateur exact, choix du critère |
| `examples/vpp_study_common.py`, `vpp_meta_overview.py`, `vpp_{integration,lhs,targeted,validation}_study.py` | études et figures ; résultats dans `examples/results/vpp_*` |

```powershell
cd Multifidelity-GP-Clean\vpp_acceleration
..\.venv-mds\Scripts\python -m pytest tests\test_vpp_coupling.py      # 7 tests, ~30 s
cd examples
..\..\.venv-mds\Scripts\python vpp_study_common.py        # vérité (une fois, 4 min)
..\..\.venv-mds\Scripts\python vpp_integration_study.py   # étude 1 (~20 min, 1 processus)
..\..\.venv-mds\Scripts\python vpp_lhs_study.py           # étude 2 (~10 min, 4 processus)
..\..\.venv-mds\Scripts\python vpp_targeted_study.py      # étude 3 (~50 min, 3 processus, l'ALC global est lent)
..\..\.venv-mds\Scripts\python vpp_validation_study.py    # étude 4 (~2 min)
```
