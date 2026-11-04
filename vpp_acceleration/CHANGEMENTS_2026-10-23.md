# Les changements du 23/10/2026 : ce qui a changé dans la recherche, et pourquoi

Ce document explique les modifications décidées le 23/10, dans l'ordre où le code les rencontre. Pour chacune, on donne ce
qu'elle fait, pourquoi, où elle se trouve, et sur quoi elle s'appuie. Les mots employés sont ceux de la note
[`doc/selection_points_vpp.tex`](doc/selection_points_vpp.tex) :

- la **titulaire** est la meilleure section du moment ;
- un **état** est une section avec son attitude d'équilibre VPP, à une condition ;
- les **points de contrôle** sont les endroits où l'on veut que le surrogate soit juste ;
- les **candidats** sont les endroits où l'on pourrait lancer un calcul de code.

Commit : `eb300ec` (branche `feat/vpp-acceleration`). Les tests nouveaux sont dans
[`tests/test_search_changes.py`](tests/test_search_changes.py), et la suite complète passe (61 tests).

## 1. Les candidats : seulement les nuages autour des états

**Avant.** Les candidats étaient de trois sortes :
- 500 points proches des 20 états ;
- 256 points pris n'importe où (une section quelconque, une attitude quelconque) ;
- jusqu'à 512 points déjà calculés en BF, sur lesquels on pouvait ajouter la HF au coût 2.

**Maintenant.** Il ne reste que les points proches, dans un nuage autour de chaque état
(`VPPSectionOptimizer.candidates`).

**Pourquoi.** Au 3e run, 100 % des 816 choix étaient des candidats proches, et 2 HF seulement sur 252 ont porté sur un
point BF d'un lot précédent. Les deux autres familles ne servaient à rien et compliquaient le code. Une HF reste possible
sur un point choisi en BF dans le même lot.

**Références.** Picheny et al. (2010, IMSE ciblé) : on place les calculs là où l'on veut la précision.
Gramacy & Lee (2009) : l'ALC sur un ensemble fini de candidats.

## 2. Une graine propre aux sections criblées

**Avant.** Les 256 sections tirées au criblage dépendaient de la graine générale du run (`seed`, 0 par défaut). Les
runs 2 et 3 ont donc criblé exactement les mêmes sections, et trouvé la même section à la même mise à jour.

**Maintenant.** Une option `--screen-seed` sert seulement à ce tirage. Par défaut, elle est tirée au hasard au lancement
et notée dans `config.json`. La graine de la PCA et du plan de départ ne change pas : les runs restent comparables dans
le même espace de scores.

**Référence.** Pas de littérature spécifique : c'est la pratique ordinaire des tirages indépendants entre essais.

## 3. La distribution d'usage dans la recherche (version complète)

**L'entrée.** `--search-profile <fichier.json>` (format de `src/operating_profile.py`, par exemple
`profiles/guess_bimodal.json`). Les 5 conditions de même poids deviennent les nœuds de la distribution, chacun avec sa
probabilité w_c : 18 nœuds avec la distribution supposée. Sans l'option, on garde les 5 conditions de même poids.

La distribution intervient à quatre endroits.

1. **Le choix de la titulaire** (`update`). La vitesse d'une section est la moyenne pondérée Σ w_c V_c. Elle est prise
   seulement **sur les conditions résolues pour les 4 sections comparées**. Sinon, une section dont la VPP échoue à une
   condition lente paraîtrait plus rapide. La part de probabilité comparée est notée à chaque mise à jour
   (`probability_compared`).
2. **Le criblage** (`screen`). Le gain attendu devient μ = Σ_c w_c ΔV_c / Σ_c w_c. L'incertitude devient
   σ² = Σ_c w_c² Var(V_c) / (Σ_c w_c)², la variance d'une moyenne pondérée de termes indépendants. Le score reste
   μ + κσ. On ne compte que les conditions où l'état de la titulaire est résolu.
3. **Les sensibilités** (`sensitivities`). Elles sont calculées à chaque condition résolue de la titulaire.
   Une condition sans équilibre garde sa valeur précédente, nulle au départ, et n'entre plus nulle part.
4. **Le critère de raffinement** (`point_weights`, puis `choose` et `batched_imse_gains`). Avant, chaque coefficient avait
   un seul poids, la moyenne sur les conditions de (∂V/∂C_k)². Maintenant, **chaque point de contrôle** a son poids
   pour chaque coefficient :

   ω_ik = w_c · (∂V/∂C_k)_c² / (nombre de points de son état), où c est la condition de l'état du point.

   - **La division par le nombre de points** fait qu'un état compte pour la probabilité de sa condition, même si
     quelques-uns de ses points ont été écartés.
   - **Le gain d'un candidat** devient la somme pondérée des baisses de variance aux points de contrôle, au lieu de leur
     moyenne simple.
   - **Les conséquences :** le raffinement va vers les conditions probables, et chaque condition y compte avec sa propre
     sensibilité. Au 3e run, ∂V/∂Fx allait de 84 (45/12) à 302 (90/14) : une moyenne globale gommait cet écart.

**Le coût.** La VPP complète passe de 4 × 5 = 20 à 4 × 18 = 72 solves par mise à jour. Les sensibilités montent à 18 × 6
× 2 solves, une mise à jour sur trois. Le pool de 30 processus VPP l'absorbe : une mise à jour prend de 1 à 3 minutes.

**À savoir.** Avec la distribution supposée, environ 28 % de la probabilité (le portant profond par petit temps) n'a pas
d'équilibre VPP. C'est vrai aussi avec la RS de production, et c'est un réglage de la VPP à revoir. Ces conditions sont
ignorées, et la part comparée est notée.

**Références.**
- Huyse, Padula, Lewis & Li (2002) : un profil optimisé sur l'espérance de la distribution des conditions, plutôt qu'en
  un point.
- Drela (1998) : un profil optimisé sur peu de points se dégrade entre eux, d'où l'optimisation multipoint.
- Bect et al. (2012) et Picheny et al. (2010) : un critère de réduction de variance intégré contre une mesure de
  probabilité.

## 4. Raffiner aussi la section : la recherche locale

**Avant.** Les 256 sections criblées étaient tirées dans toute la famille. Les points de contrôle et les candidats
gardaient la section de leur état, et seule l'attitude variait. Une fois qu'aucune section tirée au hasard ne battait plus
la titulaire, l'algorithme passait son temps à raffiner autour d'elle, sans chercher plus près.

**Maintenant**, avec `--local-fraction 0.5`, il y a deux parties.
1. **La moitié des sections criblées est tirée autour de la titulaire** (`RealFoilProblem.local_sections`), dans une
   « région de confiance » de rayon r, en scores PCA blanchis (1 = un écart type de la famille). Les tirages sont
   section = titulaire + r·g/√k, avec g gaussien : la distance typique vaut r. On ne garde que les sections dont la
   section décodée reste dans la famille. L'autre moitié reste tirée dans toute la famille, pour continuer d'explorer.
2. **Le rayon s'adapte** (`update`) :
   - il est multiplié par 1,5 quand la titulaire change : on progresse, on regarde plus large ;
   - il est multiplié par 0,7 quand elle est confirmée : on resserre autour d'elle ;
   - il reste entre 0,1 et 3, et part de 1 (`--radius`).
3. **Les nuages du critère varient aussi la section** (`RealFoilProblem.cloud`). Un point sur deux de chaque nuage
   déplace la section du même rayon. L'autre moitié garde la section de l'état, pour le match entre les 4 sections.
   Les points dont la section sort de la famille sont retirés. Le surrogate apprend ainsi comment les coefficients
   changent quand on modifie un peu la section près des concurrentes, ce dont la recherche locale a besoin pour bien les
   juger.

**Références.**
- Conn, Gould & Toint (2000) : les méthodes à région de confiance, et la règle d'adaptation du rayon.
- Eriksson et al. (2019, TuRBO) : la même idée en optimisation bayésienne, efficace en grande dimension.
- Astudillo & Frazier (2019) : l'optimisation bayésienne de fonctions composites, notre cas (on modélise le torseur, et
  la VPP calcule la vitesse).

## 5. Garde-fou : des hyperparamètres figés au-delà d'une taille de plan

`--max-refit-size 2500`. Au-delà de 2 500 points, les hyperparamètres ne sont plus réestimés, et le modèle se
reconditionne seulement sur les nouveaux points. Une réestimation coûte O(n³) : environ 2,5 min à 1 045 points au
3e run, et de l'ordre de l'heure vers 3 000 points. Sans ce garde-fou, un run de 8 h y passerait plusieurs heures.

## 6. Les figures : un `index.html` par run

`real_run_figures.write_index` écrit `figures/index.html` dans chaque run. Cette page donne un lien vers chaque figure,
avec ce qu'elle montre, et le résumé du run. `write_runs_index` écrit `runs/index.html`, qui renvoie à l'index de chaque
run. Ces pages se régénèrent à la fin de chaque run et de chaque analyse sur une distribution.

## Ce qui n'a pas changé

- **Le reste de la méthode :** le co-krigeage, le gain ALC exact, le choix par lots (« kriging believer »), le bonus
  κσ, la FSI.
- **Les anciennes options :** sans les nouvelles options, le comportement est celui d'avant, à deux différences près.
  Les candidats lointains et les points BF sans HF ont disparu. Le poids d'un point de contrôle utilise la sensibilité
  de **sa** condition, au lieu de la moyenne sur les conditions (point 3.4).
- **`choose`** accepte toujours un poids par sortie, la forme d'avant, utilisée par les études sur méta-surrogate.
  Les deux formes donnent le même choix quand elles sont équivalentes, et un test le vérifie.

## Le test court

Le dossier est [`runs/2026-10-23_1643_smoke_search_profile_local`](runs/2026-10-23_1643_smoke_search_profile_local).
Il a tourné avec la distribution supposée réduite à 8 conditions, en 6 min, sans erreur.
- **Les conditions sans équilibre** (le portant à 139°) sont écartées.
- **La recherche locale** fournit de 0 à 3 des 3 concurrentes à chaque mise à jour.
- **Le surrogate** est à 0,005 nd près de NPLLT × XFOIL au meilleur profil.

**Un point à surveiller dans le run long.** La titulaire a changé à chaque mise à jour, en gagnant de 0,025 à 0,06 nd,
donc le rayon est monté d'emblée à son maximum. C'est attendu quand le modèle a peu de points. Mais les concurrentes,
choisies avec un bonus d'incertitude et moins raffinées que la titulaire, peuvent avoir une vitesse surestimée : c'est
la « malédiction du vainqueur ». Si la titulaire ne se stabilise pas, on exigera qu'une concurrente gagne plus que son
incertitude pour la remplacer et élargir le rayon, comme le seuil d'amélioration de TuRBO.

## Corrections du 26/10, après le week-end

1. **L'analyse sur une distribution d'usage partait d'un seul point de départ, et c'était faux**
   (`scripts/profile_analysis.py`).
   - **Le problème.** Le solve VPP (SLSQP) trouve un maximum local de la vitesse. Depuis le point de départ fixe de la
     VPP (12 nd, voiles à 0,6), il peut s'arrêter sur un équilibre lent mais « réussi ».
   - **L'exemple.** Au meilleur profil du run long, à 48/16, on trouvait 15,7 nd (gîte 2,7°, voiles réduites à 0,35).
     En partant de l'équilibre de la titulaire, on trouve 19,2 nd (gîte 5,9°, voiles pleines). Les nœuds à 16 nd de vent
     étaient touchés, et l'analyse donnait −0,23 nd au lieu de +0,16 nd.
   - **La correction** (`robust_solve`) se fait en deux passages, et garde le plus rapide des équilibres réussis :
     1. chaque nœud est résolu depuis le point de départ fixe et deux départs génériques (près puissant, portant) ;
     2. il est résolu à nouveau depuis les équilibres trouvés au premier passage par toutes les sections au même nœud.

     Pour cela, `VPPPool.solve` accepte des départs supplémentaires. **La recherche elle-même n'est pas modifiée :** ses
     solves partent déjà des équilibres de la titulaire.
   - **Les analyses refaites :** celles des runs 3 et 4.
2. **Le résumé automatique comparait des nombres incomparables** (`run_real_vpp.summarize`).
   - **Le problème.** La ligne « start section → incumbent » mettait face à face la vitesse vue par le *premier*
     surrogate et celle vue par le *dernier*, et avec la distribution, sur des conditions résolues différentes.
   - **La correction.** Le résumé compare désormais les sections validées sur les mêmes conditions, résolues pour toutes,
     avec leurs probabilités, et avec la vitesse corrigée contre le code HF. Le `summary.txt` du run long a été réécrit.
3. **Le rapport CFD donnait des GCI pour des grandeurs qui ne convergent pas**
   (`runs/2026-10-23_weekend/cfd_mesh_report.py`).
   - **La correction.** Le rapport calcule maintenant R = ε₂₁/ε₃₂ (Celik et al., 2008), et ne donne le GCI que si
     0 < R < 0,9.
   - **Les avertissements ajoutés :** des rapports de raffinement réels sous 1,3, la compensation entre traînée de
     pression et de frottement, et des y+ obtenus loin des y+ visés.

**Une proposition pour la recherche, non appliquée sans ton accord.** Le premier solve de la titulaire part du point de
départ fixe, comme l'analyse fautive. Lui donner les mêmes départs multiples éviterait qu'un équilibre lent se propage
par les départs à chaud.

## Références

- V. Picheny, D. Ginsbourger, O. Roustant, R. Haftka, N.-H. Kim, *Adaptive designs of experiments for accurate
  approximation of a target region*, J. Mech. Design 132(7), 2010.
- R. Gramacy, H. Lee, *Adaptive design and analysis of supercomputer experiments*, Technometrics 51(2), 2009.
- L. Huyse, S. Padula, R. M. Lewis, W. Li, *Probabilistic approach to free-form airfoil shape optimization under
  uncertainty*, AIAA Journal 40(9), 2002.
- M. Drela, *Pros and cons of airfoil optimization*, Frontiers of Computational Fluid Dynamics, 1998.
- J. Bect, D. Ginsbourger, L. Li, V. Picheny, E. Vazquez, *Sequential design of computer experiments for the estimation
  of a probability of failure*, Statistics and Computing 22, 2012.
- A. Conn, N. Gould, P. Toint, *Trust-Region Methods*, SIAM, 2000.
- D. Eriksson, M. Pearce, J. Gardner, R. Turner, M. Poloczek, *Scalable global optimization via local Bayesian
  optimization*, NeurIPS 2019.
- R. Astudillo, P. Frazier, *Bayesian optimization of composite functions*, ICML 2019.
