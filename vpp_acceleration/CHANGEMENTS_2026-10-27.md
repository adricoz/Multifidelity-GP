# Les changements du 27/10/2026 : convergence, redémarrages, lots plus rapides

Ce document explique les quatre modifications décidées le 27/10 après l'examen du 4e run
([`runs/2026-10-23_1702_real_search_profile_local`](runs/2026-10-23_1702_real_search_profile_local)). Pour chacune, il
donne ce qui a été constaté, ce qui change, où cela se trouve dans le code, et comment c'est vérifié. Les mots sont ceux de
[`CHANGEMENTS_2026-10-23.md`](CHANGEMENTS_2026-10-23.md) : la **titulaire** est la meilleure section du moment ; une
**région** est une recherche locale autour d'une titulaire.

Les tests nouveaux sont dans [`tests/test_convergence_and_batch.py`](tests/test_convergence_and_batch.py) ; la suite
complète passe (66 tests). **Sans les nouvelles options, le comportement est celui d'avant, à deux différences près :**
- le choix des lots, plus rapide pour le même résultat (point 2) ;
- le premier solve de la titulaire, qui essaie maintenant plusieurs départs (point 4).

## Ce que montrait le 4e run

- **La recherche a convergé vers 70 min.** Dans les 6 h 40 suivantes, la titulaire n'a gagné que 0,012 nd au total. À
  la fin, les 4 sections résolues à chaque mise à jour étaient quasi identiques (12,836 nd chacune).
- **Un lot prenait 7 min en fin de run :** environ 2,5 min de mise à jour (VPP) et 4 min de choix.
- **Une sensibilité aberrante est revenue trois fois :** ∂V/∂Fx à la condition 8 valait 224 au lieu de 117, à 98, 166 et
  252 min, chaque fois pendant 3 mises à jour. C'est le même défaut qu'au 3e run.
- **Le rayon a mis 223 min à atteindre son minimum**, parce que chaque micro-changement l'élargissait.
- **La vitesse affichée par mise à jour n'est pas comparable d'une mise à jour à l'autre :** elle est calculée sur les
  conditions résolues pour les 4 sections, qui couvrent 67, 72 ou 78 % de la probabilité selon la mise à jour.

## 1. La convergence d'une région, puis le redémarrage

**Le gain d'une mise à jour** (`update`, champ `gain` de `updates.json`) est la vitesse pondérée de la nouvelle
titulaire moins celle de l'ancienne. Les deux sont résolues dans la même mise à jour, avec le même surrogate et sur les
mêmes conditions. Il vaut 0 quand la titulaire est gardée.

**La règle du rayon** (`--min-gain 0.002`). Un changement de titulaire qui gagne moins de 0,002 nd compte comme une
confirmation : le rayon se resserre au lieu de s'élargir. La titulaire prend quand même la meilleure section.

**La convergence** (`converged`, `--converge 10 0.01`). Une région a convergé dans l'un de ces deux cas :
- ses 10 dernières mises à jour ont gagné moins de 0,01 nd au total ;
- sa titulaire s'approche à moins de 1 (en scores blanchis) d'un optimum déjà trouvé : c'est le même bassin.

Rejouée sur les gains du 4e run, la première condition se déclenche à la mise à jour 28, vers 70 min. Ce qui a suivi ne
valait que 0,012 nd.

**Le redémarrage** (`restart`, `--restarts 2`). Quand une région a convergé et qu'il reste des redémarrages :
1. sa titulaire est gardée comme **optimum local**, avec ses états ;
2. on crible 1 024 sections de toute la famille avec le score du criblage μ + κσ (`first_order_scores`) ;
3. parmi celles qui sont à au moins 2 de tous les optima (`--restart-distance`), la meilleure devient la titulaire
   d'une nouvelle région ;
4. le rayon repart de sa valeur initiale, et les sensibilités sont recalculées à la mise à jour suivante ;
5. **toutes les données restent :** le surrogate ne perd rien.

Sans redémarrage restant, la boucle s'arrête. Le temps limite reste une borne.

**La comparaison finale** (`compare_optima`). Les optima et la dernière titulaire sont résolus de nouveau avec le
surrogate final. Chaque solve part de tous les équilibres connus à la condition et des départs génériques. La meilleure
vitesse pondérée, prise sur les conditions résolues pour toutes les sections, désigne la section finale. Les autres
optima sont validés comme elle contre les codes (`region_<n>` dans `validation.csv`), et le tableau est écrit dans
`results/optima.json`.

**Pourquoi.** La région 1 du 4e run était dans un coin de la famille : son premier mode PCA vaut −5,1, alors que les
sections tirées dans la famille restent entre −3,1 et +3,0. Le criblage global n'avait jamais placé de section lointaine
dans le trio des concurrentes. Le temps passé après la convergence peut donc servir à chercher un autre bassin. Si les
redémarrages reviennent au même optimum, on gagne la confiance qu'il est global dans la famille.

**Références.** Eriksson et al. (2019, TuRBO) : redémarrer une région de confiance quand elle a convergé, en gardant les
données. Conn, Gould & Toint (2000) : le test de progrès suffisant dans la règle du rayon.

## 2. Le choix des lots, exact et 2,8 fois plus rapide

**Avant.** Après chaque choix d'un lot, le point choisi était ajouté aux données avec sa moyenne prédite (le « kriging
believer »). Puis les 7 co-krigeages étaient reconditionnés **sur toutes les données**. À 3 437 points, cela coûtait 12,4 s
par choix, dont 2,7 s par modèle pour reconstruire la matrice de covariance.

**Maintenant** (`sequential_design.append_observations`, `TorsorCoKriging.believe`), on prolonge le facteur de Cholesky
des lignes des nouveaux points :

L' = [[L, 0], [wᵀ, E]], avec w = L⁻¹c et EEᵀ = c_nouveau + bruit − wᵀw.

Le coût est en n² par point, au lieu de n² d pour la matrice et n³ pour son facteur. La valeur ajoutée étant la moyenne
prédite, la moyenne a posteriori ne change pas, et α' = (α, 0) exactement. Les covariances a posteriori sont celles d'un
reconditionnement complet.

**La vérification.**
- Un test unitaire compare les moyennes et les covariances jointes à celles d'un reconditionnement complet, et le lot
  de 6 choix à celui de l'ancienne méthode.
- Sur le surrogate final du 4e run (3 437 points), un lot de 16 choix donne **les mêmes 16 choix** que l'ancienne
  méthode, avec des rendements égaux à 4·10⁻⁹ près. Le lot passe **de 269 s à 97 s** : le reconditionnement descend de
  173 s à 2 s, et le calcul du critère sur les candidats reste à 95 s.

**La suite possible.** Ces 95 s viennent du recalcul, à chaque choix, des covariances blanchies des points de contrôle et
des candidats. On pourrait les prolonger de la même façon, pour encore environ 80 s par lot. Ce n'est pas fait.

## 3. Les sensibilités : la médiane des 3 dernières évaluations

**Maintenant** (`sensitivities`, `--sensitivity-median 3`), chaque entrée de ∂V/∂C_k est la médiane de ses 3 dernières
évaluations. Un saut isolé est ignoré, et une vraie évolution est suivie avec une évaluation de retard. L'historique
repart de zéro à chaque nouvelle région. Le journal signale les valeurs à plus de 30 % de la médiane.

**Pourquoi.** Le solve VPP dépend du chemin parcouru. Un solve perturbé qui tombe sur un autre équilibre double une
différence finie : c'est le cas de ∂V/∂Fx à la condition 8 au 4e run. Au carré dans les poids du critère, cette valeur
quadruplait le poids de cette condition pendant 3 mises à jour.

**À savoir.** Avec 2 évaluations seulement, la médiane est la moyenne : un saut à la 2e évaluation compte pour moitié.

## 4. Le premier solve de la titulaire : plusieurs départs

**Maintenant** (`update`), quand une titulaire n'a pas encore d'état (début de la recherche, redémarrage), chaque
condition est résolue depuis le départ fixe de la VPP **et** depuis les deux départs génériques (`GENERIC_GUESSES`, qui
étaient dans `profile_analysis.py`). On garde l'équilibre réussi le plus rapide. Les solves suivants partent des états
de la titulaire, comme avant.

**Pourquoi.** Depuis le seul départ fixe, le solve peut s'arrêter sur un équilibre lent mais « réussi » (48/16 :
15,7 nd au lieu de 19,2 nd, correction du 26/10). Cet équilibre se propagerait ensuite par les départs à chaud.

## Les autres changements

- **Un champ `update_s`** dans `history.csv` donne le temps de la mise à jour dans chaque lot. `select_s` comprend
  toujours le réajustement, la mise à jour et le choix.
- **Les champs `gain` et `region`** dans `updates.json`, et `region` dans `history.csv`.
- **La figure de convergence** trace une courbe par région.

## Le test court

Le dossier est [`runs/2026-10-27_1219_smoke_restarts`](runs/2026-10-27_1219_smoke_restarts). Il a tourné en 3,2 min sans
erreur, sur 2 conditions, avec des lots de 4 et une règle de convergence volontairement lâche (`--converge 2 0.5
--restarts 1`). Il a parcouru tout le chemin nouveau :
1. la région 0 converge à la 2e mise à jour ;
2. la région 1 part d'une section à 3,3 des optima ; ses sensibilités sont recalculées, et la médiane écarte 2 valeurs ;
3. la région 1 converge, et la boucle s'arrête faute de redémarrage ;
4. la comparaison finale donne 15,329 nd pour la région 0 et 15,361 nd pour la région 1, qui l'emporte ;
5. la validation couvre la meilleure section, le profil actuel et l'optimum de la région 0, et `optima.json` est écrit.

Ces chiffres ne disent rien de la qualité : le plan n'a que 46 points BF et 7 HF.

## Le run de contrôle

Le dossier est [`runs/2026-10-27_1224_control_restarts`](runs/2026-10-27_1224_control_restarts/README.md). Il reprend
les réglages et la graine du 4e run, et dure 2 h.
- **Le même résultat en 2 h au lieu de 8 h** : 13,080 nd contre 13,084 nd en vitesse validée.
- **La région 0 converge à 92 min.** Le redémarrage revient vers le même coin de la famille en 20 min.
- **Le débit n'a pas gagné**, car le plan reste sous 1 100 points. C'est maintenant la mise à jour VPP qui limite un
  run court.

## Références

- D. Eriksson, M. Pearce, J. Gardner, R. Turner, M. Poloczek, *Scalable global optimization via local Bayesian
  optimization*, NeurIPS 2019.
- A. Conn, N. Gould, P. Toint, *Trust-Region Methods*, SIAM, 2000.
- D. Ginsbourger, R. Le Riche, L. Carraro, *Kriging is well-suited to parallelize optimization*, in *Computational
  Intelligence in Expensive Optimization Problems*, Springer, 2010 (le « kriging believer »).
