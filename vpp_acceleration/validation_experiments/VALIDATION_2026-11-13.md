# 13/11/2026 : validation de l'algorithme d'optimisation (sans CFD)

Compte rendu écrit automatiquement le 13/11/2026 à 15:11 par `night_report.py` — série de tests **terminée**. Plan approuvé le 12/11, lancé le 13/11 ; critères fixés avant de lancer.

## Le tableau des tests

| # | Test | Critère | Résultat | Verdict |
|---|---|---|---|---|
| 1 | Validation croisée avec zone d'exclusion, run du 11/11 | écart réduit ≤ 1,3 à 1 score (niveau 2) | niveau 1 : 0,94 à r = 0, 1,09 à r = 1, 1,00 à r = 2 ; niveau 2 : 0,83 à r = 0, 0,72 à r = 1, 0,72 à r = 2 | **réussi** |
| 1 | Validation croisée avec zone d'exclusion, run sans CFD | écart réduit ≤ 1,3 à 1 score (niveau 2) | niveau 1 : 0,93 à r = 0, 1,09 à r = 1, 1,05 à r = 2 ; niveau 2 : 0,80 à r = 0, 0,88 à r = 1, 0,69 à r = 2 | **réussi** |
| 2 | Certificat, run du 11/11, niveau 2 | borne haute ≤ 0,02 nd (avec κ_loin) | κ = 1 : +0,042 nd ; κ = 2 : +0,068 nd ; κ_loin = 1,00 : +0,042 nd | **échoué** |
| 2 | Certificat, run du 11/11, niveau CFD | borne haute ≤ 0,02 nd (avec κ_loin) | κ = 1 : +0,022 nd ; κ = 2 : +0,040 nd ; κ_loin = 1,00 : +0,022 nd | **échoué** |
| 2 | Certificat, run sans CFD | borne haute ≤ 0,02 nd (avec κ_loin) | κ = 1 : +0,004 nd ; κ = 2 : +0,012 nd ; κ_loin = 1,00 : +0,004 nd | **réussi** |
| 2 | Sections de borne haute maximale, résolues par la VPP sur la vérité du jumeau | aucune ne bat le meilleur optimum de référence de plus de 0,02 nd | meilleure : -0,0652 nd (5 sections) | **réussi** |
| 3 | Jugement croisé (premier ordre, niveau 2) : l'optimum de l'autre run vu par chaque modèle | écart ≤ 0,02 nd | juge 11/11 : +0,013 ± 0,002 nd ; juge sans CFD : -0,006 ± 0,002 nd (distance entre optima 1,52 score) | **réussi** |
| 3 | Jugement exact (VPP) : optimum du run sans CFD sur la vérité du jumeau (modèle du 11/11, niveau 2) | à moins de 0,02 nd du meilleur optimum de référence | -0,0008 nd, à 0,51 score | **réussi** |
| 4 | Tests existants rejoués sur le run sans CFD | KKT non rejetées (p > 0,05), montée de gradient < 0,01 nd | KKT : p = 0,00 ; montée de gradient au plus +0,0029 nd ; regret attendu 0,001 nd, P(mieux de > 0,02 nd) = 0 % | **échoué** |
| 5 | Jumeau numérique : la méthode retrouve-t-elle l'optimum connu ? | perte ≤ 0,02 nd | perte +0,0019 nd, à 1,53 score du meilleur optimum de la recherche directe | **réussi** |
| 6 | Verdict des codes (AVL × XFOIL à des sections éloignées) | aucune section éloignée ne bat la meilleure | non lancé cette nuit | pas de résultat |
| 7 | Profil d'usage (run du 11/11) | gain positif pour tout poids de mode entre 0 et 3 fois l'actuel | gain +0,393 nd ; par mode : upwind (39 %) : -0,01 nd ; downwind (39 %) : +0,66 nd ; downwind_tail (9 %) : +0,88 nd ; manual (14 %) : +0,48 nd ; gain minimal sur le balayage +0,214 nd | **réussi** |

Lecture des critères : « écart réduit » = erreur de prédiction divisée par l'écart-type annoncé (1 = incertitude juste, au-dessus = modèle trop sûr de lui). κ_loin est cet écart réduit à 1 score d'éloignement ; il sert à corriger l'incertitude du certificat. Les lignes « premier ordre » utilisent les états VPP figés de la meilleure section et sont indicatives pour des sections éloignées ; les lignes « VPP » sont exactes sur le modèle.

## À quoi sert chaque test, et ce qu'il montrera

La question : la section trouvée (+0,39 nd de vitesse moyenne sur la section actuelle) est-elle le meilleur choix du domaine, à 0,02 nd près, ou seulement le meilleur de ce que la méthode a regardé ? Il n'y a pas de descente de gradient sur les vrais codes, et le co-krigeage est entraîné sur des points **ciblés** autour de quelques sections : il pourrait être juste là où il a des données et se tromper ailleurs sans le savoir. L'affirmation « pas de meilleur optimum ailleurs » repose sur une chaîne de maillons ; chaque test en vérifie un.

| Maillon | Test |
|---|---|
| Le modèle est juste là où il a des données | déjà fait : validation croisée classique, validation de fin de run par les codes |
| Son incertitude est honnête **loin** des données | 1 |
| Avec cette incertitude, aucune région ne peut cacher une meilleure section | 2 |
| Le résultat ne dépend pas de l'histoire du run (ciblage, graine, données) | 3 et 4 |
| L'algorithme retrouve un optimum connu | 5 |
| Le modèle ne se trompe pas sur des sections éloignées, d'après les codes | 6 (non lancé) |
| Le gain ne dépend pas du profil d'usage choisi | 7 |

**1. Validation croisée avec zone d'exclusion.** Pour un point test, on retire *tous* les points dont la section est à moins de r scores (distance dans l'espace des 10 modes ACP), on le prédit avec ce qui reste, et on divise l'erreur par l'écart-type annoncé (« écart réduit »). Rayons 0 ; 0,25 ; 0,5 ; 1 ; 2 ; 3 ; 300 points tests par niveau ; les 6 efforts. À r = 0 c'est la validation croisée habituelle, optimiste parce qu'un point voisin presque identique reste dans les données ; à r ≥ 0,5 on se met dans la situation du criblage : juger une section qu'on n'a jamais calculée. *Ce que ça montre* : un écart réduit proche de 1 (ou en dessous) veut dire que, loin des données, le modèle sait qu'il ne sait pas : quand il annonce « rien de mieux ici » avec une petite incertitude, on peut le croire. Au-dessus de 1, il est trop sûr de lui, et le facteur mesuré à 1 score (κ_loin) sert à gonfler l'incertitude du certificat. Critère : ≤ 1,3 à 1 score au niveau 2.

**2. Certificat (borne haute maximale).** On cherche, sur tout le domaine admissible (mêmes contraintes que le run, Ixx/v compris), la section qui maximise *gain prédit + κ × écart-type du gain* par rapport à la meilleure section, avec de nombreux départs (sections du run et optima locaux). Le gain est la vitesse moyenne pondérée par le profil, au premier ordre autour des états VPP de la meilleure section, avec la covariance exacte du modèle. Fait pour κ = 1, 2, 3 et κ_loin, au niveau AVL × XFOIL et au niveau CFD du 11/11, et au niveau AVL × XFOIL du run sans CFD. *Ce que ça montre* : si même cette borne optimiste reste sous +0,02 nd, alors **nulle part** dans le domaine le modèle ne laisse de place à une section nettement meilleure, y compris dans les régions qu'il connaît mal (là où σ est grand, la borne monte d'elle-même). C'est la réponse directe à « comment savoir qu'il n'y a pas d'optimum ailleurs ». Si la borne dépasse, la figure dit **où** (distance à la meilleure section) : c'est là qu'il faudrait calculer. Les sections qui réalisent la borne sont ensuite résolues par la vraie VPP sur la vérité du jumeau, ce qui enlève l'approximation au premier ordre.

**3. Jugement croisé des deux runs.** Le run du 11/11 (3 niveaux, CFD) et le run sans CFD (2 niveaux) ont des données, des graines et des ciblages différents, dans le même espace de sections. Chaque modèle évalue la meilleure section de l'autre (écart et incertitude), puis la VPP exacte évalue la section du run sans CFD sur le modèle du 11/11. *Ce que ça montre* : si chaque modèle place l'optimum de l'autre à moins de 0,02 nd du sien, le ciblage n'a pas décidé du résultat ; deux apprentissages différents s'accordent. Si un modèle juge l'optimum de l'autre nettement moins bon alors que l'autre le juge meilleur, c'est le signe d'un biais de ciblage.

**4. Tests existants rejoués sur le run sans CFD.** Test KKT (le gradient de la vitesse est-il nul, aux contraintes actives près ?), montée de gradient depuis la meilleure section, regret attendu. *Ce que ça montre* : que les conclusions « optimum local sur le modèle » du 11/11 se reproduisent sur un second run, et ne sont pas un accident d'un seul jeu de données.

**5. Jumeau numérique.** On remplace les codes aéro par une fonction **connue** et gratuite : la moyenne du modèle du 11/11 (niveau 1 et niveau 2). Sur elle, on cherche le vrai optimum *directement* : montée à région de confiance, chaque pas vérifié par la vraie VPP, depuis plusieurs départs (meilleure section, section actuelle, optima locaux du run, une section au hasard). En parallèle, la méthode complète repart **de zéro** sur cette vérité (plan initial, ciblage, criblage, redémarrages, convergence, mêmes options que le run sans CFD, autre graine). On compare les vitesses vraies. *Ce que ça montre* : c'est le seul test où l'on connaît la réponse. Perte ≤ 0,02 nd : l'algorithme retrouve l'optimum. Perte plus grande avec une section éloignée : il s'est arrêté dans un autre bassin (criblage ou redémarrages insuffisants). Section éloignée mais même vitesse : la vérité a une crête plate, l'optimum n'est pas unique et ce n'est pas grave. Si les départs de la recherche directe arrivent à des vitesses différentes, la vérité a plusieurs bassins et le tableau du jumeau dit lesquels comptent. Limite : la vérité est un krigeage, plus lisse que les vrais codes ; le test juge l'algorithme, pas la régularité des codes (c'est le rôle des tests 1 et 6).

**6. Verdict des codes (non lancé aujourd'hui).** AVL × XFOIL réels aux sections du certificat. Il montrerait si le modèle se trompe sur des sections éloignées d'après les vrais codes. À décider ensemble.

**7. Profil d'usage et direction active.** Le gain est recalculé à partir des 49 vitesses de la validation, en multipliant le poids de chaque mode (près, portant, vent arrière, manuel) de 0 à 3 fois, les autres renormalisés ; plus la carte du gain par condition et la vitesse prédite de toutes les sections calculées le long de la direction qui porte la vitesse. *Ce que ça montre* : si le gain garde son signe quel que soit le profil, le choix de section ne dépend pas d'un profil d'usage qu'on a fixé à la main ; la carte montre où le gain se fait (et où l'on perd) ; la coupe montre une bosse ou plusieurs.

**Ensemble.** Si 1, 2, 3, 4 et 5 réussissent : le modèle est honnête loin des données, il ne laisse de place nulle part à une section meilleure de plus de 0,02 nd, deux runs indépendants s'accordent, et l'algorithme retrouve un optimum connu. Cela ne prouve pas que les codes (ni la CFD) sont d'accord avec le modèle loin des données : c'est le test 6, puis une vérification CFD de la meilleure section contre la section actuelle.

## Détail

- Jumeau numérique : [twin_experiments/README.md](../twin_experiments/README.md) et ses figures.
- Figures du run du 11/11 : [figures/index.html](figures/index.html) ; du run sans CFD : [figures/nocfd/index.html](figures/nocfd/index.html).
- Résultats bruts : `results/*.json` (11/11), `results/nocfd/*.json`, `../twin_experiments/results/*.json`.
- Journaux : `logs/night/` (un fichier par étape, `night.log`, `status.json`, `git.log`).

## Déroulement

| Étape | État | Début | Durée (min) |
|---|---|---|---|
| `cv_1111` | terminée | 09:30 | 3 |
| `twin_method` | terminée | 09:30 | 240 |
| `summary_1111` | terminée | 09:30 | 3 |
| `twin_reference` | terminée | 09:30 | 318 |
| `profile_1111` | terminée | 09:30 | 0 |
| `certificate_1111_level2` | terminée | 09:33 | 106 |
| `certificate_1111_top` | terminée | 09:33 | 106 |
| `calibration_nocfd` | terminée | 11:19 | 1 |
| `gradient_nocfd` | terminée | 11:19 | 11 |
| `slices_nocfd` | terminée | 11:19 | 17 |
| `regret_nocfd` | terminée | 11:19 | 14 |
| `dimension_nocfd` | terminée | 11:19 | 20 |
| `levels_nocfd` | terminée | 11:19 | 11 |
| `cv_nocfd` | terminée | 11:39 | 5 |
| `summary_nocfd` | terminée | 11:39 | 3 |
| `profile_nocfd` | terminée | 11:39 | 0 |
| `certificate_nocfd` | terminée | 11:39 | 58 |
| `cross` | terminée | 12:37 | 1 |
| `twin_compare` | terminée | 14:49 | 23 |
| `twin_report` | terminée | 15:11 | 0 |

## Notes

- 09:30 night started (stop 13/11 16:45, dry False)
- 09:30 run without CFD finished: full use of the machine
- 15:11 codes verdict (AVL x XFOIL at far sections) not run: not implemented
- 15:11 night finished

## Limites à garder en tête

- Le jumeau a pour vérité un krigeage : il juge l'algorithme, pas la régularité des vrais codes.
- La validation croisée avec zone d'exclusion retire des données : à grand rayon il en reste peu, et l'erreur mesurée est celle d'un modèle plus pauvre que le vrai (test sévère).
- Le certificat et le jugement croisé « premier ordre » gèlent les états VPP.
- Aucun calcul CFD ; les codes aéro n'ont pas été rappelés sauf si la ligne 6 le dit.
