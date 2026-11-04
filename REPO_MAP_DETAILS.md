# REPO_MAP_DETAILS — Multifidelity-GP-Clean

> Carte générée automatiquement par `repo_map.py` (modèle local `qwen-science`) le 2026-10-19, 133 fichiers.
> Résumés automatiques : vérifier dans le code avant de s'appuyer sur un détail.


## `./`

### `README.md`
Documente de référence du dépôt implémentant le processus gaussien multi-fidélité non imbriqué (NN-MF-GP) et l'algorithme d'optimisation globale efficace (EGO) selon Sacher et al. (2021).
- `ExperimentData` : gère la génération de plans d'expériences (LHS imbriqués ou non) et le stockage sécurisé des observations multi-fidélités.
- `MultifidelityModel` : moteur mathématique calculant les hyperparamètres (MLE/MAP) et la corrélation $\rho$ pour l'approximation récursive des niveaux de fidélité.
- `EGOOptimizer` : contrôle le cycle d'optimisation Ask-and-Tell en maximisant la fonction de mérite et en sauvegardant l'état.
- `CoKriging` : surrogat exact pour les designs imbriqués, incluant la validation par leave-one-out et les scores de prédiction.
- Dépend de : `numpy`, `scipy`, `matplotlib`, `plotly`, `neuralfoil`, `aerosandbox`.

### `requirements-benchmark.txt`
Définit les dépendances de l'environnement virtuel dédié au benchmark `benchmarks/hartmann_benchmark.ipynb`.
- `numpy`, `scipy`, `matplotlib`, `plotly`, `pandas` : bibliothèques de base pour le calcul et la visualisation.
- `scikit-learn`, `smt`, `botorch`, `GPy`, `emukit` : bibliothèques de processus gaussiens et co-kriging multi-fidélité comparées à mfego.
- `neuralfoil`, `aerosandbox` : résolution du problème aérodynamique de l'aile profilée.
- `jupyter`, `ipykernel`, `nbconvert`, `nbclient`, `pytest` : exécution des notebooks et tests.
- Dépend de : `mfego`, `benchmarks/cokriging`.

### `requirements.txt`
Déclare les dépendances Python minimales requises pour le fonctionnement du projet, testées avec Python 3.13 et 3.14.
- numpy>=1.24 : moteur mathématique central.
- scipy>=1.9 : fournit les algorithmes qmc, L-BFGS-B et DE vectorisé.
- matplotlib>=3.5 : génère les graphiques statiques PNG.
- neuralfoil>=0.2.0 : module optionnel spécifique à l'exemple d'hydrofoil.
- Dépend de : bibliothèques externes (numpy, scipy, matplotlib, plotly, pytest, neuralfoil, aerosandbox).


## `analysis/`

Le dossier `analysis` regroupe des scripts de diagnostic et d'analyse de performance (`scripts/`) pour valider la correction du modèle multifidélité, ainsi qu'un rapport détaillé (`RAPPORT_ANALYSE.md`) sur l'implémentation de l'optimisation NN-MF-EGO. Il documente les corrections théoriques et numériques, comparant les versions initiale et corrigée du code via des études comme `rho_study.py`.

### `analysis/RAPPORT_ANALYSE.md`
Rapport d'analyse détaillé de l'implémentation de l'optimisation globale efficace multi-fidélité (NN-MF-EGO) de Sacher et al. (2021), documentant les corrections théoriques, numériques et de robustesse.
- **Synthèse priorisée** : Liste des écarts critiques (ρ figé, bornes absolues, reconstruction JSON incorrecte) et leur état de correction.
- **Axe 1 — Cohérence avec la théorie** : Confrontation équation par équation avec les références [S] et [LG], détaillant les fixes pour l'estimation de ρ et l'acquisition.
- **Axe 2 — Bornes d'hyperparamètres** : Analyse des bornes actives sur les sorties non normalisées et justification de la paramétrisation log.
- **Axe 3 — Fonctions de simulation** : Vérification des constantes de Hartmann 6D et des mappings de niveaux pour les hydrofoils.
- Dépend de : `mfego/`, `example/`, `legacy/`, `analysis/scripts/`, `references/multifidelity_opt.pdf`, `references/Multi_fidelity_Gaussian_process_regressi.pdf`


## `analysis/scripts/`

Le dossier `analysis/scripts` contient des utilitaires partagés (`_common.py`) et des scripts de diagnostic (`check_env.py`, `check_hyperparams.py`, `check_reconstruction.py`, `check_simulators.py`, `check_theory.py`) pour valider la correction du modèle multifidélité. Il inclut également des analyses de performance (`profile_perf.py`, `speedup_breakdown.py`) et des études théoriques (`rho_study.py`) comparant les versions initiale et corrigée du code.

### `analysis/scripts/_common.py`
Fournit des utilitaires partagés pour les scripts d'analyse, incluant le chargement dynamique des versions du code et des fonctions de test analytiques.
- `load_mfego`: importe les modules `src` des versions "current" ou "initial" en injectant un shim pour `traitlets` si nécessaire.
- `forrester_lf`, `forrester_hf`, `hartmann6`, `hartmann_level`: implémentent les fonctions de référence pour la validation multi-fidélité.
- `save_figure`, `save_results`: exportent les figures Plotly interactives et les résultats numériques au format JSON.
- `make_data`: initialise un objet `ExperimentData` avec un plan d'expériences LHS et les évaluations des fonctions.
- Dépend de : `numpy`, `mfego` (modules `kernels`, `data_management`, `surrogate_models`, `acquisition`, `optimizer`, `simulator`, `visualization`).

### `analysis/scripts/check_env.py`
Script de vérification de l'environnement (versions Python, dépendances, signatures) pour le rapport.
- `interpreters`: Détecte les interpréteurs Python disponibles via le lanceur `py` ou le système actuel.
- `import_checks`: Teste l'import de `src.data_management` pour chaque combinaison interpréteur/version de code.
- `package_metadata`: Récupère les versions et contraintes `Requires-Python` des paquets listés dans `PACKAGES`.
- `signatures`: Extrait les signatures des fonctions tierces (NeuralFoil, scipy) utilisées par le cadre.
- Dépend de : `src.data_management`, `analysis/scripts/_common`, `neuralfoil`, `scipy`, `plotly`.

### `analysis/scripts/check_hyperparams.py`
Script de diagnostic comparant les bornes de recherche des hyperparamètres et l'équivariance d'échelle entre les versions initiale et corrigée du modèle multifidélité.
- `INITIAL_BOUNDS`: Définit les intervalles de recherche des hyperparamètres pour la version initiale.
- `current_bounds`: Récupère les bornes dynamiques définies dans le module `surrogate_models` de la version courante.
- `data_from_main`: Charge les données expérimentales (X/Y) depuis des fichiers JSON versionnés dans la branche `main` via `git show`.
- `active_bounds`: Identifie les hyperparamètres de chaque niveau de fidélité qui atteignent les limites de leur boîte de recherche.
- `__main__`: Orchestre le processus de test sur trois jeux de données, calcule les erreurs de sensibilité à l'échelle et génère un tableau récapitulatif interactif.
- Dépend de : `_common`, `scipy.stats`, `plotly.graph_objects`, `numpy`, `json`.

### `analysis/scripts/check_reconstruction.py`
Script de vérification de la reconstruction des surfaces de réponse depuis les fichiers JSON avant et après correction.
- `forrester_run`: exécute une itération EGO courte sur Forrester et retourne le chemin du JSON de sauvegarde.
- `compare`: compare le modèle en mémoire avec celui reconstruit par `ModelVisualizer` et calcule les métriques d'erreur.
- `scatter_check`: valide les coordonnées tracées lors de la visualisation 2D d'un modèle 3D.
- Dépend de : `forrester_hf`, `forrester_lf`, `load_mfego`, `make_data`, `save_figure`, `save_results`

### `analysis/scripts/check_simulators.py`
Script de vérification des fonctions de simulation (Hartmann 6D et Hydrofoil/NeuralFoil) pour valider les constantes, corrélations multi-fidélité et faisabilité des bornes.
- `hartmann_checks`: Compare les minima et valeurs aux points optimaux des constantes Hartmann initiales, corrigées et standard (Sacher Eq. 30).
- `hartmann_correlations`: Calcule les coefficients de corrélation entre les niveaux de fidélité de la séquence Hartmann (Eqs. 31-32) et génère des nuages de points.
- `hydrofoil_checks`: Évalue la carte de faisabilité du bracket de recherche de racine, la traînée des deux niveaux et le minimum de référence haute fidélité sur la grille de design.
- `load_example_module`: Importe dynamiquement des modules d'exemples depuis des chemins relatifs pour les tests.
- Dépend de : `_common`, `aerosandbox`, `neuralfoil`, `scipy`, `plotly`.

### `analysis/scripts/check_theory.py`
Script de vérification théorique comparant le code initial et corrigé sur les axes de précision, vraisemblance et variance.
- `surrogate_accuracy`: compare l'erreur RMSE et la couverture des bandes de confiance sur les fonctions Forrester et Hartmann.
- `rho_profile_likelihood`: calcule la vraisemblance négative du paramètre rho pour identifier l'optimum par rapport au modèle additif.
- `variance_reduction_and_aei`: évalue l'erreur relative de la formule de réduction de variance et le facteur AEI selon le rapport de bruit.
- `reference_value_f_best`: détermine la meilleure valeur effective observée via la fonction d'acquisition.
- Dépend de : `mfego` (via `load_mfego`), `_common` (données et utilitaires), `scipy.stats`, `plotly`.

### `analysis/scripts/profile_perf.py`
Script de profilage des performances numériques et de la mémoire comparant les versions initiale et corrigée du code.
- `timeit`: Mesure le temps d'exécution minimal sur plusieurs itérations.
- `covariance_timings`: Compare le calcul de la matrice de covariance (boucle vs vectorisé) pour différentes tailles d'échantillons.
- `gp_fit_timings`: Évalue le temps de calibration et le nombre de matrices de covariance générées par un processus gaussien.
- `ego_iteration_and_memory`: Mesure le temps par itération EGO et la mémoire tracée via `tracemalloc`.
- Dépend de : `_common`, `numpy`, `plotly`, `tracemalloc`

### `analysis/scripts/rho_study.py`
Étudie le traitement de rho (fixe, profilé, hybride) via la précision du substitut et la convergence de l'optimisation.
- MODES : Définit les configurations d'estimation de rho (fixed, profiled, hybrid).
- case_definition : Retourne les bornes, fonctions de niveaux, coûts et minimum pour un cas spécifique.
- accuracy_task : Évalue la précision du modèle (RMSE, NLPD, couverture) pour une combinaison de cas, points et mode.
- ego_task : Exécute une itération d'optimisation EGO et enregistre l'historique d'erreurs et de coûts.
- decision : Applique la règle de décision pour déterminer le mode par défaut selon les performances.
- Dépend de : _common (forrester_hf, hartmann6, load_mfego), plotly, scipy.stats, numpy.

### `analysis/scripts/speedup_breakdown.py`
Script d'analyse qui décompose l'accélération du code corrigé via une ablation cumulée et des micro-benchmarks sur le problème Hartmann 6D.
- `ablation`: exécute la cascade d'ablation en activant/désactivant les optimisations N1, N3, N4, N6, N8 pour mesurer leur impact temporel.
- `micro_benchmarks`: isole et mesure les performances de chaque mécanisme (matrice de covariance, ajustement GP, prédiction, mérite).
- `figures`: génère les graphiques Plotly du waterfall et des comparaisons des mécanismes.
- `run_iterations`: mesure le temps réel d'exécution de N itérations EGO avec un simulateur spécifique.
- Dépend de : `_common` (pour `load_mfego`, `hartmann6`, `make_data`), `numpy`, `plotly`.


## `benchmarks/`

Le dossier `benchmarks` évalue les versions v1 à v3 du modèle `CoKriging` via des sous-dossiers dédiés comme `cokriging`, `surrogate_v2` et `surrogate_v3`. Il compare les performances à des bibliothèques concurrentes (BoTorch, scikit-learn) sur des problèmes de Forrester et Hartmann. Des fichiers partagés, notamment `bench_lib.py` et `figure_style.py`, standardisent les tests et la génération de figures Plotly.

### `benchmarks/bench_lib.py`
Bibliothèque d'aide pour le benchmark Hartmann multi-fidélité comparant mfego, scikit-learn, SMT et BoTorch.
- `HartmannMF` : problème de benchmark multi-fidélité 6D avec séquence U_k et coûts.
- `Surrogate` : interface commune pour les modèles substituts (fit, predict, rhos).
- `MfegoSurrogate` : wrapper pour le modèle multi-fidélité de mfego.
- `BotorchSurrogate` : wrapper pour les modèles GP de BoTorch (SF et MF).
- `run_mfego` : boucle d'optimisation EGO multi-fidélité sous budget de coût.
- `run_botorch` : boucle d'optimisation BoTorch (qLogEI ou Knowledge Gradient).
- Dépend de : mfego (src.acquisition, src.cokriging, src.data_management, src.kernels, src.optimizer, src.simulator, src.surrogate_models), scipy, numpy, et optionnellement smt, botorch, GPy, emukit, sklearn.

### `benchmarks/figure_style.py`
Fournit le style commun (palette, grilles, légendes) et la gestion du mode sombre pour les figures de benchmarks interactives.
- `INK`, `CATEGORICAL` : Définit les couleurs claires/sombres validées et la palette catégorielle.
- `DARK_JS` : Script JavaScript injecté pour basculer automatiquement les couleurs en mode sombre.
- `style` : Applique la mise en page standard (titres, marges, axes, légendes) à une figure Plotly.
- `hline` : Ajoute une ligne de référence horizontale avec annotation sur un sous-plot spécifique.
- `save` : Exporte la figure en HTML avec la dépendance Plotly.js via CDN.
- Dépend de : `plotly.graph_objects`, `numpy`.

### `benchmarks/hartmann_benchmark.ipynb`
Notebook de benchmark comparant mfego à scikit-learn, SMT et BoTorch sur le problème multi-fidélité de Hartmann 6D.
- `bl.HartmannMF` : définit le problème avec deux niveaux de fidélité et des coûts spécifiques.
- `bl.run_parallel` : exécute en parallèle les tâches d'évaluation de précision et d'optimisation bayésienne.
- `accuracy` : table de données mesurant RMSE, NLPD et couverture des modèles sur des points de test.
- `results` : agrégat des erreurs d'optimisation et du temps par itération pour chaque méthode.
- Dépend de : `bench_lib`, `analysis/results/speedup_breakdown.json`, `benchmarks/figures`.


## `benchmarks/cokriging/`

Le dossier `benchmarks/cokriging` contient des scripts de validation et d'évaluation de la classe `CoKriging` de mfego v1, incluant des tests sur les problèmes de Forrester et Hartmann. Les fichiers clés sont `run_cokriging_benchmark.py`, `forrester_checks.py` et `airfoil_problem.py`, accompagnés du rapport `RAPPORT_COKRIGING.md`. Ces benchmarks comparent les performances à emukit et au modèle récursif, tandis que `make_figures.py` génère les visualisations Plotly des résultats.

### `benchmarks/cokriging/RAPPORT_COKRIGING.md`
Rapport de benchmark évaluant la classe `CoKriging` de mfego v1 par rapport à emukit et au modèle récursif sur des plans imbriqués.
- `CoKriging` : implémente le co-krigeage conditionné à toutes les données via la matrice de covariance jointe V.
- `benchmarks/cokriging/run_cokriging_benchmark.py` : script principal exécutant les comparaisons de précision et de calibration.
- `benchmarks/cokriging/explore_rho_centering.py` : analyse du biais de ρ et test de la correction par centrage.
- `benchmarks/cokriging/airfoil_problem.py` : définition du problème NACA 4 chiffres avec modèles NeuralFoil.
- Dépend de : `mfego/src/cokriging.py`, `bench_lib.py`, `results/`, `figures/`.

### `benchmarks/cokriging/airfoil_problem.py`
Benchmarks de co-kriging multi-fidélité pour la traînée de sections NACA 4 chiffres via NeuralFoil et XFOIL.
- `AirfoilMF`: classe de données définissant le problème multi-fidélité (niveaux xxsmall/xxxlarge) et générant des designs LHS.
- `neuralfoil_cd`: calcule les compteurs de traînée vectorisés à l'aide de la bibliothèque `neuralfoil`.
- `xfoil_cd`: évalue la traînée d'une section individuelle via `aerosandbox` et XFOIL.
- `kulfan_parameters`: convertit les paramètres géométriques en poids CST pour le calcul aérodynamique.
- Dépend de : `optim_neuralfoil`, `aerosandbox`, `neuralfoil`, `scipy.stats`.

### `benchmarks/cokriging/explore_rho_centering.py`
Script de benchmark explorant le biais du rho profilé dans CoKriging en comparant une normalisation centrée avec l'implémentation actuelle.
- `CenteredGP` : sous-classe de `GaussianProcess` qui centre le niveau inférieur pour que la constante du résidu suive rho.
- `CenteredCoKrigingSurrogate` : wrapper `CoKrigingSurrogate` intégrant `CenteredGP` et bornes de recherche pour rho.
- `task` : exécute une prédiction sur un jeu de données imbriqué et retourne les métriques de performance et la valeur de rho.
- `main` : orchestre les tests parallèles, génère le CSV de résultats et affiche les médianes et taux de supériorité.
- Dépend de : `bench_lib`, `run_cokriging_benchmark`, `src.cokriging`, `src.kernels`, `src.surrogate_models`.

### `benchmarks/cokriging/forrester_checks.py`
Script de validation de la CoKriging sur le problème 1D de Forrester, testant l'estimation de rho avec peu de points haute fidélité et la présence de bruit basse fidélité.
- `f_low`, `f_high` : fonctions analytiques basse et haute fidélité du problème.
- `nested_forrester` : génère des données imbriquées avec option de bruit sur la fidélité basse.
- `V1` : configuration des hyperparamètres (noyau SE, priorités v1) pour les modèles.
- `CoKriging` : modèle utilisé pour l'ajustement et la prédiction.
- Dépend de : `src.cokriging`, `src.data_management`, `src.kernels`, `src.surrogate_models`.

### `benchmarks/cokriging/make_figures.py`
Génère des figures interactives Plotly HTML pour le benchmark de co-kriging à partir des résultats CSV.
- `figure_nested`: compare les métriques d'exactitude (RMSE, NLPD, couverture) entre les modèles sur des conceptions imbriquées.
- `figure_calibration`: visualise les diagrammes de fiabilité et l'histogramme des erreurs standardisées pour évaluer la variance prédite.
- `figure_verification`: valide la proposition 3.1 (équivalence CoKriging/modèle récursif) et l'exactitude numérique par rapport à emukit.
- `figure_infill`: compare les performances des stratégies d'infilling (NN-MF-EGO) sur des métriques de précision et de variance.
- `figure_rho`: analyse le biais du profil de rho et l'impact du centrage sur l'estimation.
- Dépend de : `figure_style`, données CSV dans `benchmarks/cokriging/results/`.

### `benchmarks/cokriging/run_cokriging_benchmark.py`
Script de benchmark évaluant la co-criging conditionnée sur des données multi-fidélité (Hartmann 6D et NACA 4-digit).
- `nested_task`: Évalue l'exactitude et la calibration des modèles sur des plans expérimentaux emboîtés, incluant une vérification d'exactitude pour emukit.
- `infill_task`: Compare les performances du modèle EGO, du conditionnement exact et de la co-criging après complétion des données basse fidélité.
- `scores`: Calcule les métriques de précision (RMSE, NLPD) et de couverture des intervalles de confiance.
- `exact_conditioning`: Crée un objet `CoKriging` exact à partir des hyperparamètres d'un modèle `MultifidelityModel` récursif.
- Dépend de : `src.cokriging`, `src.kernels`, `src.surrogate_models`, `bench_lib`, `airfoil_problem`.


## `benchmarks/map_hartmann/`

Le dossier `benchmarks/map_hartmann` contient le benchmark de l'estimation MAP pour le problème multifidélité de Hartmann 6D. Les fichiers clés `run_map_benchmark.py` et `run_prior_sensitivity.py` comparent la MAP au MLE et à BoTorch, tandis que `priors.py` définit les distributions de longueur d'échelle. Le script `make_figures.py` génère des figures Plotly interactives, et `RAPPORT_MAP.md` documente les résultats comparatifs.

### `benchmarks/map_hartmann/RAPPORT_MAP.md`
Rapport de benchmark comparant l'estimation MAP (avec prior) au MLE et à BoTorch pour le problème multifidélité de Hartmann 6D.
- `GaussianProcess.negative_log_likelihood` : implémentation de la pénalité MAP via l'option `use_map` et la constante `LENGTHSCALE_PRIOR`.
- `run_map_benchmark.py` : script principal évaluant la précision et l'optimisation des modèles mfego et BoTorch.
- `priors.py` : définition des lois a priori (InvGamma, Gamma, LogNormale) et de la classe `PriorGP`.
- `results/` : stockage des résultats bruts (CSV, JSON) et des figures interactives.
- Dépend de : `mfego/src/surrogate_models.py`, `bench_lib.HartmannMF`, `scipy.stats`, `botorch`.

### `benchmarks/map_hartmann/make_figures.py`
Génère des figures interactives Plotly (HTML) pour l'étude MAP sur Hartmann 6D à partir des résultats de benchmarks.
- `figure_accuracy`: visualise RMSE, NLPD et couverture en fonction de n_HF pour MAP, MLE et BoTorch.
- `figure_optimization`: compare la convergence, l'erreur finale et la part d'évaluations BF des méthodes.
- `figure_sensitivity`: analyse l'impact du choix de la prior sur la précision et l'optimisation.
- `figure_prior_shapes`: affiche les densités des priors testées et les longueurs de corrélation de référence.
- `main`: orchestre la génération des sept figures en lisant les fichiers CSV et JSON du dossier results.
- Dépend de : `priors`, données dans `benchmarks/map_hartmann/results/`

### `benchmarks/map_hartmann/priors.py`
Définit les priors de longueur d'échelle (InvGamma, Gamma, LogNormal) et les classes de modèles GP associées pour l'étude MAP sur Hartmann.
- PRIORS : dictionnaire de configurations de priors nommées avec leurs familles et paramètres.
- distribution : retourne la distribution SciPy figée correspondant à un prior donné.
- mode : calcule le mode de la densité du prior pour identifier la valeur cible du MAP.
- PriorGP : sous-classe de GaussianProcess ajoutant le log-prior sur les longueurs d'échelle à la NLL.
- PriorMultifidelityModel : construit un modèle multi-fidélité où chaque niveau utilise PriorGP.
- Dépend de : src.surrogate_models, scipy.stats, numpy, bench_lib.

### `benchmarks/map_hartmann/run_map_benchmark.py`
Benchmark de l'estimation MAP par rapport au MLE et BoTorch sur le problème multi-fidélité Hartmann 6D.
- `accuracy_task`: Évalue la précision des substituts (RMSE, NLPD) sur 2000 points de test.
- `optimization_task`: Exécute des optimisations avec un budget de coût égal pour chaque méthode.
- `summarize_accuracy`: Agrège les métriques de précision et compare les performances MAP/MLE.
- `summarize_optimization`: Analyse les erreurs finales et le nombre d'itérations des optimisations.
- Dépend de : `bench_lib`, `src.surrogate_models`

### `benchmarks/map_hartmann/run_prior_sensitivity.py`
Script de benchmark évaluant la sensibilité de l'estimation MAP de mfego à la prior de lengthscale sur le problème Hartmann 6D multi-fidélité.
- `accuracy_task` : calcule les métriques de précision et les lengthscales ajustées pour chaque prior et nombre de points HF.
- `optimization_task` : exécute l'optimisation NN-MF-EGO avec différents priors pour générer les historiques de performance.
- `reference_task` : détermine les lengthscales de référence via la maximisation de vraisemblance scikit-learn sur 300 points.
- `mle_diagnostic_task` : compare le NLL du fit MLE et du point MAP pour diagnostiquer l'optimiseur.
- `main` : orchestre les tâches parallèles, génère les tableaux statistiques et exporte les résultats CSV/JSON.
- Dépend de : `bench_lib`, `priors`, `run_map_benchmark`, `src.kernels`, `src.surrogate_models`, `scipy.stats`, `sklearn`.


## `benchmarks/priors_all/`

Le dossier `benchmarks/priors_all` contient un script de benchmark (`run_priors_benchmark.py`) évaluant les stratégies de priors (MAP-all, MAP-l, MLE) pour l'ajustement de modèles de processus gaussien sur le problème Hartmann 6D multifidélité. Il inclut un script de génération de figures interactives Plotly (`make_figures.py`) et un rapport détaillé (`RAPPORT_PRIORS.md`) comparant ces approches.

### `benchmarks/priors_all/RAPPORT_PRIORS.md`
Rapport de benchmark comparant les stratégies de priors (MAP-all, MAP-l, MLE) pour l'ajustement de modèles de processus gaussien sur le problème Hartmann 6D multifidélité.
- `DEFAULT_PRIORS` : Définit les distributions par défaut (InvGamma, LogNormale) pour les longueurs, variances de signal/biais et le bruit.
- `neg_log_prior` : Calcule le terme de log-vraisemblance négative et sa dérivée analytique pour l'optimisation MAP.
- `GaussianProcess` : Intègre les priors dans la fonction de vraisemblance via le paramètre `priors`.
- `MultifidelityModel` : Propage les priors à tous les niveaux de fidélité du modèle.
- Dépend de : `mfego/src/surrogate_models.py`, `benchmarks/map_hartmann/priors.py`, `bench_lib`.

### `benchmarks/priors_all/make_figures.py`
Génère des figures interactives Plotly (HTML) pour le benchmark des priors sur tous les hyperparamètres, à partir des résultats de `run_priors_benchmark.py`.
- `main`: orchestre la lecture des données CSV/JSON et l'appel des six fonctions de génération de figures.
- `line_with_band`: trace des médianes avec bandes interquartiles pour les comparaisons de performance.
- `figure_accuracy`: visualise la précision des surrogats (RMSE, NLPD, couverture) en fonction de `n_HF`.
- `figure_hyperparameters`: affiche les valeurs ajustées des hyperparamètres via des diagrammes en boîte.
- `figure_optimization`: compare les trajectoires d'optimisation et les erreurs de recommandation finales.
- `figure_prior_shapes`: superpose les densités des priors par défaut et les valeurs ajustées.
- Dépend de : `bench_lib`, `figure_style`, `src.surrogate_models`, `plotly`, `pandas`, `scipy.stats`.

### `benchmarks/priors_all/run_priors_benchmark.py`
Script de benchmark évaluant les a priori sur tous les hyperparamètres de mfego sur le problème Hartmann 6D multi-fidélité.
- `accuracy_task`: calcule les métriques de précision (rmse_rel, nlpd) et extrait les hyperparamètres ajustés.
- `optimization_task`: exécute des runs d'optimisation avec un budget de coût fixe pour comparer les méthodes.
- `main`: orchestre les tâches de précision, de grands échantillons et d'optimisation en parallèle.
- Dépend de : `bench_lib`, `src.surrogate_models`


## `benchmarks/surrogate_v2/`

Le dossier `benchmarks/surrogate_v2` contient les scripts et rapports de qualification du modèle de substitution `CoKriging` v2. Il inclut des outils de benchmark comme `run_benchmark.py` et `run_selection.py` pour comparer les performances aux bibliothèques concurrentes et à la v1. Des scripts de visualisation (`make_figures.py`) et d'exploration (`explore_ego_rho.py`) complètent l'analyse, appuyés par la bibliothèque partagée `v2_lib.py`.

### `benchmarks/surrogate_v2/RAPPORT_V2.md`
Rapport de benchmark documentant la qualification, la sélection des défauts et la comparaison de la v2 du modèle de substitution `CoKriging` par rapport à la concurrence et à la v1.
- `CoKriging` : modèle de substitution par co-krigeage multi-fidélité, amélioré par le noyau Matérn 5/2 et la correction du biais de ρ.
- `MultifidelityModel` : modèle utilisé pour l'optimisation EGO, conservant les paramètres de la v1 par défaut.
- `Matern52Kernel` : noyau analytique avec gradient, défini comme défaut pour `CoKriging`.
- `leave_one_out` : méthode de validation interne en forme fermée, évaluée pour sa fiabilité de calibration.
- Dépend de : `mfego/src/surrogate_models.py`, `mfego/src/kernels.py`, `mfego/src/validation.py`, `benchmarks/cokriging/airfoil_problem.py`, `pipelines/bdtoolbox_foil/config.py`.

### `benchmarks/surrogate_v2/explore_ego_rho.py`
Script de benchmark comparant les variantes mfego v2 et v1 sur Hartmann 6D pour évaluer l'impact du noyau et du paramètre rho.
- VARIANTS : dictionnaire définissant les configurations de noyaux (Matern52Kernel, SquaredExponentialKernel) et d'options pour chaque variante testée.
- task : exécute une optimisation MF-EGO avec un budget de 250 évaluations et retourne les métriques d'erreur et la part de bas-fidélité.
- main : orchestre l'exécution parallèle des tâches, sauvegarde les résultats dans ego_rho.csv et affiche les médianes par variante.
- Dépend de : v2_lib, bench_lib, src.kernels.

### `benchmarks/surrogate_v2/make_figures.py`
Script de génération de figures Plotly interactives pour les benchmarks mfego v2 à partir des résultats CSV/JSON.
- `figure_selection`: compare les variantes de prior via dNLPD et RMSE.
- `figure_nested`: affiche les médianes d'erreurs et de couverture des modèles imbriqués.
- `figure_calibration`: visualise la calibration de la variance prédite.
- `figure_loo`: confronte les scores leave-one-out aux scores de test.
- `figure_infill`: compare les boîtes à moustaches des stratégies d'infilling.
- `figure_optimization`: trace l'évolution de la meilleure observation observée.
- `figure_ego_rho`: analyse l'impact du rho corrigé sur la part de basse fidélité.
- Dépend de : `v2_lib`, `figure_style`, `run_selection`, `pandas`, `plotly`, `scipy`.

### `benchmarks/surrogate_v2/run_benchmark.py`
Script de benchmark final comparant mfego v2 à d'autres bibliothèques de GP sur des problèmes multi-fidélité.
- `models`: définit les substituts (CoKriging, Emukit, BoTorch, SMT) pour les conceptions imbriquées.
- `nested_task`: évalue l'exactitude et la calibration des modèles sur un ensemble de test.
- `infill_task`: analyse les données d'infilling générées par NN-MF-EGO v1 et v2.
- `optimization_task`: exécute l'optimisation de Hartmann 6D avec un budget fixe de 250.
- `main`: orchestre l'exécution parallèle des tâches et génère les rapports CSV et TXT.
- Dépend de : `v2_lib`, `bench_lib`, `src.cokriging`, `src.surrogate_models`, `src.validation`.

### `benchmarks/surrogate_v2/run_selection.py`
Script de benchmark pour sélectionner les nouveaux défauts de mfego v2 en comparant diverses variantes de prior et de noyau contre la version v1.
- `VARIANTS` : Dictionnaire définissant les configurations testées (centrage de rho, priors normaux, types de noyaux).
- `fit_task` : Fonction exécutant l'ajustement du modèle et le calcul des scores de prédiction sur un jeu de données spécifique.
- `decide` : Applique la règle de décision pour choisir la variante minimisant dNLPD sous contrainte de RMSE.
- `main` : Orchestre la génération des tâches, l'exécution parallèle et la sauvegarde des résultats.
- Dépend de : `v2_lib`, `bench_lib`, `src.validation`.

### `benchmarks/surrogate_v2/v2_lib.py`
Bibliothèque de fonctions utilitaires partagées pour les benchmarks mfego v2, gérant les problèmes, jeux de test et plans d'expériences.
- PROBLEMS : Dictionnaire de classes de problèmes (Hartmann, airfoil, airfoil_xfoil)
- v1_options / v2_options : Configuration des hyperparamètres pour les variantes de modèles
- test_set : Génération et mise en cache des points de test LHS
- designs : Calcul et stockage des plans d'expériences imbriqués et indépendants
- Dépend de : bench_lib, airfoil_problem, src.cokriging, src.kernels, src.surrogate_models


## `benchmarks/surrogate_v3/`

Dossier dédié aux benchmarks de la version v3 du modèle de substitution multifidélité `mfego`. Il contient `run_benchmark.py` pour exécuter les comparaisons, `v3_lib.py` pour définir les variantes, `make_figures.py` pour générer les figures Plotly et `RAPPORT_V3.md` pour le rapport comparatif. Il s'appuie sur les problèmes définis dans `surrogate_v2`.

### `benchmarks/surrogate_v3/RAPPORT_V3.md`
Rapport de benchmark comparant les versions v1, v2 et v3 du modèle de substitution multifidélité d'`mfego`.
- **Résumé et décision** : La v3 (ρ appris, prior log-normale) devient le défaut de `CoKriging` pour les plans imbriqués.
- **Méthodologie** : La v3 utilise L-BFGS-B avec un gradient analytique, contrairement à la forme fermée de la v2.
- **Résultats** : La v3 offre la meilleure calibration (NLPD) sur Hartmann, mais la v1 reste supérieure pour l'optimisation EGO.
- **Défauts appliqués** : `CoKriging` adopte la v3 ; `MultifidelityModel` conserve la v1 par défaut.
- Dépend de : `mfego/src/surrogate_models.py`, `mfego/src/cokriging.py`, `benchmarks/surrogate_v2/`.

### `benchmarks/surrogate_v3/make_figures.py`
Génère des figures interactives Plotly (HTML) comparant les benchmarks mfego v1/v2/v3 à partir des résultats de `run_benchmark.py`.
- `figure_priors` : visualise les densités des priors sur rho (normales et lognormales) avec leurs bornes.
- `figure_rho` : affiche les boxplots des rho ajustés pour chaque variante et problème, indiquant la part des ajustements aux bornes.
- `figure_nested` : trace les métriques de précision (RMSE, NLPD, couverture) sur les designs imbriqués avec intervalles interquartiles.
- `figure_optimization` : compare la performance d'optimisation (coût vs meilleure observation) des variantes EGO et concurrents.
- `figure_ego` : détaille les erreurs de recommandation, parts d'évaluations LF et qualité du modèle d'infill pour chaque variante.
- Dépend de : `v3_lib`, `figure_style`, données CSV/JSON dans `v3.RESULTS`.

### `benchmarks/surrogate_v3/run_benchmark.py`
Script de benchmark comparant mfego v1/v2/v3 aux bibliothèques GP concurrentes sur des designs imbriqués et des problèmes d'optimisation.
- `nested_task` : évalue l'exactitude et la calibration des modèles sur les jeux de test.
- `optimization_task` : exécute l'optimisation de Hartmann 6D avec différents budgets et méthodes.
- `infill_task` : analyse les données d'infilling, part basse-fidélité et précision finale.
- `decide_surrogate` : applique la règle 1 pour sélectionner la variante CoKriging par défaut.
- `decide_ego` : applique la règle 2 pour choisir la variante EGO par défaut.
- Dépend de : `v3_lib`, `v2_lib`, `bench_lib`, `src.surrogate_models`, `src.validation`

### `benchmarks/surrogate_v3/v3_lib.py`
Définit les variantes de benchmark mfego v3 (v1, v2, v3) et réutilise les problèmes de surrogate_v2.
- VARIANTS : dictionnaire des configurations d'options pour les modèles MultifidelityModel et CoKriging.
- lognormal_mode_1 : génère les paramètres d'une loi a priori LogNormale avec mode à 1 pour rho.
- centred : construit les options de MAP avec centrage de rho et noyau spécifié.
- at_bound : vérifie si rho atteint la limite de son intervalle de recherche.
- Dépend de : v2_lib, src.kernels


## `example/cokriging/`

Exemple de construction d'un surrogat multi-fidélité et de sa qualification sans jeu de test sur le problème de Forrester 1D.

### `example/cokriging/main.py`
Exemple de construction d'un surrogat multi-fidélité et de sa qualification sans jeu de test sur le problème de Forrester 1D.
- `f_low` / `f_high` : définition des fonctions de fidélité basse et haute (Eq. 17 de Sacher et al.)
- `MultifidelityModel` : modèle récursif avec hyperparamètres appris sous a priori MAP
- `CoKriging` : modèle de co-krigeage conditionné sur les données imbriquées
- `prediction_scores` : calcul des métriques de qualification leave-one-out (Q², couverture)
- Dépend de : `src.cokriging`, `src.data_management`, `src.kernels`, `src.surrogate_models`, `src.validation`


## `example/hartmann_6d/`

Le dossier `example/hartmann_6d` contient un exemple d'application pour l'optimisation multi-fidélité. Il inclut `Hartmann6d.py`, qui définit la fonction de test Hartmann 6D et ses variantes, ainsi que `hartmann_6d.py`, le script principal exécutant l'optimisation EGO.

### `example/hartmann_6d/Hartmann6d.py`
Implémente la fonction de test Hartmann 6D et ses variantes multi-fidélité pour l'optimisation.
- `Hartmann6D`: calcule la valeur exacte de la fonction Hartmann 6D à partir de constantes alpha, A et P.
- `f_l`: approxime la fonction avec un niveau de fidélité `k` via une itération de type Newton.
- `evaluate_fidelity`: sélectionne la fidélité appropriée (basse ou haute) selon le niveau demandé.
- Dépend de : `numpy`

### `example/hartmann_6d/hartmann_6d.py`
Script principal exécutant un processus d'optimisation EGO multi-fidélité sur la fonction de Hartmann 6D.
- `FunctionSimulator` : sous-classe de `BaseSimulator` évaluant la fonction à différents niveaux de fidélité via `evaluate_fidelity`.
- `__main__` : orchestre la création du modèle `MultifidelityModel`, l'initialisation des données et l'exécution de 30 itérations d'optimisation.
- `ModelVisualizer` : génère les graphiques de convergence et de surface de réponse, y compris les versions interactives.
- Dépend de : `mfego.src` (acquisition, kernels, optimizer, surrogate_models, visualization) et `.Hartmann6d`.


## `example/hydrofoil_optim/`

Le dossier `example/hydrofoil_optim` contient des exemples d'optimisation multi-fidélité pour hydrofoils. Il inclut `hydrofoil_optim.py` pour minimiser la traînée via EGO et `optim_neuralfoil.py` pour l'optimisation de profils hydrodynamiques par réseaux de neurones.

### `example/hydrofoil_optim/hydrofoil_optim.py`
Script principal exécutant l'optimisation multi-fidélité EGO pour minimiser la traînée d'un hydrofoil.
- `FunctionSimulator`: sous-classe de `BaseSimulator` évaluant la traînée via `aerosandbox` et `objective_function`.
- `EGOOptimizer`: orchestre la boucle d'optimisation itérative avec le modèle et l'acquisition.
- `MultifidelityModel`: modèle de substitution multi-fidélité utilisant `SquaredExponentialKernel`.
- `ModelVisualizer`: génère les graphiques de convergence et de surface de réponse (statiques et interactifs).
- Dépend de : `mfego/src` (modules `acquisition`, `data_management`, `kernels`, `optimizer`, `run_utils`, `simulator`, `surrogate_models`, `visualization`), `aerosandbox`, `numpy`.

### `example/hydrofoil_optim/optim_neuralfoil.py`
Module d'optimisation de profils hydrodynamiques via simulations aérodynamiques par réseaux de neurones.
- `foil_mid_fidelity`: Calcule les coefficients de traînée et portance pour un profil NACA donné à une fidélité spécifiée.
- `generate_continuous_naca4`: Génère les coordonnées mathématiques continues d'un profil NACA 4 chiffres.
- `objective_function`: Résout l'angle d'attaque pour une portance cible et évalue la traînée en gérant les échecs de calcul.
- `find_optimal_foil`: Parcourt un espace de camber et épaisseur pour identifier le profil optimal à un angle donné.
- Dépend de : `aerosandbox`, `neuralfoil`, `numpy`, `scipy.optimize`.


## `legacy/legacy_mfego_initial/`

Dossier d'archivage conservant une copie figée du framework MF-EGO antérieure aux corrections théoriques. Il contient le noyau fonctionnel dans `mfego/` (avec `main.py` et `src/`) ainsi que des cas d'usage concrets dans `example/` (notamment `hartmann_6d/` et `hydrofoil_optim/`). La documentation est assurée par `README.md` et `README_initial.md`, garantissant la traçabilité de l'implémentation initiale du projet.

### `legacy/legacy_mfego_initial/README.md`
Copie figée du framework `mfego` antérieure aux corrections théoriques et numériques, conservée pour la traçabilité.
- `mfego/main.py` et `mfego/src/*.py` : code source initial du framework.
- `example/hartmann_6d` et `example/hydrofoil_optim` : scripts d'exemple initiaux.
- `README_initial.md` : documentation d'origine du projet.
- Dépend de : modules `analysis/`, fichier `analysis/RAPPORT_ANALYSE.md`, bibliothèque `traitlets`.

### `legacy/legacy_mfego_initial/README_initial.md`
Documente de référence détaillant l'architecture et l'usage du framework NN-MF-GP pour l'optimisation multi-fidélité.
- `ExperimentData`: Gère le Design of Experiments (DoE) et l'intégrité des données par niveau de fidélité.
- `MultifidelityModel`: Implémente l'approximation récurrente non imbriquée et l'optimisation des hyperparamètres.
- `EGOOptimizer`: Contrôle le processus Ask-and-Tell et la persistance de l'état via JSON.
- `Simulator` : Interface abstraite séparant la logique physique du moteur mathématique.
- Dépend de : `numpy`, `scipy`, `matplotlib`, `neuralfoil`, `aerosandbox`.


## `legacy/legacy_mfego_initial/example/hartmann_6d/`

Le dossier `legacy/legacy_mfego_initial/example/hartmann_6d` contient l'implémentation et l'exécution d'une optimisation EGO multi-fidélité sur la fonction de Hartmann 6D. Les fichiers clés sont `Hartmann6d.py`, qui définit la fonction et son approximation, et `hartmann_6d.py`, le script principal de l'optimisation. Cette sous-section sert d'exemple concret au sein du module `legacy` du projet `Multifidelity-GP-Clean`.

### `legacy/legacy_mfego_initial/example/hartmann_6d/Hartmann6d.py`
Implémente la fonction de Hartmann 6D et son approximation multi-fidélité pour l'optimisation.
- `Hartmann6D` : calcule la valeur de la fonction de Hartmann 6D à partir d'un point en R^6.
- `f_l` : approxime la fonction via une méthode itérative dépendant du paramètre de fidélité `k`.
- `evaluate_fidelity` : sélectionne la fidélité appropriée (basse ou haute) selon le niveau `level` et le maximum `L`.
- Dépend de : numpy

### `legacy/legacy_mfego_initial/example/hartmann_6d/hartmann_6d.py`
Script principal exécutant une optimisation EGO multi-fidélité sur la fonction de Hartmann 6D.
- FunctionSimulator : sous-classe de BaseSimulator évaluant la fonction de Hartmann avec gestion des erreurs.
- __main__ : orchestre la boucle EGO en initialisant les données, le modèle et l'optimiseur.
- data : instance d'ExperimentData générant le design initial et stockant les observations.
- ego : instance d'EGOOptimizer lançant les itérations d'optimisation.
- vizualizer : instance de ModelVisualizer générant les graphiques de convergence et de surface de réponse.
- Dépend de : Hartmann6d, src.acquisition, src.data_management, src.kernels, src.optimizer, src.simulator, src.surrogate_models, src.visualization.


## `legacy/legacy_mfego_initial/example/hydrofoil_optim/`

Ce dossier contient le script `hydrofoil_optim.py`, qui exécute une optimisation globale efficace multi-fidélité (EGO) pour la conception d'une hydrofoil, ainsi que le module `optim_neuralfoil.py`, fournissant des fonctions d'optimisation basées sur des simulations aérodynamiques par réseaux de neurones.

### `legacy/legacy_mfego_initial/example/hydrofoil_optim/hydrofoil_optim.py`
Script principal exécutant une optimisation globale efficace multi-fidélité (EGO) pour la conception d'une hydrofoil.
- FunctionSimulator : évalue la traînée et la portance via aerosandbox en générant des profils NACA.
- __main__ : configure les bornes, les coûts et le nombre de niveaux de fidélité pour l'optimisation.
- EGOOptimizer : orchestre la boucle d'optimisation itérative avec 40 itérations.
- ModelVisualizer : génère les graphiques de convergence et de surface de réponse.
- Dépend de : aerosandbox, numpy, src.acquisition, src.data_management, src.kernels, src.optimizer, src.simulator, src.surrogate_models, src.visualization, .optim_neuralfoil.

### `legacy/legacy_mfego_initial/example/hydrofoil_optim/optim_neuralfoil.py`
Module de fonctions pour l'optimisation de designs d'hydrofoils utilisant des simulations aérodynamiques basées sur des réseaux de neurones.
- `foil_mid_fidelity`: Calcule les coefficients de traînée et portance pour un profil d'aile donné à un niveau de fidélité spécifique.
- `generate_continuous_naca4`: Génère les coordonnées mathématiques continues d'un profil NACA 4 chiffres.
- `objective_function`: Évalue la fonction objectif en résolvant l'angle d'attaque cible et gérant les niveaux de fidélité du modèle.
- `find_optimal_foil`: Recherche le profil NACA optimal en évaluant une gamme de valeurs de camber et d'épaisseur.
- Dépend de : `aerosandbox`, `neuralfoil`, `numpy`, `scipy.optimize`


## `legacy/legacy_mfego_initial/mfego/`

Le dossier `legacy/legacy_mfego_initial/mfego` héberge l'implémentation du framework MF-EGO. Il comprend le script principal `main.py` et le sous-dossier `src/` contenant l'algorithme d'optimisation, les fonctions d'acquisition, les modèles de substitution, les noyaux, la gestion des données, la base des simulateurs et la visualisation. Cette structure forme le noyau fonctionnel de l'optimisation multi-fidélité du projet.

### `legacy/legacy_mfego_initial/mfego/main.py`
Script principal exécutant un processus d'optimisation globale efficace multi-fidélité (EGO) avec visualisation des résultats.
- `FunctionSimulator`: classe dérivée de `BaseSimulator` évaluant une fonction quadratique bruitée sur deux niveaux de fidélité.
- `ExperimentData`: initialise les données expérimentales, les bornes de conception et les coûts associés aux différents niveaux.
- `MultifidelityModel`: configure le modèle de substitution multi-fidélité utilisant le noyau `SquaredExponentialKernel`.
- `EGOOptimizer`: orchestre la boucle d'optimisation itérative sur 10 itérations.
- `ModelVisualizer`: génère les graphiques de convergence et de réponse à partir du fichier JSON de sauvegarde.
- Dépend de : `src.acquisition`, `src.data_management`, `src.kernels`, `src.optimizer`, `src.simulator`, `src.surrogate_models`, `src.visualization`.


## `legacy/legacy_mfego_initial/mfego/src/`

Ce dossier contient l'implémentation du framework MF-EGO, incluant l'algorithme d'optimisation (`optimizer.py`), la fonction de mérite (`acquisition.py`) et les modèles de substitution (`surrogate_models.py`). Il gère également les données (`data_management.py`), les noyaux (`kernels.py`), la base abstraite des simulateurs (`simulator.py`) et la visualisation (`visualization.py`).

### `legacy/legacy_mfego_initial/mfego/src/acquisition.py`
Implémente la fonction de mérite pour l'algorithme EGO multi-fidélité.
- AcquisitionFunction: classe encapsulant le calcul de la fonction d'acquisition.
- evaluate_merit: évalue la fonction de mérite en combinant l'Expected Improvement, le ratio de coût et le ratio d'information.
- Dépend de : numpy, scipy.stats, src.data_management, src.surrogate_models

### `legacy/legacy_mfego_initial/mfego/src/data_management.py`
Gère les données expérimentales pour l'optimisation multi-fidélité, en assurant la modification unique des données.
- `ExperimentData`: Classe principale gérant les bornes, coûts et structures de données par niveau de fidélité.
- `generate_initial_design`: Génère un plan expérimental initial utilisant l'échantillonnage latin hypercube.
- `is_already_evaluated`: Vérifie si un point a déjà été évalué à un niveau spécifique via la distance minimale.
- `add_observation`: Ajoute de nouvelles observations aux dictionnaires de coordonnées, valeurs et métriques.
- Dépend de : numpy, scipy.stats, traitlets

### `legacy/legacy_mfego_initial/mfego/src/kernels.py`
Fournit les fonctions de noyau (covariance) et les classes associées pour le processus gaussien multi-fidélité.
- cov_fct : calcule la covariance entre deux points via le noyau exponentiel carré.
- base_covariance_matrix : construit la matrice de covariance complète pour le jeu d'entraînement.
- k_l_vector : calcule le vecteur de covariance croisée entre un point candidat et l'entraînement.
- Kernel : classe de base gérant les hyperparamètres et définissant l'interface des méthodes de covariance.
- SquaredExponentialKernel : implémente le noyau exponentiel carré en héritant de Kernel.
- Dépend de : numpy

### `legacy/legacy_mfego_initial/mfego/src/optimizer.py`
Implémente l'algorithme d'optimisation globale efficace (EGO) pour l'optimisation multi-fidélité.
- EGOOptimizer: Gère la boucle d'optimisation ask/tell, le suivi des coûts et l'historique des meilleures solutions.
- _find_next_point: Maximise la fonction d'acquisition via l'optimisation par évolution différentielle pour sélectionner le point suivant.
- save_state / load_state: Sérialise et restaure l'état de l'optimiseur (données, hyperparamètres GP) dans un fichier JSON.
- NumpyEncoder: Encodage JSON personnalisé pour convertir les tableaux et scalaires NumPy en types Python natifs.
- Dépend de : src.acquisition, src.simulator, src.surrogate_models, numpy, scipy.optimize

### `legacy/legacy_mfego_initial/mfego/src/simulator.py`
Définit la classe de base abstraite pour les simulateurs dans le cadre d'optimisation multi-fidélité.
- BaseSimulator : classe abstraite que les utilisateurs doivent hériter pour implémenter leurs propres simulateurs.
- evaluate : méthode abstraite obligeant les sous-classes à retourner la valeur à minimiser et un dictionnaire de valeurs réelles.
- Dépend de : abc

### `legacy/legacy_mfego_initial/mfego/src/surrogate_models.py`
Module principal implémentant les classes de processus gaussiens et de modèles multi-fidélité.
- `GaussianProcess`: gère l'ajustement des hyperparamètres par minimisation de la log-vraisemblance négative et la prédiction de la moyenne et de la variance.
- `MultifidelityModel`: combine plusieurs niveaux de fidélité via des coefficients de corrélation pour produire une prédiction finale.
- `fit` (dans `MultifidelityModel`): entraîne séquentiellement un processus gaussien par niveau de fidélité en soustrayant la prédiction du niveau inférieur.
- `predict` (dans `MultifidelityModel`): calcule la prédiction agrégée et les variances individuelles de chaque niveau.
- Dépend de : `src.data_management`, `src.kernels`, `scipy`, `numpy`.

### `legacy/legacy_mfego_initial/mfego/src/visualization.py`
Module de visualisation pour le framework MF-EGO permettant de recharger un modèle depuis JSON et de tracer les surfaces de réponse et la convergence.
- `ModelVisualizer`: Reconstitue le modèle en mémoire et génère les graphiques.
- `_rebuild_model_in_memory`: Injecte les hyperparamètres et recalcule les matrices inverses sans ré-entraînement.
- `plot_convergence`: Trace l'évolution du meilleur point en fonction du coût cumulé.
- `plot_response_1d`: Visualise la prédiction 1D avec les bandes d'incertitude.
- `plot_response_surface_2d`: Génère la surface de réponse 2D pour deux paramètres spécifiés.
- Dépend de : `src.kernels.SquaredExponentialKernel`, `src.surrogate_models.MultifidelityModel`, `matplotlib`, `numpy`.


## `legacy/legacy_nested/`

Le dossier `legacy/legacy_nested` contient l'implémentation de l'optimisation multi-fidélité imbriquée (Nested MF) basée sur le schéma de Le Gratiet, appliquée à la fonction de Hartmann 6D. Les fichiers clés incluent `main.py` pour l'exécution, `nested_mf_optimizer.py` pour l'algorithme d'optimisation, et `nested_mf_covariance.py` pour les fonctions de covariance. Ce module fournit également des outils d'échantillonnage via `nested_mf_sampling.py` et des définitions de la fonction cible dans `Hartmann6d.py`.

### `legacy/legacy_nested/Hartmann6d.py`
Implémente la fonction de Hartmann 6D et des approximations multi-fidélités pour l'optimisation.
- `Hartmann6D`: calcule la valeur exacte de la fonction de Hartmann en 6 dimensions.
- `f_l`: génère une approximation de la fonction via un processus itératif contrôlé par le paramètre `k`.
- `evaluate_fidelity`: sélectionne le niveau de fidélité approprié (bas ou haut) selon les paramètres `level` et `L`.
- Dépend de : `numpy`

### `legacy/legacy_nested/README.md`
Documentation du projet d'optimisation multi-fidélité EGO basé sur le schéma récursif de Le Gratiet.
- `main.py` : point d'entrée orchestrant l'expérience et l'interface utilisateur.
- `nested_mf_optimizer.py` : cœur de l'algorithme, boucle EGO et fonctions d'acquisition.
- `nested_mf_covariance.py` : gestion des noyaux de covariance et matrices spatiales.
- `nested_mf_sampling.py` : génération d'expériences emboîtées et gestion des résidus.
- `Hartmann6d.py` : simulateur de fonction de référence en 6 dimensions.
- Dépend de : modules du projet (`main.py`, `nested_mf_optimizer.py`, `nested_mf_covariance.py`, `nested_mf_sampling.py`, `Hartmann6d.py`).

### `legacy/legacy_nested/main.py`
Script principal exécutant l'optimisation multi-fidélité imbriquée sur la fonction Hartmann 6D.
- LogTee : classe dupliquant la sortie standard vers la console et un fichier journal.
- evaluate_fidelity : évalue la fonction cible selon le niveau de fidélité demandé.
- plot_ego_results : génère les graphiques de convergence et de projection spatiale des résultats.
- __main__ : gère l'analyse des arguments, l'initialisation du design expérimental et le lancement de l'optimiseur.
- Dépend de : Hartmann6d, nested_mf_sampling, nested_mf_optimizer, numpy, matplotlib.

### `legacy/legacy_nested/nested_mf_covariance.py`
Définit les fonctions de covariance pour un processus gaussien multi-fidélité imbriqué.
- `Cov_fct`: calcule la covariance ponctuelle entre deux échantillons via une fonction radiale.
- `base_covariance_matrix`: construit la matrice de covariance de base à partir des hyperparamètres.
- `nested_mf_covariance`: évalue la covariance agrégée multi-fidélité en sommant les contributions de chaque niveau.
- `nested_mf_covariance_matrix`: génère la matrice de covariance complète pour un ensemble de points.
- `k_l_vector`: calcule le vecteur de covariance entre un nouveau point et les points existants.
- Dépend de : numpy

### `legacy/legacy_nested/nested_mf_optimizer.py`
Implémente l'optimisation multi-fidélité imbriquée (Nested MF) via processus gaussiens et critères d'amélioration.
- `log_likelihood_mf`: Calcule la log-vraisemblance pour l'estimation des hyperparamètres.
- `predict_nested_mf`: Prédit la moyenne et la variance de manière récursive sur les niveaux de fidélité.
- `merit_nested`: Évalue la fonction de mérite combinant l'amélioration attendue et la réduction de variance.
- `run_nested_mf_ego`: Orchestre la boucle d'optimisation EGO (entraînement, sélection du point, évaluation).
- Dépend de : `Hartmann6d`, `nested_mf_covariance`, `nested_mf_sampling`, `scipy.optimize`, `scipy.stats`.

### `legacy/legacy_nested/nested_mf_sampling.py`
Module de génération d'échantillonnage multi-fidélité par Hypercubes Latins imbriqués et calcul de résidus pour processus gaussiens.
- `generate_nested_lhs`: Génère des designs LHS imbriqués où chaque niveau de fidélité est un sous-ensemble du précédent.
- `Delta_Y_l`: Calcule les résidus inter-fidélités en soustrayant la contribution du niveau inférieur pondérée par la corrélation.
- `extract_subpart_vector`: Récupère les observations du niveau inférieur correspondant exactement aux coordonnées du niveau supérieur.
- `is_already_evaluated`: Vérifie si un point est déjà présent dans un jeu de données existant via une tolérance de distance.
- Dépend de : `numpy`, `scipy.stats.qmc`


## `legacy/legacy_non_nested/`

Le dossier `legacy/legacy_non_nested` implémente l'algorithme MF-EGO non imbriqué pour l'optimisation bayésienne multi-fidélité. Il contient des modules clés comme `non_nested_mf_optimizer.py`, `non_nested_mf_covariance.py` et `non_nested_mf_sampling.py`, ainsi que le script principal `main.py` dédié à la conception d'ailes NACA. Le sous-dossier `tests/` fournit des scripts de validation, tandis que `surrogate_vizu.py` permet la visualisation des surfaces de réponse.

### `legacy/legacy_non_nested/Hartmann6d.py`
Implémente la fonction de Hartmann 6D et des approximations multi-fidélités pour l'optimisation.
- `Hartmann6D`: calcule la valeur exacte de la fonction de Hartmann en 6 dimensions.
- `f_l`: génère une approximation de la fonction via un processus itératif contrôlé par le paramètre `k`.
- `evaluate_fidelity`: sélectionne le niveau de fidélité approprié (bas ou haut) selon les paramètres `level` et `L`.
- Dépend de : `numpy`

### `legacy/legacy_non_nested/README.md`
Documentation du dépôt implémentant l'algorithme MF-EGO non imbriqué (Non-Nested) en Python.
- `generate_non_nested_lhs` : Génère des plans d'expériences indépendants par niveau de fidélité.
- `predict_mf_mean_up_to_level` : Effectue la prédiction croisée en remplaçant les valeurs manquantes par l'espérance mathématique.
- `merit_non_nested` : Évalue la fonction de mérite pour sélectionner le niveau d'évaluation optimal.
- `log_likelihood_mf` : Garantit la stabilité numérique des calculs de vraisemblance via des opérations vectorielles strictes.
- Dépend de : `main.py`, `non_nested_mf_optimizer.py`, `nested_mf_covariance.py`, `custom_fluid_functions.py`, `scipy`, `numpy`.

### `legacy/legacy_non_nested/main.py`
Script principal d'optimisation multi-fidélité non imbriquée appliquée à la conception d'ailes NACA continues.
- `LogTee` : duplique la sortie standard vers la console et un fichier de journalisation.
- `evaluate_fidelity` : évalue l'aile via `generate_continuous_naca4` et `objective_function` avec pénalité sur la portance.
- `plot_ego_results` : génère les graphiques de convergence et de projection spatiale des points d'évaluation.
- Bloc `__main__` : gère l'argumentaire, initialise le design expérimental, exécute `run_non_nested_mf_ego` et exporte les résultats.
- Dépend de : `non_nested_mf_optimizer`, `non_nested_mf_sampling`, `tests.optim_neuralfoil`, `aerosandbox`, `neuralfoil`, `numpy`, `matplotlib`.

### `legacy/legacy_non_nested/non_nested_mf_covariance.py`
Module legacy pour le calcul de noyaux de covariance gaussienne non imbriqués.
- `Cov_fct`: calcule le noyau exponentiel carré entre deux points individuels.
- `base_covariance_matrix`: construit la matrice de covariance n x n complète pour un jeu de données.
- `k_l_vector`: génère le vecteur de covariance croisée entre un point candidat et l'ensemble d'entraînement.
- Dépend de : numpy

### `legacy/legacy_non_nested/non_nested_mf_optimizer.py`
Module d'optimisation multi-fidélité non imbriquée (MF-EGO) pour l'optimisation bayésienne.
- `log_likelihood_mf` : calcule la log-vraisemblance via décomposition de Cholesky pour la stabilité numérique.
- `predict_non_nested_mf` : prédit récursivement la moyenne et la variance du modèle multi-fidélité.
- `merit_non_nested` : évalue la fonction de mérite combinant AEI, coût et réduction d'incertitude.
- `run_non_nested_mf_ego` : orchestre la boucle itérative d'entraînement, d'acquisition et d'évaluation.
- Dépend de : `non_nested_mf_covariance`, `non_nested_mf_sampling`, `scipy.optimize`, `scipy.stats`, `numpy`.

### `legacy/legacy_non_nested/non_nested_mf_sampling.py`
Génère des échantillonnages Latin Hypercube non imbriqués et calcule des résidus pour le modélisation multi-fidélité.
- `generate_non_nested_lhs`: Crée des designs LHS indépendants pour chaque niveau de fidélité via `scipy.stats.qmc`.
- `Delta_Y_l`: Calcule les résidus en soustrayant la corrélation du niveau inférieur aux observations actuelles.
- `extract_subpart_vector`: Identifie les observations du niveau inférieur correspondant spatialement aux points du niveau supérieur.
- `is_already_evaluated`: Vérifie si un point est déjà présent dans un ensemble d'entraînement existant.
- Dépend de : `numpy`, `scipy.stats`

### `legacy/legacy_non_nested/surrogate_vizu.py`
Script de visualisation de la surface de réponse d'un processus gaussien en utilisant des données d'entraînement et des hyperparamètres.
- `squared_exponential_kernel`: Calcule la matrice de covariance entre deux ensembles de points.
- `predict_gp`: Prédit la moyenne du processus gaussien pour de nouveaux points.
- Chargement des données : Lit les données d'entraînement et les hyperparamètres depuis `ego_backup.json`.
- Génération de la grille : Crée une grille 50x50 pour la prédiction et fixe les dimensions supplémentaires à leur valeur optimale.
- Visualisation : Produit une surface 3D et une carte de contour 2D sauvegardée dans `response_surface.png`.
- Dépend de : `ego_backup.json`, modules `numpy`, `matplotlib`, `mpl_toolkits`.


## `legacy/legacy_non_nested/tests/`

Le dossier `legacy/legacy_non_nested/tests` contient des scripts de test pour des méthodes non imbriquées, incluant `custom_fluid_functions.py` pour le calcul du coefficient de traînée et `optim_neuralfoil.py` pour l'optimisation de profils via NeuralFoil.

### `legacy/legacy_non_nested/tests/custom_fluid_functions.py`
Fournit une fonction d'objectif multi-fidélité pour le calcul du coefficient de traînée d'une aile.
- `foil_mid_fidelity`: Calcule le coefficient de traînée via `neuralfoil` pour un profil donné.
- `objective_function`: Évalue la traînée en sélectionnant la taille du modèle selon le niveau de fidélité.
- Dépend de : `neuralfoil`, `aerosandbox`, `numpy`.

### `legacy/legacy_non_nested/tests/optim_neuralfoil.py`
Script de test pour l'optimisation de profils d'ailes utilisant le module NeuralFoil.
- `foil_mid_fidelity`: calcule les coefficients de traînée et portance pour un profil NACA donné.
- `generate_continuous_naca4`: génère les coordonnées mathématiques d'un profil NACA 4 chiffres.
- `objective_function`: détermine l'angle d'attaque optimal pour atteindre une portance cible via `root_scalar`.
- `find_optimal_foil`: itère sur les cambers et épaisseurs pour identifier le profil optimal.
- Dépend de : `neuralfoil`, `aerosandbox`, `numpy`, `scipy.optimize`.


## `mfego/`

Le dossier `mfego` contient le cœur du framework d'optimisation multi-fidélité, avec `src/` pour l'implémentation (`optimizer.py`, `acquisition.py`, `cokriging.py`, `kernels.py`) et `main.py` pour l'exécution d'un exemple. Il fournit les modèles de substitution et les outils de simulation, s'intégrant au projet `Multifidelity-GP-Clean` comme module central de calcul.

### `mfego/main.py`
Script principal exécutant une optimisation EGO multi-fidélité sur un exemple de Forrester.
- `FunctionSimulator`: simulateur de démonstration évaluant une fonction quadratique avec bruit selon deux niveaux de fidélité.
- `MultifidelityModel`: modèle de substitution multi-fidélité utilisant un noyau `SquaredExponentialKernel` et une estimation MAP.
- `EGOOptimizer`: orchestrateur de l'algorithme d'optimisation itérative avec sauvegarde de l'état.
- `ModelVisualizer`: module de visualisation générant des graphiques statiques et interactifs de convergence et de réponse.
- Dépend de : `src.acquisition`, `src.data_management`, `src.kernels`, `src.optimizer`, `src.run_utils`, `src.simulator`, `src.surrogate_models`, `src.visualization`.


## `mfego/src/`

Le dossier `mfego/src` constitue le cœur du framework MF-EGO, regroupant l'implémentation de l'optimisation globale efficace multi-fidélité via `optimizer.py` et `acquisition.py`. Il fournit les modèles de substitution, notamment le co-kriging auto-régressif (`cokriging.py`) et les noyaux gaussiens (`kernels.py`), gérés par `data_management.py`. Des modules auxiliaires comme `validation.py`, `visualization.py` et `simulator.py` assurent la qualification, l'affichage et la simulation des expériences.

### `mfego/src/acquisition.py`
Implémente la fonction de mérite de l'algorithme EGO pour la sélection de points d'échantillonnage multi-fidélité.
- `AcquisitionFunction`: classe principale gérant le calcul des mérites et la mise à jour de la meilleure solution effective.
- `update`: calcule la meilleure solution effective (Eq. 19) à partir des points d'entraînement et des prédictions du modèle.
- `evaluate_merits_batch`: évalue vectoriellement les mérites (AEI pondérés par les coûts et ratios d'information) pour plusieurs niveaux de fidélité.
- `feasibility`: attribut optionnel appliquant des contraintes géométriques en annulant les mérites des points infeasibles.
- Dépend de : `src.data_management.ExperimentData`, `src.surrogate_models.MultifidelityModel`, `numpy`, `scipy.stats`.

### `mfego/src/cokriging.py`
Surrogate de co-kriging auto-régressif pour des designs emboîtés, conditionné sur toutes les observations sans infilling.
- `CoKriging`: Classe principale gérant l'ajustement niveau par niveau, la prédiction et la sérialisation.
- `nested_index`: Identifie l'index des points d'un niveau supérieur dans le niveau inférieur.
- `missing_points_for_nesting`: Détecte les points manquants pour rendre un design non emboîtement compatible.
- `predict_batch`: Calcule la moyenne et la variance conditionnées via la décomposition de Cholesky.
- Dépend de : `src.data_management`, `src.kernels`, `src.surrogate_models`.

### `mfego/src/data_management.py`
Module dédié à la gestion exclusive des données expérimentales pour l'optimisation multi-fidélité.
- `ExperimentData`: classe principale gérant les dictionnaires de points (`x_dict`), observations (`y_dict`) et métriques par niveau de fidélité.
- `generate_initial_design`: crée un plan d'expériences initial en hypercube latin, avec option de filtrage par faisabilité et sélection maximin.
- `generate_nested_design`: génère un plan d'expériences imbriqué où chaque niveau est un sous-ensemble bien réparti du précédent.
- `greedy_maximin`: fonction utilitaire sélectionnant un sous-ensemble de points bien répartis parmi des candidats via une approche gloutonne.
- Dépend de : `numpy`, `scipy.stats.qmc`.

### `mfego/src/kernels.py`
Définit les fonctions de covariance et les classes de noyaux pour le processus gaussien multi-fidélité.
- `cov_fct` : calcule la covariance exponentielle carrée entre deux points individuels.
- `base_covariance_matrix` : construit la matrice de covariance complète pour un ensemble d'entraînement.
- `k_l_vector` : calcule le vecteur de covariance croisée entre un nouveau point et les points d'entraînement.
- `cross_covariance_matrix` : version par lots du vecteur de covariance croisée pour plusieurs nouveaux points.
- `pairwise_sq_diff` : calcule les différences carrées coordonnées entre toutes les paires de points.
- `Kernel` : classe de base gérant les hyperparamètres et les méthodes abstraites de covariance.
- `SquaredExponentialKernel` : implémente le noyau RBF avec gradients analytiques.
- `Matern52Kernel` : implémente le noyau Matérn 5/2 avec gradients analytiques.
- Dépend de : numpy

### `mfego/src/optimizer.py`
Module implémentant l'algorithme d'optimisation globale efficace (EGO) pour l'optimisation multi-fidélité.
- `EGOOptimizer`: classe principale gérant la boucle d'optimisation ask/tell, la sauvegarde d'état et l'évaluation.
- `_find_next_point`: maximise la fonction d'acquisition via l'évolution différentielle pour sélectionner le point suivant.
- `NumpyEncoder`: encodeur JSON personnalisé convertissant les types NumPy en types Python natifs.
- `run`: exécute la boucle d'optimisation automatique avec gestion de la convergence et des points aléatoires.
- Dépend de : `src.acquisition`, `src.simulator`, `src.surrogate_models`.

### `mfego/src/run_utils.py`
Fournit des utilitaires de gestion d'exécution : création d'un horodaté, configuration du journalisation et mesure du temps de calcul.
- `RunContext`: dataclass définissant les chemins (annuaire, journal) partagés par une exécution.
- `create_run`: crée l'annuaire horodaté unique pour les sorties de l'exécution.
- `setup_logging`: configure le logger racine avec le format bdFoil et l'encodage UTF-8.
- `RunTimer`: gestionnaire de contexte affichant le temps total de calcul, même en cas d'exception.
- Dépend de : `logging`, `os`, `time`, `dataclasses`, `datetime`, `pathlib`, `zoneinfo`.

### `mfego/src/simulator.py`
Définit la classe de base abstraite pour les simulateurs dans le cadre d'optimisation multi-fidélité.
- BaseSimulator : classe abstraite que les utilisateurs doivent hériter pour implémenter leurs propres simulateurs.
- evaluate : méthode abstraite obligeant les sous-classes à retourner la valeur à minimiser et un dictionnaire de valeurs réelles.
- Dépend de : abc

### `mfego/src/surrogate_models.py`
Implémente les classes de processus gaussien et de modèle multi-fidélité avec estimation des hyperparamètres.
- `GaussianProcess`: Gère l'ajustement, la prédiction et l'optimisation des hyperparamètres (MAP/MLE) d'un processus gaussien unique.
- `MultifidelityModel`: Combine plusieurs niveaux de fidélité pour des prédictions récursives et vectorisées.
- `neg_log_prior` / `check_priors` : Valide et calcule les log-a-priori des hyperparamètres.
- `load_surrogate` : Charge un modèle pré-entraîné depuis un fichier JSON.
- Dépend de : `src.data_management.ExperimentData`, `src.kernels.Kernel`, `scipy.linalg`, `scipy.optimize`.

### `mfego/src/validation.py`
Évalue la qualification d'un substitut en calculant la précision et la calibration des prédictions gaussiennes par rapport aux valeurs réelles.
- `prediction_scores`: Calcule les métriques de performance incluant `rmse_rel`, `q2`, `nlpd`, `z_rms` et les taux de couverture (`coverage_50`, `coverage_90`, `coverage_95`, `coverage_99`).
- Dépend de : `numpy`, `scipy.stats`

### `mfego/src/visualization.py`
Module de visualisation du framework MF-EGO qui recharge un modèle entraîné depuis un fichier JSON pour tracer les surfaces de réponse et la convergence sans réentraîner les processus gaussiens.
- `ModelVisualizer`: classe principale qui reconstruit le modèle en mémoire via `_rebuild_model_in_memory` et génère les graphiques statiques (matplotlib) et interactifs (plotly).
- `_load_json`: charge l'état du modèle depuis le fichier JSON spécifié.
- `_rebuild_model_in_memory`: reconstruit l'instance `MultifidelityModel` en injectant les hyperparamètres et les données d'entraînement.
- `plot_convergence`, `plot_response_1d`, `plot_response_surface_2d`: méthodes de génération des graphiques statiques PNG.
- `plot_convergence_interactive`, `plot_response_1d_interactive`, `plot_response_surface_2d_interactive`: méthodes de génération des graphiques interactifs HTML.
- Dépend de : `src.kernels.SquaredExponentialKernel`, `src.surrogate_models.MultifidelityModel`, `matplotlib.pyplot`, `numpy`, `json`, `plotly` (optionnel).


## `pipelines/`

Le dossier `pipelines` orchestre l'optimisation multi-fidélité des profils d'aile en reliant le framework `mfego` aux solveurs `bdToolbox` et `bdFoil`. Il contient principalement le sous-dossier `bdtoolbox_foil` (avec `run.py`, `simulator.py`, `solvers.py`) pour la gestion des géométries et des objectifs, ainsi que `README.md` et `__init__.py` pour la documentation et l'initialisation du paquet.

### `pipelines/README.md`
Documente de référence détaillant le pipeline d'optimisation multi-fidélité des profils d'aile reliant mfego à bdToolbox et bdFoil.
- `pipelines/bdtoolbox_foil` : connecteur principal orchestrant l'optimisation bayésienne (NN-MF-EGO) via une configuration JSON.
- `run` : module exécutable gérant le DOE, l'optimisation, l'export des surrogates et les logs horodatés.
- `planform.py` : géométrie 3D (arc C-foil) et gestion des paramètres d'attitude pour les solvants NPLLT et AVL.
- `section_constraint.py` : application de la contrainte de portance 2D par projection sur le décalage de cambrure.
- Dépend de : bdToolbox (bdSec, PARSEC), bdFoil (XFOIL, AVL, NonPlanarSolver), NeuralFoil, aerosandbox.

### `pipelines/__init__.py`
Module d'initialisation du paquet `pipelines` reliant `mfego` aux boîtes à outils externes de conception et de simulation.
- pipelines/__init__.py : définit la documentation et la structure du module de connexion.
- Dépend de : modules externes de conception et de simulation (non spécifiés).


## `pipelines/bdtoolbox_foil/`

Le dossier `pipelines/bdtoolbox_foil` implémente un pipeline d'optimisation multi-fidélité (mfego) pour des sections d'aile via les solveurs bdToolbox et bdFoil. Il comprend des modules de gestion des géométries 2D et 3D (`geometry.py`, `planform.py`), des objectifs aérodynamiques (`objectives.py`) et des contraintes (`constraints.py`, `section_constraint.py`). Le script `run.py` orchestre l'optimisation, tandis que `simulator.py` et `solvers.py` gèrent l'évaluation des points de conception et les backends de calcul.

### `pipelines/bdtoolbox_foil/__init__.py`
Définit le pipeline d'optimisation multi-fidélité (mfego) des sections d'aile en utilisant les solveurs bdToolbox et bdFoil.
- mfego NN-MF-EGO : optimise les sections 2D via NeuralFoil et bdFoil core XFOIL.
- NeuralFoil polars : gère les polaires pour la ligne de portance non planaire.
- bdFoil core AVL : fournit les polaires XFOIL pour la section d'un plan fixe 3D (C-foil arc).
- Dépend de : bdToolbox, bdFoil, pipelines.configs

### `pipelines/bdtoolbox_foil/bridge.py`
Gère la résolution des chemins et l'importation différée des modules externes bdFoil et bdToolbox.
- `BridgePaths`: dataclass figée contenant les racines résolues des boîtes à outils.
- `resolve_paths`: détermine les chemins selon la priorité configuration, variables d'environnement, puis valeurs par défaut.
- `setup`: configure `sys.path` et la variable `BDFOIL_SOFT` pour rendre les paquets importables.
- `bdfoil_core`: importe le paquet central de bdFoil et ses sous-modules obligatoires.
- `environment_report`: compile les versions des dépendances et l'état git pour le traçage des résultats.
- Dépend de : bdFoil, bdToolbox, neuralfoil, numpy, scipy, aerosandbox, numba.

### `pipelines/bdtoolbox_foil/config.py`
Définit la structure et la validation des configurations d'optimisation de profils (2D et 3D).
- `FoilProblem` : classe de données représentant la configuration validée du problème.
- `load_config` : charge, valide et résout les chemins relatifs du fichier JSON.
- `from_dict` : construit l'objet `FoilProblem` à partir d'un dictionnaire brut.
- `_check_3d` : applique les règles spécifiques aux optimisations de section 3D.
- Dépend de : `pipelines.bdtoolbox_foil.planform`

### `pipelines/bdtoolbox_foil/constraints.py`
Définit les contraintes géométriques a priori de l'épaisseur pour la validation des points de conception.
- `section_violation`: génère un message descriptif si la contrainte d'épaisseur est violée.
- `_kulfan_thickness_weights`: extrait les poids d'épaisseur et l'épaisseur TE pour la paramétrisation Kulfan.
- `max_thickness_batch`: calcule l'épaisseur max/chord d'un lot de points, gérant les surfaces croisées.
- `feasibility_function`: fournit la fonction de faisabilité pour les hooks mfego [KC].
- Dépend de : `bridge`, `geometry` (via `THICKNESS_STATIONS`, `build_section`, `cst_thickness`, `max_thickness`, `section_parameters`).

### `pipelines/bdtoolbox_foil/geometry.py`
Génère les coordonnées des sections 2D (NACA 4, Kulfan/CST, PARSEC) à partir de variables de conception normalisées.
- `Section` : dataclass stockant coordonnées, paramètres physiques et épaisseur.
- `to_physical` : convertit les points de design normalisés en valeurs physiques.
- `kulfan_parameters` : transforme les paramètres plats en poids Kulfan (formes "weights" ou "thickness_camber").
- `build_section` : assemble la géométrie finale selon le type de paramétrisation configuré.
- `write_xf` : exporte les coordonnées au format XFOIL/bdSec (.xf).
- Dépend de : `bridge` (module local), `aerosandbox`, `scipy`.

### `pipelines/bdtoolbox_foil/objectives.py`
Définit les objectifs d'optimisation calculés à partir d'une polaire de section, incluant la traînée à portance cible et le rapport portance/traînée maximal.
- `monotone_branch`: Identifie la portion monotone de la courbe de portance contenant l'angle d'attaque proche de zéro.
- `value_at_cl`: Interpole la valeur d'une colonne (ex. traînée) à une portance cible, retournant NaN si hors branche.
- `evaluate_objective`: Calcule la valeur minimisée de l'objectif et les métriques associées (portance max, angles, coefficients) pour une polaire donnée.
- Dépend de : `solvers.Polar`, `numpy`.

### `pipelines/bdtoolbox_foil/planform.py`
Génère la géométrie 3D fixe d'une voile à profil unique (C-foil) sous forme de fichier CSV bdGeometry.
- `CFoilArc` : dataclass définissant les paramètres géométriques et validant la configuration.
- `arc_geometry` : calcule les coordonnées du bord d'attaque le long de l'arc circulaire.
- `stations` : construit les tableaux de planform (chord, twist, MS) pour les stations.
- `write_planform_csv` : écrit le fichier CSV de géométrie en mètres.
- `attitude_angles` : convertit l'attitude (heel, trunk_cant) en cant total.
- Dépend de : `numpy`

### `pipelines/bdtoolbox_foil/run.py`
Script d'optimisation multi-fidélité de profils d'aile utilisant les solveurs bdToolbox / bdFoil via le framework mfego.
- `run_optimization`: orchestre la boucle NN-MF-EGO, la génération de DOE, l'export des résultats et la visualisation.
- `verify_optimum`: évalue le meilleur point du substitut au niveau le plus coûteux pour valider la solution finale.
- `calibrate_costs`: mesure le temps de calcul moyen par niveau sur des designs aléatoires pour ajuster les coûts relatifs.
- `main`: point d'entrée CLI gérant les arguments, la configuration et la création du dossier d'exécution.
- Dépend de : `src.acquisition`, `src.data_management`, `src.optimizer`, `src.surrogate_models`, `src.visualization`, `pipelines.bdtoolbox_foil.bridge`, `pipelines.bdtoolbox_foil.config`, `pipelines.bdtoolbox_foil.constraints`, `pipelines.bdtoolbox_foil.geometry`, `pipelines.bdtoolbox_foil.simulator`.

### `pipelines/bdtoolbox_foil/section_constraint.py`
Gère la contrainte d'égalité sur la portance 2D d'une section en résolvant le décalage de cambrure delta via la méthode de Brent.
- `CamberProjector` : classe principale qui résout le paramètre delta pour atteindre la cible CL2d.
- `ProjectionResult` : dataclass stockant le delta résolu, la portance et la confiance du modèle.
- `ConstraintFailure` : exception levée si la cible de portance est inatteignable dans les bornes définies.
- `settings` : fonction fusionnant les paramètres de configuration avec les valeurs par défaut.
- Dépend de : `bridge`, `geometry`

### `pipelines/bdtoolbox_foil/simulator.py`
Simulateur d'optimisation de foil pour mfego, évaluant des points de conception via des solveurs 2D ou 3D.
- `BdToolboxFoilSimulator`: classe principale héritant de `BaseSimulator`, gérant l'évaluation et les erreurs.
- `_trim`: itération sécante pour ajuster l'attitude afin d'égaliser la portance.
- `_evaluate_section_on_planform`: calcule l'objectif de traînée avec correction de portance.
- `_json_friendly`: convertit les métriques NumPy en types JSON compatibles.
- Dépend de : `mfego`, `bridge`, `constraints`, `geometry`, `objectives`, `solvers`, `planform`, `section_constraint`.

### `pipelines/bdtoolbox_foil/solvers.py`
Définit les backends de solveurs 2D et 3D pour le calcul des polaires et coefficients aérodynamiques.
- `NeuralFoilBackend`: Calcule les polaires de section via le modèle NeuralFoil.
- `XfoilCoreBackend`: Génère les polaires XFOIL via le moteur bdFoil core.
- `NpLltBackend`: Résout la ligne de portance non plane (NPLLT) avec polaires NeuralFoil ou XFOIL.
- `AvlCoreBackend`: Exécute la campagne AVL avec polaires XFOIL et modèle visqueux profil.
- `make_backend`: Fabrique l'instance de solveur selon le niveau de fidélité configuré.
- Dépend de : `bridge`, `geometry`, `planform`, `bdFoil`.


## `sandbox/`

Carnet d'expérimentation reproduisant l'optimisation bayésienne mono-fidélité et multi-fidélité imbriquée (N-MF) sur la fonction de Hartmann 6D.

### `sandbox/gaussian_process_singlefidelity_sandbox.ipynb`
Carnet d'expérimentation reproduisant l'optimisation bayésienne mono-fidélité et multi-fidélité imbriquée (N-MF) sur la fonction de Hartmann 6D.
- `Hartmann6D` : implémentation de la fonction objectif de test en 6 dimensions.
- `Cov_fct` et `C_Mtx` : calcul des fonctions et matrices de covariance pour le Processus Gaussien.
- `log_likelihood` : évaluation de la vraisemblance marginale pour l'optimisation des hyperparamètres.
- `nested_mf_covariance` : calcul de la covariance spécifique au modèle multi-fidélité imbriqué.
- `run_nested_mf_ego` : boucle principale de l'algorithme EGO multi-fidélité.
- Dépend de : `numpy`, `scipy`, `matplotlib`.


## `studies/cfoil_intens_kulfan/`

Le dossier `studies/cfoil_intens_kulfan` contient un pipeline d'optimisation multifidélité pour la section C-foil d'Intens SY, documenté par `GUIDE_PIPELINE.md` et `RAPPORT.md`. Il s'appuie sur une bibliothèque partagée `study_lib.py` et un module de style `plotkit.py` pour exécuter des étapes séquentielles, de la vérification `step0_check_setup.py` à la génération de figures `step4_make_figures.py`, incluant la calibration des solveurs et l'optimisation bayésienne.

### `studies/cfoil_intens_kulfan/GUIDE_PIPELINE.md`
Guide pédagogique détaillant le pipeline d'optimisation multifidélité de la section du C-foil d'Intens SY.
- `CamberProjector` : élimine la contrainte d'égalité CL2d(0°) = 0,45 par projection sur δ.
- `MultifidelityModel` : implémente le processus gaussien récursif avec estimation MAP des hyperparamètres.
- `optimizer.py` : gère la boucle ask/tell EGO et la sélection du niveau de fidélité via la fonction de mérite.
- `config_cfoil.json` : centralise les paramètres physiques, géométriques et numériques de l'étude.
- Dépend de : `pipelines/bdtoolbox_foil` (geometry, solvers, planform), `mfego` (surrogate_models, optimizer).

### `studies/cfoil_intens_kulfan/RAPPORT.md`
Rapport de calibration du GP multifidélité et des solveurs NPLLT × NeuralFoil pour l'optimisation de la section C-foil d'Intens SY.
- Résumé : synthèse des résultats d'optimisation (gain −0,9 % à portance égale) et choix du MAP IG(3, 2) comme estimateur par défaut.
- Vérification du MAP : correction de l'intégration dans le pipeline et ajout de diagnostics pour remplacer le MLE.
- Pipeline 3D : réécriture des backends `NpLltBackend` et `AvlCoreBackend` pour corriger les erreurs de clés et intégrer la contrainte CL2d(0°).
- Calibration des solveurs : validation de NPLLT × NeuralFoil xxlarge (L2) comme niveau basse fidélité fiable face à AVL × XFOIL (L3).
- Dépend de : `Multifidelity-GP-Clean`, `bdFoil`, `NeuralFoil`, `aerosandbox`, `numba`.

### `studies/cfoil_intens_kulfan/plotkit.py`
Module de style commun pour les figures interactives Plotly de l'étude cfoil_intens_kulfan.
- `INK` / `CATEGORICAL` / `SEQUENTIAL` : palettes de couleurs validées pour les modes clair/sombre et les niveaux de fidélité.
- `style` : applique la mise en forme standard (grille, légende, typographie) aux objets `go.Figure`.
- `save` : exporte les figures en fichiers HTML autonomes avec script de bascule automatique en mode sombre.
- `DARK_JS` : script JavaScript injecté pour remplacer les couleurs claires par leurs équivalents sombres.
- Dépend de : `plotly.graph_objects`, `json`, `pathlib`.

### `studies/cfoil_intens_kulfan/step0_check_setup.py`
Script de vérification préliminaire (STEP 0) qui valide l'environnement, la géométrie 3D, les profils et les solveurs avant les calculs intensifs.
- `check_environment`: Vérifie les versions logicielles et l'existence des exécutables XFOIL/AVL.
- `check_arc_against_bdfoil`: Compare l'arc généré aux références bdFoil et aux formules analytiques.
- `check_core_geometry`: Valide les valeurs de référence, la position des stations et la cohérence du fichier AVL.
- `check_section_geometry`: Vérifie l'épaisseur CST, l'élévation des poids Kulfan et la faisabilité des designs.
- `check_projection` / `check_solvers`: Teste la projection CL2d et l'évaluation des objectifs aux trois niveaux.
- Dépend de : `pipelines.bdtoolbox_foil` (bridge, geometry, constraints, planform), `study_lib`, `plotkit`, `aerosandbox`.

### `studies/cfoil_intens_kulfan/step1_calibrate_solvers.py`
Script de calibration multi-fidélité évaluant trois solveurs sur un design commun pour vérifier leur corrélation et coût.
- `evaluate_design`: évalue chaque point de design aux trois niveaux et collecte les métriques.
- `agreement`: calcule les corrélations (Pearson, Spearman, Kendall) et l'ajustement linéaire entre les niveaux.
- `costs`: mesure le temps d'exécution moyen par niveau et par étape de résolution.
- `noise_floor`: estime le bruit numérique via des perturbations infinitésimales du design.
- `mesh_convergence`: analyse la convergence de la grille NPLLT et AVL sur la section de référence.
- `main`: orchestre l'évaluation, le calcul des statistiques, le bruit et la génération des figures.
- Dépend de : `study_lib`, `plotkit`, `pipelines.bdtoolbox_foil` (constraints, geometry, solvers).

### `studies/cfoil_intens_kulfan/step2_validate_gp.py`
Étape 2 : validation et calibrage du processus gaussien multi-fidélité avant l'optimisation.
- `choose_estimator`: sélectionne MLE ou MAP via test de Wilcoxon, vérification grande-n et courbe d'apprentissage.
- `leave_one_out`: évalue la précision (RMSE, NLPD, couverture) sur les points L3 exclus.
- `large_n_check`: teste la robustesse des estimateurs sur 150 sections L1 supplémentaires.
- `learning_curve`: compare la précision en fonction du nombre de points L3 pour MLE et MAP.
- `main`: orchestre les tests, sauvegarde les résultats (`step2_choice.json`) et génère les figures.
- Dépend de : `study_lib`, `src.surrogate_models`, `src.kernels`, `plotkit`, `scipy.stats`.

### `studies/cfoil_intens_kulfan/step3_run_optimization.py`
Script de lancement de l'optimisation bayésienne multi-fidélité (NN-MF-EGO) pour la section C-foil, préparant les données des étapes 1 et 2.
- `apply_calibration`: injecte l'estimateur GP (étape 2) et les coûts relatifs des niveaux (étape 1) dans la configuration du problème.
- `holdout`: évalue la prédiction finale du substitut sur les sections L3 de l'étape 1 pour une vérification indépendante.
- `main`: orchestre l'exécution via `run_optimization`, gère les arguments de ligne de commande et sauvegarde les résultats.
- Dépend de : `pipelines.bdtoolbox_foil.run`, `src.surrogate_models`, `src.run_utils`, `study_lib`, `step2_validate_gp`, `results/step1_summary.json`, `results/step2_choice.json`.

### `studies/cfoil_intens_kulfan/step4_make_figures.py`
Étape 4 du pipeline d'optimisation : vérification des candidats prometteurs en L3, génération des figures de convergence, surfaces de réponse et tableau de bord final.
- `verify_candidates` : évalue à L3 les sections non testées (meilleur du surrogat, minima Sobol, meilleurs L1/L2) et enregistre `step4_l3_checks.json`.
- `figure_convergence` : trace la meilleure traînée L3 et l'ordre des évaluations par niveau de fidélité.
- `figure_slices_1d` / `figure_slices_2d` : visualise la surface de réponse (moyenne ± 2σ) le long des variables et par paires.
- `write_index` : génère le tableau de bord HTML `figures/index.html` avec les liens et les indicateurs clés.
- Dépend de : `study_lib`, `plotkit`, `pipelines.bdtoolbox_foil`, `src.surrogate_models`, `step0_check_setup`.

### `studies/cfoil_intens_kulfan/study_lib.py`
Bibliothèque partagée pour les scripts d'étude C-foil, fournissant la configuration, la gestion du simulateur, les échantillonnages faisables et la journalisation.
- `load_problem` : charge la configuration du problème de profil via `load_config`.
- `make_simulator` : instancie le `BdToolboxFoilSimulator` avec les backends multi-fidélité.
- `baseline_parameters` : dérive les paramètres de base en ajustant une section Kulfan sur le fichier XF de référence.
- `feasible_design` : génère un design d'expérimentation espace-remplissant et faisable par maximin.
- `setup_logging` : configure la journalisation vers la console et les fichiers de résultats.
- Dépend de : `pipelines.bdtoolbox_foil`, `src.data_management`, `src.optimizer`, `config_cfoil.json`.


## `studies/cfoil_intens_kulfan_trim/`

Ce dossier contient l'étude d'optimisation de la section C-foil d'Intens SY visant la réduction de la traînée à portance constante. Il comprend les scripts `step1_run_optimization.py` et `step2_analysis.py` pour l'optimisation bayésienne multi-fidélité et son analyse, ainsi que `study_trim.py` pour la configuration partagée. Le fichier `RAPPORT_TRIM.md` documente les résultats de cette étude spécifique.

### `studies/cfoil_intens_kulfan_trim/RAPPORT_TRIM.md`
Rapport d'optimisation de la section du C-foil d'Intens SY visant la réduction de la traînée à portance constante par ajustement du yaw.
- `config_cfoil_trim.json` : définit les paramètres de l'optimisation, les contraintes de portance et le trim.
- `step1_run_optimization.py` : exécute la mesure des coûts et la boucle d'optimisation multifidélité.
- `step2_analysis.py` : réalise la vérification AVL, l'analyse statistique et l'export des résultats.
- `study_trim.py` : fournit les outils communs de gestion des chemins et des niveaux de fidélité.
- Dépend de : `simulator.py`, `solvers.py`, `run.py`, `config.py`, `constraints.py`, `results/analysis.json`.

### `studies/cfoil_intens_kulfan_trim/step1_run_optimization.py`
Lance l'optimisation bayésienne multi-fidélité de la section de profil C-foil Intens SY en minimisant la traînée à portance exacte.
- `measure_costs`: Mesure les temps de calcul relatifs des niveaux de simulation via des sections aléatoires.
- `main`: Orchestre la configuration, la mesure des coûts et l'exécution de l'optimisation via `run_optimization`.
- `st.load_problem`: Charge la définition du problème depuis `config_cfoil_trim.json`.
- Dépend de : `study_trim`, `pipelines.bdtoolbox_foil.run`, `src.run_utils`

### `studies/cfoil_intens_kulfan_trim/step2_analysis.py`
Analyse de l'optimisation à la portance de la foil de référence en lisant les résultats de l'étape 1.
- `main`: orchestre la vérification AVL x XFOIL, l'analyse de la correction du premier ordre, la précision du substitut et la génération des figures.
- `evaluate_both`: évalue une section aux deux niveaux de fidélité avec le trim pour obtenir la traînée à la portance de référence.
- `collect_candidates`: identifie les sections prometteuses (optima du substitut, meilleures sections mesurées, variabilité inter-exécutions).
- `leave_one_out`: mesure la précision du substitut final sur les points AVL en conservant les points NPLLT.
- `run_statistics`: résume les statistiques du trim, les temps de calcul et la correction de la traînée de la zone racine.
- Dépend de : `study_trim`, `pipelines.bdtoolbox_foil`, `src.data_management`, `src.kernels`, `src.surrogate_models`.

### `studies/cfoil_intens_kulfan_trim/study_trim.py`
Outil partagé pour l'étude de trim (second test C-foil), gérant les chemins, la configuration, les niveaux et le chargement des exécutions d'optimisation.
- `load_problem` : retourne l'instance `FoilProblem` spécifique à cette étude.
- `setup_logging` : configure la journalisation console et fichier dans le répertoire `results`.
- `load_run` : charge l'état, les coûts cumulés, les métriques et le substitut entraîné d'une exécution.
- `LEVEL_NAME`, `LEVEL_COLOR` : définissent les noms, couleurs et symboles des niveaux de solvants.
- Dépend de : `plotkit`, `study_lib`, `src.surrogate_models`, `config_cfoil_trim.json`.


## `studies/cfoil_intens_kulfan_trim_2/`

Le dossier `studies/cfoil_intens_kulfan_trim_2` contient une étude d'optimisation multi-fidélité de la section Kulfan d'un C-foil, incluant les rapports `RAPPORT_NPLLT_AVL_XFOIL.md` et `RAPPORT_TRIM.md`. Les scripts `step0_check_setup.py`, `step1_run_optimization.py` et `step2_analysis.py` orchestrent l'initialisation, l'optimisation bayésienne de la traînée et l'analyse des résultats. Le module `study_trim.py` définit la configuration spécifique de cette étude de trim.

### `studies/cfoil_intens_kulfan_trim_2/RAPPORT_NPLLT_AVL_XFOIL.md`
Rapport d'évaluation de la fiabilité de mfego et de la concordance NPLLT × XFOIL / AVL × XFOIL pour l'optimisation aérodynamique.
- mfego : optimiseur multifidélité qui guide la recherche mais dont l'erreur de prédiction (~0,33 % de CD) dépasse le gain visé (0,19 %).
- NPLLT × XFOIL : modèle de travail recommandé pour la portance visqueuse et le trim, offrant une rapidité supérieure à AVL.
- AVL × XFOIL : outil de contre-vérification des effets 3D, inadapté à la prédiction précise du yaw de trim.
- Dépend de : étude `cfoil_intens_kulfan_trim_2`, sections Kulfan, polaires XFOIL, NeuralFoil xxxlarge.

### `studies/cfoil_intens_kulfan_trim_2/RAPPORT_TRIM.md`
Rapport technique détaillant l'optimisation multi-fidélité de la section Kulfan d'un C-foil sous contraintes de réalisabilité et de portance.
- `study_trim.YawModel` : régresseur quadratique résolvant la cambrure moyenne δ pour atteindre un yaw cible.
- `YawProjector` : projette les sections proposées dans l'espace faisable en ajustant δ via le modèle.
- `config_cfoil_trim.json` : définit la géométrie fixe, la marge de yaw [3,8 ; 4,2]° et les paramètres d'écoulement.
- `step0_check_setup.py` : ajuste le modèle du yaw et valide la robustesse des polaires XFOIL.
- Dépend de : `NeuralFoil`, `XFOIL`, `AVL`, `bdToolbox`, `pipelines/bdtoolbox_foil/solvers.py`.

### `studies/cfoil_intens_kulfan_trim_2/step0_check_setup.py`
Script d’initialisation de l’optimisation : ajuste le modèle de lacet de niveau 1 et vérifie la configuration sans optimisation.
- fit_yaw_model : ajuste le modèle de lacet sur des trims de niveau 1 et estime les erreurs par validation croisée.
- check_reference : valide la section de référence contre aerosandbox et résout le décalage de cambrure via le modèle de lacet.
- feasible_box : calcule la boîte des points faisables avec une marge pour proposer de nouvelles bornes.
- xfoil_robustness : évalue la robustesse des polaires XFOIL sur plusieurs configurations de paramètres.
- main : orchestre les vérifications, génère les figures et sauvegarde les résultats dans step0_checks.json.
- Dépend de : study_trim, pipelines.bdtoolbox_foil, src.data_management, aerosandbox.

### `studies/cfoil_intens_kulfan_trim_2/step1_run_optimization.py`
Script d'optimisation bayésienne multi-fidélité pour minimiser la traînée d'une section de profil C-foil à portance de référence.
- `measure_costs`: Mesure les temps d'évaluation par niveau avec un cache XFOIL temporaire vide pour estimer les coûts relatifs.
- `run_failures`: Analyse les échecs d'évaluation (XFOIL, trim) à partir du fichier `ego_backup.json`.
- `main`: Orchestre le processus, charge la configuration, exécute l'optimisation via `run_optimization` et sauvegarde les résultats.
- Dépend de : `study_trim`, `pipelines.bdtoolbox_foil.run`, `src.run_utils`.

### `studies/cfoil_intens_kulfan_trim_2/step2_analysis.py`
Évalue les sections prometteuses d'une étude d'optimisation à trois niveaux de fidélité, vérifie leur faisabilité et génère des statistiques et visualisations.
- `main`: orchestre la lecture des résultats, l'évaluation multi-niveaux, la collecte des candidats et l'export des figures.
- `collect_candidates`: identifie les sections optimales du substitut et les meilleures mesures dans la bande de tangage.
- `evaluate_levels`: calcule la traînée pénalisée des sections via les simulateurs NPLLT, XFOIL et AVL.
- `figure_comparison` : génère les graphiques comparatifs de performance et les tableaux de bord interactifs.
- Dépend de : `study_trim`, `pipelines.bdtoolbox_foil`, `src.data_management`, `src.kernels`, `src.surrogate_models`.

### `studies/cfoil_intens_kulfan_trim_2/study_trim.py`
Module de configuration et d'outils partagés pour l'étude de trim (troisième test C-foil), définissant la géométrie, les contraintes et la simulation.
- `YawModel` : modèle quadratique régressif du yaw trimmé (Niveau 1) permettant de résoudre l'offset de cambrure (`delta`) pour atteindre le yaw cible.
- `section_properties` : calcule les propriétés géométriques exactes (Ixx, Iyy, épaisseurs locales) à partir des poids Kulfan.
- `feasibility_function` : vérifie les contraintes connues (épaisseur min, rigidité, arrondi du bord d'attaque, confiance NeuralFoil) pour l'optimisation.
- `TrimBandSimulator` : simulateur multi-niveaux appliquant une pénalité sur le coût si le yaw trimmé sort de la bande admissible définie.
- Dépend de : `study_lib`, `plotkit`, `pipelines.bdtoolbox_foil`, `src.data_management`, `src.surrogate_models`.


## `tests/`

Le dossier `tests` contient les tests unitaires du projet, orchestrés par `conftest.py`. Il valide les composants de `src` (acquisition, kernels, `GaussianProcess`, `MultifidelityModel`, `optimizer`) ainsi que les pipelines `bdtoolbox`. Les fichiers spécifiques incluent `test_cokriging.py`, `test_data_management.py` et `test_single_fidelity.py` pour vérifier la cohérence du framework mono-fidélité.

### `tests/conftest.py`
Fournit des fixtures partagées et des utilitaires pour les tests unitaires du projet.
- `ForresterSimulator` : implémente le simulateur Forrester (fidélités basse et haute) pour les tests.
- `forrester_data` / `single_fidelity_data` : fixtures générant des données expérimentales initiales à deux ou un niveau.
- `make_optimizer` : fabrique une instance complète de `EGOOptimizer` avec modèle, acquisition et simulateur.
- `in_tmp` : fixture changeant le répertoire de travail pour isoler les écritures de fichiers.
- Dépend de : `src.acquisition`, `src.data_management`, `src.kernels`, `src.optimizer`, `src.simulator`, `src.surrogate_models`.

### `tests/test_acquisition.py`
Tests unitaires de `src/acquisition.py` validant la fonction de mérite (équations 19-24).
- `FakeModel` : modèle factice renvoyant des prédictions prescrites pour isoler les formules de mérite.
- `acquisition_with` : fabrique une instance d'`AcquisitionFunction` avec des données et coûts spécifiques.
- `test_ei_matches_monte_carlo` : vérifie que la fonction de mérite réduit à l'EI en l'absence de bruit.
- `test_aei_factor_and_latent_variance_reduction` : valide le facteur AEI et la réduction de variance latente.
- `test_variance_reduction_matches_block_inverse_update` : compare la réduction de variance à la mise à jour par inverse de bloc.
- `test_r2_product_and_cost_ratio_three_levels` : teste le produit R2 et le ratio de coûts sur trois niveaux.
- `test_merit_vanishes_without_improvement` : confirme l'annulation du mérite sans amélioration possible.
- `test_batch_and_single_point_merits_agree` : assure la cohérence entre évaluations par lot et point par point.
- `test_effective_best_solution` : vérifie la détermination de `x_best` et `f_best` selon Huang et al.
- Dépend de : `src.acquisition`, `src.data_management`, `src.kernels`, `src.surrogate_models`, `numpy`, `scipy.stats`.

### `tests/test_benchmark_lib.py`
Tests unitaires de `benchmarks/bench_lib.py` pour la validation des fonctions de benchmark et des substituts multi-fidélité.
- `test_hartmann_problem_matches_the_example_functions`: vérifie la cohérence des fonctions Hartmann avec les exemples de référence.
- `test_doe_and_costs`: valide la génération de plans d'expériences et le calcul des coûts de fidélité.
- `test_accuracy_metrics`: teste les métriques de précision (RMSE, couverture) pour les substituts MFEGO et Sklearn.
- `test_mfego_optimization_loop_respects_the_budget`: assure le respect du budget d'optimisation et la monotonie des coûts.
- `test_nested_doe_and_cokriging_surrogate`: vérifie les plans d'expériences imbriqués et le comportement du substitut CoKriging.
- Dépend de : `benchmarks/bench_lib.py`, `example/hartmann_6d/Hartmann6d.py`, `conftest.py`

### `tests/test_cokriging.py`
Tests unitaires de la classe `CoKriging` et de la génération de plans imbriqués.
- `test_joint_covariance_is_eq_3_11` : vérifie la structure matricielle de la covariance conjointe.
- `test_equivalence_with_the_recursive_formula` : valide l'équivalence avec la formule récursive de la littérature.
- `test_to_dict_from_dict_round_trip` : teste la sérialisation et le déchargement de l'état JSON.
- `test_v3_defaults` : contrôle les hyperparamètres par défaut de la version 3 (noyau, priors).
- Dépend de : `src.cokriging`, `src.data_management`, `src.surrogate_models`, `src.kernels`.

### `tests/test_data_management.py`
Tests unitaires de `src/data_management.py` validant l'absence de dépendance à `traitlets`, la génération de designs initiaux bornés et reproductibles, ainsi que la gestion des observations invalides.
- `test_module_does_not_depend_on_traitlets`: Vérifie par analyse AST que le module n'importe pas `traitlets`.
- `test_initial_design_in_bounds_and_reproducible`: Assure que les points générés respectent les bornes et la graine par défaut.
- `test_add_observation_and_failed_points`: Confirme l'exclusion des valeurs NaN/inf des données d'entraînement et le suivi des échecs.
- `test_best_observation_without_valid_point`: Valide le retour de `None` et NaN si aucune observation valide n'existe.
- Dépend de : `src.data_management`, `numpy`, `scipy.stats.qmc`, `ast`, `pathlib`

### `tests/test_export_visualization.py`
Tests unitaires de l'export du modèle substitut et de la visualisation.
- test_export_and_load_surrogate : Vérifie la cohérence des prédictions entre le modèle en mémoire et celui rechargé depuis "surrogate.json".
- test_visualizer_reads_old_json_files : Assure la rétrocompatibilité avec les fichiers JSON anciens manquant des clés spécifiques.
- test_grid_follows_the_bounds : Valide que la grille de tracé respecte les bornes du problème plutôt que [0, 1].
- test_scatter_uses_the_columns_of_the_points : Confirme que les points de dispersion utilisent les colonnes correctes des données.
- Dépend de : src/visualization, src/surrogate_models, src/optimizer, src/data_management, src/kernels

### `tests/test_gaussian_process.py`
Tests unitaires de la classe `GaussianProcess` dans `src/surrogate_models`, couvrant les vraisemblances, les a priori, le profilage de `rho` et les gradients analytiques.
- `test_nll_matches_multivariate_normal` : vérifie que la log-vraisemblance négative correspond au calcul par `scipy.stats.multivariate_normal`.
- `test_neg_log_prior_matches_scipy` : compare les valeurs et dérivées des a priori (`invgamma`, `gamma`, `lognormal`) avec `scipy.stats`.
- `test_rho_prior_closed_form_is_the_minimum` : valide que le profilage de `rho` minimise l'objectif avec un a priori normal.
- `test_analytical_gradient_matches_finite_differences` : confirme l'exactitude du gradient analytique via `scipy.optimize.check_grad`.
- Dépend de : `src.surrogate_models`, `src.kernels`, `conftest`

### `tests/test_kernels.py`
Tests unitaires de la module `src/kernels.py` validant les noyaux exponentiel carré et Matérn 5/2.
- `test_vectorized_matrix_equals_reference_loop` : Vérifie que la matrice vectorisée est identique à la boucle de référence `cov_fct`.
- `test_matrix_is_symmetric_positive_semidefinite` : Assure la symétrie et la semi-définition positive de la matrice de covariance.
- `test_log_gradients_match_finite_differences` : Compare les gradients log-paramétriques aux différences finies centrales.
- `test_matern52_matrix_and_cross_covariance` : Valide la formule du noyau Matérn 5/2 contre une référence analytique.
- `test_default_batch_cross_covariance_of_base_class` : Confirme le fonctionnement par lot du noyau de base implémentant uniquement le vecteur.
- Dépend de : `src.kernels`, `numpy`, `pytest`

### `tests/test_multifidelity.py`
Tests unitaires de `src/surrogate_models.MultifidelityModel` (formulation non imbriquée).
- `fitted_model` : construit et ajuste un modèle `MultifidelityModel` avec des noyaux spécifiés.
- `test_rho_is_estimated_on_forrester` : vérifie que le paramètre `rho` estimé correspond à la valeur théorique (2.0).
- `test_recursive_prediction_formula` : valide les équations de prédiction récursive (moyenne et variance) pour chaque niveau de fidélité.
- `test_to_dict_from_dict_round_trip` : assure la cohérence de la sérialisation et de la désérialisation de l'état du modèle.
- `test_learned_rho_model_round_trip` : teste le chargement et la sauvegarde des priors sur `rho` (lognormale) et la stabilité des prédictions.
- Dépend de : `src.surrogate_models`, `src.kernels`, `src.data_management`, `conftest`

### `tests/test_optimizer.py`
Tests unitaires de `src/optimizer.py` couvrant la boucle EGO, le mode ask/tell et la persistance JSON.
- `test_doe_cost_is_counted`: Vérifie que le coût du DOE initial est correctement comptabilisé.
- `test_ask_tell_mode_counts_the_doe`: Valide la séquence ask/tell et le suivi des coûts.
- `test_state_is_consistent_after_run`: Assure la cohérence du modèle sauvegardé avec l'état en mémoire.
- `test_save_and_load_state`: Teste la restauration complète de l'état depuis le fichier JSON.
- `test_seeded_runs_are_reproducible`: Garantit la reproductibilité des exécutions avec une graine fixe.
- `test_failed_simulations_do_not_break_the_run`: Vérifie la robustesse face aux simulations échouées (NaN).
- Dépend de : `src.data_management.ExperimentData`, `src.visualization.ModelVisualizer`, `conftest`.

### `tests/test_pipeline_bdtoolbox.py`
Tests unitaires du pipeline de profil d'aile mfego <-> bdToolbox (pipelines/bdtoolbox_foil).
- `base_config` : construit la configuration de référence pour les tests.
- `FakeBackend` : simulateur analytique de polaires avec gestion des échecs.
- `test_shipped_configurations_are_valid` : valide les fichiers JSON de configuration livrés.
- `test_objectives_on_an_analytical_polar` : vérifie le calcul des objectifs et les erreurs de convergence.
- `test_end_to_end_run_with_fake_backends` : exécute le pipeline complet (DOE, EGO, export) avec des backends factices.
- `test_npllt_xfoil_polars_share_the_avl_spec_and_cache` : assure la compatibilité des spécifications entre backends NPLLT et AVL.
- Dépend de : `pipelines.bdtoolbox_foil` (bridge, geometry, objectives, config, run, simulator, solvers), `src.run_utils`.

### `tests/test_pipeline_options.py`
Unit tests verifying the Gaussian Process options (kernel, priors, rho centering) in the foil pipeline configuration.
- `config`: Construct the base dictionary for a 2D section optimization problem with NACA4 parametrization.
- `AnalyticalBackend`: Fake solver backend generating analytical polars to simulate solver responses.
- `test_defaults_of_the_new_options`: Assert that default configuration values match mfego v1 behavior.
- `test_invalid_options_are_rejected`: Verify that invalid option overrides raise a `ValueError`.
- `test_options_reach_the_model`: Ensure configuration options correctly propagate to the GP model and surrogate state.
- Dépend de : `pipelines.bdtoolbox_foil`, `src.run_utils`

### `tests/test_run_utils.py`
Tests unitaires de `src/run_utils.py` couvrant la création de répertoires horodatés, la gestion des journaux et la mesure du temps de calcul.
- `test_run_directory_and_log_name`: valide la convention de nommage bdFoil pour les répertoires `runs` et les fichiers de log.
- `test_two_runs_in_the_same_second_do_not_collide`: vérifie l'unicité des identifiants lors de créations simultanées.
- `test_log_contains_the_total_computation_time`: confirme l'écriture du temps total et du support UTF-8 dans les logs.
- `test_total_time_is_written_when_the_run_fails`: assure la persistance du temps de calcul même en cas d'exception.
- `test_format_duration`: teste la fonction `format_duration` pour la conversion des durées.
- Dépend de : `src.run_utils` (RunTimer, create_run, format_duration, setup_logging)

### `tests/test_simulators.py`
Tests unitaires des simulateurs d'exemple (Hartmann 6D, hydrofoil).
- `test_hartmann_standard_minimum`: Vérifie la valeur minimale de la fonction de Hartmann.
- `test_hartmann_multifidelity_sequence`: Valide la séquence de fidélité multi-niveaux.
- `test_hydrofoil_level_to_model_mapping`: Assure le mapping correct des niveaux vers les modèles NeuralFoil.
- `test_hydrofoil_failure_returns_nan`: Confirme le retour de NaN en cas d'échec de calcul.
- Dépend de : `example/hartmann_6d/Hartmann6d.py`, `example/hydrofoil_optim/optim_neuralfoil.py`, `neuralfoil`, `aerosandbox`.

### `tests/test_single_fidelity.py`
Vérifie la cohérence du framework mono-fidélité (L=1) incluant modèle, mérite, boucle EGO, export JSON et visualisation.
- test_single_level_model_is_a_plain_gp: Valide que le modèle mono-niveau équivaut à un processus gaussien simple.
- test_single_level_merit_is_the_aei: Confirme que la fonction de mérite correspond à l'AEI ajustée.
- test_single_fidelity_end_to_end: Teste le cycle complet d'optimisation, export, rechargement et génération de graphiques.
- test_single_fidelity_2d_plots: Vérifie les surfaces de réponse 2D statiques et interactives.
- Dépend de : src.acquisition, src.data_management, src.kernels, src.surrogate_models, src.visualization, conftest.

