# Les runs de vpp_acceleration

Un dossier par run, pour ne jamais mélanger les tests, les données, les résultats et les figures (règle du
22/10/2026, outil [`src/run_dir.py`](../src/run_dir.py)) :

```
runs/<AAAA-MM-JJ_HHMM>_<nom>/
    config.json      tous les réglages du run, écrits au départ
    run.log          progression et temps
    data/            evaluations.csv (un calcul de code par ligne : niveau, entrées, sorties, temps, statut),
                     pca.json, codes/ (configurations bdFoil, cache des polaires XFOIL)
    results/         tableaux et JSON du run (history.csv, updates.json, validation.csv, best_section.json,
                     summary.txt…)
    figures/         figures HTML
    README.md        lecture du run
```

Les dossiers datés du 21/10 regroupent les sorties des études faites avant cette règle, rangées après coup avec
leurs noms de fichiers d'origine. Les scripts qui les produisent écrivent désormais directement dedans.

| Dossier | Contenu | Scripts | Document |
| --- | --- | --- | --- |
| `2026-10-21_cokriging_benchmarks` | co-krigeage séquentiel : Forrester, Hartmann et profil, test région, bibliothèques (emukit, BoTorch), ρ(x), estimation conjointe, partage des longueurs | `examples/{forrester,benchmark,region,library,rho,joint,share}_study.py` | [`../README.md`](../README.md) § 0 |
| `2026-10-21_section_pca` | PCA de la section : famille C-foil à 8 paramètres, famille à 16 poids lisses (k = 12) | `examples/section_pca_study.py` | [`../README.md`](../README.md) § 0 |
| `2026-10-21_meta_vpp_cible` | méta-surrogate 16D : vérité VPP, N_sobol, plan LHS, ciblé contre ALC global et LHS, validation | `examples/vpp_*_study.py` | [`../RAPPORT_VPP_CIBLE.md`](../RAPPORT_VPP_CIBLE.md) |
| `2026-10-22_1003_step0_calibration` | 1er recalage des codes contre la RS de production, avec le MC2 symétrique (raté, voir le suivant) | `scripts/real_step0_calibration.py --section mc2` | — |
| `2026-10-22_1006_step0_calibration_reference` | recalage avec le profil cambré de référence, Cpmin = minimum sur l'envergure : forces et moments bons, Cpmin faux | `scripts/real_step0_calibration.py` | [`../RAPPORT_VPP_CIBLE.md`](../RAPPORT_VPP_CIBLE.md) § 10 bis |
| `2026-10-22_1009_step0_calibration_reference` | idem avec le Cpmin de bdFoil (20e percentile) : **feu vert** pour les forces et les moments, sans conversion | `scripts/real_step0_calibration.py` | [`../RAPPORT_VPP_CIBLE.md`](../RAPPORT_VPP_CIBLE.md) § 10 bis |
| `2026-10-22_1016_smoke` | test court de la chaîne réelle (4,8 min) | `scripts/run_real_vpp.py --smoke` | — |
| `2026-10-22_1021_real_1h` | **premier vrai run** (44 min) : NPLLT × NeuralFoil / AVL × XFOIL, section Dim-Reduction, VPP MDS | `scripts/run_real_vpp.py` | [`README.md`](2026-10-22_1021_real_1h/README.md) du dossier |
| `2026-10-22_1126_smoke_current_box` | test court du 2e run : boîte autour du profil actuel, Ixx ≥ Ixx actuel | `scripts/run_real_vpp.py --smoke --box current --box-scale 0.5` | — |
| `2026-10-22_1131_real_1h30_current_box` | **2e vrai run** (82 min, gain +0,18 nd sur le profil actuel) : boîte de demi-largeur Dim-Reduction centrée sur le profil actuel, Ixx ≥ Ixx actuel, départ = profil actuel | `scripts/run_real_vpp.py --box current --box-scale 0.5 --time-limit-min 80` | [`README.md`](2026-10-22_1131_real_1h30_current_box/README.md) du dossier |
| `2026-10-22_1507_smoke_npllt_xfoil_fsi` | test court du 3e run : HF = NPLLT × XFOIL, foil flexible, lots parallèles ; a révélé des sensibilités fausses (solves de base et décalés dans des processus différents), corrigé | `scripts/run_real_vpp.py --smoke --hf npllt_xfoil --fsi --batch 16 …` | — |
| `2026-10-22_1514_smoke2_npllt_xfoil_fsi` | idem après correction : sensibilités stables, 144 choix en 3 min | idem | — |
| `2026-10-23_weekend` | week-end du 23 au 25/10 : l'enchaînement (orchestrateur) du run long, puis l'étude de maillage CFD (6 cas, OpenFOAM v2512). Traînée convergée en partie par compensation, portance non convergée (couches de prismes à 72-78 % de couverture), domaine trop petit (−0,04 nd) | `orchestrate.ps1`, `cfd_mesh_study.sh`, `cfd_mesh_report.py` | [`README.md`](2026-10-23_weekend/README.md), [`cfd/report.md`](2026-10-23_weekend/cfd/report.md) |
| `2026-10-23_1702_real_search_profile_local` | **4e vrai run** (8 h 10) : recherche sur la distribution supposée (18 nœuds), recherche locale de la section. Nouveau meilleur profil, encore moins cambré, Ixx à la contrainte | `scripts/run_real_vpp.py … --search-profile profiles/guess_bimodal.json --local-fraction 0.5 …` (voir `orchestrate.ps1`) | [`README.md`](2026-10-23_1702_real_search_profile_local/README.md) du dossier |
| `2026-10-27_1219_smoke_restarts` | test court des changements du 27/10 : convergence d'une région et redémarrage, lots par prolongement de Cholesky, médiane des sensibilités, premier solve à départs multiples | `scripts/run_real_vpp.py --smoke … --converge 2 0.5 --restarts 1 …` | [`../CHANGEMENTS_2026-10-27.md`](../CHANGEMENTS_2026-10-27.md) |
| `2026-10-27_1224_control_restarts` | **5e vrai run** (2 h), contrôle des changements du 27/10 avec les réglages et la graine du 4e : même résultat (−0,005 nd) en 2 h au lieu de 8 h ; région 0 convergée à 92 min, redémarrage revenu vers le même coin de la famille | `scripts/run_real_vpp.py … --converge 10 0.01 --restarts 2 --min-gain 0.002 --sensitivity-median 3` | [`README.md`](2026-10-27_1224_control_restarts/README.md) du dossier |
| `2026-10-27_1636_smoke_bdfoil_v1` | premier test court sur **bdfoil_v1** (`pipelines/CHANGEMENTS_2026-10-27_bdfoil_v1.md`) : chaîne complète sans erreur, mais 16 BF sur 66 en échec à cause de la coupure de confiance NeuralFoil de bdfoil_v1 (0,90), d'où l'option `neuralfoil_min_confidence = 0` du niveau BF ; ce dossier a tourné avec la coupure | `scripts/run_real_vpp.py --smoke … --hf npllt_xfoil --fsi --batch 4 --workers 4 --vpp-workers 4` | — |
| `2026-10-29_0943_smoke_profile_v2_box3` | test court du 8e run : `profile_v2` (49 conditions), boîte à l'échelle 3 ; aucun échec de code | `scripts/run_real_vpp.py --smoke --box current --box-scale 3 --fsi --batch 4 --workers 6 --vpp-workers 6 --search-profile profiles/profile_v2.json …` | — |
| `2026-10-29_1002_profile_v2_box3_3h` | **8e vrai run** (2 h 49, toute la machine) : profil `profile_v2` (TWS [5, 17], points sur les bords), boîte à l'échelle 3, donc seule la contrainte Ixx retient la section. +0,256 nd sur la section de départ contre +0,235 pour le meilleur du 7e run ; boîte plus active (échelle 2,13 nécessaire) ; cambrure encore plus basse et reculée ; **extrado plus convexe** (à vérifier) ; gagne au travers, perd un peu au près serré par vent fort | `scripts/run_real_vpp.py --box current --box-scale 3 --hf avl --fsi --batch 16 --workers 28 --vpp-workers 28 --time-limit-min 140 --search-profile profiles/profile_v2.json …` | [`README.md`](2026-10-29_1002_profile_v2_box3_3h/README.md) du dossier |
| `2026-10-28_1747_profile_v1_box075` | **7e vrai run** (8 h, machine partagée avec la CFD) : 2 niveaux (NPLLT × NF / AVL × XFOIL itéré), profil `profile_v1` (bosses 40° et 80°, 39 conditions), boîte élargie à 0,75. **Nouveau bassin**, encore moins cambré : +0,208 nd sur la section actuelle contre +0,156 pour le meilleur du 6e run ; la section est au bord de la boîte (active) et sur la contrainte Ixx | `scripts/run_real_vpp.py --box current --box-scale 0.75 --hf avl --fsi --batch 16 --n-low 256 --n-high 64 --search-profile profiles/profile_v1.json …` | [`README.md`](2026-10-28_1747_profile_v1_box075/README.md) du dossier |
| `2026-10-27_1739_long_3levels_avl_iter` | **6e vrai run** (6 h 20) : 3 niveaux (NPLLT × NF / NPLLT × XFOIL / AVL × XFOIL itéré J8) sur bdfoil_v1. **Même section** que les runs 4 et 5 (13,088 nd contre 13,091 et 13,083, distance PCA 0,95-1,30), gain +0,16 nd ; le niveau 2 coûte presque autant qu'AVL sans être plus précis | `scripts/run_real_vpp.py --levels nf_llt xfoil_llt avl --n-design 256 64 32 --costs 1 2 2.5 …` | [`README.md`](2026-10-27_1739_long_3levels_avl_iter/README.md) du dossier |
| `2026-10-23_1643_smoke_search_profile_local` | test court des changements du 23/10 : recherche pondérée sur la distribution supposée (8 nœuds), recherche locale en région de confiance, graine de criblage propre, candidats proches seulement | `scripts/run_real_vpp.py --smoke … --search-profile <8 nœuds> --local-fraction 0.5` | — |
| `2026-10-22_1740_smoke_profile` | test court de l'option `--profile` (analyse sur la distribution d'usage en fin de run) et de la sauvegarde `results/surrogate.pkl` | `scripts/run_real_vpp.py --smoke … --profile profiles/guess_bimodal.json` | — |
| `2026-10-22_1519_real_npllt_xfoil_fsi` | **3e vrai run** : HF = NPLLT × XFOIL (AVL × XFOIL en validation), foil flexible (correction de la RS FSI de production), lots de 16 en parallèle (28 + 30 processus), même boîte que le 2e run | `scripts/run_real_vpp.py --box current --box-scale 0.5 --hf npllt_xfoil --hf-cost 2 --fsi --batch 16 --workers 28 --vpp-workers 30 --n-low 256 --n-high 64 --time-limit-min 65 --compare …` | [`README.md`](2026-10-22_1519_real_npllt_xfoil_fsi/README.md) du dossier : même profil que le 2e run, gain +0,24 nd (NPLLT × XFOIL) / +0,22 nd (AVL), Cpmin −2,57 → −0,87 |

Lancer un nouveau run (environnement MDS, **vérifier la charge de la machine avant**) :

```powershell
cd Multifidelity-GP-Clean\vpp_acceleration
$env:PYTHONPATH = "scripts"
..\.venv-mds\Scripts\python scripts\run_real_vpp.py --name <nom> --time-limit-min 42 --workers 6
```
