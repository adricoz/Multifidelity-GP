# PCA sur le domaine convexe des sections (2026-11-04)

97401 sections tirées uniformément dans le domaine (hit-and-run depuis la section actuelle), src/section_domain.py. Résumé complet : results/summary.json.

| mode | variance % | borne − | borne + | contrainte − | contrainte + |
|---|---|---|---|---|---|
| 1 | 55.72 | -3.25 | +1.49 | leading_edge | leading_edge |
| 2 | 24.66 | -2.37 | +3.12 | kulfan_box | leading_edge |
| 3 | 8.49 | -1.08 | +1.73 | leading_edge | upper_curvature |
| 4 | 5.97 | -2.39 | -0.41 | leading_edge | lower_curvature |
| 5 | 2.08 | -1.26 | -0.22 | lower_curvature | leading_edge |
| 6 | 1.75 | -2.59 | +1.16 | upper_curvature | lower_curvature |
| 7 | 0.56 | -1.73 | +1.26 | leading_edge | upper_curvature |
| 8 | 0.34 | +0.02 | +1.21 | lower_curvature | lower_curvature |
| 9 | 0.18 | -1.65 | +0.57 | upper_curvature | upper_curvature |
| 10 | 0.14 | -0.53 | +1.14 | lower_curvature | lower_curvature |
| 11 | 0.05 | -1.58 | +0.21 | upper_curvature | upper_curvature |
| 12 | 0.04 | -1.47 | -0.26 | lower_curvature | lower_curvature |
| 13 | 0.01 | -2.17 | +0.14 | upper_curvature | upper_curvature |
| 14 | 0.01 | -1.17 | +0.92 | lower_curvature | lower_curvature |
| 15 | 0.00 | -2.07 | +0.54 | upper_curvature | upper_curvature |
| 16 | 0.00 | -3.02 | +1.98 | kulfan_box | lower_curvature |

Segments entre sections admissibles de la boîte qui sortent du domaine : 0.0 % ({'kulfan_box': 0.0, 'crossing': 0.0, 'upper_curvature': 0.0, 'lower_curvature': 0.0, 'leading_edge': 0.0, 'ixx_v': 0.0, 'ixx': 0.0}).

## Validation (scripts/validate_section_pca.py)

Domaine : Ixx >= 0 x actuelle (0.0000e+00) et Ixx/v >= 0.95 x actuelle (1.0419e-03), plus les contraintes linéaires du 04/11. Résultat global : **tous les contrôles OK**.

| | contrôle | valeur | seuil |
|---|---|---|---|
| OK | section actuelle dans le domaine | True | vrai |
| OK | decode(encode(actuelle)) exact (écart max des poids) | 3.191891195797325e-16 | < 1e-10 |
| OK | 4000 sections de la box : contraintes violées | {'kulfan_box': 0, 'crossing': 0, 'upper_curvature': 0, 'lower_curvature': 0, 'leading_edge': 0, 'ixx_v': 0, 'ixx': 0} | 0 partout |
| OK | variance gardée par k = 10 | 99.88969108029937 | >= 99.8 % |
| OK | fraction admissible de la box (uniforme) | 1.717 | >= 0.3 % (ancienne : 1.72 %) |

Sections de la box : Ixx / actuelle p5/p50/p95 [1.045, 1.777, 3.118], Ixx/v / actuelle [1.011, 1.425, 2.039]; à moins de 1 % de la limite : Ixx 0.0 %, Ixx/v 0.7 %.
Contre la PCA 2026-11-05_1212_section_pca_convex_tmax2_0p9 : angles principaux entre les sous-espaces à 10 modes [0.0, 0.0, 0.0, 0.0, 0.6, 1.2, 1.8, 2.5, 3.9, 4.8] deg ; erreur de reconstruction des sections de la nouvelle box dans l'ancienne base p50/p95/max [5e-05, 0.0001, 0.00017] (corde 1 ; nouvelle base : 1.7e-16) ; 0 % des sections de l'ancienne box violent Ixx >= 0.

| meilleure section | Ixx / actuelle | Ixx/v / actuelle | dans le nouvel espace |
|---|---|---|---|
| convexe 05/11 (FSI) (2026-11-05_1259_convex_envelope_k10_long) | 0.833 | 0.901 | False |
| xtr 5 % (2026-11-09_1057_xtr005_rigid_convex_k10) | 0.856 | 0.902 | False |
| xtr 3 % (2026-11-09_1411_xtr003_rigid_convex_k10) | 0.862 | 0.903 | False |
| xtr 10 % depuis 5 % (2026-11-09_1325_xtr010_from5_rigid_convex_k10) | 0.899 | 0.900 | False |
| run 8 (2026-10-29_1002_profile_v2_box3_3h) | 1.004 | 0.982 | False |

Figures : figures/validation.html (variance, Ixx et Ixx/v, box par mode, sections), figures/pca_explorer.html (explorateur interactif).
