# PCA sur le domaine convexe des sections (2026-11-04)

97425 sections tirées uniformément dans le domaine (hit-and-run depuis la section actuelle), src/section_domain.py. Résumé complet : results/summary.json.

| mode | variance % | borne − | borne + | contrainte − | contrainte + |
|---|---|---|---|---|---|
| 1 | 56.23 | -3.19 | +1.43 | leading_edge | leading_edge |
| 2 | 23.74 | -2.44 | +3.14 | kulfan_box | leading_edge |
| 3 | 8.86 | -1.10 | +1.68 | leading_edge | upper_curvature |
| 4 | 5.95 | -2.44 | -0.53 | leading_edge | lower_curvature |
| 5 | 2.11 | -1.08 | -0.07 | lower_curvature | leading_edge |
| 6 | 1.76 | -2.55 | +1.21 | upper_curvature | upper_curvature |
| 7 | 0.57 | -1.74 | +1.27 | leading_edge | upper_curvature |
| 8 | 0.35 | +0.02 | +1.19 | lower_curvature | lower_curvature |
| 9 | 0.18 | -1.79 | +0.54 | upper_curvature | upper_curvature |
| 10 | 0.14 | -0.55 | +1.07 | lower_curvature | lower_curvature |
| 11 | 0.05 | -1.62 | +0.17 | upper_curvature | upper_curvature |
| 12 | 0.04 | -1.37 | -0.19 | lower_curvature | lower_curvature |
| 13 | 0.01 | -2.05 | +0.22 | upper_curvature | upper_curvature |
| 14 | 0.01 | -1.01 | +1.07 | lower_curvature | lower_curvature |
| 15 | 0.00 | -1.98 | +0.55 | upper_curvature | upper_curvature |
| 16 | 0.00 | -3.05 | +2.04 | kulfan_box | lower_curvature |

Segments entre sections admissibles de la boîte qui sortent du domaine : 0.1 % ({'kulfan_box': 0.0, 'crossing': 0.0, 'upper_curvature': 0.0, 'lower_curvature': 0.0, 'leading_edge': 0.0, 'ixx_v': 0.0, 'ixx': 0.15}).

## Validation (scripts/validate_section_pca.py)

Domaine : Ixx >= 0.95 x actuelle (6.3816e-05) et Ixx/v >= 0.9 x actuelle (9.8704e-04), plus les contraintes linéaires du 04/11. Résultat global : **tous les contrôles OK**.

| | contrôle | valeur | seuil |
|---|---|---|---|
| OK | section actuelle dans le domaine | True | vrai |
| OK | decode(encode(actuelle)) exact (écart max des poids) | 3.885780586188048e-16 | < 1e-10 |
| OK | 4000 sections de la box : contraintes violées | {'kulfan_box': 0, 'crossing': 0, 'upper_curvature': 0, 'lower_curvature': 0, 'leading_edge': 0, 'ixx_v': 0, 'ixx': 0} | 0 partout |
| OK | variance gardée par k = 10 | 99.88931438995775 | >= 99.8 % |
| OK | fraction admissible de la box (uniforme) | 1.821 | >= 0.3 % (ancienne : 1.72 %) |

Sections de la box : Ixx / actuelle p5/p50/p95 [1.049, 1.793, 3.213], Ixx/v / actuelle [1.01, 1.436, 2.087]; à moins de 1 % de la limite : Ixx 0.5 %, Ixx/v 0.1 %.
Contre la PCA 2026-11-05_1212_section_pca_convex_tmax2_0p9 : angles principaux entre les sous-espaces à 10 modes [0.0, 0.0, 0.0, 0.0, 0.5, 0.7, 1.5, 2.6, 4.3, 7.7] deg ; erreur de reconstruction des sections de la nouvelle box dans l'ancienne base p50/p95/max [5e-05, 0.0001, 0.00013] (corde 1 ; nouvelle base : 1.4e-16) ; 3 % des sections de l'ancienne box violent Ixx >= 0.95.

| meilleure section | Ixx / actuelle | Ixx/v / actuelle | dans le nouvel espace |
|---|---|---|---|
| convexe 05/11 (FSI) (2026-11-05_1259_convex_envelope_k10_long) | 0.833 | 0.901 | False |
| xtr 5 % (2026-11-09_1057_xtr005_rigid_convex_k10) | 0.856 | 0.902 | False |
| xtr 3 % (2026-11-09_1411_xtr003_rigid_convex_k10) | 0.862 | 0.903 | False |
| xtr 10 % depuis 5 % (2026-11-09_1325_xtr010_from5_rigid_convex_k10) | 0.899 | 0.900 | False |
| run 8 (2026-10-29_1002_profile_v2_box3_3h) | 1.004 | 0.982 | False |

Figures : figures/validation.html (variance, Ixx et Ixx/v, box par mode, sections), figures/pca_explorer.html (explorateur interactif).
