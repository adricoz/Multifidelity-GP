# PCA sur le domaine convexe des sections (2026-11-04)

97488 sections tirées uniformément dans le domaine (hit-and-run depuis la section actuelle), src/section_domain.py. Résumé complet : results/summary.json.

| mode | variance % | borne − | borne + | contrainte − | contrainte + |
|---|---|---|---|---|---|
| 1 | 58.77 | -2.99 | +1.63 | leading_edge | upper_curvature |
| 2 | 22.54 | -2.37 | +3.03 | lower_curvature | leading_edge |
| 3 | 8.16 | -1.09 | +1.65 | leading_edge | upper_curvature |
| 4 | 5.62 | -2.44 | -0.50 | leading_edge | lower_curvature |
| 5 | 1.97 | -1.08 | -0.05 | lower_curvature | leading_edge |
| 6 | 1.66 | -2.61 | +1.11 | upper_curvature | upper_curvature |
| 7 | 0.53 | -1.77 | +1.24 | leading_edge | upper_curvature |
| 8 | 0.33 | -0.03 | +1.16 | lower_curvature | lower_curvature |
| 9 | 0.17 | -1.66 | +0.59 | upper_curvature | upper_curvature |
| 10 | 0.14 | -0.57 | +1.08 | lower_curvature | lower_curvature |
| 11 | 0.05 | -1.56 | +0.19 | upper_curvature | upper_curvature |
| 12 | 0.04 | -1.46 | -0.26 | lower_curvature | lower_curvature |
| 13 | 0.01 | -2.05 | +0.22 | upper_curvature | upper_curvature |
| 14 | 0.01 | -1.09 | +0.99 | lower_curvature | lower_curvature |
| 15 | 0.00 | -1.92 | +0.62 | upper_curvature | upper_curvature |
| 16 | 0.00 | -2.97 | +2.08 | kulfan_box | lower_curvature |

Segments entre sections admissibles de la boîte qui sortent du domaine : 0.1 % ({'kulfan_box': 0.0, 'crossing': 0.0, 'upper_curvature': 0.0, 'lower_curvature': 0.0, 'leading_edge': 0.0, 'ixx_v': 0.1}).
