# PCA sur le domaine convexe des sections (2026-11-04)

97377 sections tirées uniformément dans le domaine (hit-and-run depuis la section actuelle), src/section_domain.py. Résumé complet : results/summary.json.

| mode | variance % | borne − | borne + | contrainte − | contrainte + |
|---|---|---|---|---|---|
| 1 | 56.06 | -3.10 | +1.64 | leading_edge | upper_curvature |
| 2 | 24.30 | -2.44 | +3.03 | kulfan_box | leading_edge |
| 3 | 8.74 | -1.03 | +1.73 | leading_edge | upper_curvature |
| 4 | 5.75 | -2.40 | -0.39 | leading_edge | lower_curvature |
| 5 | 2.07 | -1.19 | -0.14 | lower_curvature | leading_edge |
| 6 | 1.73 | -2.50 | +1.01 | upper_curvature | upper_curvature |
| 7 | 0.57 | -1.67 | +1.28 | leading_edge | upper_curvature |
| 8 | 0.35 | -0.03 | +1.14 | lower_curvature | lower_curvature |
| 9 | 0.18 | -1.59 | +0.65 | upper_curvature | upper_curvature |
| 10 | 0.14 | -0.39 | +1.25 | lower_curvature | lower_curvature |
| 11 | 0.05 | -1.51 | +0.24 | upper_curvature | upper_curvature |
| 12 | 0.04 | -1.51 | -0.32 | lower_curvature | lower_curvature |
| 13 | 0.01 | -2.13 | +0.15 | upper_curvature | upper_curvature |
| 14 | 0.01 | -1.15 | +0.96 | lower_curvature | lower_curvature |
| 15 | 0.00 | -1.96 | +0.61 | upper_curvature | upper_curvature |
| 16 | 0.00 | -2.94 | +2.13 | kulfan_box | lower_curvature |

Segments entre sections admissibles de la boîte qui sortent du domaine : 0.1 % ({'kulfan_box': 0.0, 'crossing': 0.0, 'upper_curvature': 0.0, 'lower_curvature': 0.0, 'leading_edge': 0.0, 'ixx_v': 0.05}).
