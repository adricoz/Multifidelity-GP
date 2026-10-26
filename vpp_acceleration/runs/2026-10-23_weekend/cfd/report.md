# Étude de maillage CFD du C-foil (week-end du 23/10/2026)

Lecture automatique de `cfd/results.csv` par `cfd_mesh_report.py`. Plan de l'étude et justification : `README.md` du dossier.

## Les calculs

| case           |   cells_M |   layer_percent |   yplus_avg |   yplus_max |    Fx |         Fy |        Fz |         Mx |   Fx_drift_last200 |   mesh_s |   solve_s |
|:---------------|----------:|----------------:|------------:|------------:|------:|-----------:|----------:|-----------:|-------------------:|---------:|----------:|
| G2_medium_yp50 |     3.809 |            75.2 |       93.7  |        1864 | -2254 | -5.445e+04 | 2.941e+04 | -9.121e+04 |           1.13e-05 |      493 |      5851 |
| G1_coarse_yp50 |     2.229 |            77.9 |      107.5  |        2238 | -2269 | -5.394e+04 | 2.909e+04 | -8.989e+04 |           4.53e-06 |      437 |      3275 |
| G3_fine_yp50   |     7.224 |            72.3 |       86.41 |        1467 | -2250 | -5.501e+04 | 2.972e+04 | -9.22e+04  |           5.76e-06 |     1702 |     11604 |
| Y100_medium    |     3.663 |            78.9 |      121.5  |        1608 | -2245 | -5.456e+04 | 2.947e+04 | -9.141e+04 |           1.42e-05 |      869 |      5456 |
| Y30_medium     |     3.994 |            75.2 |       82.64 |        1911 | -2255 | -5.442e+04 | 2.94e+04  | -9.115e+04 |           3.82e-05 |      492 |      6379 |
| D2_coarse_yp50 |     2.406 |            79.4 |      105.1  |        2268 | -2297 | -5.416e+04 | 2.926e+04 | -9.077e+04 |           6.04e-07 |      302 |      3493 |

## Convergence en maillage (GCI, Celik et al. 2008)

Rapports de raffinement réels : 1.196 (G1 → G2) et 1.238 (G2 → G3), sous le minimum de 1.3 recommandé : le nombre de cellules croît moins vite que le cube du rapport de fond (surface et couches).

Le GCI n'est donné que si les écarts entre maillages diminuent nettement (R = ε21 / ε32 entre 0 et 0.9).

| quantity    |        coarse |        medium |          fine |   step_coarse_medium_% |   step_medium_fine_% |      R | verdict                               |   observed_order |   extrapolated |   GCI_fine_% |   GCI_medium_% |
|:------------|--------------:|--------------:|--------------:|-----------------------:|---------------------:|-------:|:--------------------------------------|-----------------:|---------------:|-------------:|---------------:|
| Fx          | -2269         | -2254         | -2250         |                -0.6407 |              -0.2078 | 0.3223 | converge                              |            6.815 |  -2248         |      0.07936 |          0.339 |
| Fx_pressure | -1291         | -1260         | -1244         |                -2.379  |              -1.265  | 0.5191 | converge                              |            4.306 |  -1234         |      1.063   |          2.631 |
| Fx_viscous  |  -977.9       |  -994.1       | -1005         |                 1.654  |               1.133  | 0.6959 | converge                              |            2.779 |  -1019         |      1.73    |          3.165 |
| Fy          |    -5.394e+04 |    -5.445e+04 |    -5.501e+04 |                 0.9476 |               1.041  | 1.109  | diverge : pas de GCI                  |          nan     |    nan         |    nan       |        nan     |
| Fz          |     2.909e+04 |     2.941e+04 |     2.972e+04 |                 1.13   |               1.048  | 0.9378 | hors régime asymptotique : pas de GCI |          nan     |    nan         |    nan       |        nan     |
| Mx          |    -8.989e+04 |    -9.121e+04 |    -9.22e+04  |                 1.475  |               1.082  | 0.7444 | converge                              |            2.43  |     -9.365e+04 |      1.969   |          3.343 |
| My          |     1.421e+04 |     1.434e+04 |     1.445e+04 |                 0.8699 |               0.8008 | 0.9285 | hors régime asymptotique : pas de GCI |          nan     |    nan         |    nan       |        nan     |
| Mz          |     1.793e+04 |     1.812e+04 |     1.831e+04 |                 1.051  |               1.08   | 1.038  | diverge : pas de GCI                  |          nan     |    nan         |    nan       |        nan     |

**Compensation.** La traînée de pression et la traînée de frottement varient en sens contraires : la convergence de la traînée totale est en partie une compensation. Leurs propres GCI disent la vraie incertitude.

Équivalent vitesse (sensibilités du run 3 à 45/16) : G2 → G3 -0.011 nd, G1 → G2 -0.013 nd.

## Résolution de paroi (même maillage, y+ différents)

| case           |   yplus_target |   yplus_avg |   layer_percent |    Fx |   Fx_pressure |   Fx_viscous |        Fz |         Mx |
|:---------------|---------------:|------------:|----------------:|------:|--------------:|-------------:|----------:|-----------:|
| Y30_medium     |             30 |       82.64 |            75.2 | -2255 |         -1271 |       -983.7 | 2.94e+04  | -9.115e+04 |
| G2_medium_yp50 |             50 |       93.7  |            75.2 | -2254 |         -1260 |       -994.1 | 2.941e+04 | -9.121e+04 |
| Y100_medium    |            100 |      121.5  |            78.9 | -2245 |         -1254 |       -990.7 | 2.947e+04 | -9.141e+04 |

Le y+ moyen est celui de toute la surface du foil, zones sans couches comprises (y+ maximal de 1500 à 2300) : il ne mesure pas le y+ des zones à couches. Les y+ obtenus restent loin des y+ visés, et l'écart entre ces cas ne mesure que la plage obtenue.

## Domaine (D2, deux fois plus grand, contre G1)

Fx +1.25 %, Fy +0.42 %, Fz +0.61 %, Mx +0.98 %; équivalent vitesse -0.037 nd.

