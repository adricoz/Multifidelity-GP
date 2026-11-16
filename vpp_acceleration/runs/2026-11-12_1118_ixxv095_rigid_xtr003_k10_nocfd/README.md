# Run sans CFD, comparé au run avec CFD (12/11/2026)

**Demande d'Adrian (12/11).** Refaire l'optimisation sans le niveau CFD 3D. Tout le reste est identique au run
`2026-11-11_1245_ixxv095_rigid_xtr003_k10_cfd3d` :

- même PCA (`2026-11-11_1054_section_pca_convex_ixxv095`, k = 10, Ixx/v ≥ 0,95) ;
- NPLLT × NeuralFoil + AVL × XFOIL à xtr 3 %, profile_v2 ;
- mêmes graines, mêmes tailles de plan, 3 redémarrages ;
- sans repli XFOIL à 5 % (`--no-xtr-fallback`), comme le run CFD.

Lanceur : `runs/launch_2026-11-12_ixxv095_nocfd.ps1`.

| | Run avec CFD (11/11) | Ce run, sans CFD |
|---|---|---|
| Durée | 15,4 h (GPU pour la CFD) | 12,5 h |
| Plan final par niveau | 1214 / 280 / 6 CFD | 1188 / 320 |
| Meilleure section, validée en AVL × XFOIL | 13,260 nd | 13,266 nd |
| Section actuelle MC2, même validation | 12,930 nd | 12,952 nd |
| Gain | +0,33 nd | +0,31 nd |
| Optima des régions (prédits par le surrogate) | 13,158-13,161 nd | 13,255-13,266 nd |
| Écart du surrogate à AVL sur la meilleure section | RMS 0,12 nd | RMS 0,0003 nd |

**Lecture.**

- **Les deux gains sont égaux aux erreurs près.** La vitesse validée de MC2 elle-même varie de 0,02 nd d'un run
  à l'autre.
- **Les vitesses prédites sont plus basses d'environ 0,1 nd dans le run avec CFD.** Le surrogate y suit la CFD,
  qui donne environ 12 % de traînée de plus qu'AVL (Fx).
- **Les deux sections ne sont pas les mêmes**, ce qui montre que l'optimum est plat :
  - distance de 1,5 σ dans la PCA, écart local de 1,1 % de corde au plus (intrados) ;
  - la section de ce run est un peu plus cambrée (cambrure max 1,9 % contre 1,6 %) et un peu plus mince
    (t/c 12,0 % contre 12,4 %).
- **Sur la figure Sections**, la meilleure section du run avec CFD est en tirets.
- **Validation incomplète.** Les régions 0 et 2 n'ont pas pu être validées, à cause d'échecs XFOIL à 3 %.
  Leurs vitesses de 0,000 et 0,255 nd dans le résumé sont des « nan » mal affichés.
- **Ce qui reste à faire pour trancher** entre les deux sections : des calculs CFD des deux sections et de MC2 à deux
  lacets chacune, pour comparer les traînées à portance égale.
