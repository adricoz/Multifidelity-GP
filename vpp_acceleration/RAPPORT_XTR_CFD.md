# Effet de la transition forcée (xtr) et vérification CFD des sections optimisées — 09-10/11/2026

Figures : `runs/2026-11-09_1846_xtr_study/figures/index.html` (sections, matrice croisée, gains, polaires, CFD)
et `cfd_sections.html` (CFD des sections optimisées).

## 1. Question

L'optimisation de section (C-foil Intens SY, 2 niveaux NPLLT × NeuralFoil / AVL × XFOIL, PCA convexe k = 10,
départ section actuelle MC2, foil rigide, profil profile_v2) dépend-elle de la transition turbulente forcée xtr
d'XFOIL / NeuralFoil (Ncrit 1 gardé) ? Et le gain trouvé tient-il face à la CFD (OpenFOAM k-ω SST, entièrement
turbulent, chaîne bdfoil_v1 d'Antoine) ?

## 2. Runs d'optimisation

| run | xtr | fin | meilleure section | échecs HF |
|---|---|---|---|---|
| 2026-11-09_1057_xtr005 | 5 % | convergé | 13.272 kt | faibles |
| 2026-11-09_1057_xtr010 | 10 % | convergé trop tôt (11 mises à jour) | 13.280 kt (physique 10 %) | faibles |
| 2026-11-09_1325_xtr010_from5 | 10 %, départ = best 5 % | convergé | 13.313 kt | faibles |
| 2026-11-09_1411_xtr003 | 3 % | limite 3 h, plateau (gain 0 sur les dernières mises à jour) | 13.262 kt | 26/322 (8 %) |
| xtr 1 % et 2 % | — | **invalides** : XFOIL ne converge pas sur la branche négative de la polaire (≈ 50 % / 42 % d'échecs) | — | — |

Validation du modèle de substitution contre AVL × XFOIL sur la meilleure section : RMS 0.0005 kt.

## 3. Évaluation croisée (chaque section dans la physique de chaque xtr)

Vitesse moyenne pondérée sur 99 % de la probabilité du profil [kt] :

| physique \ section | best 5 % | best 3 % | best 10 % (départ 5 %) | best 10 % (1er run) | MC2 |
|---|---|---|---|---|---|
| xtr 5 % | 13.272 | 13.277 | 13.267 | 13.235 | 12.958 |
| xtr 10 % | 13.305 | 13.303 | **13.313** | 13.280 | — |
| xtr 3 % (modèle seul*) | −0.009 | réf. | −0.013 | −0.038 | −0.321 |

\* à xtr 3 %, XFOIL échoue à Re 4.2e6 sur toutes les sections optimisées : pas de correction HF, écarts du modèle
seul par rapport à la meilleure section 3 %.

**Conclusion 1 : xtr entre 3 et 10 % ne change presque pas la section optimale** (les trois optimums se valent à
±0.01 kt dans chaque physique) ; le gain sur MC2 reste ≈ +0.31 kt quel que soit xtr. Le seul écart notable
(−0.03 kt) vient du premier run à 10 % arrêté trop tôt, pas de la physique. Les sections optimisées diffèrent de
MC2 de 25 mm au plus (corde 700 mm, cambrure 1.7-2.2 % contre 3.7 %) et entre elles de 7-10 mm, surtout à
l'intrados : plateau plat.

## 4. CFD 2D (section seule, Re 6.96e6, paire L1'+L2' extrapolée comme le cas C6 d'Antoine)

CD [1e-4] à CL = 0.1 (interpolé entre points convergés) :

| section | CFD | XFOIL 3 % | XFOIL 5 % | XFOIL 10 % |
|---|---|---|---|---|
| MC2 | 83.4 | — | 82.2 | 81.1 |
| opt 5 % | 80.4 | 78.3 | 76.6 | 74.5 |
| opt 3 % | **79.4** | 77.3 | 76.4 | 74.7 |
| opt 10 % (départ 5 %) | ≈ 80.3 (à CL 0.094) | 78.6 | 77.0 | 73.8 |

- Gain des sections optimisées sur MC2 : **−4 à −5 % en CFD**, contre −7 % (XFOIL 5 %) et −8 à −9 % (XFOIL 10 %).
  XFOIL surestime le gain 2D d'environ un facteur 2.
- Écart CFD – XFOIL : 1.5-3 % sur MC2, 4-8 % sur les sections optimisées : XFOIL favorise légèrement les
  sections optimisées.
- La CFD classe opt 3 % en tête, cohérent avec une CFD entièrement turbulente (xtr bas = plus proche).
- Points non convergés exclus : opt 5 % à 0° (limite de temps), opt 10 % à 0° (arrêté le matin) ; MC2 au-delà de
  −2° (runs d'Antoine).

## 5. CFD 3D (cas C9 d'Antoine : R9800, 23 kn, plan de symétrie coque, sans surface libre)

Comparaison **à portance égale** : MC2 à l'assiette neutre (run L1 CPU d'Antoine, 20/10) contre opt 3 % en yaw
2.593° (yaw donnant la portance de MC2 selon AVL × XFOIL xtr 5 %). Même maillage L1 (≈ 0.88 M cellules),
CPU 16 rangs, convergé par le critère d'Antoine (2 386 itérations, 56 min).

| | MC2 neutre | opt 3 %, yaw 2.59° | écart |
|---|---|---|---|
| Fz [N] | 28 026 | 27 894 | −0.5 % |
| −Fy [N] | 52 137 | 52 003 | −0.3 % |
| **traînée −Fx [N]** | **2 223** | **2 127** | **−96 N (−4.3 %)** |
| dont pression | 1 420 | 1 330 | −90 N |
| dont frottement | 804 | 797 | −7 N |
| AVL × XFOIL xtr 3 % : −Fx | 2 126 | 2 070 | −56 N (−2.7 %) |
| AVL × XFOIL xtr 5 % : −Fx | 2 108 | 2 046 | −62 N (−2.9 %) |
| AVL × XFOIL xtr 10 % : −Fx | 2 072 | 2 012 | −60 N (−2.9 %) |

**Conclusion 2 : la CFD 3D confirme le gain**, et même un peu plus grand que celui prévu par AVL × XFOIL
(−96 N contre −56 à −62 N selon xtr : le gain AVL × XFOIL ne dépend presque pas de xtr), le supplément venant de la traînée de pression (jonction, bout de foil, que la théorie
des bandes ne voit pas). En 2D le gain CFD est plus petit que celui d'XFOIL ; en 3D il est plus grand : l'écart
modèle/CFD est dominé par le 3D, pas par la transition.

Limites : L1 seulement (≈ 2.5 % d'erreur de maillage sur Fx entre L1 et L2, qui s'annule en grande partie dans
l'écart) ; assiettes différentes (neutre contre yaw 2.6°) donc répartition de charge pas identique ; AVL
surestime la portance des deux sections de ≈ 4 % (même biais).

## 6. Ce qui n'a pas marché

- **3D sur GPU (nuit)** : 4 rangs GPU en parallèle d'une 2D à 10 rangs → ≈ 4.5× plus lent que le CPU à 16 rangs
  sur ce maillage de 0.9 M cellules (MC2 L1 : 1 084 itérations en 2 h contre 2 439 en 59 min) ; aucun run des
  sections optimisées n'a convergé (L2 bloqué à 28 itérations en 5 h, cause non diagnostiquée). La chaîne GPU est
  juste (MC2 L1 GPU = CPU à 0.7 % sur Fx). Règle retenue : < 1-1.5 M cellules → CPU 16 rangs, machine libre.
- **AVL × XFOIL sur les sections optimisées (nuit)** : échec, fichiers de 321 points ; corrigé en rééchantillonnant
  à 249 points (comme MC2).
- **xtr 1 et 2 %** : XFOIL ne converge pas (voir § 2).

## 7. Suites possibles (à décider ensemble)

1. L2 à portance égale (MC2 neutre L2 d'Antoine existe, ≈ 5.6 h CPU 16 rangs) pour confirmer l'écart sur maillage fin.
2. Corriger le modèle par la CFD (calage de la traînée de pression 3D) plutôt que de baisser xtr.
3. Garder xtr 5 % (ou 3 %) pour la suite : le choix ne change pas la section, il change surtout la vitesse absolue.

## Fichiers

- `scripts/cfd_sections.py` (prepare / avl / run2d / run3d [--gpu, --yaw] / figures), `scripts/xtr_study_figures.py`,
  `scripts/robustness_checks.py cross`, options `--xtr` / `--polar-timeout` de `scripts/run_real_vpp.py`.
- Données : `data/xtr_study/cfd/` (sections .xf, avl.json, cfd2d.json, cfd3d.json), `data/xtr_study/c9_avl_xtr.json`.
- Lanceurs : `runs/launch_2026-11-09_xtr*.ps1`, `runs/launch_2026-11-09_cfd_night.ps1`,
  `runs/launch_2026-11-10_cfd_opt003_yaw.ps1`.
