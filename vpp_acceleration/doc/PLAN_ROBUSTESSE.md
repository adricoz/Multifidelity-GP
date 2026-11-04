# Plan : solidité des optimums (02/11/2026)

En attente du pipeline CFD. Contexte : note de Le Maître (optimisation_foil_GP_VPP_v01, §6.3, §10, §11.4)
et préparation `preparation_reponse_le_maitre.tex`.

## Constat

- **Statistique** : gain du run 8 = +0,256 nd, erreur de fin de run contre AVL × XFOIL = 0,013 nd
  (variance sous-estimée ×2 loin des données). Gain > 10 σ même variance doublée.
  Les deux régions du plateau (13,566 / 13,560 nd) : regret H(|d|, s) ≈ 0,005 à 0,008 nd, donc négligeable.
- **Numérique** : VPP multi-départs. Optimum parfois sur la contrainte Ixx (run 4) : afficher le multiplicateur
  (nd gagnés par unité d'Ixx, sa 5.18).
- **Physique (le vrai risque)** : le GP apprend AVL × XFOIL, pas la réalité. Extrados non convexe au run 8 :
  l'optimum peut exploiter un défaut de XFOIL. (Pas le laminaire : XFOIL tourne déjà en Ncrit = 1 avec
  transition forcée à 10 % de corde, `src/foil_codes.py`.)

## Tests sans run

1. **Tirages a posteriori (§6.3)** : 20 à 50 fonctions GP cohérentes (tirage pathwise BoTorch), VPP multi-départs
   sur les 5 à 10 meilleures sections, loi du meilleur candidat et regret. Quelques minutes, sur le GP seul.
2. **Tolérances de fabrication** : perturber les poids de Kulfan (amplitude réaliste), environ 10 évaluations
   des vrais codes par finaliste.
3. **CFD sur les 2 ou 3 finalistes** (§11.4), après l'étude de maillage.

## Résultats des tests sans run (03/11, run 8, `scripts/robustness_checks.py`)

1. **Tirages a posteriori** (`runs/2026-11-03_1041_robustness_posterior_2026-10-29_1002`) : 20 fonctions GP
   cohérentes, à σ et à 2σ, sur best, région 1, best du run 7 et départ. P(best) = 1 (20/20) aux deux échelles,
   regret nul. best − région 1 = +0,007 ± 0,002 nd, best − run 7 = +0,017 ± 0,004 nd, best − départ =
   +0,259 ± 0,006 nd (2σ). **Condition 2 de la règle d'arrêt remplie.**
2. **Tolérances** (`runs/2026-11-03_1312_robustness_tolerance_2026-10-29_1002`), AVL × XFOIL aux équilibres :
   ±0,002 nd à 0,35 mm, −0,003 / +0,007 nd à 1,05 mm. Les écarts suivent 1/Ixx (corrélation 0,96) ; les sections
   qui respectent Ixx ≥ Ixx actuel perdent 0 à 0,003 nd. Plateau, pas pic ; l'optimum est tenu par Ixx.
3. CFD des finalistes : en attente du pipeline CFD.

Limite : ces deux tests portent sur AVL × XFOIL. Le risque physique (défaut de XFOIL) reste pour la CFD et le lot B.

## Runs (environ 1 h 30 chacun sans CFD dans la boucle)

| Lot | Config | Seeds | Question tranchée |
|---|---|---|---|
| A | config actuelle | 3 | reproductibilité (même bassin ? écart de vitesse) |
| B | + contrainte de régularité de l'extrados | 3 (2 si A stable) | part du gain qui survit à une géométrie propre |

- Lot C (transition forcée) supprimé le 03/11 : c'est déjà la configuration de base.
- **Nominal** : 6 runs (3 en A, 3 en B), soit environ 9 h. **Minimum** : 5 runs, soit environ 7 h 30.
- Lancer en séquence, la nuit (machine partagée).
- **Recommandé** : CFD en validation des finalistes, pas comme niveau du GP. Les runs restent à environ 1 h 30.
  Si la CFD devient un niveau, recompter le coût avec le temps d'un calcul CFD.

## Règle d'arrêt

La solidité est tranchée si les trois conditions sont réunies :
- les 3 seeds de A tombent à moins d'environ 0,02 nd les uns des autres ;
- le regret calculé sur les tirages a posteriori reste sous 0,01 nd ;
- la CFD confirme le gain des finalistes à l'erreur près.

Si A est stable, réduire B.

## Pour la réunion avec Le Maître

Ajouter la question « Votre optimum exploite-t-il les défauts de XFOIL ? » au tableau des questions probables.
