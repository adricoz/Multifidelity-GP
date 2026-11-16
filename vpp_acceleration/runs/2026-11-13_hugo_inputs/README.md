# Données pour l'optimisation robuste de section (Hugo) — 13/11/2026

Ce dossier rassemble trois choses tirées de notre chaîne d'optimisation du C-foil d'Intens SY (VPP + substitut multi-fidélité) :

1. la distribution de la vitesse du bateau et de l'angle d'attaque du foil à l'équilibre VPP ;
2. notre PCA des sections, avec ses 4 premiers modes et leurs sensibilités de traînée ;
3. une contrainte d'épaisseur.

Tout est produit par `vpp_acceleration/scripts/hugo_inputs.py` (sous-commandes `thickness`, `pca`, `constraints`, `profile`, `sensitivities`, `xfoil_check`, `figures`). Les figures sont dans `figures/hugo_inputs.html`.

Le dossier `codes/` est un cache de polaires XFOIL interne, à ne pas transmettre.

---

## 1. Vitesse et angle d'attaque à l'équilibre VPP

### Ce qui a été calculé

- **Conditions de navigation.** 500 tirages Monte Carlo (graine 0) dans la loi analytique de notre profil `profile_v2`. C'est un mélange de normales tronquées, TWA et TWS indépendantes dans chaque mode :

| mode | poids | TWA [°] | TWS [kt] |
|---|---|---|---|
| près (upwind) | 0,387 | N(40, 10) tronquée [32, 90] | N(11, 15) tronquée [5, 17] (≈ uniforme) |
| largue (downwind) | 0,387 | N(80, 10) tronquée [32, 120] | idem |
| queue de largue | 0,086 | N(80, 25) tronquée [32, 130] | idem |
| points posés à la main | 0,14 | 13 points en bordure (TWS 5 et 17, TWA 32 et 110) et 60°/11 kt | |

- **Repérer les points posés à la main.** Ce sont des masses ponctuelles qui couvrent les bords du domaine ; leurs tirages ont `mode = manual`. Si tu ne veux que la partie continue de la loi, supprime-les.

- **Équilibre.** Notre VPP (Intens SY, MDS) avec le foil actuel :
  - section MC2 (base de la PCA, § 2), foil rigide ;
  - forces du foil données par notre substitut AVL × XFOIL, transition forcée à 3 %, n_crit 1.
  - Le VPP choisit la vitesse et l'attitude (gîte = Cant_Eff, assiette = Rake_Eff, dérive = Yaw_Eff, avec les réglages du foil à 0). L'angle d'attaque en découle.

- **Angle le long de l'envergure.** AVL × XFOIL à l'attitude d'équilibre, 103 strips du pied (s = 0, côté coque) au bout (s = 1) :
  - `incidence_geo_deg` est l'incidence géométrique ;
  - `alpha_eff_deg` est l'angle effectif, c'est-à-dire l'angle auquel la polaire 2D de la section donne le cl de la strip. Il inclut l'angle induit.

- **Reynolds.** `re_chord_vpp` = V_bateau × c / ν, avec c = 0,70 m et ν = 1,19e-6 m²/s (eau de mer à 15 °C). Notre substitut est calculé à Re fixe 5e6 ; le Re donné ici est celui de la vitesse VPP, celui que la section voit vraiment.

### Résultats

Sur 500 tirages :
- 490 trouvent un équilibre ;
- 10 n'en ont pas : tous au près par vent très faible (TWS < 6 kt, TWA < 41°), où le bateau ne vole pas dans notre VPP. Ils sont gardés dans le fichier avec `vpp_success = False` ;
- 5 sont au plafond de vitesse de 24 kt du modèle de coque (`at_speed_cap = True`), au largue par vent fort.

| grandeur | moyenne | p5 | p25 | médiane | p75 | p95 |
|---|---|---|---|---|---|---|
| vitesse bateau [kt] | 12,99 | 7,84 | 9,21 | 11,44 | 16,20 | 22,69 |
| Re sur la corde | 3,94e6 | 2,38e6 | 2,79e6 | 3,47e6 | 4,91e6 | 6,88e6 |
| **alpha équivalent [°]** | −0,81 | −2,25 | −1,67 | −1,11 | 0,10 | 1,21 |
| alpha effectif à s = 0,1 (pied) [°] | −0,45 | −2,13 | −1,45 | −0,80 | 0,61 | 1,90 |
| alpha effectif à s = 0,5 [°] | −0,72 | −2,18 | −1,59 | −1,02 | 0,22 | 1,34 |
| alpha effectif à s = 0,9 (bout) [°] | −1,74 | −2,63 | −2,27 | −1,93 | −1,18 | −0,49 |

Le cl moyen du foil vaut 0,34 (p5–p95 : 0,19–0,56).

### Ce que nous proposons d'utiliser

- **Un seul angle par condition : `alpha_eq_deg`.**
  - C'est la moyenne de l'angle effectif sur l'envergure, pondérée par la portance locale (cl·c·ds), donc l'angle « là où la force est produite ».
  - Avec la vitesse, cela donne la loi jointe (V, alpha_eq) de `profile_samples.csv`, une ligne par tirage, tirages équiprobables.
  - Les deux variables sont corrélées (figure « loi jointe ») : entre p5 et p95, le mode près est à alpha −0,8 à +1,3° et 7,5–18 kt, le mode largue à alpha −2,2 à −1,1° et 8,5–23 kt. Il vaut mieux tirer les couples que les deux lois séparément.
- **La distribution le long de l'envergure, si un angle unique ne suffit pas.**
  - Fichier `span_states.csv` : une ligne par tirage et par strip.
  - Le pied voit en moyenne +0,36° de plus que alpha_eq (de 0 à +0,8°). Le bout voit −0,93° de moins (de −0,1 à −1,8°), et l'écart grandit avec alpha (figure « écart pied / bout »).
  - Les 10 % d'envergure côté bout chutent fortement (perte de bout, jusqu'à −4° à l'extrémité).

**Limite importante.** Ces angles sont ceux de la section MC2. À l'équilibre, le VPP impose la **portance**, pas l'angle. Une section plus ou moins cambrée a un autre angle de portance nulle : notre meilleure section optimisée, à dérive égale, ne porte que 55 % de la portance de MC2. Elle volerait donc à un autre alpha. Si ton optimisation change beaucoup la cambrure, la portance requise (colonne `cl`) est une condition plus robuste que l'angle.

---

## 2. PCA des sections

### Comment la base a été construite

- **Paramétrisation.** Kulfan (CST) avec 8 poids par face, classe C(x) = √x (1 − x) (N1 = 0,5, N2 = 1), poids de bord d'attaque 0, épaisseur de bord de fuite 0,00197 c. Le vecteur w contient d'abord les 8 poids d'extrados, puis les 8 poids d'intrados (négatifs). Les coordonnées s'écrivent y = A w + b sur 200 stations uniformes, extrados puis intrados, soit 400 valeurs.
- **Échantillon.** 97 401 sections tirées uniformément dans un domaine **convexe** autour de MC2. ⚠ Ce n'est **pas** un LHS : c'est un échantillonnage *hit-and-run* (Smith 1984), uniforme dans le domaine. Contraintes du domaine (linéaires en w, sauf la dernière) :
  - boîte Kulfan = 3 × la boîte de notre étude de réduction de dimension, centrée sur MC2 ;
  - surfaces qui ne se croisent pas, t(x) ≥ 0 ;
  - extrados convexe, y_u'' ≤ 0 sur [0,01 ; 0,99] ;
  - intrados avec zones de signe fixes autour des points d'inflexion de MC2 (x₁ = 0,467, x₂ = 0,938, bandes libres de ±0,05) ;
  - bord d'attaque rond, |w₀| ≥ √(2κ) t(x) ;
  - Ixx/v ≥ 0,95 × Ixx/v(MC2 ancre), avec v = t_max/2.

  Les paramètres exacts sont dans `pca_hugo.json` → `generation`.
- **PCA.** Centrée, sur les coordonnées. Les scores sont blanchis, z_j = (y − ȳ)·v_j / σ_j, avec une variance de 1 sur l'échantillon. Les 4 premiers modes portent 94,9 % de la variance (55,7 / 24,7 / 8,5 / 6,0 %) et les 10 premiers 99,89 %.

### Utiliser les 4 premiers modes

```
w(z1..z4) = mean_weights_anchor4 + Σ_{j=1..4} σ_j z_j m_j      (16 poids Kulfan)
y(z1..z4) = mean_coordinates_anchor4 + Σ_{j=1..4} σ_j z_j v_j  (400 coordonnées)
```

- La moyenne est **ancrée** sur MC2 : z = `reference_scores_z0`[:4] = (−0,145 ; 2,052 ; 0,621 ; −1,738) redonne exactement MC2. Les modes 5 à 10 restent à leur valeur pour MC2.
- `pca_modes_weights.csv` donne σ_j m_j (déplacement des 16 poids pour une unité de score) des 10 modes, ainsi que MC2 et la moyenne ancrée.
- `pca_modes_dy.csv` donne σ_j v_j (déplacement de chaque surface en fonction de x). `pca_hugo.json` contient toutes les matrices.

Effet d'une unité de score (autour de MC2) :

| mode | variance | déplacement max | dt_max/dz | dIxx/v / Ixx/v | effet visible (figure) |
|---|---|---|---|---|---|
| 1 | 55,7 % | 1,87 % c | +1,53 % c | +27 % | épaisseur (extrados surtout) |
| 2 | 24,7 % | 1,28 % c | −1,49 % c | −17 % | intrados : cambrure / épaisseur |
| 3 | 8,5 % | 0,88 % c | +0,07 % c | −3 % | position du maximum d'épaisseur |
| 4 | 6,0 % | 0,64 % c | +0,24 % c | −2 % | intrados arrière (inflexions) |

### Domaine admissible pour les 4 scores

- **Boîte.** Celle de notre optimisation (modes 1–4) : z₁ ∈ [−2,70 ; 1,50], z₂ ∈ [−2,63 ; 2,11], z₃ ∈ [−1,48 ; 2,64], z₄ ∈ [−2,80 ; 1,83].
- **Seulement 18,8 % de cette boîte est admissible.** C'est surtout l'extrados non convexe qui élimine des sections (40 % de la boîte), puis le bord d'attaque (25 %), l'intrados (21 %), la boîte Kulfan (20 %) et Ixx/v (17 %).
- **Contraintes linéaires écrites directement en fonction des 4 scores.** `constraints_z4.csv` donne 9 556 inégalités a·z ≥ b, avec des lignes normées pour que la marge soit une distance en unités de score. Elles sont vérifiées identiques au domaine sur 20 000 points. MC2 y est à une marge de 0,53.
- La contrainte non linéaire Ixx/v est remplacée par la contrainte d'épaisseur du § 3.

### Sensibilités de la traînée

- **Méthode.** NeuralFoil xxxlarge (2D), transition 3 %, n_crit 1, à l'(alpha_eq, Re) de 300 tirages du § 1. Différences centrées de ±0,1 autour de MC2, vérifiées avec ±0,2.
- **Deux lectures.** À alpha fixé, ce que tu fais si tu imposes alpha. À CL fixé, c'est-à-dire au CL de MC2 à cet alpha, ce que fait le VPP.

| mode | dCD/dz à alpha fixé | en % du CD | dCL/dz à alpha fixé | dCD/dz à CL fixé (p5 ; p95) | en % du CD |
|---|---|---|---|---|---|
| 1 | +4,1e-4 | +4,5 % | +0,074 | +3,7e-4 (2,1e-4 ; 5,0e-4) | +4,1 % |
| 2 | −1,5e-4 | −1,7 % | +0,035 | −1,6e-4 (−3,6e-4 ; −0,2e-4) | −1,8 % |
| 3 | −0,4e-4 | −0,5 % | −0,014 | −0,3e-4 (−0,6e-4 ; −0,1e-4) | −0,4 % |
| 4 | +0,4e-4 | +0,5 % | +0,026 | +0,3e-4 (−0,8e-4 ; +1,4e-4) | +0,3 % |
| 5 à 10 | ≤ 0,3e-4 | ≤ 0,35 % | ≤ 0,013 | | |

CD de MC2 ≈ 0,0090 et CL ≈ 0,36 en moyenne sur ces conditions.

**Fiabilité.**
- Modes 1 et 2 : signe et ordre de grandeur sûrs. Le contrôle XFOIL (Re 3,5e6, alpha −1 / 0,5 / 1,5°) donne le même signe ; NeuralFoil sous-estime le mode 1 de 20 à 35 %.
- Modes 3 et 4 : seulement un ordre de grandeur, environ 0,4 % du CD par unité de score. XFOIL et NeuralFoil ne s'accordent pas sur le signe, et l'écart est sous la résolution de XFOIL.
- dCL/dz : NeuralFoil et XFOIL concordent à moins de 5 % sur les 4 modes.
- Modes 5 à 10 : aucun ne dépasse les modes 3 et 4. Le mode 5 est non linéaire, avec un résultat qui change entre ±0,1 et ±0,2. Les 4 premiers modes sont donc bien les plus influents sur la traînée, mais 3 et 4 de très peu.

### Base de la PCA et vrai MC2

La section de base (`reference_weights_mc2_anchor`) est **notre approximation Kulfan de MC2**, pas le fichier MC2 utilisé en CFD (`MC2_60_6101_1_re5M.xf`) :

| section | t/c max | à x/c | cambrure max | Ixx/v |
|---|---|---|---|---|
| MC2 fichier | 12,68 % | 0,31 | 3,3 % à 0,51 | 1,254e-3 |
| MC2 base PCA (ancre) | 12,25 % | 0,27 | 4,2 % à 0,33 | 1,097e-3 |

- Elles diffèrent de 1,9 % de corde au maximum.
- Le fichier MC2 réel n'est pas exactement représentable : un ajustement Kulfan à 16 poids laisse 0,43 % c d'écart, et sa projection sur nos modes (0,60 % c avec 4 modes) sort du domaine admissible.

---

## 3. Contrainte d'épaisseur

**t/c max ≥ 12,34 %**, à n'importe quelle position.

- **Origine.** C'est l'épaisseur maximale qui donne Ixx/v = 0,95 × Ixx/v du **vrai MC2** (fichier CFD), quand on met à l'échelle l'épaisseur de MC2 en gardant sa ligne de cambrure. Ixx est calculé par bandes verticales, cambrure incluse, avec v = t_max/2. C'est la contrainte de rigidité de notre étude, traduite en épaisseur.
- ⚠ **La base de la PCA (12,25 %) est juste en dessous de ce seuil.** L'optimisation doit donc épaissir un peu la section, ce qui se fait essentiellement par le mode 1 (+1,53 % c par unité de score).
- t/c max n'est qu'un substitut de Ixx/v, qui dépend aussi de la distribution d'épaisseur. Si besoin, la formule exacte et les valeurs de référence sont dans `thickness.json`.

---

## Fichiers

| fichier | contenu |
|---|---|
| `profile_samples.csv` | 1 ligne par tirage : TWA, TWS, mode, succès VPP, vitesse, attitude, Re, alpha_eq, alpha au pied / milieu / bout, cl moyen |
| `span_states.csv` | 1 ligne par tirage × strip : s/envergure, position, corde, incidence géométrique, alpha effectif, cl, cd, Re |
| `span_summary.csv` | quantiles du tableau du § 1 |
| `pca_hugo.json` | toute la PCA (stations, σ, modes v et m, moyennes ancrées, MC2, boîtes, génération, projection du MC2 réel) |
| `pca_modes_weights.csv`, `pca_modes_dy.csv` | directions des modes en poids Kulfan et en déplacement de surface |
| `constraints_z4.csv`, `constraints_z4.json` | contraintes linéaires a·z ≥ b sur (z1..z4) |
| `sensitivities.csv`, `sensitivities.json` | dCD/dz, dCL/dz des 10 modes (moyennes, p5, p95, pas h et 2h) |
| `xfoil_check.json` | contrôle XFOIL des sensibilités des modes 1 à 4 |
| `thickness.json` | épaisseurs, Ixx, Ixx/v et seuils équivalents |
| `figures/hugo_inputs.html` | figures |
