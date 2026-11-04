# Premier vrai run : la section du C-foil qui maximise la vitesse d'Intens SY

22/10/2026, 10 h 21 → 11 h 05 (44 min) · [`config.json`](config.json) · [`run.log`](run.log) ·
[`results/summary.txt`](results/summary.txt)

**En bref.** La chaîne complète tourne avec les vrais codes : NPLLT × NeuralFoil en BF, AVL × XFOIL en HF, et la VPP
MDS d'Intens SY.

* La recherche a amélioré la vitesse moyenne prédite de **+0,05 nd (+0,25 %)** par rapport à la section moyenne de
  départ.
* Le surrogate est **très précis là où il a raffiné.** Aux 5 états VPP de la meilleure section, l'erreur de vitesse
  contre AVL vaut **0,003 nd RMS** (0,007 nd au pire), et chaque coefficient du torseur est juste à 0,1 à 0,3 % près.
* Mais **la meilleure section n'a que 6,2 % d'épaisseur.** Il n'y avait pas de contrainte d'épaisseur. Et le profil
  actuel du C-foil (12,25 %) est **hors de la boîte Dim-Reduction** : le résultat n'est pas utilisable tel quel.

## Ce qui a tourné

| Élément | Réglage |
| --- | --- |
| Section | boîte à 16 poids de `Dim-Reduction/experiments.ipynb`, poids de bord d'attaque 0, bord de fuite 0,00197, surfaces sans croisement, NeuralFoil fiable (confiance ≥ 0,9 de 0 à 6°) |
| Réduction | PCA sur 4 000 profils, **k = 10 modes** (99,9 % de la variance géométrique). Le choix est expliqué plus bas |
| Entrées du surrogate | 10 scores + Cant_Eff, Rake_Eff, Yaw_Eff (13 dimensions, pas de Hvol) |
| Sorties | Fx, Fy, Fz, Mx, My, Mz, Cpmin (7 co-krigeages, ρ borné) |
| Codes | BF = NPLLT × NeuralFoil xxxlarge (0,18 s), HF = AVL × XFOIL (7 s en médiane, ~12 s pour un nouveau profil) ; foil de la RS de production (R 9,8 m, cant_geom 20,168°, image paroi, Re 5·10⁶), recalé à l'étape 0 |
| VPP | MDS `new_master`, variante `CFoilFSI` dont la RS dyn est remplacée par le surrogate rigide (Bending et Twisting à 0) |
| Objectif | vitesse moyenne sur 5 conditions (TWA/TWS : 45/12, 45/16, 60/16, 75/16, 90/14) |
| Départ | 120 BF (99 réussis) et 30 HF (30 réussis) |
| Boucle | critère ciblé VPP (16 points par état, 512 candidats locaux et 256 globaux), coûts 1 : 6 en temps réel (surcoût de la boucle compris), mise à jour de l'optimiseur tous les 10 pas, arrêt à 42 min |

**Pourquoi k = 10 et pas plus.** Cette famille remplit bien les 16 dimensions : il faut 15 à 16 modes pour
**représenter** fidèlement un profil quelconque. Pour **optimiser**, ce n'est pas le bon critère. Les codes évaluent
toujours le profil **décodé**, dont la géométrie est exacte, donc la troncature ne fausse aucun calcul : elle restreint
seulement la recherche au sous-espace des 10 premiers modes. Le surrogate apprend alors en 13 dimensions au lieu de 19.

## Ce qui s'est passé

**Le déroulé** ([`figures/convergence.html`](figures/convergence.html)) :

* 214 pas ; 294 calculs BF (61 s de code) et 99 calculs HF (20 min de code) ; plan final de 263 BF et 85 HF.
* L'algorithme a fait **155 choix BF et 59 choix HF**. Près des états VPP, la traînée HF compte pour la vitesse : il
  achète de la HF, ce que l'ALC global ne faisait pas sur le méta-problème.
* **Tous les points ajoutés sont près des états VPP**, à 0,6 demi-largeur de nuage en médiane
  ([`figures/points.html`](figures/points.html)). Ils forment deux grappes, une par régime : faible gîte à 12 et
  14 nd, forte gîte à 16 nd. La HF va surtout dans la seconde, où la traînée pèse le plus.
* **La meilleure section n'a changé que 3 fois** en 23 mises à jour. La vitesse prédite de la section retenue bouge
  aussi entre deux mises à jour, sans qu'elle change : à mesure que le surrogate s'affine, sa vitesse est corrigée, de
  −0,02 nd entre les mises à jour 5 et 6. **La meilleure section n'est apparue qu'à 38-40 min** : la recherche n'était
  pas convergée à l'arrêt.

**Le temps.** Le temps de la boucle se répartit ainsi :

* 42 % dans les mises à jour de l'optimiseur : 23 mises à jour, 17 min, avec les VPP des 4 sections × 5 conditions,
  plus 30 VPP de sensibilités une fois sur deux ;
* environ la moitié dans les calculs HF ;
* le reste dans le co-krigeage.

**Les échecs.**

* 21 calculs NPLLT sur 120 n'ont pas convergé dans le plan de départ ; ces points ont été écartés.
* Le même point HF a échoué 4 fois de suite (polaire XFOIL, 40 s à chaque fois). **C'est corrigé après le run :** un
  point en échec est maintenant exclu des candidats (`src/vpp_optimize.py`, `EXCLUSION_RADIUS`).

## Le résultat

| | Section moyenne (départ) | Meilleure section |
| --- | --- | --- |
| Vitesse moyenne prédite | 19,444 nd | **19,493 nd** |
| Épaisseur maximale | 9,6 % c | **6,2 % c** |

Vitesses de la meilleure section : 12,13 nd à 45/12, 18,96 nd à 45/16, 21,81 nd à 60/16, 24,00 nd à 75/16, 20,57 nd à
90/14 ([`results/best_section.json`](results/best_section.json), [`figures/sections.html`](figures/sections.html)).

**Validation contre AVL × XFOIL aux états VPP finaux** ([`results/validation.csv`](results/validation.csv),
[`figures/validation.html`](figures/validation.html)) :

| | Erreur de vitesse (1er ordre) | σ annoncé | Erreur sur Fx…Mz | Erreur sur Cpmin |
| --- | --- | --- | --- | --- |
| Meilleure section (raffinée) | **0,003 nd RMS**, 0,007 au pire | 0,009 nd | 0,1 à 0,3 % | 5 % |
| Section moyenne (peu raffinée à la fin) | 0,011 nd RMS, 0,015 au pire | 0,010 nd | 0,1 à 0,6 % | 5 % |

NPLLT seul est nettement décalé d'AVL, surtout sur Fz et My : le co-krigeage a appris à corriger ce biais. Le
raffinement a bien rendu le surrogate précis là où se prend la décision. L'erreur y est quatre fois plus faible que sur
la section de départ.

## Ce qu'il faut en retenir avant le prochain run

1. **Ajouter une contrainte d'épaisseur.** Sans elle, la vitesse pousse vers le profil le plus mince, ce qui n'est pas
   réaliste structurellement. L'épaisseur est linéaire en les poids, donc la contrainte se teste exactement dans
   `KulfanBoxFamily.contains`. Il faut une épaisseur min de 12,25 %, ou mieux un Ixx minimal, comme dans l'étude
   `cfoil_intens_kulfan_trim_2`.
2. **Revoir la boîte.** Le profil actuel du C-foil en sort : 7 poids d'extrados sont au-dessus des bornes hautes. Je
   propose une boîte qui le contient, par exemple la boîte Dim-Reduction élargie, ou centrée sur la référence.
3. **Les vitesses absolues ne sont pas celles de la VPP de production.** Notre AVL rigide donne 1,29 fois moins de
   traînée que la RS de production, qui est faite en FSI (étape 0). À 45/16, notre chaîne est donc 0,5 nd plus rapide
   que la VPP de production. **Seules les différences entre sections ont un sens.**
4. **Cpmin est encore mal appris** (5 % d'erreur, grandes barres d'incertitude). Si la cavitation devient une
   contrainte, il faudra plus de points HF ou un modèle de Cpmin dédié.
5. **Pour converger, il faut plus de temps ou des mises à jour moins chères.** Il y a deux leviers : calculer les
   sensibilités moins souvent (elles bougent peu), et résoudre les VPP de l'optimiseur en parallèle.
