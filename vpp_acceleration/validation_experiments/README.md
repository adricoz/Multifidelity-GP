# Expérimentations de validation de l'optimum

Tests de la note `doc/validation_theorique_optimum.tex`, appliqués à un run **terminé**, sans aucun
nouveau calcul des codes ni de la VPP. Tout vient de ce que le run a sauvegardé : le modèle final
(`results/surrogate.pkl`), les états VPP et les codes à ces états (`results/validation.csv`), les
sensibilités de vitesse (`run.log`).

Run analysé le 12/11/2026 : `runs/2026-11-11_1245_ixxv095_rigid_xtr003_k10_cfd3d`
(3 niveaux : NPLLT × NeuralFoil 1214 points, AVL × XFOIL 280 points, CFD 3D 6 points ; contrainte
Ixx/v ≥ 0,95 ; 48 conditions résolues sur 49).

## Lancer

```
..\.venv-mds\Scripts\python validation_experiments\experiments.py --run runs/<run>
..\.venv-mds\Scripts\python validation_experiments\figures.py
```

Environ 20 minutes par expérience sur un thread (`--only <nom>` pour en lancer une seule, en
parallèle si la machine est libre). Résultats dans `results/*.json`, figures dans
`figures/index.html`.

## L'approximation utilisée

L'objectif est linéarisé autour de la meilleure section, comme le criblage du run : les attitudes
du foil et les sensibilités `dV/dC` sont **figées** aux équilibres VPP de la meilleure section.

- Le **gradient** en la meilleure section est exact (théorème de l'enveloppe).
- Loin d'elle, c'est un modèle au premier ordre : les résultats sur des sections éloignées (regret,
  sous-espaces actifs sur tout le domaine) sont indicatifs.
- Cet objectif est linéaire en les efforts : sa loi a posteriori est gaussienne, et les covariances
  entre sections sont exactes.

## Résultats sur le run du 11/11

| Question (section de la note) | Résultat | Lecture |
|---|---|---|
| Le modèle est-il calibré entre les données ? (§3) | écart réduit 0,91–0,99 au niveau 1, 0,81–0,88 au niveau 2 ; Cpmin niveau 2 : Q² 0,69, écart réduit 1,21 | calibré, un peu prudent ; Cpmin médiocre |
| Est-il calibré près de l'optimum ? (§3) | niveau 2 contre AVL × XFOIL aux 48 états de la meilleure section : erreur 0,0005 nd, écart-type annoncé 0,0005 nd | facteur κ ≈ 1 (et non 2 comme au run 8) |
| Reste-t-il de la pente ? (§2.3) | gradient 0,048 nd/score, dont 0,005 le long des contraintes ; χ² = 0,7 pour 8 degrés de liberté | conditions KKT satisfaites |
| Qu'est-ce qui retient la section ? (§2.3) | le rayon de bord d'attaque et Ixx/v, tous deux à 0,012 score | prix de Ixx/v : 0,0015 nd par % relâché |
| Que gagnerait une montée de gradient ? (§2.3) | +0,0012 ± 0,0006 nd (rayon 0,1), +0,0020 ± 0,0019 (0,3), +0,004 ± 0,006 (1,0) | rien de mesurable |
| Une meilleure section existe-t-elle ? (§2.1) | regret attendu 0,007 nd ; P(mieux de plus de 0,01 nd) = 27 % ; de plus de 0,02 nd : 3 % | le modèle ne départage pas les 4 optima locaux |
| Le problème est-il simple ? (§4) | 1 direction porte 98 % de la variation de vitesse, 2 directions 99,3 % ; une seule courbure forte au sommet | plateau : optimum sûr en vitesse, pas en forme |
| Un niveau moins cher suffit-il ? (§5.2) | optimiser AVL × XFOIL au lieu du niveau CFD : optimum déplacé de 1,3 score, perte 0,024 ± 0,008 nd | 10 % du gain, mais repose sur 6 points CFD |

### Détails utiles

- **Le « 4 % dans ± 2σ » du résumé du run n'est pas un défaut de calibration.** Le modèle final
  prédit le niveau CFD ; la validation de fin de run le compare à AVL × XFOIL. L'écart est le
  décalage entre niveaux (8,7 % sur Fx en médiane, ρ = 1,34 sur Fx au niveau 3).
- **Le niveau CFD n'est pas vérifiable** : 6 points, tous à cant et rake nuls. Sa validation croisée
  (6 valeurs) ne prouve rien. Or c'est lui qui déplace l'optimum (dernière ligne du tableau).
- **Les 4 optima locaux sont équivalents pour le modèle** : écarts moyens de −0,003 à +0,001 nd
  avec un écart-type de 0,009 nd sur ces différences, à 1,7–2 scores de distance. Ce sont eux, et
  les sections à un rayon de 1, qui « gagnent » dans les tirages. Aucune section tirée dans tout le
  domaine ne gagne (au premier ordre).
- **Les coupes 1D** : la section admissible ne peut bouger que de 0,05 à 0,35 score dans la plupart
  des modes ; le gain admissible maximal le long d'un mode est de 0,0007 nd.
- **Direction active** : surtout le mode 1 (poids 0,94 sur tout le domaine) ; près de l'optimum, un
  mélange des modes 1 et 2, qui est aussi la direction du gradient.

## Limites

- Tout est conditionnel au modèle. La calibration n'est mesurée qu'aux niveaux 1 et 2.
- États VPP figés : pas de résolution VPP pour les sections candidates. Le test exact (VPP sur des
  tirages du modèle) est `scripts/robustness_checks.py posterior`, plus coûteux.
- Les sensibilités sont lues dans `run.log` avec deux décimales.
- La probabilité qu'une meilleure section existe porte sur 246 candidates : elle croît avec leur
  nombre (effet de sélection) et ne voit pas ce qui n'est pas dans la liste.
