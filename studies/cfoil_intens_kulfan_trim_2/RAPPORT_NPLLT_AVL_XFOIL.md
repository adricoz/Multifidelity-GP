# Évaluation de mfego et de la concordance NPLLT × XFOIL / AVL × XFOIL

Oct 15, 2026 · @Antoine

mfego fait correctement son travail d'optimiseur multifidélité, et NPLLT × XFOIL et AVL × XFOIL concordent sur les tendances : écarts de 1 à 4 % de traînée, efficacité d'envergure, traînée de profil. Les deux chaînes sont utilisables en production pour explorer, filtrer et comparer des familles de sections. Elles ne le sont pas pour départager deux sections à moins de \~0,5 % de traînée, ni, pour AVL, pour prédire le trim (le yaw).

Base : étude `cfoil_intens_kulfan_trim_2` (C-foil d'Intens SY, section Kulfan à portance 3D égale), run du 15/10/2026, 3 niveaux : NPLLT × NeuralFoil xxxlarge, NPLLT × XFOIL, AVL × XFOIL.

## mfego : ce qui marche, ce qui limite

mfego guide bien la recherche et sait estimer son incertitude, mais son erreur de prédiction (\~0,33 % de CD) dépasse le gain cherché (0,19 %).

**Ce qui marche :**

- **Le modèle sait ce qu'il ne sait pas.** En leave-one-out sur les 70 points AVL, 96 % tombent dans l'intervalle à 95 %. Les 4 sections jamais vues par AVL sont prédites à moins de 1,3σ.
- **Il trouve un optimum stable.** L'optimum du modèle et la meilleure mesure sont le même point. Les vérifications périodiques se trompent de −0,06 à +0,23 %.
- **Le mérite exploite enfin les niveaux chers** : 48 choix L2 et 44 choix L3, contre 0 à 8 aux runs précédents.
- **Le pipeline est devenu robuste** : trim exact, contraintes connues, 0 échec XFOIL sur 161 polaires.

**Ce qui limite :**

- **L'erreur de prédiction reste plus grande que le gain cherché.** Le RMSE du leave-one-out vaut 25 % de la dispersion, soit \~0,33 % de CD, pour un gain de 0,19 %. Le modèle guide la recherche mais ne peut pas certifier un gain de cet ordre : seule la re-mesure finale le fait.
- **Le niveau 1 trompe localement.** NeuralFoil xxxlarge classe bien l'ensemble de l'espace (Spearman 0,87), mais à l'envers parmi les meilleures sections (−0,74 contre AVL).
- **Les niveaux 1 et 2 sont presque redondants** sur la portance : ρ ≈ 1,01 et même yaw à 0,03° près. Ils ne diffèrent que par le modèle 2D.
- **La recherche du point suivant coûte 60 % du temps de calcul**, sans gain de précision : c'est à optimiser.

## NPLLT × XFOIL contre AVL × XFOIL

Les deux modèles 3D sont d'accord sur la physique globale et en désaccord sur la portance à attitude donnée, donc sur le trim et le classement fin des sections.

| Grandeur | Concordance | Chiffres |
| --- | --- | --- |
| Efficacité d'envergure (Cdi/CL²) | excellente | 0,03544 contre 0,03546 sur la référence (0,05 %) |
| Variations de traînée de profil | parfaite, mais pas indépendante | Pearson 1,00 sur les écarts des 8 candidates ; les deux niveaux utilisent les mêmes polaires XFOIL, cette concordance ne valide donc rien |
| Écarts de traînée de 1 à 4 % | bonne | sections de contrôle : +0,90 / +0,99 %, +3,56 / +3,15 %, +4,09 / +4,18 % |
| Portance à attitude donnée | mauvaise | CL 0,716 contre 0,672 (−6 % pour AVL) |
| Yaw de trim à portance égale | corrélé mais décalé | corrélation 0,83 sur les candidates ; AVL demande jusqu'à 0,55° de moins ; sur le run, yaw AVL de 2,96 à 4,41° contre 3,75 à 4,31° pour NPLLT × XFOIL |
| Traînée induite entre sections proches | pas d'accord | Pearson 0,13 sur les écarts |
| Classement des meilleures sections (écarts < 0,3 %) | pas d'accord | Spearman 0,38 |

Le désaccord fin vient de la portance : AVL ne voit que la ligne de cambrure, donc il trime à un autre yaw et répartit la charge autrement. L'incertitude dominante pour le choix d'une section reste le modèle 2D : NeuralFoil et XFOIL divergent de ±0,5 % sur la traînée de profil, et ce run ne dit pas lequel a raison.

## Pour la production

Les outils sont utilisables dès maintenant pour l'exploration et les tendances, pas pour trancher des écarts de moins de \~0,5 %.

**Utilisable dès maintenant :**

- explorer et filtrer des sections sous contraintes de fabrication et de structure ;
- dégager des tendances de l'ordre du % de traînée ;
- étudier la planform et l'attitude (efficacité d'envergure fiable).

**Choix d'outil recommandé :**

- **NPLLT × XFOIL comme modèle de travail** : portance visqueuse, trim cohérent, \~13 s par évaluation contre \~20 s pour AVL ;
- **AVL × XFOIL comme contre-vérification des effets 3D**, pas pour le trim ;
- **NeuralFoil xxxlarge pour le balayage rapide**.

**Pas sans validation externe :**

- tout choix qui repose sur moins de \~0,5 % de traînée, comme dans cette étude ;
- toute valeur absolue de portance ou de yaw tirée d'AVL.

**Avant usage en production, il manque :**

1. Un **arbitre 2D** : CFD (RANS ou MSES) ou essais sur 3 ou 4 sections, pour savoir qui de NeuralFoil ou d'XFOIL a raison à n\_crit 1 et transition forcée.
2. Le **benchmark NPLLT-XFOIL / AVL-XFOIL** sur des planforms plus variées : ici, une seule géométrie et une seule attitude ont été testées.
3. Des **essais de sensibilité aux choix de modélisation**, qui pèsent \~0,1 à 0,3 % chacun :
   - image « mur » de la coque ;
   - correction au pied ;
   - rayon de cœur de NPLLT ;
   - transition forcée contre transition libre.
