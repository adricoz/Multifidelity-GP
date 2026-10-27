# 5e run : le contrôle des changements du 27/10

Le 27/10, de 12 h 24 à 14 h 40 (2 h 16 : la boucle et la comparaison finale se terminent à 2 h, puis viennent la
validation et l'analyse sur la distribution). Fichiers :
[`config.json`](config.json), [`run.log`](run.log),
[`results/summary.txt`](results/summary.txt), [`results/optima.json`](results/optima.json),
[`figures/index.html`](figures/index.html), dont la comparaison avec le 4e run
([`figures/comparison.html`](figures/comparison.html), `scripts/compare_runs.py`).

**Ce qu'on teste.** Les changements du 27/10 ([`../../CHANGEMENTS_2026-10-27.md`](../../CHANGEMENTS_2026-10-27.md)) :
la convergence d'une région et le redémarrage, le choix des lots par prolongement du facteur de Cholesky, la médiane des
sensibilités, le premier solve à départs multiples.

**Les réglages.** Ce sont ceux du 4e run : même famille, même PCA, même distribution supposée, même graine de criblage
(875768254). Le plan de départ est identique (253 BF et 61 HF réussis). S'y ajoutent `--converge 10 0.01 --restarts 2
--min-gain 0.002 --sensitivity-median 3`, avec une limite de 110 min.

## En bref

- **Le résultat égale celui du 4e run en 2 h au lieu de 8 h.** Le meilleur profil fait 13,080 nd, contre 13,084 pour
  celui du 4e run, soit −0,005 nd. Le gain sur le profil actuel vaut +0,157 nd, contre +0,161. Ce sont des vitesses
  corrigées contre NPLLT × XFOIL, en moyenne pondérée sur les 13 conditions résolues pour toutes les sections (72 % de la
  probabilité). L'écart de 0,005 nd est de l'ordre de ce que le 4e run a gagné en 6 h 40 après sa convergence
  (0,012 nd).
- **La région 0 a convergé à 92 min** (33e mise à jour). La règle aurait arrêté le 4e run à 70 min, sur sa propre
  trajectoire.
- **Le redémarrage est revenu vers le même coin de la famille.** La région 1 est partie d'une section à 3,6 de l'optimum,
  prédite plus lente de 0,059 ± 0,003 nd. En 20 min, elle est remontée à 13,065 nd. Elle s'est rapprochée à 1,96 de
  l'optimum de la région 0, avec le même premier score PCA (−4,5). Cela va dans le sens d'un seul bassin, sans le
  prouver : la région 1 n'a eu que 6 mises à jour.
- **Le débit n'a pas gagné dans ce run.** 608 choix en 110 min, contre 688 au 4e run dans le même temps. Le gain du
  prolongement de Cholesky (269 → 97 s par lot à 3 437 points) ne joue que sur un grand plan. Celui-ci est resté sous
  1 100 points : le choix des 16 points y prend environ 32 s dans les deux runs. La mise à jour VPP, qui domine
  désormais, a pris 125 s contre 107 s au 4e run à taille égale. La cause n'est pas établie.

## Les vitesses validées

Vitesse de chaque état, corrigée au premier ordre de l'écart du surrogate final à NPLLT × XFOIL ; moyenne pondérée sur
les 13 conditions résolues pour toutes les sections ([`results/validation.csv`](results/validation.csv)).

| Section | Vitesse | Écart au meilleur du 4e run |
| --- | --- | --- |
| profil actuel | 12,923 nd | −0,161 nd |
| meilleur du 3e run | 13,026 nd | −0,058 nd |
| **meilleur du 4e run** | **13,084 nd** | — |
| **meilleur de ce run (région 0)** | **13,080 nd** | **−0,005 nd** |
| optimum de la région 1 (20 min) | 13,065 nd | −0,019 nd |

**Sur la distribution supposée** ([`results/profile_guess_bimodal.json`](results/profile_guess_bimodal.json), départs
multiples, 72 % de la probabilité résolue pour toutes les sections), le gain sur le profil actuel vaut :
- **+0,146 nd** pour le meilleur de ce run (+0,104 au près, +0,242 au portant) ;
- +0,158 nd pour le meilleur du 4e run ;
- +0,097 nd pour celui du 3e run.

Cette analyse le place 0,011 nd derrière le meilleur du 4e run ; la validation, 0,005 nd.

**Par condition**, le meilleur de ce run est :
- un peu plus lent au portant (−0,01 à −0,03 nd) ;
- plus rapide à 48/16 (+0,027 nd) ;
- égal au près par petit temps.

**Le profil** de la région 0 est très proche de celui du 4e run :
- distance de 1,2 en scores blanchis ;
- Ixx à la contrainte, à 0,06 % près ;
- poids de cambrure moyen 0,084, contre 0,081.

**La précision du surrogate** au meilleur profil :
- 0,0005 nd RMS contre NPLLT × XFOIL ;
- 0,016 nd RMS contre AVL ;
- |z| ≤ 2 sur 100 % des états.

## Ce que montre chaque changement

1. **La convergence et le redémarrage.**
   - Ils ont fonctionné comme prévu : la région 0 a été arrêtée quand ses gains sont tombés sous 0,01 nd pour 10 mises
     à jour.
   - Le redémarrage a choisi la meilleure section éloignée. Sa vitesse prédite (−0,059 ± 0,003 nd) montre que le
     surrogate connaît assez bien la famille loin de l'optimum pour la juger plus lente.
   - La première mise à jour de la région 1 a pris environ 5 min (départs multiples, sensibilités recalculées).
   - La comparaison finale, avec le surrogate final et sur les conditions communes à toutes les sections, a départagé
     les deux optima : 12,832 nd pour la région 0 et 12,824 nd pour la région 1 (78 % de la probabilité).
2. **Les lots par prolongement de Cholesky.** Ce run reste trop petit pour en voir l'effet (point ci-dessus). Les
   choix sont identiques à l'ancienne méthode : vérifié à 3 437 points, avec le même lot de 16, avant le run.
3. **La médiane des sensibilités.**
   - Elle a écarté de 2 à 18 valeurs à la plupart des évaluations, surtout sur les petits coefficients (M_x, M_z,
     proches de 0), et en début de run quand la titulaire bougeait beaucoup.
   - Le journal ne garde que la médiane : on ne peut pas dire si un saut du type de celui du 4e run (∂V/∂F_x doublé)
     s'est produit.
4. **Le premier solve à départs multiples.** La première mise à jour a pris 236 s au lieu de 161 s. La vitesse du profil
   actuel y vaut 12,930 nd, contre 12,926 au 4e run.

## Ce que j'en retiens

- **La règle de convergence est le vrai gain de temps.** Le même résultat est obtenu en 2 h, au lieu de 8 h dont
  6 h 40 sans progrès.
- **Le prolongement de Cholesky sert aux longs runs.** Il évite que le choix des lots redevienne le goulot au-delà
  d'environ 2 000 points.
- **Pour accélérer un run court, il faut maintenant raccourcir la mise à jour VPP.** Par exemple : ne pas résoudre une
  concurrente presque identique à la titulaire, ou réutiliser les états d'une section déjà résolue.
- **Pour trancher la question d'un autre bassin, il faut laisser une région redémarrée converger.** Il faut compter
  environ 1 h 30 par région, donc un run de 3 à 4 h avec 2 redémarrages.
