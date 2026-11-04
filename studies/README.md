# Études

Les études de ce dossier (`cfoil_intens_kulfan`, `cfoil_intens_kulfan_trim`, `cfoil_intens_kulfan_trim_2`) ont été
calculées avec l'**ancien moteur** du pipeline : bdFoil core pour AVL et XFOIL, bdFoil/NonPlanarSolver pour la ligne
portante.

Depuis le 27/10/2026, `pipelines/bdtoolbox_foil` repose sur **bdfoil_v1**, le successeur de `foil.py` écrit par
Antoine. Le détail est dans [`../pipelines/README.md`](../pipelines/README.md).

Les scripts de ces études utilisent encore des pièces internes de l'ancien moteur, par exemple `xfoil_spec`,
`bridge.bdfoil_core` et les options `neuralfoil_model` ou `nodes_count`. **Pour les relancer ou reproduire leurs
résultats, il faut partir de l'étiquette git `pre-bdfoil-v1`** :

```
git worktree add ../mfgp_pre_bdfoil_v1 pre-bdfoil-v1
```

Choisis un dossier au chemin court : les caches de polaires dépassent sinon la limite de longueur des chemins de
Windows.

Le fichier `cfoil_intens_kulfan_trim_2/config_cfoil_trim.json` sert encore de base à
`vpp_acceleration/src/foil_codes.py`, qui en remplace les niveaux. Il reste donc valide avec le nouveau moteur.
