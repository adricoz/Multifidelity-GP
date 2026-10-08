"""
[P1] mfego <-> bdToolbox pipeline for foil optimization.

Multi-fidelity Bayesian optimization (mfego NN-MF-EGO) of 2D sections with the bdToolbox / bdFoil
solvers (NeuralFoil levels, bdFoil core XFOIL), and templates of 3D appendage backends
(non-planar lifting line, bdFoil core AVL). bdToolbox and bdFoil are only read (never modified).

Entry point:  python -m pipelines.bdtoolbox_foil.run --config pipelines/configs/<config>.json
"""
