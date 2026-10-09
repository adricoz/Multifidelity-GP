"""
[P1] mfego <-> bdToolbox pipeline for foil optimization.

Multi-fidelity Bayesian optimization (mfego NN-MF-EGO) of 2D sections with the bdToolbox / bdFoil
solvers (NeuralFoil levels, bdFoil core XFOIL), and of the section of a fixed 3D planform (C-foil
arc) with the non-planar lifting line (NeuralFoil polars) and bdFoil core AVL (XFOIL polars).
bdToolbox and bdFoil are only read (never modified).

Entry point:  python -m pipelines.bdtoolbox_foil.run --config pipelines/configs/<config>.json
"""
