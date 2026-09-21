# Non-nested Multi fidelity gaussian Processes and surrogates

This repo is all about implementing a Non-Nested Multy Fidelity Gaussian Process (NN-MF-GP) togeter with an improved Efficient Global Optimization (EGO) algorithm based on the article "A Non-Nested Infilling Srategy for Multi-Fidelity Efficient Global Optimisation" by Sacher et al. (2021). It is a first approach to developping a fully fonctionning framework for Fluid Dynamics computations acceleration.

Nevertheless the current state of the project is general enough to be applicable to any appropriate case scenario where the cost of computation is the main decision factor.

The current document explains the structure of the repo and the structure of the code with step by step guide for its usage.

## 📁 Multifidelity-GP - Project Structure

```text
Multifidelity-GP/  
├──  **.gitignore**
├── charts/
│   └── 🖼️ relations.png
├── example/
│   ├── hartmann_6d/            # showcase of the convergence of the algo. with Hartmann 6D function
│   │   ├── convergence_plot.png
│   │   ├── ego_backup.json
│   │   ├── hartmann_6d.py
│   │   ├── Hartmann6d.py
│   │   ├── logfile.log
│   │   └── response_surface_2d.png
│   └── hydrofoil_optim/        # Simple case of a hydrofoil optimization
│   │   ├── convergence_plot.png
│   │   ├── ego_backup.json
│   │   ├── hydrofoil_optim.py
│   │   ├── logfile.log
│   │   ├── optim_neuralfoil.py
│   │   └── response_surface_2d.png
├── 📂legacy/                   # History of the project with first functions ...
│   ├── legacy_nested/
|   |   └── ...
│   ├── legacy_non_nested/
│   │   └── ...
├── 📂 mfego/                   # Project's main directory
│   ├── convergence_plot.png
│   ├── ego_backup.json
│   ├── logfile.log
│   ├── main.py
│   ├── response_1d.png
│   └── 📁 src/                 # Source code 
│   │   ├── 📄 acquisition.py
│   │   ├── 📄 data_management.py
│   │   ├── 📄 kernels.py
│   │   ├── 📄 optimizer.py
│   │   ├── 📄 simulator.py
│   │   ├── 📄 surrogate_models.py
│   │   └── 📄 visualization.py
├── 📖 **README.md**
├── 📂 sandbox/                 # Theoretical case with a simple Single-fidelity GP
│   └── 📄 gaussian_process_singlefidelity_sandbox.ipynb
```

The programm is object oriented (POO) with multiple classes and a hierarchy being the basis of the gaussian process and the EGO algorithm. The following diagramm gives some insights on the relations between the classes.

<img src="./charts/mfego_structure.svg" width="70%" alt="mfego structure">

## Quick guide on how to build the process

First thing to do is define the boundaries of the problem and the levels of fidelity and their cost as well as the nomber of initial points per level of fidelity:

```code
L = 2  # Number of fidelity levels
bounds = [(0.0, 1.0)] # 1D: Normalized
costs = [1.0, 10.0]  # Example costs
initial_points = [10, 4]  # Number of points for each fidelity level
```

First class to be implemented is the ExperimentData class wich is essentially a Dat-Manager class. It requires the bound and the costs as arguemnts. It is the only class allowed to modyfiy the data set. Thus used troughout the code.

```code
data = ExperimentData(bounds=bounds, costs=costs)
```

Then one needs to implement a child class of the Simulator class which is an abstract one. This was done to separate fully the 
physical/real-world case scenarios. The only method required to be implemented is the method <!-- evaluate(self, design_point, level) -->