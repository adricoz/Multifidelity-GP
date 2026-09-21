# Non-nested Multi fidelity gaussian Processes and surrogates

This repo is all about implementing a Non-Nested Multy Fidelity Gaussian Process (NN-MF-GP) togeter with an improved Efficient Global Optimization (EGO) algorithm based on the article "A Non-Nested Infilling Srategy for Multi-Fidelity Efficient Global Optimisation" by Sacher et al. (2021). It is a first approach to developping a fully fonctionning framework for Fluid Dynamics computations acceleration.

Nevertheless the current state of the project is general enough to be applicable to any appropriate case scenario where the cost of computation is the main decision factor.

The current document explains the structure of the repo and the structure of the code with step by step guide for its usage.

## 📦 Installation & Dependencies

To run the framework and the examples, you need Python 3.8+ and the following modules.

**Core Mathematical Engine:**

* `numpy` (>= 1.20.0): For matrix operations and linear algebra.
* `scipy` (>= 1.7.0): For L-BFGS-B optimization, Latin Hypercube Sampling (QMC), and Root-Finding algorithms.

**Visualization:**

* `matplotlib` (>= 3.4.0): For plotting convergence and 2D/3D response surfaces.

**Physics & Examples (Optional but required for hydrofoil showcase):**

* `neuralfoil` (>= 0.1.0): Aerodynamic/Hydrodynamic solver used as the black-box physics model.
* `aerosandbox` (>= 4.0.0): Used for airfoil generation and geometry handling.

**Installation command:**

```bash
pip install numpy scipy matplotlib neuralfoil aerosandbox
```

## Architecture & Class Methods

The program is entirely Object-Oriented (OOP). This isolates the mathematical purity of the Gaussian Processes from the physical constraints of the simulator.

1. **ExperimentData (Data Management)**

    Acts as the single source of truth for your Design of Experiments (DoE). It safely handles the training data

    * `matrices.generate_initial_design(points_per_level)`: Generates independent Latin Hypercube Samples (LHS) for each fidelity level
    * `is_already_evaluated(level, x)`: Checks if a spatial coordinate x is already present to prevent singular matrices.
    * `add_observation(level, x_new, y_new)`: Safely appends new evaluations to the internal dictionaries.

2. **Simulator (Physical Interface)**

    A completely isolated class mapping the normalized optimization space [0, 1] to physical parameters.

    * `evaluate(design_point, level)`: Translates mathematical vectors into physical shapes, handles root-finding (e.g., enforcing $C_l = 1.0$), and returns the objective value (e.g., $C_d$) or a "death penalty" for unfeasible designs.

3. **MultifidelityModel & Kernel (Math Engine)**

    Handles the non-nested recursive approximation.

    * `MultifidelityModel.fit(data)`: Sequentially optimizes the hyper-parameters ($\theta$, $\rho$, $\sigma_\epsilon$) for all discrepancy GPs by maximizing the log-marginal likelihood 
    * `MultifidelityModel.predict(x_new)`: Returns the mean $\hat{f}(x)$ and variance $\hat{\sigma}^2(x)$ using the Le Gratiet recursive formulation.
    * `Kernel.get_cross_covariance_vector(x, X)`: Optimized, vectorized spatial distance computation.

4. **GOOptimizer (The Controller)**

    Implements an Ask-and-Tell architecture, making it suitable for both fast analytical functions and days-long CFD computations.
    * `ask()`: Fits the model and maximizes the merit function (Eq. 24) to return the optimal next x and level to evaluate, without blocking the code.
    * `tell(x, level, y)`: Ingests the result from an external solver and updates the dataset.
    * `run(n_iterations)`: Automated loop combining ask and tell for fast-evaluating functions.
    * `save_state(filepath)` / `load_state(filepath)`: Exports/Imports the full optimizer state (data and hyperparameters) to JSON, ensuring you never lose progress if a cluster crashes.

5. **ModelVisualizer**
    Standalone class to visualize results without retraining models.
    * `plot_convergence()`: Plots the best high-fidelity observation against the cumulative cost.
    * `plot_response_surface_2d()`: Reloads a JSON state and plots the Gaussian Process contour map.

One can find an illusrtration of the class hierarchy in the following diagramm:

<img src="./charts/mfego_structure.svg" width="70%" alt="mfego structure">$

We also give the general structure of the repo with the main files to consider. 

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
│   │   ├── hartmann_6d.py      # essentially the main 
│   │   ├── Hartmann6d.py       # core Hartmann function
│   │   ├── logfile.log         # example logfile
│   │   └── response_surface_2d.png
│   └── hydrofoil_optim/        # Simple case of a hydrofoil optimization
│   │   ├── convergence_plot.png
│   │   ├── ego_backup.json
│   │   ├── hydrofoil_optim.py  # essentially thge main
│   │   ├── logfile.log             
│   │   ├── optim_neuralfoil.py # function called by the main using neural foil
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

## Quick guide on how to build the process

First thing to do is define the boundaries of the problem and the levels of fidelity and their cost as well as the nomber of initial points per level of fidelity:

```code
L = 2  # Number of fidelity levels
bounds = [(0.0, 1.0)] # 1D: Normalized
costs = [1.0, 10.0]  # Example costs
initial_points = [10, 4]  # Number of points for each fidelity level
```

First class to be implemented is the `ExperimentData` class wich is essentially a Dat-Manager class. It requires the bound and the costs as arguemnts. It is the only class allowed to modyfiy the data set. Thus used troughout the code.

```code
data = ExperimentData(bounds=bounds, costs=costs)
```

Then one needs to implement a child class of the `BaseSimulator` class which is an abstract one. This was done to separate fully the 
physical/real-world case scenarios. The only method required to be implemented is the method `evaluate(self, design_point, level)`. An example is being given bellow with an analytical 2 level of fidelity function:

```python
class FunctionSimulator(BaseSimulator):
        """
        A simple simulator that evaluates a quadratic function with noise.
        Subclass of BaseSimulator
        """
        def evaluate(self, design_point: list, level: int) -> float, dict:
            """
            Evaluate the simulator at a given point and fidelity level.
            This is a placeholder implementation. Replace with actual simulation code.
            """
            # Example: Eqs: (17) of the reference article.
            # It should always deal with exections...
            try:
                x = design_point[0]
                f_1 = 0.5 *(6 * x - 2)**2 * np.sin(12 * x - 4) + 10 * (x - 1)
                if level == 1:
                    return f_1, {"y": f_1}
                if level == 2:
                    f2 = 2 * f_1 - 20* (x -1)
                    return f2, {"y": f2}
                if level > 2:
                    raise ValueError(f"Invalid fidelity level: {level}. Must be 1 or 2.")
                if not isinstance(level, int):
                    raise TypeError(f"Fidelity level must be an integer, got {type(level)}.")

            except (IndexError, TypeError, ValueError) as e:
                logger.error( \
                    "Error evaluating simulator at point %s and level %s: %s", \
                     design_point, level, e)
                return np.nan, {}  # Return NaN to indicate an error in evaluation
```

It is important to state that function evaluate should return a `float` and a `dict`. One is used to optimize the process, but can/should be modified with a `log10()` function to smothen the results and convergence. The `dict` stands for later fitting pupuses in the surrogate to extract real world behaviour of the data.

We can then create an instance of the class:

```code
simu = FunctionSimulator(num_levels=L)
```

We need to create a model instance of the `MultifidelityModel` class which requires a `Kernel` for the covariance computation. Kernel is an abstract class with a child classe `SquaredExponentialKernel` already impelemnted which is exactly what is presented in the reference article. As for now we have only the `MultifidelityModel` class but one could imagine that in the futer we could extend the framework to single fidelity (SF) as well.

```python
model = MultifidelityModel(l=L, kernel_class = SquaredExponentialKernel)
```

The foundations of the optimizer are almost comple. We still need to create an instance of the `AcquisitionFunction` which is nothing else than the Merit function (Eq. 24).

```python
acq = AcquisitionFunction(model=model, data=data)
```

Once all of this has been completed, we can finally give all the pieces to the EGO optimizer (Algorithm 1 in the reference article) which is done trough an instance of the `EGOOptimizer` class:

```python
ego = EGOOptimizer(data=data, model=model, simulator=simu, acquisition=acq)
```

Remaining steps are the initialization of the data with the method:

```python
data.generate_initial_design(points_per_level=initial_points)
```

Which automatically generates a LHS with the given points/dimensions.
A quick non mendatory loop can be implemented to update the dictionnary of the mais function (y=f(x)) evaluations and metrics with initial LHS points since it is not done natively inside the code.

```python
for l in range(1, L + 1):
        y_values = []
        metrics_list = []

        print("Level %s design points: %s", l, data.x_dict[l])

        for x in data.x_dict[l]:
            y_opt, metrics = simu.evaluate(x, level=l)

            y_values.append(y_opt)
            metrics_list.append(metrics)

        data.y_dict[l] = np.array(y_values)
        data.metrics_dict[l] = metrics_list
```

Then one just needs to run the EGO algorithm with the desired amount of iterations. 

```python
_, _ = ego.run(n_iterations = 10)
```

Here we do not recall the outputs of the method since all the hystory has been saved in `"ego_backup.json"` by default.

## Optinnal: Ask/Tell scheme for long computations

```python
# 1st Ask for a point

x_next, l_next = ego.ask()
print(f"Please run CFD for {x_next} at level {l_next}")
ego.save_state("backup.json")
# ... close python, wait 3 days for the cluster to finish ...

# 2nd: Tell the result

ego.load_state("backup.json")
ego.tell(x_evaluated=x_next, level=l_next, y_result=0.015)
```

## Nota-Bene

Troughout the code, we use a logger so one could decide to save all log info in a seperate file siply via:

```python
import logging

logging.basicConfig(
    filename='logfile.log',
    level=logging.INFO,
    format=' %(levelname)s - %(message)s',
    force = True,
    )
logger = logging.getLogger(__name__)
```

