# Nested Multi-Fidelity Efficient Global Optimization (MF-EGO)

This repository implements a Nested Multi-Fidelity Efficient Global Optimization (EGO) algorithm based on Le Gratiet's recursive Gaussian Process scheme. It is designed to minimize expensive black-box functions in complex dimensions (tested here on the 6D Hartmann function) by exploiting multiple fidelity levels ($X_L \subset X_{L-1} \subset \dots \subset X_1$).

## Project Architecture

The code is fully modular and divided into 5 distinct files to separate the application case (Hartmann), data management, mathematical kernels, optimizer core, and the execution script.

### 1. `main.py` (Entry Point and Interface)
This file orchestrates the experiment. It parses terminal arguments, initializes the Design of Experiments (DoE), and runs the EGO loop.
*   **`evaluate_fidelity(x, level, L)`**: A "bridge" function that translates the fidelity level requested by the EGO algorithm into parameters understood by your simulator (here `f_l`).
*   **`plot_ego_results(X_train, Y_train, L, n_initial_hf)`**: Generates and saves diagnostic plots at the end of the run (convergence curve and 2D spatial projection of the evaluations).

### 2. `nested_mf_optimizer.py` (Algorithm Core)
Contains the Bayesian Optimization intelligence and the EGO loop itself.
*   **`run_nested_mf_ego(...)`**: The main loop. At each iteration, it (1) trains the Gaussian Process hyperparameters, (2) searches for the point maximizing the merit function via a genetic algorithm, and (3) evaluates the true function to update the datasets.
*   **`log_likelihood_mf(...)`**: Computes the marginal log-likelihood of the residuals. This is minimized to find the optimal hyperparameters ($\Theta, \rho, \sigma_\epsilon$) for each discrepancy model.
*   **`predict_base_gp(...)`**: Performs standard kriging for a single Gaussian Process (modeling the discrepancy $\Delta Y^{(l)}$ between two levels).
*   **`predict_nested_mf(...)`**: Applies Le Gratiet's recursive formulas from $l=1$ to $L$ to fuse the discrepancy models and predict the final High-Fidelity mean and variance.
*   **`merit_nested(...)`**: Implements the acquisition function (Merit). It combines the AEI, the cumulative cost ratio, and the expected variance reduction ratio.
*   **`aei_multi_fidelity(...)`**: Computes the Augmented Expected Improvement (AEI), which is the classic EI penalized by the model's local uncertainty (noise).
*   **`expected_improvement(...)`**: Computes the standard Expected Improvement (EI) based on the normal distribution.

### 3. `nested_mf_covariance.py` (Kernels and Matrices)
Groups the mathematical functions handling spatial covariance.
*   **`Cov_fct(x, y, l, t1, t2)`**: The Gaussian covariance kernel (Squared Exponential). Computes the spatial correlation between two points $x$ and $y$ based on the length scales (`l`).
*   **`base_covariance_matrix(X, Theta_l)`**: Builds the full $N \times N$ covariance matrix for a dataset $X$ associated with a single fidelity level.
*   **`k_l_vector(...)`**: Builds the cross-covariance vector between a new candidate point and the existing training points.
*   **`nested_mf_covariance(...)` / `nested_mf_covariance_matrix(...)`**: *(Historical)* Functions calculating the cumulative recursive covariance across all levels.

### 4. `nested_mf_sampling.py` (Design of Experiments Management)
Contains tools to manipulate the nested data structure.
*   **`generate_nested_lhs(d, n_levels_points)`**: Generates a strictly nested Latin Hypercube Sampling (LHS) design, ensuring that High-Fidelity points are an exact subset of Low-Fidelity points.
*   **`Delta_Y_l(Y_l, Y_l_minus_1, rho_l_minus_1)`**: Computes the residual (discrepancy) between level $l$ observations and level $l-1$ predictions adjusted by the correlation coefficient $\rho$.
*   **`extract_subpart_vector(X_higher, X_lower, Y_lower, tol)`**: Searches the level $l-1$ database to extract the $Y$ values corresponding exactly to the spatial coordinates of the points evaluated at level $l$.
*   **`is_already_evaluated(x, X_train_level, tol)`**: Checks if a candidate point has already been evaluated at a specific level in the past to avoid rerunning an expensive calculation unnecessarily.

### 5. `Hartmann6d.py` (Application Case)
The simulator (black box) used to validate the algorithm.
*   **`Hartmann6D(x)`**: The true analytical function in 6 dimensions (features 6 local minima and a global minimum of -3.322).
*   **`f_l(x, deg, k, delta)`**: The "degradable" version of the Hartmann function. The lower the parameter $k$, the rougher the approximation of reality.

---

## Basic commands

## Usage (CLI)

The project is executed from the terminal via `main.py`. The default parameters perform a run on 3 fidelity levels with 40 EGO iterations.

**Install the required dependencies:**
```bash
pip install numpy scipy matplotlib
```
**Running the EGO Algorithm**
```bash
python main.py
```
**Advanced Configuration**
```bash
--iters  # Number of EGO iterations
--levels # Total number of fidelity levels L.
--costs  # The cost of each level (space-separated, from lowest to highest).
--points # The initial DoE size for each level (space-separated, from lowest to highest).
```

**Example of usage**
```bash
python main.py --iters 50 --levels 5 --costs 1 5 20 50 100 --points 80 40 20 12 7
```
**Help Menu**
To see all available arguments and their descriptions:
```bash
python main.py -h
```

