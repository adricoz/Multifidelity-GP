# Non-Nested Multi-Fidelity Efficient Global Optimization (MF-EGO)

This repository contains the Python implementation of the **Non-Nested** Multi-Fidelity Efficient Global Optimization (MF-EGO) algorithm, based on the work of Mathieu Sacher et al. (2021) 

Unlike the nested approach, which requires High-Fidelity evaluations to be strictly accompanied by Low-Fidelity evaluations at the exact same coordinates ($X^{HF} \subset X^{LF}$), this **Non-Nested** version makes the fidelity levels completely independent. 

## New Features & Recent Changes (Non-Nested Transition)

The codebase has undergone a major refactoring to remove the geometric constraints of the previous model and ensure numerical stability:

### 1. Independent Design of Experiments (LHS)
* **Removal of Nested LHS**: Initial points are now generated completely independently for each fidelity level using the new `generate_non_nested_lhs` function (utilizing `scipy.stats.qmc.LatinHypercube`). 
* **Advantage**: Allows the injection of arbitrary historical datasets without having to run new simulations to fill spatial "holes".

### 2. Mean Field Approximation (Mean Prediction)
* **Removal of strict extraction**: The algorithm no longer looks for exact spatial observations in the lower-level dataset (`extract_subpart_vector` has been removed).
* **Implementation of cross-prediction**: Missing values are dynamically replaced by the mathematical expectation (mean) predicted by the lower-level Gaussian Process at the target coordinates (`predict_mf_mean_up_to_level`).

### 3. Merit Function & Targeted Evaluation
* **Merit Function Update**: Implementation of `merit_non_nested` (Equation 24 from Sacher et al. 2021). Variance reduction is evaluated exclusively at the candidate level, and the cost ratio is strictly individual.
* **Single Evaluation**: During step 3 of the EGO loop, **only the level selected by the merit function is evaluated**. The cascading evaluation down to lower levels has been removed.

### 4. Numerical Stability ("Anti-Broadcasting")
* The code is now fully protected against NumPy linear algebra dimension mismatch errors (broadcasting issues).
* Systematic use of `np.squeeze()` and `np.dot()` in `log_likelihood_mf` and prediction functions to guarantee strict vector multiplications, regardless of the dimension or size of the dataset.

---

##  Project Structure

* `main.py`: Main script serving as the entry point. Handles LHS initialization, parameter normalization, and the orchestration loop.
* `non_nested_mf_optimizer.py`: Core of the algorithm. Contains hyperparameter inference, merit maximization, and non-nested recursive inference (`predict_non_nested_mf`).
* `nested_mf_covariance.py`: Definition of the optimized and vectorized covariance kernels (Squared Exponential) for matrix computations.
* `custom_fluid_functions.py`: "Black-box" simulator (based on Faltinsen equations) evaluating the hydrodynamic drag of the foil.

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
--iters   # Number of EGO iterations
--levels  # Total number of fidelity levels L.
--costs   # The cost of each level (space-separated, from lowest to highest).
--points  # The initial DoE size for each level (space-separated, from lowest to highest).
--logname # Name of the log file to save console outputs.
```

**Example of usage**
```bash
python main.py --iters 5 --levels 3 --costs 1 2 3 --points 5 5 5 --logname test_nn.log
```
**Help Menu**
To see all available arguments and their descriptions:
```bash
python main.py -h
```
