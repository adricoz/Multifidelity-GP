import argparse
import numpy as np
import matplotlib.pyplot as plt

# for printing in a logfile 
import argparse
import sys

# Local imports
from Hartmann6d import f_l # Hartman for testing and calibration purpuses
from non_nested_mf_sampling import generate_non_nested_lhs
from non_nested_mf_optimizer import run_non_nested_mf_ego

#from tests.custom_fluid_functions import objective_function
from tests.optim_neuralfoil import objective_function, generate_continuous_naca4
import neuralfoil as nf
import aerosandbox as asb

#imports fopr dealing with files and directories
import os
import json
# ==========================================
# LOGGING CLASS
# ==========================================
class LogTee:
    """Duplicates the flux sys.stdout to write in the console and the logging file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # console output
        self.log.write(message)  # write to log file

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==========================================
# TARGET FUNCTION ADAPTATION
# ==========================================
# def evaluate_fidelity(x, level, L):
#     """
#     Evaluates the Hartmann 6D function at the requested fidelity level.
#     """
#     if not (1 <= level <= L):
#         raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")
    
#     if level == L:
#         # High Fidelity (True function)
#         return f_l(x, deg=6, k=np.inf)
#     else:
#         # Low Fidelity approximations (k matches the level)
#         return f_l(x, deg=6, k=level, delta=0.05)

# def evaluate_fidelity(x_normalized, level, L, target_cl=1.0,coordinates_only=False):
#     """
#     Evaluates the custom Fluid function using normalized inputs [0, 1].
#     """
#     if not (1 <= level <= L):
#         raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")
#     #-----------------------------------------------------------------

#     # scalar extraction
#     # angle of attack
#     lower_bound = -5
#     upper_bound = 15
#     #-----------------------------------------------------------------
#     #naca profile generation
#     # we create our own naca profile
#     camber = int(x_normalized[0] * 7) + 2  # Camber between 1 and 10
#     pos_camber = int(3)
#     thickness = int(x_normalized[1] * 9) + 8  # Thickness between 8 and 17
#     if thickness < 10:
#         naca_string = f"naca{camber:.0f}{pos_camber:.0f}0{thickness:.0f}"
#     else:
#         naca_string = f"naca{camber:.0f}{pos_camber:.0f}{thickness:.0f}"
#     # example naca string: "naca6412"
#     #-----------------------------------------------------------------

#     if coordinates_only:
#                 #print (f"Coordinates (non-normalized) Evaluating at alpha: {np.rad2deg(alpha_phys):.4f} deg, NACA: {naca_string}, Level: {level}/{L}")
#                 cd, cl, alpha = objective_function(naca_string=naca_string, target_cl=target_cl, level=level, L=L)
#                 print (f"Coordinates (non-normalized) of best point for NACA: {naca_string}, Cl: {cl:.4f}, Alpha: {alpha:.4f}, Level: {level}/{L}")
#                 print(f"best value of the drag coefficient : {cd:.4f}")
#     else:
#         cd, cl , _ = objective_function(naca_string=naca_string, target_cl=target_cl, level=level, L=L)
#         return cd + 10*np.abs(cl - target_cl)  # Penalize deviation from target Cl


def evaluate_fidelity(x_normalized, level, L, target_cl=1.0, coordinates_only=False):
    """
    Evaluates a continuous NACA airfoil for the Gaussian Process, 
    but also computes and displays its closest classic "4-digit NACA" equivalent.
    """
    if not (1 <= level <= L):
        raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")

    # Continuous Denormalization (for the GP math) ---
    m_camber = 0.02 + x_normalized[0] * (0.09 - 0.02)    # Exact camber (between 2% and 9%)
    p_position = 0.3                                     # Fixed position of maximum camber (30%)
    t_thickness = 0.08 + x_normalized[1] * (0.17 - 0.08) # Exact thickness (between 8% and 17%)
    #alpha_phys = 0 + x_normalized[2] * (15 - 0)          # Exact angle of attack (between -5° and 15°)
    alpha_phys = 5.00
    # Reconstructing the standard NACA name (for human readability) ---
    # We round to the nearest integer to find the classic NACA equivalent
    camber_int = int(round(m_camber * 100))      # ex 0.0423 -> 4
    pos_int = int(round(p_position * 10))        # ex., 0.3 -> 3
    thick_int = int(round(t_thickness * 100))    # ex, 0.128 -> 13
    
    # Formatting: if thickness < 10, add a leading zero (ex, 09)
    naca_string = f"naca{camber_int}{pos_int}{thick_int:02d}"

    # Highly precise name (optional, to differentiate very similar airfoils in the GP)
    exact_name = f"naca_{m_camber*100:.2f}_{pos_int}_{t_thickness*100:.2f}"

    # Airfoil Object Creation ---
    coords = generate_continuous_naca4(m_camber, p_position, t_thickness)
    custom_naca = asb.Airfoil(name=exact_name, coordinates=coords)

    #  Evaluation and Output ---
    if coordinates_only:
        # Pass the custom_naca object to NeuralFoil
        cd, cl, alpha = objective_function(airfoil_obj=custom_naca,alpha = alpha_phys, level=level, L=L)
        
        print(f"==================================================")
        print(f" BEST AIRFOIL FOUND (Level {level}/{L})")
        print(f"==================================================")
        print(f"Closest standard NACA  : ** {naca_string.upper()} **")
        print(f"Exact values found     : Camber {m_camber*100:.2f}%, Thickness {t_thickness*100:.2f}%")
        print(f"Aerodynamic performance: Cl = {cl:.4f} | Cd = {cd:.4f} | Alpha = {alpha:.2f}°")
        print(f"==================================================")
        
        return cd 
    else:
        cd, cl, _ = objective_function(airfoil_obj=custom_naca, alpha=alpha_phys, level=level, L=L)
        
        # penalty on the target Cl to guide the optimizer
        weight = 10.0
        merit = cd + weight * (cl - target_cl)**2
        return float(np.log10(merit))

# ==========================================
# VISUALIZATION FUNCTION
# ==========================================
def plot_ego_results(X_train, Y_train, L, n_initial_hf, location = "problem_data/ego_convergence_results.png"):
    """
    Generates and saves the analysis plots for the Multi-Fidelity EGO.
    """
    # ensures that dimension is > 1
    if X_train[L].shape[1] < 2:
        print("No plot can be generated for dimensions < 2. Skipping plotting.")
        return
    
    y_hf = Y_train[L]
    best_y = np.minimum.accumulate(y_hf)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- PLOT 1: CONVERGENCE ---
    ax1.plot(best_y, marker='o', linestyle='-', color='darkblue', linewidth=2, label='Best Y observed')
    ax1.axvline(x=n_initial_hf - 1, color='red', linestyle='--', label='End of Initial DoE')
    ax1.set_xlabel("Total number of High-Fidelity evaluations")
    ax1.set_ylabel("Target function value")
    ax1.set_title("EGO Convergence")
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_xlim(1, len(best_y))
   # ax1.set_ylim(best_y[n_initial_hf]*0.9, best_y[n_initial_hf-2]*1.1)
    ax1.legend()
    
    # --- PLOT 2: 2D SPATIAL PROJECTION ---
    dim_a, dim_b = 0, 1 
    
    ax2.scatter(X_train[1][:, dim_a], X_train[1][:, dim_b], 
                color='lightblue', s=50, alpha=0.6, label='Low Fidelity (Level 1)')
    ax2.scatter(X_train[L][:, dim_a], X_train[L][:, dim_b], 
                color='darkblue', marker='*', s=150, edgecolor='white', label=f'High Fidelity (Level {L})')
    
    best_idx = np.argmin(Y_train[L])
    ax2.scatter(X_train[L][best_idx, dim_a], X_train[L][best_idx, dim_b], 
                color='red', marker='*', s=200, edgecolor='black', label='Global Optimum Found')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(f"Dimension {dim_a + 1}")
    ax2.set_ylabel(f"Dimension {dim_b + 1}")
    ax2.set_title("Spatial Distribution (2D Projection)")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(location, dpi=300)
    print(f"\n=> Plot saved as '{location}'.")

# ==========================================
# EXECUTION BLOCK (WITH PARSER)
# ==========================================
if __name__ == "__main__":

    # --- 1. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Run Nested Multi-Fidelity EGO on Hartmann 6D.")
    
    parser.add_argument('--iters', type=int, default=40, 
                        help='Number of EGO iterations to perform (default: 40).')
    
    parser.add_argument('--levels', type=int, default=3, 
                        help='Total number of fidelity levels L (default: 3).')
    
    parser.add_argument('--costs', type=float, nargs='+', default=[1.0, 10.0, 100.0], 
                        help='Costs for each level from lowest to highest. Example: --costs 1 10 100')
    
    parser.add_argument('--points', type=int, nargs='+', default=[60, 20, 8], 
                        help='Number of initial DoE points per level. Example: --points 60 20 8')
    
    parser.add_argument('--logname', type=str, default="problem_data/execution.log", 
                        help='Name of the log file (default: problem_data/execution.log).')

    args = parser.parse_args()
    
    # Redirect stdout to log file
    sys.stdout = LogTee(args.logname)

    # --- 2. SECURITY CHECKS ---
    if len(args.costs) != args.levels:
        parser.error(f"Number of costs provided ({len(args.costs)}) must match the number of levels ({args.levels}).")
    if len(args.points) != args.levels:
        parser.error(f"Number of point counts provided ({len(args.points)}) must match the number of levels ({args.levels}).")
        
    d = 2
    bounds =[(0.0, 1.0) for _ in range(d)]
    
    print("\n==========================================")
    print("   EXPERIMENT CONFIGURATION")
    print("==========================================")
    print(f" - Iterations : {args.iters}")
    print(f" - Levels (L) : {args.levels}")
    print(f" - Costs      : {args.costs}")
    print(f" - Initial DoE: {args.points} points")
    print("==========================================\n")
    
    # --- 3. INITIALIZATION ---
    print("Generating the initial Non-Nested Latin Hypercube Design...")
    X_nested = generate_non_nested_lhs(d, args.points)
    
    X_train = {}
    Y_train = {}
    
    # In this case it is simply the Hartmann function, but we can easily adapt it to any other function
    def target_function(x, level):
        return evaluate_fidelity(x, level, L=args.levels)
    
    # setting up the training data for each level
    for i, x_lvl in enumerate(X_nested):
        l = i + 1 
        X_train[l] = x_lvl
        print(f"Evaluating initial Level {l} ({x_lvl.shape[0]} points)...")
        Y_train[l] = np.array([target_function(x, l) for x in x_lvl])
        
    print(f"\nInitial best High-Fidelity point: {np.min(Y_train[args.levels]):.4f}")
    
    # --- 4. RUN OPTIMIZATION ---
    print("\n==========================================")
    print("   STARTING MULTI-FIDELITY EGO")
    print("==========================================")
    
    X_train_final, Y_train_final, thetas, rhos, noises = run_non_nested_mf_ego(
        X_train=X_train, 
        Y_train=Y_train, 
        L=args.levels, 
        costs=args.costs, 
        bounds=bounds, 
        n_iterations=args.iters, 
        true_function=target_function
    )

    print("\n==========================================")
    print("   OPTIMIZATION COMPLETED")
    print("==========================================")
    print(f"Absolute best point found: {np.min(Y_train_final[args.levels]):.8f}")
    #print (f"Coordinates (normalized) of the best point found: {X_train_final[args.levels][np.argmin(Y_train_final[args.levels])]}")
    best_coord = X_train_final[args.levels][np.argmin(Y_train_final[args.levels])]
    evaluate_fidelity(best_coord, level=args.levels, L=args.levels, coordinates_only=True)

    #writing the bests accumulated points to a text file independent of the dimension of the problem
    #createing the directory if it does not exist

    if not os.path.exists("problem_data"):
        os.makedirs("problem_data")

    with open("problem_data/best_points.txt", "w") as f:
        f.write("Best points found during the optimization:\n")

        for l in range(1, args.levels + 1):
            best_idx = np.argmin(Y_train_final[l])
            best_x = X_train_final[l][best_idx]
            best_y = Y_train_final[l][best_idx]
            f.write(f"Level {l}: Best X = {np.array(best_x).round(4).tolist()}, Best Y = {best_y:.8f}\n")
    # writing the hyperparameters of the problesm to a text file
    with open("problem_data/hyperparameters.txt", "w") as f:
        f.write("Hyperparameters of the Gaussian Process:\n")
        
        for l in range(args.levels):
            rho_val = rhos[l] if l < len(rhos) else "N/A"
            f.write(f"Level {l+1}: Theta = {thetas[l]}, Rho = {rho_val}, Noise = {noises[l]}\n")
    
    # json files for python
    export_data = {
        "best_points": {},
        "hyperparameters": {},
        "training_data": {}
    }
    
    for l in range(1, args.levels + 1):
        best_idx = np.argmin(Y_train_final[l])
        export_data["best_points"][f"level_{l}"] = {
            "x": np.array(X_train_final[l][best_idx]).tolist(),
            "y": float(Y_train_final[l][best_idx])
        }

    for l in range(args.levels):
        export_data["hyperparameters"][f"level_{l+1}"] = {
            "theta": np.array(thetas[l]).tolist(),
            "noise": float(noises[l]),
            "rho": float(rhos[l]) if l < len(rhos) else None
        }

    for l in range(1, args.levels + 1):
        export_data["training_data"][f"level_{l}"] = {
            "X_train": np.array(X_train_final[l]).tolist(),
            "Y_train": np.array(Y_train_final[l]).tolist()
        }

    with open("problem_data/optimization_results.json", "w") as f:
        json.dump(export_data, f, indent=4)

    print("\n=> Best points and hyperparameters saved to TXT (for reading) and JSON (for reloading).")


    # --- 5. VISUALIZATION ---
    print("\nGenerating final plots...")
    n_initial_hf = args.points[-1]
    plot_ego_results(X_train_final, Y_train_final, args.levels, n_initial_hf)



