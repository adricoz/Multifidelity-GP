import argparse
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from Hartmann6d import f_l
from nested_mf_sampling import generate_nested_lhs
from nested_mf_optimizer import run_nested_mf_ego

# ==========================================
# TARGET FUNCTION ADAPTATION
# ==========================================
def evaluate_fidelity(x, level, L):
    """
    Evaluates the Hartmann 6D function at the requested fidelity level.
    """
    if not (1 <= level <= L):
        raise ValueError(f"Error: Fidelity level {level} is not defined. Must be between 1 and {L}.")
    
    if level == L:
        # High Fidelity (True function)
        return f_l(x, deg=6, k=np.inf)
    else:
        # Low Fidelity approximations (k matches the level)
        return f_l(x, deg=6, k=level, delta=0.05)

# ==========================================
# VISUALIZATION FUNCTION
# ==========================================
def plot_ego_results(X_train, Y_train, L, n_initial_hf):
    """
    Generates and saves the analysis plots for the Multi-Fidelity EGO.
    """
    y_hf = Y_train[L]
    best_y = np.minimum.accumulate(y_hf)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- PLOT 1: CONVERGENCE ---
    ax1.plot(best_y, marker='o', linestyle='-', color='darkblue', linewidth=2, label='Best Y observed')
    ax1.axvline(x=n_initial_hf - 1, color='red', linestyle='--', label='End of Initial DoE')
    ax1.set_xlabel("Total number of High-Fidelity evaluations")
    ax1.set_ylabel("Target function value")
    ax1.set_title("EGO Convergence (Hartmann 6D)")
    ax1.grid(True, linestyle=':', alpha=0.7)
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
    plt.savefig("ego_convergence_results.png", dpi=300)
    print("\n=> Plot saved as 'ego_convergence_results.png'.")
    plt.show()

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

    args = parser.parse_args()
    
    # --- 2. SECURITY CHECKS ---
    if len(args.costs) != args.levels:
        parser.error(f"Number of costs provided ({len(args.costs)}) must match the number of levels ({args.levels}).")
    if len(args.points) != args.levels:
        parser.error(f"Number of point counts provided ({len(args.points)}) must match the number of levels ({args.levels}).")
        
    d = 6
    bounds = [(0.0, 1.0) for _ in range(d)]
    
    print("\n==========================================")
    print("   EXPERIMENT CONFIGURATION")
    print("==========================================")
    print(f" - Iterations : {args.iters}")
    print(f" - Levels (L) : {args.levels}")
    print(f" - Costs      : {args.costs}")
    print(f" - Initial DoE: {args.points} points")
    print("==========================================\n")
    
    # --- 3. INITIALIZATION ---
    print("Generating the initial Nested Latin Hypercube Design...")
    X_nested = generate_nested_lhs(d, args.points)
    
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
    
    X_train_final, Y_train_final, thetas, rhos, noises = run_nested_mf_ego(
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
    print(f"Absolute best HF point found: {np.min(Y_train_final[args.levels]):.4f}")
    
    # --- 5. VISUALIZATION ---
    print("\nGenerating final plots...")
    n_initial_hf = args.points[-1]
    plot_ego_results(X_train_final, Y_train_final, args.levels, n_initial_hf)



