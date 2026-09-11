import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def squared_exponential_kernel(X1, X2, theta):
    """Calculates the covariance matrix between two sets of points"""
    d = X1.shape[1]
    
    # FIX: Ensure theta has the exact same size as the number of physical dimensions 'd'
    # If the GP outputted 5 thetas but we only have 3 dimensions, we keep the first 3.
    theta_d = theta[:d] 
    
    dist = np.sum(theta_d * (X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
    return np.exp(-dist)

def predict_gp(X_test, X_train, Y_train, theta, noise):
    """Predicts the mean of the Gaussian Process for new points"""
    K = squared_exponential_kernel(X_train, X_train, theta)
    K += np.eye(len(X_train)) * noise
    
    K_s = squared_exponential_kernel(X_test, X_train, theta)
    
    K_inv_Y = np.linalg.solve(K, Y_train)
    mu = K_s.dot(K_inv_Y)
    
    return mu

# ==========================================
# 1. LOADING DATA
# ==========================================
print("Loading model data...")
with open("problem_data/optimization_results.json", "r") as f:
    data = json.load(f)

X_train = np.array(data["training_data"]["level_1"]["X_train"])
Y_train = np.array(data["training_data"]["level_1"]["Y_train"]).reshape(-1, 1)

# Using the correct key based on your current JSON structure
theta = np.array(data["hyperparameters"]["level_1"]["theta"])
noise = data["hyperparameters"]["level_1"]["noise"]

# Number of actual dimensions in the dataset (will be 3)
d = X_train.shape[1]
print(f"Detected {d} dimensions in the training data.")

# ==========================================
# 2. GRID CREATION (SURFACE)
# ==========================================
print("Calculating the response surface (50x50 grid)...")
grid_size = 50
x0 = np.linspace(0, 1, grid_size)
x1 = np.linspace(0, 1, grid_size)
X0_mesh, X1_mesh = np.meshgrid(x0, x1)

# We create X_test with 'd' dimensions
X_test = np.zeros((grid_size * grid_size, d))
X_test[:, 0] = X0_mesh.ravel()
X_test[:, 1] = X1_mesh.ravel()

# If there are more than 2 dimensions (e.g., Alpha), we freeze them at the optimal value
best_idx = np.argmin(Y_train)
best_x = X_train[best_idx]

for i in range(2, d):
    X_test[:, i] = best_x[i]
    print(f"Freezing dimension {i+1} at its optimal value: {best_x[i]:.4f}")

# Prediction
Z_pred = predict_gp(X_test, X_train, Y_train, theta, noise)
Z_mesh = Z_pred.reshape(X0_mesh.shape)

# ==========================================
# 3. VISUALIZATION
# ==========================================
print("Generating the plot...")
fig = plt.figure(figsize=(14, 6))

# --- Plot 1: 3D Surface ---
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X0_mesh, X1_mesh, Z_mesh, cmap='viridis', edgecolor='none', alpha=0.8)

# We only plot the (x0, x1) coordinates of the training points
ax1.scatter(X_train[:, 0], X_train[:, 1], Y_train, color='red', s=50, label='Evaluated Points', zorder=5)

ax1.set_title(f"Response Surface\n(Dim 3+ fixed to optimal)")
ax1.set_xlabel("Param 1 (Normalized)")
ax1.set_ylabel("Param 2 (Normalized)")
ax1.set_zlabel("Objective (Cd + Penalty)")
ax1.view_init(elev=30, azim=-45)

# --- Plot 2: Heatmap (2D Contour) ---
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X0_mesh, X1_mesh, Z_mesh, levels=50, cmap='viridis')
ax2.scatter(X_train[:, 0], X_train[:, 1], color='red', s=30, edgecolors='black', label='Observations')

# Star for the optimal point
ax2.scatter(best_x[0], best_x[1], color='gold', marker='*', s=200, edgecolors='black', label='Optimum Found')

ax2.set_title("Top View (Contour Plot)")
ax2.set_xlabel("Param 1 (Normalized)")
ax2.set_ylabel("Param 2 (Normalized)")
ax2.legend()
fig.colorbar(contour, ax=ax2, label="Objective Value")

plt.tight_layout()
plt.savefig("problem_data/response_surface.png", dpi=300)
plt.show()