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
with open("ego_backup.json", "r") as f:
    data = json.load(f)

X_train_1 = np.array(data["X_dict"]["1"])
Y_train_1 = np.array(data["Y_dict"]["1"]).reshape(-1, 1)
X_train_2 = np.array(data["X_dict"]["2"])
Y_train_2 = np.array(data["Y_dict"]["2"]).reshape(-1, 1)

noise = 1e-6

# Using the correct key based on your current JSON structure
theta = np.array(data["gp_params"][1])

# Number of actual dimensions in the dataset (will be 3)
d = X_train_1.shape[1]
print(f"Detected {d} dimensions in the training data.")

# ==========================================
# 2. GRID CREATION (SURFACE)
# ==========================================
print("Calculating the response surface (50x50 grid)...")
grid_size = 50
x0 = np.linspace(0, 1, grid_size)
X_test = np.linspace(0, 1, grid_size).reshape(-1, 1)

# Prediction
y_pred_1 = predict_gp(X_test, X_train_1, Y_train_1, theta, noise)
y_pred_2 = predict_gp(X_test, X_train_2, Y_train_2, theta, noise)

# ==========================================
# 3. VISUALIZATION
# ==========================================
print("Generating the plot...")
fig = plt.figure(figsize=(14, 6))

plt.figure(figsize=(6,6))
plt.plot(X_test, y_pred_1, label="GP Level 1", color="blue")
plt.scatter(X_train_1, Y_train_1, color="blue", marker="o", label="Observations Level 1")
plt.plot(X_test, y_pred_2, label="GP Level 2", color="red")
plt.scatter(X_train_2, Y_train_2, color="red", marker="x", label="Observations Level 2")
plt.title("Gaussian Process Predictions for Two Fidelity Levels")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("gp_predictions.png", dpi=300)
plt.close()