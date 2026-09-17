"""
Module for vizualisation purposes
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import GaussianProcess


def data_loader(filename: str):
    """
    Load data from a json state file of the Gaussian Process
    Args:
        filename (str): Path to the json file.
    Returns:
        dict: Dictionary containing the data from the json file.
    """
    print("Loading model data...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return {}

if __name__ == "__main__":
    # Example usage
    FILENAME = "ego_backup.json"
  
    data = data_loader(FILENAME)
    if data:
        print("Data loaded successfully.")

    # Extracting data for visualization
    X_dict_1 = np.array(data["X_dict"]["1"]).reshape(-1, 1)
    Y_dict_1 = np.array(data["Y_dict"]["1"])
    X_dict_2 = np.array(data["X_dict"]["2"]).reshape(-1, 1)
    Y_dict_2 = np.array(data["Y_dict"]["2"])
    gp_params = data["gp_params"]

    # Initializing the gaussian process model
    kernel_1 = SquaredExponentialKernel()
    kernel_2 = SquaredExponentialKernel()
    gp_1 = GaussianProcess(kernel=kernel_1)
    gp_2 = GaussianProcess(kernel=kernel_2)

    gp_1.fit(X_dict_1, Y_dict_1)
    gp_2.fit(X_dict_2, Y_dict_2)

    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_pred_1 = []
    y_pred_2 = []
    for x in x_test:
        y_pred_1.append(gp_1.predict(x))
        y_pred_2.append(gp_2.predict(x))
    # Plotting the results
    plt.figure(figsize=(6,6))
    plt.plot(x_test, y_pred_2, label="GP Level 2", color="red")
    plt.scatter(X_dict_2, Y_dict_2, color="red", marker="x", label="Observations Level 2")
    plt.title("Gaussian Process Predictions for Two Fidelity Levels")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("gp_predictions.png", dpi=300)
    plt.close()