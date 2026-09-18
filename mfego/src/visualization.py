"""
Module de visualisation pour le framework MF-EGO.
Permet de recharger un modèle depuis un fichier JSON et de tracer les surfaces de réponse
et la convergence sans ré-entraîner les Processus Gaussiens.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import MultifidelityModel


class ModelVisualizer:
    """
    Class to plot the results of the GP processes
    """
    def __init__(self, json_filepath: str, num_levels: int):
        self.json_filepath = json_filepath
        self.num_levels = num_levels
        self.state = self._load_json()
        self.model = self._rebuild_model_in_memory()

    def _load_json(self) -> dict:
        """Load the state dictionary from the JSON file."""
        with open(self.json_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _rebuild_model_in_memory(self) -> MultifidelityModel:
        """
        Reconstruct the MultifidelityModel and its GP components in memory.
        This method injects the hyperparameters (including the noise) and recalculates
        the inverse matrices without calling .fit()[cite: 18].
        """
        model = MultifidelityModel(self.num_levels, SquaredExponentialKernel)
        model.rhos = self.state.get("rhos", [1.0] * (self.num_levels - 1))

        # Rebuild each GP with the loaded data level by level
        for l in range(self.num_levels):
            level_str = str(l + 1)
            x_train = np.array(self.state["X_dict"][level_str])
            y_train_raw = np.array(self.state["Y_dict"][level_str])

            # Inject gemoetrical hyperparams
            gp_params = np.array(self.state["gp_params"][l])
            model.gps[l].kernel.set_params(gp_params)
            model.gps[l].noise = self.state.get("noises", [1e-6] * self.num_levels)[l]

            if l == 0:
                target_y = y_train_raw
            else: # this is essentially Eq. 13 of the reference article
                rho = model.rhos[l - 1]
                f_prev = np.array([model._predict_up_to(x.reshape(1, -1), l)[0] for x in x_train])
                target_y = y_train_raw - rho * f_prev

            gp = model.gps[l]
            gp.x_train = x_train
            gp.y_train = np.squeeze(target_y)

            # Inverse matrix recalculation
            covariance_matrix = gp.kernel.get_covariance_matrix(gp.x_train) \
                                       + gp.noise * np.eye(len(gp.x_train))
            gp.l_chol = np.linalg.cholesky(covariance_matrix)
            gp.k_inv = np.linalg.solve(gp.l_chol.T,
                                       np.linalg.solve(gp.l_chol,
                                      np.eye(len(gp.x_train))))

        return model

    def plot_convergence(self, save_path: str = "convergence_plot.png", target: float = None) -> None:
        """Best point as a function of the cumulative cost."""
        cost_history = self.state.get("cost_history", [])
        best_y_history = self.state.get("best_y_history", [])

        if not cost_history or not best_y_history:
            print("No convergence history found in the JSON file.")
            return
        if target is not None and not isinstance(target, (int, float)):
            print(f"Invalid target value: {target}. It must be a numeric type.")
            return

        plt.figure(figsize=(10, 6))

        plt.title("Cost vs best HF observation")
        plt.xlabel("Cumulative cost")
        if target is not None:
            plt.step(cost_history, np.abs(best_y_history - target* np.ones_like(best_y_history)), where='post', color='b',
                              linewidth=2, marker='o')
            plt.yscale('log')
            plt.ylabel("Absolute distance to target")
        else:
            plt.step(cost_history, best_y_history, where='post', color='b',
                              linewidth=2, marker='o')
            plt.ylabel("Best HF observation")

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        print(f"Convergence graph saved as : {save_path}")
        plt.close()

    def plot_response_1d(self, grid_size: int = 100, save_path: str = "response_1d.png") -> None:
        """Plot the 1D response of the model with uncertainty bands."""
        dim = self.model.gps[0].x_train.shape[1]
        if dim != 1:
            print(f"Error: The model is not 1D (dim={dim}). This function only supports 1D models.")
            return

        x_test = np.linspace(0, 1, grid_size).reshape(-1, 1)

        y_pred = []
        y_std = []

        for x in x_test:

            f_hat, sigma2_hat, _ = self.model.predict(x.reshape(1, -1))
            y_pred.append(f_hat)
            y_std.append(np.sqrt(max(sigma2_hat, 1e-12)))

        y_pred = np.array(y_pred)
        y_std = np.array(y_std)

        plt.figure(figsize=(10, 6))
        plt.plot(x_test, y_pred, 'b-', label='Predicted Mean (HF)', linewidth=2)
        plt.fill_between(x_test.flatten(), 
                         y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, 
                         alpha=0.2, color='blue', label='Uncertainty (95%)')

        hf_points_x = np.array(self.state["X_dict"][str(self.num_levels)])
        hf_points_y = np.array(self.state["Y_dict"][str(self.num_levels)])
        if hf_points_x.size > 0:
            plt.scatter(hf_points_x, hf_points_y, color='red', 
                        marker='x', s=60, label="Real HF Evaluations", zorder=5)

        plt.title("Prediction of the MF-EGO")
        plt.xlabel("X normalized")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"1D response plot saved as : {save_path}")
        plt.close()

    def plot_response_surface_2d(self, param_x_idx: int = 0, param_y_idx: int = 1, 
                                 grid_size: int = 50, fixed_values: list = None, 
                                 save_path: str = "response_surface_2d.png") -> None:
        """
        Generates a 2D response surface of the model for two specified parameters.
        Other parameters can be fixed at specified values.
        """
        x_vals = np.linspace(0, 1, grid_size)
        y_vals = np.linspace(0, 1, grid_size)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

        dim = self.model.gps[0].x_train.shape[1]
        x_test = np.zeros((grid_size * grid_size, dim))

        if fixed_values is None:
            fixed_values = [0.5] * dim

        for i in range(dim):
            if i == param_x_idx:
                x_test[:, i] = x_mesh.ravel()
            elif i == param_y_idx:
                x_test[:, i] = y_mesh.ravel()
            else:
                x_test[:, i] = fixed_values[i]

        # Predictions
        z_pred = []
        for x in x_test:
            f_hat, _, _ = self.model.predict(x.reshape(1, -1))
            z_pred.append(f_hat)
    
        z_mesh = np.array(z_pred).reshape(x_mesh.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(x_mesh, y_mesh, z_mesh, levels=50, cmap='viridis')
        plt.colorbar(contour, label="Predicted Value (HF)")

        # Hystory of evolution
        print("string", str(self.num_levels))
        hf_points = np.array(self.state["X_dict"][str(self.num_levels)])
        if hf_points.size > 0:
            print(f"HF points shape: {hf_points.shape}")
            plt.scatter(hf_points[param_x_idx], hf_points[param_y_idx],
                        color='red', marker='x', label="HF Evaluations")

        plt.title(f"Response Surface (Dimensions {param_x_idx} & {param_y_idx})")
        plt.xlabel(f"Parameter {param_x_idx}")
        plt.ylabel(f"Parameter {param_y_idx}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        print(f"Surface de réponse 2D sauvegardée sous : {save_path}")
        plt.close()
