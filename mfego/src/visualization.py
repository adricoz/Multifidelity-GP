"""
Visualization module of the MF-EGO framework.
Reloads a trained model from a JSON file and plots the response surfaces and the convergence
without re-training the Gaussian Processes.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from src.kernels import SquaredExponentialKernel
from src.surrogate_models import MultifidelityModel

# [FIX-X3] markers/colors used for the observations of each fidelity level
LEVEL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
LEVEL_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',
                'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']


class ModelVisualizer:
    """
    Class to plot the results of the GP processes
    """
    def __init__(self, json_filepath: str, num_levels: int = None):
        self.json_filepath = json_filepath
        self.state = self._load_json()
        # [FIX-X3] num_levels is read from the JSON when available (argument kept for
        # compatibility with the previous signature and the old JSON files)
        self.num_levels = int(self.state.get("num_levels") or num_levels
                              or len(self.state["X_dict"]))
        self.model = self._rebuild_model_in_memory()
        dim = self.model.gps[0].x_train.shape[1]
        # [FIX-X3] plotting domain = problem bounds (was hard-coded to [0, 1])
        self.bounds = [tuple(b) for b in self.state.get("bounds") or [(0.0, 1.0)] * dim]

    def _load_json(self) -> dict:
        """Load the state dictionary from the JSON file."""
        with open(self.json_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _rebuild_model_in_memory(self) -> MultifidelityModel:
        """
        Reconstruct the MultifidelityModel and its GP components in memory.
        This method injects the hyperparameters (including the noise) and recalculates
        the inverse matrices without calling .fit().
        [FIX-X3] Uses the exact snapshot of the trained model ("surrogate" entry, same code
        as load_surrogate) so that the plots show the model that was actually trained. The
        previous reconstruction (data + hyperparameters of the previous fit) is kept for old
        JSON files only.
        """
        if self.state.get("surrogate"):
            return MultifidelityModel.from_dict(self.state["surrogate"])

        model = MultifidelityModel(self.num_levels, SquaredExponentialKernel)
        model.rhos = self.state.get("rhos", [1.0] * (self.num_levels - 1))

        # Rebuild each GP with the loaded data level by level
        for l in range(self.num_levels):
            level_str = str(l + 1)
            x_train = np.atleast_2d(np.array(self.state["X_dict"][level_str], dtype=float))
            y_train_raw = np.array(self.state["Y_dict"][level_str], dtype=float)
            # [FIX-R4] failed evaluations are not part of the training data
            valid = np.isfinite(y_train_raw)
            x_train, y_train_raw = x_train[valid], y_train_raw[valid]
            model.train_data[l + 1] = (x_train, y_train_raw)

            # Inject the kernel hyperparameters
            gp_params = np.array(self.state["gp_params"][l])
            model.gps[l].kernel.set_params(gp_params)
            model.gps[l].noise = self.state.get("noises", [1e-6] * self.num_levels)[l]

            if l == 0:
                target_y = y_train_raw
            else: # this is essentially Eq. 13 of the reference article
                rho = model.rhos[l - 1]
                f_prev = model.predict_batch(x_train, level=l)[0]
                target_y = y_train_raw - rho * f_prev

            # [FIX-N4] same factorization code as the training (GaussianProcess.condition)
            model.gps[l].condition(x_train, target_y)

        return model

    # ---------------------------------------------------------------------------------------
    # [FIX-X3] helpers shared by the static (matplotlib) and interactive (plotly) plots
    # ---------------------------------------------------------------------------------------
    def _level_label(self, level: int) -> str:
        """Readable name of a fidelity level (coherent for a single-fidelity model)."""
        if self.num_levels == 1:
            return "Single fidelity"
        if level == self.num_levels:
            return f"HF (level {level})"
        return f"Level {level}"

    def _observations(self, level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Valid (x, y) observations of a level and the x of its failed evaluations."""
        x = np.atleast_2d(np.array(self.state["X_dict"].get(str(level), []), dtype=float))
        y = np.array(self.state["Y_dict"].get(str(level), []), dtype=float).reshape(-1)
        if y.size == 0:
            return np.empty((0, self.model.gps[0].x_train.shape[1])), y, y
        valid = np.isfinite(y)
        return x[valid], y[valid], x[~valid]

    def _best_observed_point(self) -> np.ndarray:
        """Best observed point of the highest level (center of the domain if none)."""
        x, y, _ = self._observations(self.num_levels)
        if y.size == 0:
            return np.array([(lo + hi) / 2.0 for lo, hi in self.bounds])
        return x[int(np.argmin(y))]

    def _convergence_data(self, target: float = None) -> tuple[np.ndarray, np.ndarray, str]:
        """Cost history and best value (or distance to target) history."""
        cost_history = np.array(self.state.get("cost_history", []), dtype=float)
        best_y_history = np.array(self.state.get("best_y_history", []), dtype=float)
        if target is None:
            return cost_history, best_y_history, "Best HF observation"
        # [FIX-X3] tiny floor so that a zero distance does not break the log scale
        distance = np.maximum(np.abs(best_y_history - target), 1e-16)
        return cost_history, distance, "Absolute distance to target"

    def _grid_1d(self, grid_size: int) -> np.ndarray:
        """1D grid over the problem bounds."""
        return np.linspace(self.bounds[0][0], self.bounds[0][1], grid_size).reshape(-1, 1)

    def _grid_2d(self, param_x_idx: int, param_y_idx: int, grid_size: int,
                 fixed_values: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D grid over the bounds of two parameters, the other ones being fixed."""
        x_vals = np.linspace(*self.bounds[param_x_idx], grid_size)
        y_vals = np.linspace(*self.bounds[param_y_idx], grid_size)
        x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

        dim = self.model.gps[0].x_train.shape[1]
        if len(fixed_values) != dim:
            raise ValueError(f"fixed_values must have {dim} components, got {len(fixed_values)}.")
        x_test = np.tile(np.asarray(fixed_values, dtype=float), (grid_size * grid_size, 1))
        x_test[:, param_x_idx] = x_mesh.ravel()
        x_test[:, param_y_idx] = y_mesh.ravel()
        return x_mesh, y_mesh, x_test

    # ---------------------------------------------------------------------------------------
    # Static plots (matplotlib, PNG)
    # ---------------------------------------------------------------------------------------
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

        cost, values, ylabel = self._convergence_data(target)
        plt.figure(figsize=(10, 6))

        plt.title("Cost vs best HF observation")
        plt.xlabel("Cumulative cost")
        plt.step(cost, values, where='post', color='b', linewidth=2, marker='o')
        if target is not None:
            plt.yscale('log')
        plt.ylabel(ylabel)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        print(f"Convergence graph saved as : {save_path}")
        plt.close()

    def plot_response_1d(self, grid_size: int = 100, save_path: str = "response_1d.png",
                         show_levels: bool = True) -> None:
        """Plot the 1D response of the model with uncertainty bands.
        [FIX-X3] grid over the bounds, observations of every level, optional surrogate of
        each intermediate level (recursive formulation)."""
        dim = self.model.gps[0].x_train.shape[1]
        if dim != 1:
            print(f"Error: The model is not 1D (dim={dim}). This function only supports 1D models.")
            return

        x_test = self._grid_1d(grid_size)
        y_pred, y_var, _ = self.model.predict_batch(x_test)
        y_std = np.sqrt(np.maximum(y_var, 1e-12))

        plt.figure(figsize=(10, 6))
        top_label = self._level_label(self.num_levels)
        plt.plot(x_test, y_pred, 'b-', label=f'Predicted mean - {top_label}', linewidth=2)
        plt.fill_between(x_test.flatten(),
                         y_pred - 1.96 * y_std, y_pred + 1.96 * y_std,
                         alpha=0.2, color='blue', label='Uncertainty (95%)')

        if show_levels:
            for l in range(1, self.num_levels):
                mean_l = self.model.predict_batch(x_test, level=l)[0]
                plt.plot(x_test, mean_l, '--', color=LEVEL_COLORS[(l - 1) % 8],
                         label=f"Predicted mean - {self._level_label(l)}")

        for l in range(1, self.num_levels + 1):
            x_obs, y_obs, _ = self._observations(l)
            if y_obs.size > 0:
                plt.scatter(x_obs[:, 0], y_obs, color=LEVEL_COLORS[(l - 1) % 8],
                            marker=LEVEL_MARKERS[(l - 1) % 8], s=50, zorder=5,
                            edgecolors='k', label=f"Evaluations - {self._level_label(l)}")

        plt.title("Prediction of the MF-EGO")
        plt.xlabel("x")
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
        [FIX-X3] default fixed values = best observed HF point (was 0.5 for every parameter).
        """
        if fixed_values is None:
            fixed_values = self._best_observed_point()
        x_mesh, y_mesh, x_test = self._grid_2d(param_x_idx, param_y_idx, grid_size,
                                               fixed_values)

        # Predictions
        # [FIX-N6] one vectorized prediction for the whole grid
        z_mesh = self.model.predict_batch(x_test)[0].reshape(x_mesh.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(x_mesh, y_mesh, z_mesh, levels=50, cmap='viridis')
        plt.colorbar(contour, label=f"Predicted value - {self._level_label(self.num_levels)}")

        # Hystory of evolution
        for l in range(1, self.num_levels + 1):
            points, _, _ = self._observations(l)
            if points.size > 0:
                # [FIX-X3] columns of the parameters (was points[idx] = rows, wrong points)
                plt.scatter(points[:, param_x_idx], points[:, param_y_idx],
                            color=LEVEL_COLORS[(l - 1) % 8], marker=LEVEL_MARKERS[(l - 1) % 8],
                            edgecolors='k', label=f"Evaluations - {self._level_label(l)}")

        plt.title(f"Response Surface (Dimensions {param_x_idx} & {param_y_idx})")
        plt.xlabel(f"Parameter {param_x_idx}")
        plt.ylabel(f"Parameter {param_y_idx}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        print(f"Response surface saved as : {save_path}")
        plt.close()

    # ---------------------------------------------------------------------------------------
    # [FIX-X3] Interactive plots (plotly, HTML). plotly is an optional dependency.
    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _plotly():
        """Lazy import of plotly (only needed for the interactive plots)."""
        try:
            import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel
            from plotly.subplots import make_subplots  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise ImportError("Interactive plots require plotly: pip install plotly") from error
        return go, make_subplots

    def plot_convergence_interactive(self, save_path: str = "convergence_plot.html",
                                     target: float = None,
                                     include_plotlyjs: str = "cdn") -> None:
        """Interactive version of plot_convergence (HTML file)."""
        go, _ = self._plotly()
        cost, values, ylabel = self._convergence_data(target)
        if cost.size == 0:
            print("No convergence history found in the JSON file.")
            return
        fig = go.Figure(go.Scatter(x=cost, y=values, mode="lines+markers",
                                   line={"shape": "hv"}, name=ylabel))
        fig.update_layout(title="Cost vs best HF observation", xaxis_title="Cumulative cost",
                          yaxis_title=ylabel, yaxis_type="log" if target is not None else None,
                          template="plotly_white")
        fig.write_html(save_path, include_plotlyjs=include_plotlyjs)
        print(f"Interactive convergence graph saved as : {save_path}")

    def plot_response_1d_interactive(self, grid_size: int = 200,
                                     save_path: str = "response_1d.html",
                                     include_plotlyjs: str = "cdn") -> None:
        """Interactive version of plot_response_1d (HTML file): mean, 95% band, surrogate of
        each level and observations of every level."""
        go, _ = self._plotly()
        dim = self.model.gps[0].x_train.shape[1]
        if dim != 1:
            print(f"Error: The model is not 1D (dim={dim}). This function only supports 1D models.")
            return
        x_test = self._grid_1d(grid_size)
        y_pred, y_var, _ = self.model.predict_batch(x_test)
        y_std = np.sqrt(np.maximum(y_var, 1e-12))
        x_flat = x_test[:, 0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([x_flat, x_flat[::-1]]),
                                 y=np.concatenate([y_pred + 1.96 * y_std,
                                                   (y_pred - 1.96 * y_std)[::-1]]),
                                 fill="toself", fillcolor="rgba(31,119,180,0.2)",
                                 line={"width": 0}, name="Uncertainty (95%)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_flat, y=y_pred, mode="lines", line={"color": "#1f77b4"},
                                 name=f"Predicted mean - {self._level_label(self.num_levels)}"))
        for l in range(1, self.num_levels):
            fig.add_trace(go.Scatter(x=x_flat, y=self.model.predict_batch(x_test, level=l)[0],
                                     mode="lines", line={"dash": "dash"},
                                     name=f"Predicted mean - {self._level_label(l)}"))
        for l in range(1, self.num_levels + 1):
            x_obs, y_obs, _ = self._observations(l)
            fig.add_trace(go.Scatter(x=x_obs[:, 0] if y_obs.size else [], y=y_obs,
                                     mode="markers", marker={"size": 9, "line": {"width": 1}},
                                     name=f"Evaluations - {self._level_label(l)}"))
        fig.update_layout(title="Prediction of the MF-EGO", xaxis_title="x",
                          yaxis_title="Target value", template="plotly_white")
        fig.write_html(save_path, include_plotlyjs=include_plotlyjs)
        print(f"Interactive 1D response saved as : {save_path}")

    def plot_response_surface_2d_interactive(self, param_x_idx: int = 0, param_y_idx: int = 1,
                                             grid_size: int = 60, fixed_values: list = None,
                                             save_path: str = "response_surface_2d.html",
                                             include_plotlyjs: str = "cdn") -> None:
        """Interactive version of plot_response_surface_2d (HTML file): predicted mean and
        predicted standard deviation side by side, with the observations of every level."""
        go, make_subplots = self._plotly()
        if fixed_values is None:
            fixed_values = self._best_observed_point()
        x_mesh, y_mesh, x_test = self._grid_2d(param_x_idx, param_y_idx, grid_size,
                                               fixed_values)
        mean, var, _ = self.model.predict_batch(x_test)

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f"Predicted mean - {self._level_label(self.num_levels)}", "Predicted std"))
        fig.add_trace(go.Contour(x=x_mesh[0], y=y_mesh[:, 0], z=mean.reshape(x_mesh.shape),
                                 colorscale="Viridis", colorbar={"x": 0.45}, name="mean"),
                      row=1, col=1)
        fig.add_trace(go.Contour(x=x_mesh[0], y=y_mesh[:, 0],
                                 z=np.sqrt(np.maximum(var, 0.0)).reshape(x_mesh.shape),
                                 colorscale="Magma", name="std"), row=1, col=2)
        for l in range(1, self.num_levels + 1):
            points, values, _ = self._observations(l)
            if values.size == 0:
                continue
            for col in (1, 2):
                fig.add_trace(go.Scatter(
                    x=points[:, param_x_idx], y=points[:, param_y_idx], mode="markers",
                    marker={"symbol": l - 1, "size": 9, "line": {"width": 1}},
                    text=[f"y = {v:.6g}" for v in values], showlegend=col == 1,
                    legendgroup=str(l), name=f"Evaluations - {self._level_label(l)}"),
                    row=1, col=col)
        fig.update_xaxes(title_text=f"Parameter {param_x_idx}")
        fig.update_yaxes(title_text=f"Parameter {param_y_idx}")
        fig.update_layout(title=f"Response surface (dimensions {param_x_idx} & {param_y_idx}, "
                                f"other parameters fixed at {np.round(fixed_values, 3).tolist()})",
                          template="plotly_white")
        fig.write_html(save_path, include_plotlyjs=include_plotlyjs)
        print(f"Interactive response surface saved as : {save_path}")
