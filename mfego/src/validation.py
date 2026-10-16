"""
[QUAL] Qualification of a surrogate: accuracy and calibration of Gaussian predictions
(mean, variance) against true values, from a test set or from a leave-one-out
(CoKriging.leave_one_out).

Example:
    y, mean, var = model.leave_one_out()          # or predictions on a test set
    scores = prediction_scores(y, mean, var)
    print(scores["q2"], scores["coverage_95"])
"""
import numpy as np
from scipy.stats import norm


def prediction_scores(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> dict:
    """
    Accuracy and calibration of Gaussian predictions.

    Args:
    - y_true: numpy array (m,), true values.
    - mean, var: numpy arrays (m,), predicted mean and variance (noise included or not,
      consistently with y_true).
    Returns: dict with
    - rmse_rel: RMSE / std(y_true) (0 = perfect, 1 = as good as the mean of y_true);
    - q2: 1 - rmse_rel^2, coefficient of determination (1 = perfect, 0 = the mean);
    - nlpd: mean negative log predictive density (lower is better; it penalizes an
      underestimated variance);
    - coverage_50, coverage_90, coverage_95, coverage_99: share of the points inside the
      central predicted interval of that probability (ideal: the probability);
    - z_rms: root mean square of the standardized errors z = (y - mean) / std (ideal 1;
      > 1: variance too small or biased mean, < 1: variance too large).
    """
    y_true, mean = np.asarray(y_true, dtype=float), np.asarray(mean, dtype=float)
    std = np.sqrt(np.maximum(np.asarray(var, dtype=float), 1e-300))
    z = (y_true - mean) / std
    rmse_rel = float(np.sqrt(np.mean((mean - y_true) ** 2)) / np.std(y_true))
    scores = {"rmse_rel": rmse_rel, "q2": 1.0 - rmse_rel ** 2,
              "nlpd": float(np.mean(0.5 * np.log(2.0 * np.pi * std ** 2) + 0.5 * z ** 2)),
              "z_rms": float(np.sqrt(np.mean(z ** 2)))}
    for probability in (0.5, 0.9, 0.95, 0.99):
        scores[f"coverage_{round(100 * probability)}"] = float(
            np.mean(np.abs(z) <= norm.ppf(0.5 + probability / 2.0)))
    return scores
