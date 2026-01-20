"""
This folder is not supposed to work as a package,
but I think it's a good place to put some methods
that are shared by the CP implementations
to reduce code duplication
"""

import numpy as np
import numpy.typing as npt

from training import extract_system_state_features
from utils.data_structures import TimestepData, Trajectory


def extract_test_features(
    traj: Trajectory, td: TimestepData
) -> npt.NDArray[np.float64]:
    """
    Extracts the test features for a given timstep

    Args:
        traj: Trajectory from where we take the hourly stats
        td: Current TimestepData containing the current timestep data

    Returns:
        numpy array with the extracted features
    """
    obs = td.forecast_observation
    forecast_horizon = td.horizon
    action_influenced = td.is_action_influenced
    forecasted_rho_values = td.forecasted_rho

    test_features = extract_system_state_features(
        obs,
        forecasted_rho_values,
        forecast_horizon,
        traj.hourly_stats,
        action_influenced,
    )
    return test_features


def calculate_difficulty_estimate(
    neighbour_errors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculates difficulty estimates based on neighbour errors

    Uses the mean absolute error of neighbours as a difficulty estimate.

    Args:
        neighbour_errors: errors of k nearest neighbours (shape (k, n_lines))

    Returns:
        Difficulty estimate per line (shape (n_lines,))
    """
    difficulty_estimates = np.maximum(
        np.mean(neighbour_errors, axis=0),
        1e-6,  # avoid division by zero
    )
    return difficulty_estimates


def update_adaptive_alphas(
    adaptive_alphas: npt.NDArray[np.float64],
    actual_values: list[float],
    predicted_lower: list[float],
    predicted_upper: list[float],
    alpha: float,
    gamma: float,
) -> npt.NDArray[np.float64]:
    """
    Updates adaptive alpha values using the ACI (Adaptive Conformal Inference) rule

    The ACI algorithm adapts significance levels based on recent coverage performance:
        - If actual is outside interval (violation): alpha increases -> wider intervals
        - If actual is inside interval (no violation): alpha decreases -> tighter intervals

    Update rule: alpha_new[i] = alpha_old[i] + gamma * (alpha_target - violation[i])

    Args:
        adaptive_alphas: Current alpha values per line, shape (n_lines,)
        actual_values: True rho values just observed, shape (n_lines,)
        predicted_lower: Lower bounds predicted for these values, shape (n_lines,)
        predicted_upper: Upper bounds predicted for these values, shape (n_lines,)
        alpha: Base significance level (target miscoverage rate), e.g., 0.1 for 90% coverage
        gamma: Adaptation rate (higher = more responsive but less stable)

    Returns:
        Updated adaptive alphas clipped to [0, 1], (shape (n_lines,))
    """
    y = np.array(actual_values)
    lower = np.array(predicted_lower)
    upper = np.array(predicted_upper)

    # 1 if outside interval (violation), 0 if inside
    violations = ((y < lower) | (y > upper)).astype(float)

    # ACI update: increase alpha after violations, decrease after coverage
    updated_alphas = adaptive_alphas + gamma * (alpha - violations)

    # we clip because np.quantile has to be called with a quantile value in [0, 1]
    return np.clip(updated_alphas, 0.0, 1.0)


def compute_adaptive_quantiles(
    calibration_conformity_scores: npt.NDArray[np.float64],
    adaptive_alphas: npt.NDArray[np.float64],
    n_lines: int,
) -> npt.NDArray[np.float64]:
    """
    Calculates current quantiles using the current adaptive alphas

    For ACI methods, each line has its own adaptive alpha value that adjusts
    based on recent coverage performance. This function computes the corresponding
    quantile for each line from the calibration conformity scores.

    Args:
        calibration_conformity_scores: 2D array of shape (n_samples, n_lines)
        adaptive_alphas: current adaptive alpha values (one per line)
        n_lines: number of power lines

    Returns:
        Array of quantiles (one per line)
    """
    cur_quantiles = np.array(
        [
            np.quantile(
                calibration_conformity_scores[:, line_idx],
                1 - adaptive_alphas[line_idx],
            )
            for line_idx in range(n_lines)
        ]
    )

    return cur_quantiles


__all__ = [
    "update_adaptive_alphas",
    "compute_adaptive_quantiles",
    "calculate_difficulty_estimate",
    "extract_test_features",
]
