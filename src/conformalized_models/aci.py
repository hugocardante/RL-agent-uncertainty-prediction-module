import numpy as np
import numpy.typing as npt

from conformalized_models import compute_adaptive_quantiles, update_adaptive_alphas
from utils.data_structures import Trajectory


class ACIConformalPredictor:
    """
    Adaptive Conformal Inference for trajectory-based forecasting.
    This is essentially the 'vanilla method' to which we add the ACI update rule.

    Based on Gibbs & Candès (2021): "Adaptive Conformal Inference Under Distribution Shift"

    Adapts the significance level (alpha) per power line based on recent coverage
    performance, allowing real-time response to distribution shifts.
    """

    def __init__(self, alpha: float, gamma: float, n_lines: int) -> None:
        """
        Initialize ACI Conformal Predictor

        Args:
            alpha: Base significance level (e.g., 0.1 for 90% coverage)
            gamma: Adaptation rate (higher = more responsive, less stable)
            n_lines: Number of power lines
        """
        self.alpha: float = alpha
        self.n_lines: int = n_lines
        self.gamma: float = gamma

        # init adaptive alphas (all lines start with adaptive_alpha = alpha)
        self.adaptive_alphas: npt.NDArray[np.float64] = np.full(
            self.n_lines, self.alpha
        )

        self.calibration_nonconf_scores: npt.NDArray[np.float64] = None

    def fit(
        self,
        forecast_array: npt.NDArray[np.float64],
        actual_array: npt.NDArray[np.float64],
    ) -> None:
        """
        Trains the model on calibration data

        Args:
            forecast_array: Forecast rho values (n_samples, n_lines)
            actual_array: Actual rho values (n_samples, n_lines)
        """

        if actual_array.shape[0] == 0:
            raise ValueError("No data for ACI model..")

        # calibration nonconformity scores
        self.calibration_nonconf_scores = np.abs(actual_array - forecast_array)

    def predict(
        self, trajectory: Trajectory
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates prediction intervals for entire trajectory

        The ACI method uses line-specific adaptive alpha values that have been
        updated based on recent coverage performance
        This allows automatic adjustment of interval widths when distribution
        shifts occur

        Args:
            trajectory: Trajectory with forecast sequence

        Returns:
            Tuple of (lower_bounds, upper_bounds), each a list of 12 predictions
        """
        all_lower_bounds: list[list[float]] = []
        all_upper_bounds: list[list[float]] = []

        # Calculate current quantiles using adaptive alphas
        quantiles = compute_adaptive_quantiles(
            self.calibration_nonconf_scores, self.adaptive_alphas, self.n_lines
        )

        # Apply the same quantiles to all steps in the trajectory
        for step_rho_values in trajectory.forecast_trajectory:
            f_i = np.array(step_rho_values[: self.n_lines])
            lower_bounds = np.maximum(0, f_i - quantiles)
            upper_bounds = f_i + quantiles

            all_lower_bounds.append(lower_bounds.tolist())
            all_upper_bounds.append(upper_bounds.tolist())

        return all_lower_bounds, all_upper_bounds

    def update_alphas(
        self,
        actual_values: list[float],
        predicted_lower: list[float],
        predicted_upper: list[float],
    ) -> None:
        """
        Updates adaptive alpha values based on recent coverage performance
        Calls the update_adaptive_alphas

        Args:
            actual_values: True rho values just observed
            predicted_lower: Lower bounds predicted for these values
            predicted_upper: Upper bounds predicted for these values
        """
        if len(actual_values) != self.n_lines:
            raise ValueError(
                f"ACI: Expected {self.n_lines} actual values, got {len(actual_values)}"
            )

        self.adaptive_alphas = update_adaptive_alphas(
            self.adaptive_alphas,
            actual_values,
            predicted_lower,
            predicted_upper,
            self.alpha,
            self.gamma,
        )


def compute_model(
    calib_episodes_data: tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ],
    n_lines: int,
    alpha: float,
    gamma: float,
) -> ACIConformalPredictor | None:
    """
    Creates and trains the ACI model

    Args:
        calib_episodes_data: Tuple of (features, forecasts, actuals)
        n_lines: Number of power lines
        alpha: Base significance level
        gamma: step size parameter

    Returns:
        Trained model or None if insufficient data
    """
    # we dont need the features array for the aci model
    _, forecast_array, actual_array = calib_episodes_data

    print(f"Creating ACI Conformal Predictor with alpha={alpha}, gamma={gamma}")

    model = ACIConformalPredictor(alpha=alpha, gamma=gamma, n_lines=n_lines)
    model.fit(forecast_array, actual_array)

    return model
