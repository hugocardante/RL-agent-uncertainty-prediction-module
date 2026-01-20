import numpy as np
import numpy.typing as npt

from utils.data_structures import Trajectory


class VanillaConformalPredictor:
    """
    Vanilla Split Conformal Prediction

    The simplest conformal prediction method that uses absolute errors
    to create symmetric prediction intervals around forecasts.
    """

    def __init__(self, alpha: float, n_lines: int) -> None:
        """
        Initialize Vanilla Symmetric Conformal Predictor

        Args:
            alpha: Significance level (e.g., 0.1 for 90% coverage)
            n_lines: Number of power lines
        """
        self.alpha: float = alpha
        self.n_lines: int = n_lines
        self.per_line_quantiles: npt.NDArray[np.float64] = np.zeros(self.n_lines)

    def fit(
        self,
        forecast_array: npt.NDArray[np.float64],
        actual_array: npt.NDArray[np.float64],
    ) -> None:
        """
        Trains the model on calibration data

        Calculates the (1-alpha) quantile of nonconformity scores for each line

        Args:
            forecast_array: Forecast rho values (n_samples, n_lines)
            actual_array: Actual rho values (n_samples, n_lines)
        """

        # Compute absolute errors as nonconformity scores
        nonconformity_scores = np.abs(actual_array - forecast_array)

        for line_idx in range(self.n_lines):
            scores_for_line = nonconformity_scores[:, line_idx]
            self.per_line_quantiles[line_idx] = np.quantile(
                scores_for_line, 1 - self.alpha
            )

    def predict(
        self, trajectory: Trajectory
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates prediction intervals for entire trajectory

        This model's logic is context-independent, so it applies the same
        symmetric thresholds to all steps in the trajectory.

        Args:
            trajectory: Trajectory object with the forecasts

        Returns:
            Tuple of (lower_bounds, upper_bounds), each a list of 12 predictions
        """
        all_lower_bounds: list[list[float]] = []
        all_upper_bounds: list[list[float]] = []

        for timestep_forecast_values in trajectory.forecast_trajectory:
            f_i = np.array(timestep_forecast_values[: self.n_lines])

            lower_bounds = np.maximum(0, f_i - self.per_line_quantiles)
            upper_bounds = f_i + self.per_line_quantiles

            all_lower_bounds.append(lower_bounds.tolist())
            all_upper_bounds.append(upper_bounds.tolist())

        return all_lower_bounds, all_upper_bounds


def compute_model(
    calib_episodes_data: tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ],
    n_lines: int,
    alpha: float,
) -> VanillaConformalPredictor | None:
    """
    Creates and trains the Vanilla model

    Args:
        calib_episodes_data: Tuple of (features, forecasts, actuals)
        n_lines: Number of power lines
        alpha: Significance level

    Returns:
        Trained model or None if insufficient data
    """
    # We dont need the features_array for the vanilla model
    _, forecast_array, actual_array = calib_episodes_data

    print(f"Creating Vanilla Conformal Predictor with alpha={alpha}")

    vanilla_predictor = VanillaConformalPredictor(alpha=alpha, n_lines=n_lines)
    vanilla_predictor.fit(forecast_array, actual_array)

    return vanilla_predictor
