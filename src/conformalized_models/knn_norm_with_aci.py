import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from conformalized_models import (
    calculate_difficulty_estimate,
    compute_adaptive_quantiles,
    extract_test_features,
    update_adaptive_alphas,
)


class CPwithKNNPredictorwithACI:
    """
    Conformal predictor with uncertainty scalars using k-Nearest Neighbours

    This normalized version uses difficulty estimation based on neighbour errors to
    create adaptive intervals that scale with local prediction difficulty.

    Based on Renkema et. all: "Conformal prediction for stochastic decision-making
                               of PV power in electricity markets""

    (experimentally added the ACI update rule to this predictor)
    """

    def __init__(self, alpha: float, k: int, gamma: float, n_lines: int) -> None:
        """
        Initialize the Conformal Predictor

        Args:
            alpha: Significance level (e.g., 0.1 for 90% coverage)
            k: Number of nearest neighbours to use
            n_lines: Number of power lines
        """
        self.alpha: float = alpha
        self.k: int = k
        self.n_lines: int = n_lines
        self.gamma: float = gamma
        self.adaptive_alphas: npt.NDArray[np.float64] = np.full(
            self.n_lines, self.alpha
        )
        self.knn_model: NearestNeighbors = None
        self.scaler: StandardScaler = None
        self.error_array: npt.NDArray[np.float64] = None
        self.normalized_nonconf_scores: npt.NDArray[np.float64] = None

    def fit(
        self,
        features_array: npt.NDArray[np.float64],
        forecast_array: npt.NDArray[np.float64],
        actual_array: npt.NDArray[np.float64],
    ) -> None:
        """
        Trains the model on calibration data

        Args:
            features_array: System state features (n_samples, n_features)
            forecast_array: Forecast rho values (n_samples, n_lines)
            actual_array: Actual rho values (n_samples, n_lines)
        """

        self.error_array = np.abs(forecast_array - actual_array)

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_array)

        self.knn_model = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 because we'll exclude self
            algorithm="auto",
        )
        self.knn_model.fit(features_scaled)

        self._compute_quantiles(features_scaled)

    def _compute_quantiles(self, features_scaled: npt.NDArray[np.float64]) -> None:
        """
        Computes normalized quantiles by processing all calibration points

        Args:
            features_scaled: Scaled feature matrix

        """
        n_samples = len(features_scaled)
        normalized_scores_all = np.zeros((n_samples, self.n_lines))

        # using batches to balance speed and memory
        # instead of going point by point, we do it in a batch
        # saving time when making the call to kneighbors, but
        # not using excessive memory by doing all at the same time
        # This number is arbitrary and can be increased for other computers
        batch_size = min(config.BATCH_SIZE, n_samples)

        for start_idx in tqdm(
            range(0, n_samples, batch_size), desc="Processing normalized quantiles"
        ):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_size_actual = end_idx - start_idx

            batch_features = features_scaled[start_idx:end_idx]
            _, bni = self.knn_model.kneighbors(batch_features)
            # Exclude self by taking out the first neighbour with [:, 1:]
            batch_neighbour_indices = bni[:, 1:]

            for i in range(batch_size_actual):
                sample_idx = start_idx + i
                neighbour_indices = batch_neighbour_indices[i]

                normalized_scores_all[sample_idx] = self._compute_normalized_errors(
                    sample_idx, neighbour_indices
                )

        self.normalized_nonconf_scores = normalized_scores_all

    def _compute_normalized_errors(
        self, sample_idx: int, neighbour_indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes normalized errors using neighbour difficulty estimation

        Args:
            sample_idx: Index of the sample to normalize
            neighbour_indices: Indices of k nearest neighbours

        Returns:
            Normalized errors for this sample
        """
        neighbour_errors = self.error_array[neighbour_indices]  # Shape: (k, n_lines)

        # Mean absolute error of neighbours as difficulty estimate
        difficulty_estimates = calculate_difficulty_estimate(neighbour_errors)

        sample_errors = self.error_array[sample_idx]  # Shape: (n_lines,)
        normalized_scores = sample_errors / difficulty_estimates

        return normalized_scores

    def predict(self, trajectory) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates prediction intervals for the trajectory

        For each step, finds k nearest neighbours, estimates local difficulty,
        and scales the normalized quantile by the difficulty estimate

        Args:
            trajectory: Trajectory with forecast sequence

        Returns:
            Tuple of (lower_bounds, upper_bounds), each a list of 12 predictions
        """
        all_lower_bounds = []
        all_upper_bounds = []

        norm_quant_aci = compute_adaptive_quantiles(
            self.normalized_nonconf_scores,
            self.adaptive_alphas,
            self.n_lines,
        )

        for ts in trajectory.timestamps:
            td = trajectory.timesteps[ts]
            forecasted_rho_values = td.forecasted_rho

            test_features = extract_test_features(trajectory, td)
            test_features_scaled = self.scaler.transform([test_features])

            # finds the indices of the k nearest neighbours (no need to exclude self during testing)
            _, indices = self.knn_model.kneighbors(
                test_features_scaled, n_neighbors=self.k
            )

            # kneighbors returns shape (n_queries, k) but here we only have one query (n_queries = 1), because
            # we pass [test_features], then we take the only element with indices[0]
            indices = indices[0]

            # Estimate difficulty based on neighbour errors
            neighbour_errors = self.error_array[indices]  # Shape: (k, n_lines)
            difficulty_estimates = calculate_difficulty_estimate(neighbour_errors)

            # Scale normalized quantile by difficulty estimate (with the normalized with aci..)
            thresholds = difficulty_estimates * norm_quant_aci

            # Apply thresholds to current forecast
            f_i = np.array(forecasted_rho_values[: self.n_lines])
            lower_bounds = np.maximum(0, f_i - thresholds)
            upper_bounds = f_i + thresholds

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

        Args:
            actual_values: True rho values just observed
            predicted_lower: Lower bounds predicted for these values
            predicted_upper: Upper bounds predicted for these values
        """
        if len(actual_values) != self.n_lines:
            raise ValueError(
                f"kNN(norm) ACI: Expected {self.n_lines} actual values, got {len(actual_values)}"
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
    k: int,
    gamma: float,
) -> CPwithKNNPredictorwithACI | None:
    """
    Creates and trains a conformal predictor using k-NN for uncertainty scalars (and aci update rule)

    Args:
        calib_episodes_data: Tuple of (features, forecasts, actuals)
        n_lines: Number of power lines
        alpha: Significance level
        k: Number of nearest neighbours
        gamma: step size parameter

    Returns:
        Trained model or None if insufficient data
    """
    features_array, forecast_array, actual_array = calib_episodes_data

    print(f"Creating Conformal Predictor using k-NN (k = {k} with ACI)")

    knn_predictor = CPwithKNNPredictorwithACI(
        alpha=alpha, k=k, gamma=gamma, n_lines=n_lines
    )
    knn_predictor.fit(features_array, forecast_array, actual_array)

    return knn_predictor
