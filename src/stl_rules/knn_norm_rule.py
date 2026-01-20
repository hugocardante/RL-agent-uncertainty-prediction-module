"""
STL Rule: Using k-NN as uncertainty scalars

Specification: psi := G[1,12](rho ≤ threshold)
Robustness: psi(rho,t) = min_{k=1,...,12} (threshold - rho_{t+k})
"""

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from conformalized_models import calculate_difficulty_estimate
from training import extract_trajectory_features
from utils.data_structures import Trajectory


class KNNRuleNormalized:
    """
    STL rule using k-NN (normalised scores)

    The idea is that during calibration it normalizes each point's score by its neighbors' mean,
       then computes a global quantile of normalized scores (per line). Then at test time it estimates
       difficulty from neighbors, then denormalizes the global quantile using that difficulty.
    """

    def __init__(self, n_lines: int, alpha: float, k: int):
        """
        Initializes the rule (using normalized k-NN scores)

        Args:
            n_lines: number of power lines in the grid
            alpha: significance level
            k: number of nearest neighbors (defaults to config.KNN_PERCENTAGE of samples)
        """
        self.n_lines: int = n_lines
        self.alpha: float = alpha
        self.k: int = k
        self.threshold: float = config.RHO_SAFETY_THRESHOLD

        # Calibration data
        # shape: (n_samples, n_features)
        self.calibration_features: npt.NDArray[np.float64] = None
        # shape: (n_samples, n_lines)
        self.calibration_scores: npt.NDArray[np.float64] = None
        self.knn_model: NearestNeighbors = None
        self.scaler: StandardScaler = None
        # Global normalized quantiles (one per line)
        self.C_normalized_quantiles: npt.NDArray[np.float64] = None  # shape: (n_lines,)

    def fit(self, all_completed_traj: list[Trajectory]):
        """
        Calibrates the rule by computing normalized quantiles per line

        Args:
            all_completed_traj: list of completed Trajectory objects
        """
        calibration_data = []

        for trajectory in all_completed_traj:
            features = extract_trajectory_features(trajectory, self.n_lines)
            if features is None:
                continue

            # Compute calibration scores for all lines
            scores_per_line = []
            for line_idx in range(self.n_lines):
                predicted_robustness = trajectory.compute_stl_robustness(
                    line_idx, use_predicted=True
                )
                actual_robustness = trajectory.compute_stl_robustness(
                    line_idx, use_predicted=False
                )

                score = predicted_robustness - actual_robustness
                scores_per_line.append(score)

            calibration_data.append((features, scores_per_line))

        if not calibration_data:
            print("    Warning: No calibration data for kNN-norm rule")
            return

        self.calibration_features = np.array([f for f, _ in calibration_data])
        self.calibration_scores = np.array([s for _, s in calibration_data])

        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(self.calibration_features)

        # k+1 because we exclude the point itself during calibration
        self.knn_model = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto")
        self.knn_model.fit(features_scaled)

        # Compute global normalized quantiles
        self.C_normalized_quantiles = self._compute_normalized_quantiles(
            features_scaled
        )

        n_features = self.calibration_features.shape[1]
        print(
            f"    kNN Normalized: k={self.k}, n_lines={self.n_lines}, alpha={self.alpha}, "
            f"n_features={n_features} ({len(self.calibration_scores)} samples)"
        )

    def _compute_normalized_quantiles(
        self, features_scaled: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes normalized quantiles by processing all calibration points

        For each calibration point:
            1. Find k nearest neighbours (excluding itself)
            2. Compute difficulty estimate = mean of neighbour scores (per line)
            3. Normalize: score / difficulty_estimate

        Then compute (1-alpha) quantile of normalized scores for each line

        Args:
            features_scaled: scaled feature matrix (shape (n_samples, n_features))

        Returns:
            Normalized quantile values per line (shape (n_lines,))
        """
        n_samples = len(features_scaled)
        # An array to save the normalized non-conformity scores for each
        # datapoint in the calibration dtatset
        normalized_scores_all = np.zeros((n_samples, self.n_lines))

        # batches are used so that we can balance speed and memory
        batch_size = min(config.BATCH_SIZE, n_samples)

        for start_idx in tqdm(
            range(0, n_samples, batch_size), desc="knn_norm_rule quantiles.."
        ):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_features = features_scaled[start_idx:end_idx]

            _, bni = self.knn_model.kneighbors(batch_features)
            # get k+1 neighbors, exclude self with [:, 1:]
            batch_neighbour_indices = bni[:, 1:]

            for i, neighbor_indices in enumerate(batch_neighbour_indices):
                sample_idx = start_idx + i
                normalized_scores_all[sample_idx] = self._compute_normalized_scores(
                    sample_idx, neighbor_indices
                )

        # (1-alpha) quantile for each line
        normalized_quantiles = np.quantile(
            normalized_scores_all, 1 - self.alpha, axis=0
        )
        return normalized_quantiles

    def _compute_normalized_scores(
        self, sample_idx: int, neighbor_indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes normalized scores for a single calibration point

        Args:
            sample_idx: index of the calibration point
            neighbor_indices: indices of k nearest neighbours

        Returns:
            Normalized scores for this sample, shape (n_lines,)
        """
        # shape: (k, n_lines)
        neighbour_scores = self.calibration_scores[neighbor_indices]

        # since it is a magnitude, we want this to be absolute
        neighbour_scores_abs = np.abs(neighbour_scores)

        # Difficulty estimate: mean of neighbour scores per line
        # using function from the conformal models (since it is the same logic)
        difficulty_estimates = calculate_difficulty_estimate(neighbour_scores_abs)

        # This sample's scores: shape (n_lines,)
        sample_scores = self.calibration_scores[sample_idx]

        # Normalized scores
        normalized_scores = sample_scores / difficulty_estimates

        return normalized_scores

    def predict(self, trajectory) -> dict[int, bool] | None:
        """
        Verifies trajectory safety for all lines using normalized thresholds

        Args:
            trajectory: Trajectory object to verify

        Returns:
            Dictionary mapping line_idx to boolean safety prediction,
            or None if not calibrated
        """
        if self.knn_model is None or self.C_normalized_quantiles is None:
            return None

        test_features = extract_trajectory_features(trajectory, self.n_lines)
        if test_features is None:
            return None

        test_features_scaled = self.scaler.transform([test_features])

        # finds the indices of the k nearest neighbours (no need to exclude self during testing)
        _, indices = self.knn_model.kneighbors(test_features_scaled, n_neighbors=self.k)

        # kneighbors returns shape (n_queries, k) but here we only have one query (n_queries = 1), because
        # we pass [test_features], then we take the only element with indices[0]
        indices = indices[0]

        # shape: (k, n_lines)
        neighbour_scores = self.calibration_scores[indices]

        # since it is a magnitude, we want this to be absolute
        neighbour_scores_abs = np.abs(neighbour_scores)

        # shape: (n_lines,)
        difficulty_estimates = calculate_difficulty_estimate(neighbour_scores_abs)

        # denormalize: C_line = difficulty * normalized_quantile
        C_values = difficulty_estimates * self.C_normalized_quantiles

        # Generate results for all lines
        safety_results = {}

        for line_idx in range(self.n_lines):
            C_line = float(C_values[line_idx])

            predicted_robustness = trajectory.compute_stl_robustness(
                line_idx, use_predicted=True
            )
            adjusted_robustness = predicted_robustness - C_line
            is_safe = adjusted_robustness > 0

            safety_results[line_idx] = is_safe

        return safety_results


def compute_stl(
    all_completed_traj: list[Trajectory], n_lines: int, alpha: float, k: int
) -> KNNRuleNormalized | None:
    """
    Creates and calibrates the rule (using k-NN normalisation)

    Args:
        all_completed_traj: list of Trajectory objects from calibration episodes
        n_lines: number of power lines in the grid
        alpha: significance level
        k: number of nearest neighbours

    Returns:
        Calibrated KNNRuleNormalized object, or None if calibration fails
    """

    rule = KNNRuleNormalized(n_lines, alpha, k)
    rule.fit(all_completed_traj)

    return rule
