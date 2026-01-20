"""
STL Rule: Vanilla

Specification: psi := G[1,12](rho ≤ threshold)
Robustness: psi(rho,t) = min_{k=1,...,12} (threshold - rho_{t+k})
"""

import numpy as np
import numpy.typing as npt

import config
from utils.data_structures import Trajectory


class STLVanilla:
    """
    Safety STL rule using vanilla conformal prediction
    """

    def __init__(self, n_lines: int, alpha: float):
        """
        Initializes the safety rule

        Args:
            n_lines: number of power lines in the grid
            alpha: significance level for conformal prediction
        """
        self.n_lines: int = n_lines
        self.alpha: float = alpha
        self.threshold: float = config.RHO_SAFETY_THRESHOLD

        # shape: (n_samples, n_lines)
        self.calibration_scores: npt.NDArray[np.float64] = None
        # shape: (n_lines,)
        self.C_quantiles: npt.NDArray[np.float64] = None

    def fit(self, all_completed_traj: list[Trajectory]):
        """
        Calibrates the rule by computing quantiles per line

        For each trajectory, computes the difference between predicted
        and actual robustness as the nonconformity score.

        Args:
            all_completed_traj: list of completed Trajectory objects
        """
        calibration_data = []

        # for each trajectory we compute the "robustness"
        # using the predicted rho forecasts and the real rho forecasts
        for trajectory in all_completed_traj:
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

            calibration_data.append(scores_per_line)

        if not calibration_data:
            print("Warning: No calibration data for vanilla rule")
            return

        # store calibration scores (shape: (n_samples, n_lines))
        self.calibration_scores = np.array(calibration_data)

        # (1-alpha) quantile per line
        self.C_quantiles = np.quantile(self.calibration_scores, 1 - self.alpha, axis=0)

    def predict(self, trajectory) -> dict[int, bool]:
        """
        Verifies trajectory safety for all lines

        Args:
            trajectory: Trajectory object to verify

        Returns:
            Dictionary mapping line_idx to boolean safety prediction
        """
        results = {}

        for line_idx in range(self.n_lines):
            # Compute predicted robustness for this line
            predicted_robustness = trajectory.compute_stl_robustness(
                line_idx, use_predicted=True
            )

            # line-specific quantile
            C_line = float(self.C_quantiles[line_idx])
            adjusted_robustness = predicted_robustness - C_line
            is_safe = adjusted_robustness > 0

            results[line_idx] = is_safe

        return results


def compute_stl(
    all_completed_traj: list[Trajectory], n_lines: int, alpha: float
) -> STLVanilla | None:
    """
    Creates and calibrates the rule (using vanilla)

    Args:
        all_completed_traj: list of completed Trajectory objects from calibration episodes
        n_lines: number of power lines in the grid
        alpha: significance level

    Returns:
        Calibrated STLVanilla object, or None if calibration fails
    """

    rule = STLVanilla(n_lines, alpha)
    rule.fit(all_completed_traj)

    return rule
