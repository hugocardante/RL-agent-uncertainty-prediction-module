import pandas as pd

from utils.data_structures import TrajectoryManager
from utils.global_utils import ensure_dir, get_dir_name
from utils.model_wrapper import ModelWrapper
from utils.stl_wrapper import STLWrapper


class ClassificationMetrics:
    """
    Class to hold and compute classification metrics from confusion matrix.

    Attributes:
        tp: true positives
        fp: false positives
        tn: true negatives
        fn: false negatives
        fa: false alarms
        miss: overlooked violations
        f1: f1 score
    """

    def __init__(self, tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0):
        """
        Initialize classification metrics.

        Args:
            tp: true positives (default 0)
            fp: false positives (default 0)
            tn: true negatives (default 0)
            fn: false negatives (default 0)
        """
        self.tp: int = tp
        self.fp: int = fp
        self.tn: int = tn
        self.fn: int = fn
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """computes metrics (fa, miss, f1) from confusion matrix counts."""
        self.fa: float = self._calculate_false_alarms()
        self.miss: float = self._calculate_miss()
        self.f1: float = self._calculate_f1_score()

    def _calculate_false_alarms(self) -> float:
        """
        calculates false alarm (fa)

        Returns:
            false alarms: (fp / (tp + fp))
        """
        return self.fp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def _calculate_miss(self) -> float:
        """
        calculates overlooked violations (miss)

        Returns:
            miss: (fn / (tp + fn))
        """
        return self.fn / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def _calculate_f1_score(self) -> float:
        """
        calculates f1 score (copied from sklearn docs)

        Returns:
            f1 score
        """
        return (
            (2 * self.tp) / (2 * self.tp + self.fp + self.fn)
            if self.tp > 0 and self.fp > 0 and self.fn > 0
            else 0.0
        )

    def update(self, predicted_unsafe: bool, actually_unsafe: bool) -> None:
        """
        updates metrics based on a single prediction

        given what was predicted and what actually happened, increment the appropriate counter
        we want true positives to be predicted unsafe and actually unsafe, which would mean
        predicting failure and actuallly being failure

        Args:
            predicted_unsafe: Whether the prediction indicated unsafe condition
            actually_unsafe: Whether the actual outcome was unsafe
        """
        if predicted_unsafe and actually_unsafe:
            self.tp += 1
        elif predicted_unsafe and not actually_unsafe:
            self.fp += 1
        elif not predicted_unsafe and actually_unsafe:
            self.fn += 1
        else:  # predicted_safe and actually_safe
            self.tn += 1

        self._compute_metrics()

    def add_metrics(self, other: "ClassificationMetrics") -> None:
        """
        adds another ClassificationMetrics object to this one
        this is only usefull to aggregate metrics

        Args:
            other: Another ClassificationMetrics instance to add
        """
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn
        self._compute_metrics()

    def to_dict(self) -> dict[str, float | int | str]:
        """
        converts metrics to dictionary for exporting the csvs

        Returns:
            dictionary with all metrics (tp, fp, tn, fn, fa, miss, f1)
        """
        return {
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "fa": self.fa,
            "miss": self.miss,
            "f1": self.f1,
        }


def aggregate_metrics(
    all_episode_metrics: list[dict[str, ClassificationMetrics]],
    predictor_type: str,
) -> dict[str, ClassificationMetrics]:
    """
    aggregates metrics for multiple episodes

    Args:
        all_episode_metrics: list of metric dictionaries, one per episode
        predictor_type: name for logging (e.g., "rule", "model")

    Returns:
        dictionary mapping predictor_name to aggregated ClassificationMetrics
    """
    all_episode_metrics = [ep for ep in all_episode_metrics if ep]

    if not all_episode_metrics:
        print(f"No valid episode stats to aggregate for {predictor_type}s")
        return {}

    # all unique predictor names (predictor can be a conformal prediction model
    # or it can be a stl verifier)
    predictor_names = set()
    for episode_metrics in all_episode_metrics:
        predictor_names.update(episode_metrics.keys())

    if not predictor_names:
        print(f"No {predictor_type} names found in episode stats")
        return {}

    print(
        f"Aggregating across {len(all_episode_metrics)} episodes "
        f"with {len(predictor_names)} unique {predictor_type}s"
    )

    aggregated = {}
    for predictor_name in predictor_names:
        aggregated[predictor_name] = ClassificationMetrics()

        for episode_metrics in all_episode_metrics:
            if predictor_name in episode_metrics:
                aggregated[predictor_name].add_metrics(episode_metrics[predictor_name])

    print(f"Successfully aggregated {len(aggregated)} {predictor_type}s")
    return aggregated


def export_classification(
    metrics_dict: dict[str, ClassificationMetrics],
    output_path: str,
    column_name: str = "predictor_name",
    n_episodes: int | None = None,
) -> None:
    """
    exports classification results to csv

    Args:
        metrics_dict: dictionary mapping predictor name to ClassificationMetrics
        output_path: path to save csv
        column_name: name for the predictor column (e.g., "rule_name" for stl, "model_name" for conformal models)
        n_episodes: number of episodes (for aggregated results)
    """
    base_columns = ["TP", "FP", "TN", "FN", column_name, "f1", "fa", "miss"]

    rows = []
    for predictor_name, metrics in metrics_dict.items():
        # get all metrics like tp, tn, fp, fn, fa, miss, f1
        row = metrics.to_dict()
        row[column_name] = predictor_name

        if n_episodes is not None:
            row["n_episodes"] = n_episodes

        rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=base_columns)

    ensure_dir(get_dir_name(output_path))
    df.to_csv(output_path, index=False)


def analyze_episode_conformal_safety(
    trajectory_manager: TrajectoryManager,
    predictors: dict[str, ModelWrapper],
    n_lines: int,
) -> dict[str, ClassificationMetrics]:
    """
    analyzes conformal predictions (safety violations) for one episode

    a trajectory is unsafe for a line if any timestep's upper bound exceeds the threshold
    this threshold is the value of rho we consider dangerous to be at, which normally is 0.95

    Args:
        trajectory_manager: TrajectoryManager containing trajectories with predictions
        predictors: dictionary of trained conformal models
        n_lines: integer number of power lines in the grid

    Returns:
        dictionary mapping model_name to ClassificationMetrics
    """
    completed_trajectories = trajectory_manager.get_completed_trajectories()
    model_metrics = {}

    for traj in completed_trajectories:
        for model_name in predictors.keys():
            for line_idx in range(n_lines):

                predicted_unsafe = not traj.is_conf_predicted_safe(model_name, line_idx)
                actually_unsafe = not traj.is_actually_safe(line_idx)

                if model_name not in model_metrics:
                    model_metrics[model_name] = ClassificationMetrics()

                model_metrics[model_name].update(predicted_unsafe, actually_unsafe)

    return model_metrics


def analyze_episode_stl_safety(
    trajectory_manager: TrajectoryManager,
    verifiers: dict[str, STLWrapper],
    n_lines: int,
) -> dict[str, ClassificationMetrics]:
    """
    analyzes stl predictions to compute tp, fp, tn, fn per rule

    compares STL predictions (forecasts) against the actual truth (actual data)

    Args:
        trajectory_manager: TrajectoryManager containing trajectories with STL results
        verifiers: dictionary of STLWrapper
        n_lines: integer number of power lines in the grid

    Returns:
        dictionary mapping rule_name to ClassificationMetrics
    """
    completed_trajectories = trajectory_manager.get_completed_trajectories()
    rule_metrics = {}

    for traj in completed_trajectories:
        for rule_name in verifiers.keys():
            for line_idx in range(n_lines):

                predicted_unsafe = not traj.is_stl_predicted_safe(rule_name, line_idx)
                actually_unsafe = not traj.is_actually_safe(line_idx)

                if rule_name not in rule_metrics:
                    rule_metrics[rule_name] = ClassificationMetrics()

                rule_metrics[rule_name].update(predicted_unsafe, actually_unsafe)

    return rule_metrics
