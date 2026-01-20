import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from utils.classification import (
    ClassificationMetrics,
    aggregate_metrics,
    analyze_episode_conformal_safety,
    analyze_episode_stl_safety,
    export_classification,
)
from utils.data_structures import Trajectory, TrajectoryManager
from utils.global_utils import ensure_dir, get_dir_name, get_env_details
from utils.model_wrapper import ModelWrapper, get_enabled_models
from utils.parallel import TestEpisodeWorkerReturn
from utils.stl_wrapper import STLWrapper, get_enabled_stl_rules


@dataclass
class ModelMetrics:
    """
    struct for calculated metrics of a conformal prediction model
    for coverage and width stats for each of the powerlines
    for a single conformal prediction model

    Attributes:
        coverage: coverage percentage per line (all timesteps)
        action_inf_coverage: coverage percentages per line (only action-influenced timesteps)
        width: average prediction interval widths per line
        name: name of the conformal prediction model
    """

    coverage: list[float]
    action_inf_coverage: list[float]
    width: list[float]
    name: str


@dataclass
class HorizonMetrics:
    """
    Struct for calculated metrics at a specific forecast horizon.

    Attributes:
        horizon: the forecast step (1 to config.HORIZON)
        coverage: coverage percentage at this horizon
        width: average prediction interval width at this horizon
    """

    horizon: int
    coverage: float
    width: float


class ConformalMetrics:
    """
    Class for computing conformal prediction metrics (coverage and width)
    from trajectory data (per-model and also per-horizon).

    Attributes:
        line_names: list (of str) with the names of each power line
        n_lines: integer number of power lines
        model_metrics: Dictionary mapping model names to ModelMetrics (none if not yet calculated)
        horizon_metrics: Dictionary mapping model names to per-horizon metrics (none if not yet calculated)
        n_trajectories: Integer number of trajectories processed
    """

    def __init__(self, line_names: list[str]) -> None:
        """
        Args:
            line_names: we pass only a list with the power line names,
            the rest is calculated by the methodsj
        """
        self.line_names: list[str] = line_names
        self.n_lines: int = len(line_names)
        self.model_metrics: dict[str, ModelMetrics] | None = None
        self.horizon_metrics: dict[str, list[HorizonMetrics]] | None = None
        self.n_trajectories: int = 0

    def _calculate_metrics(
        self,
        trajectories: list[Trajectory],
        model_name: str,
        only_action_influenced: bool,
    ) -> tuple[list[float], list[float]]:
        """
        calculates coverage and width metrics for a model for all trajectories.
        Args:
            trajectories: list of trajectory objects to analyze
            model_name: string name of the model
            only_action_influenced: if true, only includes timesteps that were action-influenced

        Returns:
            Tuple of (coverage_per_line, width_per_line) as lists of floats
        """
        total_counts = 0
        in_interval_counts = np.zeros(self.n_lines)
        total_width_sums = np.zeros(self.n_lines)

        for traj in trajectories:
            for ts in traj.timestamps:
                # timestep_data for this timestep
                td = traj.timesteps[ts]

                # if it's only for asction_influenced_periods, and this timestep is not
                # then we continue
                if only_action_influenced and not td.is_action_influenced:
                    continue

                # it does not make sense to calculate if the timestep
                # is not complete, so it's skipped
                if not td.is_complete():
                    continue

                actuals = td.real_rho
                lower, upper = traj.get_predictions_at_timestamp(ts, model_name)

                if lower and upper:
                    for line_idx in range(self.n_lines):
                        in_interval = (
                            lower[line_idx] <= actuals[line_idx] <= upper[line_idx]
                        )
                        in_interval_counts[line_idx] += int(in_interval)
                        total_width_sums[line_idx] += upper[line_idx] - lower[line_idx]
                    total_counts += 1

        # this can happen for example if the episode ends before config.WARMUP timesteps
        # have passed, and we just return a list of nan to avoid the warnings (but this if can be removed)
        if total_counts == 0:
            nan_array = [float("nan")] * self.n_lines
            return nan_array, nan_array

        coverage_per_line = (in_interval_counts / total_counts) * 100
        avg_width_per_line = total_width_sums / total_counts

        return coverage_per_line.tolist(), avg_width_per_line.tolist()

    def calculate_model_metrics(
        self,
        trajectories: list[Trajectory],
    ) -> dict[str, ModelMetrics]:
        """
        calculates model metrics from trajectory data

        Args:
            trajectories: List of Trajectory objects to analyze

        Returns:
            Dictionary mapping model names to ModelMetrics objects
        """

        self.n_trajectories = len(trajectories)
        enabled_models = get_enabled_models()
        metrics = {}

        for model_name in enabled_models:
            # for all timesteps
            coverage, width = self._calculate_metrics(
                trajectories, model_name, only_action_influenced=False
            )

            # for only action-influenced timesteps
            # at the moment we are ignoring the width for these periods
            # but it is usable in the future
            action_inf_coverage, _ = self._calculate_metrics(
                trajectories, model_name, only_action_influenced=True
            )
            metrics[model_name] = ModelMetrics(
                coverage=coverage,
                action_inf_coverage=action_inf_coverage,
                width=width,
                name=model_name,
            )

        self.model_metrics = metrics
        return metrics

    def calculate_horizon_metrics(
        self,
        trajectories: list[Trajectory],
    ) -> dict[str, list[HorizonMetrics]]:
        """
        Coverage and width metrics for each forecast horizon step

        Args:
            trajectories: List of Trajectory objects to analyze

        Returns:
            Dictionary mapping model names to lists of HorizonMetrics
        """

        enabled_models = get_enabled_models()
        # these buckets are to collect the metrics for each forecast horizon
        # so that we can then make save it in horizon metrics
        horizon_buckets = {
            model_name: {
                h: {"coverage": [], "width": []} for h in range(1, config.HORIZON + 1)
            }
            for model_name in enabled_models
        }

        for traj in trajectories:
            for ts in traj.timestamps:
                # timestep data
                td = traj.timesteps[ts]
                if not td.is_complete():
                    continue

                actuals = td.real_rho
                horizon = td.horizon

                for model_name in enabled_models:
                    # lower bound and upper bound for this specific timestep
                    lower, upper = traj.get_predictions_at_timestamp(ts, model_name)
                    if lower and upper:
                        # if the actuals were covered
                        covered_flags = (np.array(lower) <= np.array(actuals)) & (
                            np.array(actuals) <= np.array(upper)
                        )
                        widths = np.array(upper) - np.array(lower)
                        horizon_buckets[model_name][horizon]["coverage"].extend(
                            covered_flags
                        )
                        horizon_buckets[model_name][horizon]["width"].extend(widths)

        horizon_metrics = {model_name: [] for model_name in enabled_models}
        for model_name in enabled_models:
            for h in range(1, config.HORIZON + 1):
                cov = horizon_buckets[model_name][h]["coverage"]
                wid = horizon_buckets[model_name][h]["width"]
                horizon_metrics[model_name].append(
                    HorizonMetrics(
                        horizon=h,
                        # coverage in percentage
                        coverage=float(np.mean(cov) * 100) if cov else float("nan"),
                        # averge width
                        width=float(np.mean(wid)) if wid else float("nan"),
                    )
                )

        self.horizon_metrics = horizon_metrics
        return horizon_metrics


# NOTE: Classification logic is experimental because it might not make sense
# to evaluate CP (conformal models or STL) by using these metrics, since
# we are talking about probabilities..
# so this might be removed in the future
class ClassificationAnalyser:
    """
    classification (TP/TN/FP/FN) analysis for conformal and STL classification
    per episode and also aggregated
    """

    def __init__(self, output_dir: str) -> None:
        """
        Args:
            output_dir: only takes the string path to output directory
        """
        self.output_dir: str = output_dir
        self.n_lines: int
        self.line_names: list[str]
        self.n_lines, self.line_names = get_env_details()
        # episode metrics for aggregation
        self.conformal_episode_metrics: list[dict[str, ClassificationMetrics]] = []
        self.stl_episode_metrics: list[dict[str, ClassificationMetrics]] = []

    def export_episode_conformal(
        self,
        trajectory_manager: TrajectoryManager,
        predictors: dict[str, ModelWrapper],
        episode_dir: str,
    ) -> None:
        """
        exports conformal classification for a single episode

        Args:
            trajectory_manager: TrajectoryManager with predictions.
            predictors: Dictionary mapping model name to prediction models.
            episode_dir: path to save results.
        """

        # first we call analyze
        metrics = analyze_episode_conformal_safety(
            trajectory_manager, predictors, self.n_lines
        )
        export_classification(
            metrics,
            os.path.join(episode_dir, "conformal_classification.csv"),
            column_name="model_name",
        )
        self.conformal_episode_metrics.append(metrics)

    def export_episode_stl(
        self,
        trajectory_manager: TrajectoryManager,
        verifiers: dict[str, STLWrapper],
        episode_dir: str,
    ) -> None:
        """
        exports stl classification for a single episode

        Args:
            trajectory_manager: TrajectoryManager with STL results.
            verifiers: Dictionary mapping verifier name to STL verifiers.
            episode_dir: path to save results.
        """

        # first analyze
        metrics = analyze_episode_stl_safety(
            trajectory_manager, verifiers, self.n_lines
        )
        export_classification(
            metrics,
            os.path.join(episode_dir, "stl_classification.csv"),
            column_name="rule_name",
        )
        self.stl_episode_metrics.append(metrics)

    def export_classification_agg_metrics(self) -> None:
        """
        aggregates and exports classification metrics across all episodes
        for both conf and stl
        """
        if self.conformal_episode_metrics:
            aggregated = aggregate_metrics(
                self.conformal_episode_metrics, predictor_type="model"
            )
            output_path = os.path.join(
                self.output_dir, "conformal_classification_aggregated.csv"
            )
            export_classification(
                aggregated,
                output_path,
                column_name="model_name",
                n_episodes=len(self.conformal_episode_metrics),
            )

        if self.stl_episode_metrics:
            aggregated = aggregate_metrics(
                self.stl_episode_metrics, predictor_type="rule"
            )
            output_path = os.path.join(
                self.output_dir, "stl_classification_aggregated.csv"
            )
            export_classification(
                aggregated,
                output_path,
                column_name="rule_name",
                n_episodes=len(self.stl_episode_metrics),
            )


class DataExporter:
    """
    exporter for writting trajectory data and metrics to csv files.
    """

    def export_config(
        self,
        alpha: float,
        line_names: list[str],
        model_names: list[str],
        stl_rule_names: list[str],
        output_path: str,
    ) -> None:
        """
        exports config parameters to csv

        Args:
            alpha: Float alpha value for target coverage
            line_names: List of string line names
            model_names: List of model names
            output_path: String path for config csv
            stl_rule_names: list of string STL rule names
        """
        data = [
            {"parameter": "target_coverage", "value": 100 * (1 - alpha)},
            {"parameter": "rho_safety_threshold", "value": config.RHO_SAFETY_THRESHOLD},
        ]

        for idx, name in enumerate(line_names):
            data.append({"parameter": f"line_{idx}_name", "value": name})

        for idx, name in enumerate(model_names):
            data.append({"parameter": f"model_{idx}_name", "value": name})

        if stl_rule_names:
            for idx, name in enumerate(stl_rule_names):
                data.append({"parameter": f"stl_rule_{idx}_name", "value": name})

        self._write_csv(data, output_path)

    def export_conformal_comparison(
        self,
        model_metrics: dict[str, ModelMetrics],
        n_lines: int,
        output_path: str,
    ) -> None:
        """
        exports conformal comparison metrics to csv

        Args:
            model_metrics: Dictionary mapping model names to ModelMetrics
            n_lines: integer number of line
            output_path: string path for the comparison csv file
        """
        rows = []
        for model_name, metrics in model_metrics.items():
            for line_idx in range(n_lines):
                rows.append(
                    {
                        "model_name": model_name,
                        "line_index": line_idx,
                        "coverage": metrics.coverage[line_idx],
                        "width": metrics.width[line_idx],
                        "action_inf_coverage": metrics.action_inf_coverage[line_idx],
                    }
                )
        self._write_csv(rows, output_path)

    def export_timeseries(
        self,
        trajectories: list[Trajectory],
        model_metrics: dict[str, ModelMetrics],
        n_lines: int,
        output_path: str,
        stl_rule_names: list[str] | None = None,
    ) -> None:
        """
        export the timeseries data to csv

        Args:
            trajectories: List of Trajectory objects
            model_metrics: Dictionary mapping model names to ModelMetrics objects
            n_lines: integer number of lines
            output_path: String path for timeseries csv
            stl_rule_names: Optional list of STL rule names
        """
        rows = []
        for traj in trajectories:
            for ts in traj.timestamps:
                timestep_data = traj.timesteps[ts]
                if not timestep_data.is_complete():
                    continue

                actuals = timestep_data.real_rho
                forecasts = timestep_data.forecasted_rho

                for line_idx in range(n_lines):
                    row: dict[str, str | int | float | bool | None] = {
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                        "line_index": line_idx,
                        "true_rho": actuals[line_idx],
                        "forecast_rho": forecasts[line_idx],
                        "action_influenced_flag": timestep_data.is_action_influenced,
                    }

                    # conformal model columns
                    for model_name in model_metrics:
                        lower, upper = traj.get_predictions_at_timestamp(ts, model_name)
                        if lower and upper:
                            row[f"{model_name}_lower_bound"] = lower[line_idx]
                            row[f"{model_name}_upper_bound"] = upper[line_idx]
                            row[f"{model_name}_covered"] = (
                                lower[line_idx] <= actuals[line_idx] <= upper[line_idx]
                            )
                            row[f"{model_name}_predicted_safe"] = (
                                traj.is_conf_predicted_safe(model_name, line_idx)
                            )

                    # stl rule columns
                    if stl_rule_names:
                        for rule_name in stl_rule_names:
                            row[f"{rule_name}_predicted_safe"] = (
                                traj.is_stl_predicted_safe(rule_name, line_idx)
                            )

                    rows.append(row)

        self._write_csv(rows, output_path)

    def export_horizon_data(
        self,
        horizon_metrics: dict[str, list[HorizonMetrics]],
        output_path: str,
    ) -> None:
        """
        exports horizon analysis data to csv

        Args:
            horizon_metrics: dictionary mapping model names to lists of horizon metric dicts
            output_path: string path for horizon csv
        """
        rows = []
        for model_name, horizon_data in horizon_metrics.items():
            for step in horizon_data:
                rows.append(
                    {
                        "horizon": step.horizon,
                        "model_name": model_name,
                        "coverage": step.coverage,
                        "width": step.width,
                    }
                )
        self._write_csv(rows, output_path)

    def export_aggregated_comparison(
        self,
        all_conformal_metrics: list["ConformalMetrics"],
        output_path: str,
    ) -> None:
        """
        aggregates and exports the comparison metrics (coverage, width, action_inf_coverage)
        for the episodes

        Args:
            all_conformal_metrics: List of ConformalMetrics from each episode
            output_path: string path for horizon csv
        """
        if not all_conformal_metrics:
            self._write_csv([], output_path)

        # we get the line names from any conforma_metrics object
        line_names = all_conformal_metrics[0].line_names

        # collect metrics per (model, line)
        # (model_name, line_idx) -> list of {coverage, width, action_inf_coverage}
        metrics_dict = {}

        for cm in all_conformal_metrics:
            if cm.model_metrics is None:
                continue
            for model_name, metrics in cm.model_metrics.items():
                for line_idx in range(len(line_names)):
                    key = (model_name, line_idx)
                    if key not in metrics_dict:
                        metrics_dict[key] = []
                    metrics_dict[key].append(
                        {
                            "coverage": metrics.coverage[line_idx],
                            "width": metrics.width[line_idx],
                            "action_inf_coverage": metrics.action_inf_coverage[
                                line_idx
                            ],
                        }
                    )

        # aggregated stats for the agg csv
        rows = []
        for (model_name, line_idx), values in metrics_dict.items():
            coverages = [v["coverage"] for v in values]
            widths = [v["width"] for v in values]
            action_inf_coverages = [v["action_inf_coverage"] for v in values]

            rows.append(
                {
                    "model_name": model_name,
                    "line_index": line_idx,
                    "coverage_mean": np.mean(coverages),
                    "coverage_std": np.std(coverages),
                    "width_mean": np.mean(widths),
                    "width_std": np.std(widths),
                    "action_inf_coverage_mean": np.mean(action_inf_coverages),
                    "action_inf_coverage_std": np.std(action_inf_coverages),
                    "n_episodes": len(values),
                }
            )

        self._write_csv(rows, output_path)

    @staticmethod
    def _write_csv(
        data: list[dict[str, int | float | bool | str]], output_path: str
    ) -> None:
        """
        writes the data to the csv file

        Args:
            data: List of dictionaries to write
            output_path: String path for output file
        """
        ensure_dir(get_dir_name(output_path))
        pd.DataFrame(data).to_csv(output_path, index=False)


def generate_csvs(
    alpha: float,
    results: list[TestEpisodeWorkerReturn],
    line_names: list[str],
    output_dir: str,
    stl_verifiers: dict[str, STLWrapper] | None = None,
    conf_predictors: dict[str, ModelWrapper] | None = None,
) -> None:
    """
    exports per-episode and aggregated csvs for plotting

    Args:
        alpha: Alpha value used for config export
        results: List of (episode_num, traj_manager, success) tuples
        line_names: List of line names
        output_dir: Base output directory
        stl_verifiers: STL verifiers dict (optional)
        conf_predictors: predictors dict (optional)
    """

    # return early if no conf or stl predictors
    if conf_predictors is None and stl_verifiers is None:
        return

    print("Exporting csvs..")
    ensure_dir(output_dir)

    exporter = DataExporter()
    stl_rule_names = get_enabled_stl_rules()
    model_names = get_enabled_models()

    aggregated_dir = ensure_dir(output_dir, "aggregated_csvs")
    classifier = ClassificationAnalyser(aggregated_dir)

    n_lines = len(line_names)

    metrics_for_aggregation = []

    for result in results:
        ep_num, traj_manager = result.episode_num, result.traj_man
        if not traj_manager:
            continue

        trajectories = traj_manager.get_all_trajectories()
        if not trajectories:
            continue

        episode_dir = ensure_dir(output_dir, f"episode_{ep_num}")
        csv_dir = ensure_dir(episode_dir, "csvs")

        # Compute metrics
        cm = ConformalMetrics(line_names)
        model_metrics = cm.calculate_model_metrics(trajectories)
        horizon_metrics = cm.calculate_horizon_metrics(trajectories)

        # Export episode data
        exporter.export_config(
            alpha=alpha,
            line_names=line_names,
            model_names=model_names,
            stl_rule_names=stl_rule_names,
            output_path=os.path.join(csv_dir, "config.csv"),
        )

        exporter.export_conformal_comparison(
            model_metrics,
            n_lines,
            os.path.join(csv_dir, "conformal_data.csv"),
        )

        exporter.export_horizon_data(
            horizon_metrics,
            os.path.join(csv_dir, "conformal_data_horizon.csv"),
        )

        exporter.export_timeseries(
            trajectories,
            model_metrics,
            n_lines,
            os.path.join(csv_dir, "timeseries.csv"),
            stl_rule_names=stl_rule_names,
        )

        # the episode only is aggregated if has sufficient data (see has_sufficient_data)
        if traj_manager.has_sufficient_data():
            metrics_for_aggregation.append(cm)

            if conf_predictors:
                classifier.export_episode_conformal(
                    traj_manager, conf_predictors, csv_dir
                )
            if stl_verifiers:
                classifier.export_episode_stl(traj_manager, stl_verifiers, csv_dir)

    # aggregated exports
    if metrics_for_aggregation:
        exporter.export_aggregated_comparison(
            metrics_for_aggregation,
            os.path.join(aggregated_dir, "aggregated_comparison.csv"),
        )

        # aggregated config
        exporter.export_config(
            alpha=alpha,
            line_names=line_names,
            model_names=model_names,
            output_path=os.path.join(aggregated_dir, "config.csv"),
            stl_rule_names=stl_rule_names,
        )

    classifier.export_classification_agg_metrics()

    print("Export is complete")
