import datetime

import numpy as np
from grid2op.Observation import BaseObservation

import config
from utils.model_wrapper import ModelWrapper
from utils.stl_wrapper import STLWrapper


class SimpleObservation:
    """
    A more memory-efficient way of representing a Grid2Op observation.

    Stores only essential data needed for conformal prediction, significantly
    reducing memory usage compared to full Grid2Op observation objects.
    We can add more attributes as we like..

    Trade-off: We lose access to the full Grid2Op observation but gain
    some memory savings
    """

    def __init__(self, obs: BaseObservation):
        """
        Create a SimpleObservation from a Grid2Op observation
        Args:
            Grid2Op observation with the environment state
        """
        self.rho: list[float] = obs.rho.tolist()
        self.load_p: list[float] = obs.load_p.tolist()
        self.load_q: list[float] = obs.load_q.tolist()
        self.gen_p: list[float] = obs.gen_p.tolist()
        self.gen_v: list[float] = obs.gen_v.tolist()
        self.hour_of_day: int = obs.hour_of_day
        self.day_of_week: int = obs.day_of_week
        self.minute_of_hour: int = obs.minute_of_hour
        self.day: int = obs.day
        self.month: int = obs.month
        self.year: int = obs.year
        self.current_step: int = obs.current_step


class TimestepData:
    """
    A struct for all data related to a single timestamp

    Args:
        forecasted_rho: list of the forecasted rho values
        forecast_observation: SimpleObservation object
        is_action_influenced: boolean flag saying if this timestep is action-influenced or not
        horizon: integer specifying what is the forecast horizon for this timestep
    """

    def __init__(
        self,
        forecasted_rho: list[float],
        forecast_observation: SimpleObservation,
        is_action_influenced: bool,
        horizon: int,
    ):
        self.forecasted_rho: list[float] = forecasted_rho
        self.forecast_observation: SimpleObservation = forecast_observation
        self.is_action_influenced: bool = is_action_influenced
        self.horizon: int = horizon
        self.real_rho: list[float] = []
        self.real_observation: SimpleObservation | None = None

        # conformal intervals for this specific datapoint
        # {model_name -> {"upper or lower" -> upper/lower vals}
        self.predictions: dict[str, dict[str, list[float]]] = {}

    def is_complete(self) -> bool:
        """
        check if real observations have been added to this timestep
        Returns:
            boolean indicating if real_rho has been set
        """
        return len(self.real_rho) > 0


class Trajectory:
    """
    A forecast event containing up to config.HORIZON timestep predictions (each being a TimestepData object)
    """

    def __init__(
        self,
        agent_forecasts: dict[datetime.datetime, list[float]],
        obs_dict: dict[datetime.datetime, SimpleObservation],
        is_action_influenced: bool,
    ):
        """
        Start a trajectory from forecast data
        Args:
            agent_forecasts: dictionary mapping datetime objects to forecasted rho values
            obs_dict: dictionary mapping datetime objects to simpleobservation objects
            is_action_influenced: boolean flag that says if this Trajectory was the result of the RL agent action
        """
        self.timesteps: dict[datetime.datetime, TimestepData] = {}
        for i, ts in enumerate(agent_forecasts.keys()):
            self.timesteps[ts] = TimestepData(
                forecasted_rho=agent_forecasts[ts],
                forecast_observation=obs_dict[ts],
                is_action_influenced=is_action_influenced,
                horizon=i + 1,
            )
        self.timestamps: list[datetime.datetime] = list(self.timesteps.keys())

        # The idea here is to map:
        # Rule name -> {line_idx -> bool}
        self.stl_safety_results: dict[str, dict[int, bool]] = {}

    @property
    def forecast_trajectory(self) -> list[list[float]]:
        """
        gets the forecasted rho values for all timesteps
        Returns:
            list of forecasted rho value lists, one per timestep
        """
        return [self.timesteps[ts].forecasted_rho for ts in self.timestamps]

    @property
    def hourly_stats(self) -> list[float]:
        """
        computes summary statistics from the forecast trajectory

        A trajectory typically is one hour of forecasts. For each power line,
        we compute the mean and std of its forecasted rho values across all timestamps,
        then summarize into 4 features:

            mean of per-line averages, std of per-line averages, mean of per-line stds and max of per-line stds

        Returns:
            list of floats with some statistics about the hour for the entire traj, or empty list if no forecast data.
        """
        if not self.timestamps:
            return []

        arr = np.array(self.forecast_trajectory)
        # calculated for each one of the lines
        # by avereging/std the values of each forecast horizon
        # for each one of the lines
        per_line_avg = np.mean(arr, axis=0)
        per_line_std = np.std(arr, axis=0)

        # we calculate some metrics from those two arrays
        return [
            float(np.max(per_line_avg)),
            float(np.mean(per_line_avg)),
            float(np.std(per_line_avg)),
            float(np.mean(per_line_std)),
            float(np.max(per_line_std)),
        ]

    def compute_stl_robustness(
        self, line_idx: int, use_predicted: bool = True
    ) -> float:
        """
        computes STL robustness for a specific power line
        Args:
            line_idx: index of the power line to compute robustness for
            use_predicted: boolean flag to use forecasted rho (True) or real rho (False)
        Returns:
            minimum robustness value across all timesteps (or nan if data unavailable..)
        """

        if use_predicted:
            rho_list = [self.timesteps[ts].forecasted_rho for ts in self.timestamps]
        else:
            rho_list = [
                self.timesteps[ts].real_rho
                for ts in self.timestamps
                if self.timesteps[ts].is_complete()
            ]

        # rho_values is an array of size at most config.FORECAST_HORIZON
        # which in each position of the array, holds an array of rho_values for
        # each of the lines
        rho_values = np.array(rho_list)

        # NOTE: The h function used is h(x) = 0.95 - x, if the RHO_SAFETY_THRESHOLD is 0.95
        h_values = config.RHO_SAFETY_THRESHOLD - rho_values[:, line_idx]
        result = float(np.min(h_values))

        return result

    def add_real_data(
        self,
        timestamp: datetime.datetime,
        real_rho: list[float],
        real_obs: SimpleObservation,
    ):
        """
        adds the true values/observation to a timestep

        Args:
            timestamp: datetime object identifying the timestep
            real_rho: list of actual rho values observed
            real_obs: SimpleObservation object with real observation data
        """
        if timestamp in self.timesteps:
            self.timesteps[timestamp].real_rho = real_rho
            self.timesteps[timestamp].real_observation = real_obs

    def truncate(self, truncation_timestamp: datetime.datetime):
        """
        removes all timesteps greater or equal than the specified datetime
        Args:
            truncation_timestamp: datetime object marking the truncation point
        """
        timestamps_to_remove = [
            ts for ts in self.timestamps if ts >= truncation_timestamp
        ]
        if not timestamps_to_remove:
            return

        for ts in timestamps_to_remove:
            if ts in self.timesteps:
                del self.timesteps[ts]

        self.timestamps = sorted(self.timesteps.keys())

    def is_complete(self) -> bool:
        """
        checks if all timesteps in this Trajctory have received real observations

        Returns:
            True if trajectory is complete, False otherwise
        """
        for ts in self.timestamps:
            td = self.timesteps[ts]
            if not td.is_complete():
                return False
        return True

    def generate_conformal_predictions(self, predictors: dict[str, ModelWrapper]):
        """
        generates prediction intervals for all timesteps by using the conformal predictors
        Args:
            predictors: dictionary mapping model names to ModelWrapper objects
        """
        for model_name, predictor in predictors.items():
            try:
                # we pass this specific Trajectory directly to the conformal predictor
                all_lower, all_upper = predictor.predict(self)
                for i, ts in enumerate(self.timestamps):
                    self.timesteps[ts].predictions[model_name] = {
                        "lower": all_lower[i],
                        "upper": all_upper[i],
                    }
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")

    def get_predictions_at_timestamp(
        self, timestamp: datetime.datetime, model_name: str
    ) -> tuple[list[float] | None, list[float] | None]:
        """
        gets the conformal prediction interval bounds for a specific timestep and model
        Args:
            timestamp: datetime object identifying the timestep
            model_name: name of the conformal prediction model
        Returns:
            tuple of (lower_bounds, upper_bounds) lists, or (None, None) if not available
        """
        if (
            timestamp in self.timesteps
            and model_name in self.timesteps[timestamp].predictions
        ):
            preds = self.timesteps[timestamp].predictions[model_name]
            return preds["lower"], preds["upper"]
        return None, None

    def generate_stl_predictions(self, stl_verifiers: dict[str, STLWrapper]):
        """
        generates stl predictions for this trajectory

        stores the results in self.stl_safety_results because the results are for a trajectory (safe / not safe).

        Args:
            stl_verifiers: dictionary mapping rule names to STLWrapper objects
        """
        for rule_name, verifier in stl_verifiers.items():
            try:
                # the safety results come in the form: {line_idx: bool}
                results = verifier.predict(self)

                if results is not None:
                    # we store the results for this trajectory
                    self.stl_safety_results[rule_name] = results
            except Exception as e:
                print(f"Error generating STL predictions with {rule_name}: {e}")

    def is_stl_predicted_safe(self, rule_name: str, line_idx: int) -> bool | None:
        """
        Checks if a trajectory is STL (predicted) safe for a particular rule and line.
        This is supposed to be used only after running the episodes, so that the values
        can be filled in.

        Args:
            rule_name: name of the STL rule
            line_idx: index of the power line

        Returns:
            True if safe, False if unsafe, None if no STL result exists
        """
        if rule_name not in self.stl_safety_results:
            return None

        safety_dict = self.stl_safety_results[rule_name]
        if line_idx not in safety_dict:
            return None

        return safety_dict[line_idx]

    def is_conf_predicted_safe(self, model_name: str, line_idx: int) -> bool:
        """
        This is equivalent to checking if a Trajectory (predicted) is STL safe, but
        for the conformal models (check is_stl_predicted_safe)
        We check if any timestep's upper bound prediction > threshold for a particualar line
        This is just to see if it is predicted to be safe or not, so that we can in some sense
        compare this to the results given by the STL rules..

        Args:
            model_name: Name of the prediction model
            line_idx: Index of the power line

        Returns:
            True if any prediction exceeds threshold (unsafe), False otherwise
        """
        for ts in self.timestamps:
            td = self.timesteps[ts]
            if model_name in td.predictions:
                upper = td.predictions[model_name]["upper"]
                if upper and upper[line_idx] > config.RHO_SAFETY_THRESHOLD:
                    return False
        return True

    def is_actually_safe(
        self,
        line_idx: int,
    ) -> bool:
        """
        checks if the trajectory was actually safe for a specific line
        (i.e., real_rho <= threshold for all timesteps)
        """

        for ts in self.timestamps:
            td = self.timesteps[ts]

            assert (
                td.real_rho
            ), "This should never happen, is_actually_safe is working on non-completed trajectory"
            if td.real_rho[line_idx] > config.RHO_SAFETY_THRESHOLD:
                return False
        return True


class TrajectoryManager:
    """
    Manager for managing active and completed forecast trajectories.
    This class takes care of and "saves" an entire episode.
    """

    def __init__(self, warmup_cutoff: datetime.datetime):
        self.active_trajectories: list[Trajectory] = []
        self.completed_trajectories: list[Trajectory] = []
        self.last_trajectory_added: Trajectory = None
        self.is_warmup_filtered: bool = False
        self.cutoff_time: datetime.datetime = warmup_cutoff

    def add_forecast_event(
        self,
        agent_forecasts: dict[datetime.datetime, list[float]],
        obs_dict: dict[datetime.datetime, SimpleObservation],
        is_action_inf: bool,
        is_action_driven: bool,
        conf_predictors: dict[str, ModelWrapper] | None = None,
        stl_verifiers: dict[str, STLWrapper] | None = None,
    ):
        """
        adds a new forecast trajectory to the manager

        Args:
            agent_forecasts: dictionary mapping datetime -> list of floats
            obs_dict: dictionary mapping datetime -> SimpleObservation
            is_action_inf: boolean representing if this forecast event is action influenced
            is_action_driven: boolean flag indicating if forecast was triggered by action
            conf_predictors: dictionary of conformal prediction models (for test episodes)
            stl_verifiers: dictionary of stl verifiers (for test episodes)
        """
        new_trajectory = Trajectory(agent_forecasts, obs_dict, is_action_inf)

        # For test episodes, when we get a forecast, we generate the predictions..
        if conf_predictors:
            new_trajectory.generate_conformal_predictions(conf_predictors)

        # same for the stl verifiers
        if stl_verifiers:
            new_trajectory.generate_stl_predictions(stl_verifiers)

        # truncate the any active trajectories, if this resulted from an action
        # being identified
        if is_action_driven and new_trajectory.timestamps:
            truncation_point = new_trajectory.timestamps[0]
            for trajectory in self.active_trajectories:
                trajectory.truncate(truncation_point)

        self.active_trajectories.append(new_trajectory)
        self.last_trajectory_added = new_trajectory

    def update_with_real_data(
        self,
        timestamp: datetime.datetime,
        rho_values: list[float],
        obs: SimpleObservation,
        conf_predictors: dict[str, ModelWrapper] | None = None,
    ):
        """
        updates active trajectory with real observation data
        Args:
            timestamp: datetime object for the observation
            rho_values: list of actual rho values observed
            obs: SimpleObservation object with real observation data
            conf_predictors: dictionary of conformal predictors (for test episodes)
        """

        for trajectory in self.active_trajectories:
            # It only adds the data, if the timestamp belongs to any of the active trajectories.
            trajectory.add_real_data(timestamp, rho_values, obs)

        # this is only for during testing, in which we update the adaptive alphas
        # (models that use the ACI update rule) at this point we already generated the trajectory and
        # obtained the intervals for the k horizons, and are updating this as simulation
        # real values become available
        if conf_predictors and timestamp >= self.cutoff_time:
            self.update_alphas(timestamp, rho_values, conf_predictors)

        self._archive_completed()

    def update_alphas(
        self,
        timestamp: datetime.datetime,
        rho_values: list[float],
        predictors: dict[str, ModelWrapper],
    ):
        """
        update the adaptive alphas (in methods that have the ACI update rule)

        Args:
            timestamp: datetime object for the observation
            rho_values: list of actual rho values observed
            predictors: dictionary mapping model names to ModelWrapper objects
        """

        for traj in self.active_trajectories:
            if timestamp in traj.timesteps:
                td = traj.timesteps[timestamp]
                for model_name, predictor in predictors.items():
                    if (
                        hasattr(predictor, "update_alphas")
                        and model_name in td.predictions
                    ):
                        preds = td.predictions[model_name]
                        predictor.update_alphas(
                            rho_values, preds["lower"], preds["upper"]
                        )

    def _archive_completed(self):
        """
        moves completed trajectories from active to completed list
        """
        active = []

        for traj in self.active_trajectories:
            if traj.is_complete():
                self.completed_trajectories.append(traj)
            else:
                active.append(traj)

        self.active_trajectories = active

    def filter_warmup(self):
        """
        removes warmup timesteps from all trajectories
        """
        if self.is_warmup_filtered:
            return

        # filtering each trajectory in-place
        for traj in self.get_all_trajectories():
            timestamps_to_remove = [
                ts for ts in traj.timestamps if ts < self.cutoff_time
            ]

            for ts in timestamps_to_remove:
                del traj.timesteps[ts]

            traj.timestamps = sorted(traj.timesteps.keys())

        # removing empty trajectories
        self.completed_trajectories = [
            t for t in self.completed_trajectories if t.timestamps
        ]
        self.active_trajectories = [t for t in self.active_trajectories if t.timestamps]

        # updating last_trajectory_added reference if it was filtered away
        if self.last_trajectory_added and not self.last_trajectory_added.timestamps:
            self.last_trajectory_added = (
                self.active_trajectories[-1] if self.active_trajectories else None
            )

        self.is_warmup_filtered = True

    def get_completed_trajectories(self) -> list[Trajectory]:
        """
        gets all completed trajectories from this TrajectoryManager

        Returns:
            list of completed trajectories
        """
        all_trajectories = self.get_all_trajectories()
        return [traj for traj in all_trajectories if traj.is_complete()]

    def get_all_trajectories(self) -> list[Trajectory]:
        """
        gets all trajectories from this TrajectoryManager

        Returns:
            list of all trajectories
        """
        return self.completed_trajectories + self.active_trajectories

    def has_sufficient_data(self) -> bool:
        """
        Checks if this trajectory manager has sufficient complete datapoints.
        This is essentially for plotting and statistics.
        If an episode does not have enough datapoints, it can hurt the aggregated statistics
        For example, if an episode has only two datapoints, I dont think it should count for statistics, because
        it will either have 0 coverage, 50 coverage or 100 coverage

        Returns:
            True if has_sufficient_data otherwise False
        """
        min_datapoints = config.MIN_DATAPOINTS_FOR_AGGREGATION

        all_trajectories = self.active_trajectories + self.completed_trajectories

        if not all_trajectories:
            return False

        total_datapoints = sum(
            sum(1 for ts in traj.timestamps if traj.timesteps[ts].is_complete())
            for traj in all_trajectories
        )

        return total_datapoints >= min_datapoints
