import numpy as np
import numpy.typing as npt

import config
from utils.cache import DataManager
from utils.data_structures import SimpleObservation, Trajectory, TrajectoryManager
from utils.model_wrapper import ModelWrapper, get_enabled_models
from utils.parallel import (
    ModelTrainingWorkerReturn,
    model_training_worker,
    run_parallel_jobs,
)
from utils.stl_wrapper import STLWrapper, get_enabled_stl_rules


def extract_system_state_features(
    obs: SimpleObservation,
    forecast_rho_values: list[float],
    forecast_horizon: int,
    hourly_stats: list[float],
    is_action_influenced: bool,
) -> npt.NDArray[np.float64]:
    """
    extracts feature vector for conformal prediction models and tries
    to represent the state of the system from an observation

    Args:
        obs: SimpleObservation object with current grid state
        forecast_rho_values: List of forecasted rho values
        forecast_horizon: Integer indicating the forecast step ahead
        hourly_stats: List with hourly statistics of the Trajectory that contains this timestep, or None
        is_action_influenced: Boolean saying if this timestep was action influenced

    Returns:
        numpy array of extracted features
    """
    features: list[float] = [forecast_horizon]

    # calendar features (cyclical encoding)
    hour = obs.hour_of_day
    day_of_week = obs.day_of_week
    features.extend(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * day_of_week / 7.0),
            np.cos(2 * np.pi * day_of_week / 7.0),
        ]
    )

    # rho values and statistics
    rho_values = np.array([float(r) for r in forecast_rho_values])
    features.extend(rho_values)
    features.extend(
        [
            float(np.mean(rho_values)),
            float(np.std(rho_values)),
            float(np.max(rho_values)),
            float(np.min(rho_values)),
        ]
    )

    # generation statistics
    gen_p = np.array(obs.gen_p, dtype=float)
    total_gen = np.sum(gen_p)
    features.extend([total_gen, float(np.std(gen_p)), np.max(gen_p), np.min(gen_p)])

    # load statistics
    load_p = np.array(obs.load_p, dtype=float)
    total_load = np.sum(load_p)
    features.extend([total_load, float(np.std(load_p)), np.max(load_p), np.min(load_p)])

    # load/gen balance
    features.append(total_load - total_gen)

    # action-influenced info
    features.append(1.0 if is_action_influenced else 0.0)

    # hourly stats
    features.extend(hourly_stats)

    feature_array = np.array(features, dtype=float)

    return feature_array


def extract_trajectory_features(
    traj: Trajectory, n_lines: int
) -> npt.NDArray[np.float64] | None:
    """
    extracts features from a trajectory
    at the moment it is being used for the kNN (normlised) STL rule only


    Args:
        traj: Trajectory object from which to extract the features
        n_lines: Integer number of lines
    """
    if not traj.timestamps:
        return None

    first_ts = traj.timestamps[0]
    first_obs = traj.timesteps[first_ts].forecast_observation

    features = []

    # calendar features (cyclical encoding)
    hour = first_obs.hour_of_day
    day_of_week = first_obs.day_of_week
    features.extend(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * day_of_week / 7.0),
            np.cos(2 * np.pi * day_of_week / 7.0),
        ]
    )

    # rho matrix
    rho_matrix = []
    for ts in traj.timestamps:
        forecasted_rho = traj.timesteps[ts].forecasted_rho
        if forecasted_rho and len(forecasted_rho) >= n_lines:
            rho_matrix.append(forecasted_rho[:n_lines])
        else:
            return None

    if not rho_matrix:
        return None

    rho_array = np.array(rho_matrix)

    # per-line features
    for line_idx in range(n_lines):
        line_rho = rho_array[:, line_idx]

        features.extend(
            [
                np.max(line_rho),
                np.mean(line_rho),
                line_rho[-1] - line_rho[0],  # end - begin
                np.sum(line_rho > 0.9) / len(line_rho),  # "time" in danger
                np.std(line_rho),
            ]
        )

    # system-wide features
    gen_p = np.array(first_obs.gen_p, dtype=float)
    load_p = np.array(first_obs.load_p, dtype=float)
    features.extend(
        [
            np.sum(gen_p),
            np.sum(load_p),
            np.sum(gen_p) - np.sum(load_p),
        ]
    )

    return np.array(features, dtype=float)


def prepare_calibration_data(
    calib_managers: list[TrajectoryManager],
    n_lines: int,
    model_type: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Transform calibration episodes into the training data arrays.

    Args:
        calib_managers: List of TrajectoryManager objects (one for each episode)
        n_lines: Integer number of power lines in the grid
        model_type: String identifier for the model type being trained

    Returns:
        Tuple of (features_array, forecast_array, actual_array) as numpy arrays
    """
    print(f"Processing calibration data for {model_type.upper()}...")

    final_calibration_data = []

    for episode_idx, traj_man in enumerate(calib_managers):
        traj_man.filter_warmup()
        all_trajectories = traj_man.get_all_trajectories()

        if not all_trajectories:
            print(
                f"  Episode {episode_idx + 1} has no trajectories after warmup, skipping..."
            )
            continue

        for traj in all_trajectories:
            # hourly statistics for this trajectory
            hourly_stats = traj.hourly_stats

            # Extract features from complete timesteps
            for ts in traj.timestamps:
                td = traj.timesteps[ts]

                if not td.is_complete():
                    continue

                assert td.real_observation is not None

                features = extract_system_state_features(
                    td.real_observation,
                    td.forecasted_rho,
                    td.horizon,
                    hourly_stats,
                    td.is_action_influenced,
                )

                final_calibration_data.append(
                    (
                        features,
                        np.array(td.forecasted_rho[:n_lines]),
                        np.array(td.real_rho[:n_lines]),
                    )
                )

    if not final_calibration_data:
        return np.array([]), np.array([]), np.array([])

    features_array = np.array([p[0] for p in final_calibration_data])
    forecast_array = np.array([p[1] for p in final_calibration_data])
    actual_array = np.array([p[2] for p in final_calibration_data])

    return features_array, forecast_array, actual_array


def train_model_from_storage(
    model_name: str,
    data_manager: DataManager,
    n_lines: int,
    alpha: float,
) -> ModelWrapper:
    """
    Create and train a model using the cached calibration data, referenced in the DataManager.
    Args:
        model_name: String name of the conformal prediction model
        data_manager: DataManager object containing calibration data
        n_lines: Integer number of power lines in the grid
        alpha: Float significance level for conformal prediction
    Returns:
        Fitted ModelWrapper object
    """
    model = ModelWrapper(model_name, n_lines, alpha, data_manager)
    model.fit()
    return model


def run_model_training(
    data_man: DataManager,
    n_lines: int,
    alpha: float,
) -> dict[str, ModelWrapper]:
    """
    trains all the selected conformal prediction models using the calibration data obtained in calibration
    model is loaded from cache if available (and config.IGNORE_CACHE_MODELS is False), otherwise trains and then
    saves it to cache

    Args:
        data_man: DataManager object containing calibration episodes / cached models
        n_lines: Integer number of power lines in the grid
        alpha: Float significance level for conformal prediction

    Returns:
        Dictionary mapping model names to trained ModelWrapper objects
    """
    enabled_models = get_enabled_models()
    n_episodes = config.CALIB_EPISODES

    print("Starting model training..")

    # to check which models exist in cache
    models_to_train = []
    cached_models = {}

    for model_name in enabled_models:
        if data_man.model_exists(model_name, n_episodes, alpha):
            print(f"  {model_name}: Found in cache...")
            model = data_man.load_model(model_name, n_episodes, alpha)
            # In this case, the model exists, but we want to ignore it and train it again
            # so we add it to training..
            if config.IGNORE_CACHE_MODELS:
                print(
                    "Model was cached before, but IGNORE_CACHE_MODELS is set to true in the config file"
                )
                models_to_train.append(model_name)
            elif model is not None:
                cached_models[model_name] = model
                print(f"{model_name}: Successfully loaded from cache")
            else:
                print(f"{model_name}: Failed to load, will retrain")
                models_to_train.append(model_name)
        else:
            print(f"{model_name}: Not in cache, will train")
            models_to_train.append(model_name)

    print(
        f"\nCache debug: {len(cached_models)} loaded, {len(models_to_train)} to train"
    )

    # we train the models that still need training
    newly_trained = {}
    if models_to_train:
        print(f"\nTraining {len(models_to_train)} models")
        training_args = [
            (name, data_man, n_lines, alpha, n_episodes) for name in models_to_train
        ]

        results: list[ModelTrainingWorkerReturn] = run_parallel_jobs(
            training_args,
            model_training_worker,
            max_workers=config.MAX_WORKERS_MODELS,
            desc="Model Training",
        )

        # Collecting results (models already saved by worker in the cache files)
        for result in results:
            if result.success:
                newly_trained[result.model_name] = result.trained_model

    # merging both the cached and trained models
    predictors = cached_models | newly_trained

    return predictors


def run_stl_training(
    data_man: DataManager,
    n_lines: int,
    alpha: float,
) -> dict[str, STLWrapper]:
    """
    trains all enabled stl rules using calibration data
    There is no cache yet for stl training (but it will follow the same structure)

    Args:
        data_man: DataManager object containing calibration episodes
        n_lines: Integer number of power lines in the grid
        alpha: Float significance level for conformal prediction

    Returns:
        Dictionary mapping rule names to trained STLWrapper objects
    """
    enabled_rules = get_enabled_stl_rules()

    if not enabled_rules:
        return {}

    print(f"\nTraining STL rules: {enabled_rules}")

    verifiers = {}
    for rule_name in enabled_rules:
        verifier = STLWrapper(rule_name, n_lines, alpha, data_man)
        verifier.fit()
        verifiers[rule_name] = verifier

    return verifiers
