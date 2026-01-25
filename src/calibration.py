from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from tqdm import tqdm

import config
from utils.cache import DataManager, extract_line_number
from utils.data_structures import SimpleObservation, TrajectoryManager
from utils.environment import (
    get_warmup_cutoff_time,
    is_action_empty,
    needs_new_forecast,
)
from utils.global_utils import get_env_details, get_forecaster_model
from utils.parallel import calibration_worker, run_parallel_jobs


def run_calibration_episode(
    env: Environment,
    agent: BaseAgent,
    chronic_id: int,
) -> TrajectoryManager:
    """
    Executes a single episode to collect calibration data

    Args:
        env: Grid2Op environment
        agent: CurriculumAgent (or another agent)
        model_predict: Trained prediction model for load/generation forecasting
        chronic_id: Integer chronic scenario identifier

    Returns:
        TrajectoryManager with collected data
    """
    observations = []
    last_forecasted_time = None
    expected_forecast_times = set()

    forecaster = get_forecaster_model()

    # always changing the seed, but deterministic for
    # every different run (same episode chronics)
    # no particular reason to do this, could be only config.BASE_ENV_SEED
    # but this way we have a different seed every episode
    env.seed(config.BASE_ENV_SEED + chronic_id)

    # we set the id of the chronic, so that we use the
    # particular chronic we are interested in
    env.set_id(chronic_id)
    obs = env.reset()

    # get the warmup cutoff time in the beginning of episode
    warmup_cutoff = get_warmup_cutoff_time(obs.get_time_stamp(), config.WARMUP_STEPS)

    # this trajectory manager manages this entire particular
    # episode. We can think of it like the structure that holds
    # all the trajectories in this calibration episode (see TrajectoryManager)
    trajectory_manager = TrajectoryManager(warmup_cutoff)

    done, reward, step = False, 0, 0
    n_steps = config.STEPS_TO_RUN

    with tqdm(total=n_steps, desc=f"Running calibration (chronic {chronic_id})") as pb:
        while not done and step < n_steps:
            current_datetime = obs.get_time_stamp()
            simple_obs = SimpleObservation(obs)
            action = agent.act(obs, reward, done)
            is_action = not is_action_empty(action, env)

            trajectory_manager.update_with_real_data(
                current_datetime,
                obs.rho.copy().tolist(),
                simple_obs,
            )

            force_forecast = is_action and config.PERFORM_FORECAST_ON_ACTION
            need_regular_forecast = needs_new_forecast(
                current_datetime, last_forecasted_time, expected_forecast_times
            )

            if force_forecast or need_regular_forecast:
                # when the forecast is forces, it means that
                # we detected that the agent did a non-empty action
                # so we ignore all the forecasted timesteps that we haven't been
                # to yet, and we forecast again, marking all as action influenced
                if force_forecast:
                    expected_forecast_times.clear()

                (agent_forecasts, new_timestamps, forecast_obs_dict) = (
                    forecaster.perform_forecast(obs, observations[:-1], agent)
                )

                if agent_forecasts:
                    trajectory_manager.add_forecast_event(
                        agent_forecasts,
                        forecast_obs_dict,
                        is_action,
                        is_action_driven=force_forecast,
                    )
                    expected_forecast_times = new_timestamps
                    last_forecasted_time = max(agent_forecasts.keys())

            observations.append(simple_obs)
            # This is not necessary, but can increase the memory
            # efficiency by a little bit by keeping the array smaller
            if config.MAX_OBSERVATIONS is not None:
                if len(observations) > config.MAX_OBSERVATIONS:
                    observations.pop(0)

            # Advance the real simulation
            obs, reward, done, _ = env.step(action)
            step += 1
            pb.update(1)

    return trajectory_manager


def run_ensemble_calibration(base_chronic: int) -> DataManager:
    """
    executes ensemble calibration for all power lines

    Args:
        base_chronic: Integer starting chronic id (config.BASE_CRHONIC)
    Returns:
        DataManager that contains the attacked lines
    """
    data_manager = DataManager()

    _, all_available_lines = get_env_details()

    # In this loop, line_to_attack is one of power lines.
    # this way we will attack one line at a time, and train one model
    # that has adversarial attacks only on that particular line
    for line_idx, line_name in enumerate(all_available_lines):
        attacked_line = extract_line_number(line_name)
        attacked_lines = [attacked_line]

        print(f"Calibration in ensemble mode. Line {line_idx}: {line_name}")

        episodes_to_run = []
        episodes_cached = 0

        for episode in range(config.CALIB_EPISODES):
            if data_manager.episode_exists(attacked_lines, episode):
                episodes_cached += 1
                # we run it anyway, because we are ignoring cache
                if config.IGNORE_CACHE_CALIBRATION:
                    episodes_to_run.append(episode)
            else:
                episodes_to_run.append(episode)

        if episodes_to_run:
            episode_args = [
                (
                    attacked_lines,
                    [line_name],
                    base_chronic + ep,
                    ep,
                )
                for ep in episodes_to_run
            ]

            run_parallel_jobs(
                episode_args,
                calibration_worker,
                max_workers=config.MAX_WORKERS_CALIBRATION,
                desc=f"Line {line_idx} Episodes",
            )

    # Store all line numbers that were attacked during calibration
    all_attacked_lines = [extract_line_number(line) for line in all_available_lines]
    data_manager.set_attacked_lines(all_attacked_lines)
    return data_manager


def run_single_calibration(chronic: int) -> DataManager:
    """
    Execute single-line calibration episodes (corresponding to Case Study 1 / Case Study 2).
    Case Study 1: Normal behaviour, one model trained with no attacks in any lines
    Case Study 2: Adversarial behaviour, one model is trained with attacks in the selected lines (config.CALIBRATION_LINES_ATTACKED)

    Args:
        chronic: Integer starting chronic scenario ID
    Returns:
        DataManager object that contains the attacked lines
    """
    data_manager = DataManager()

    # Extract line numbers from config
    attacked_lines = [
        extract_line_number(line) for line in config.CALIBRATION_LINES_ATTACKED
    ]

    print(f"Calibration in single mode. Attacked lines: {attacked_lines}")
    episodes_to_run = []
    episodes_cached = 0

    for episode in range(config.CALIB_EPISODES):
        # if it exists we can return it directly
        if data_manager.episode_exists(attacked_lines, episode):
            episodes_cached += 1
            # unless we want to ignore it..
            if config.IGNORE_CACHE_CALIBRATION:
                episodes_to_run.append(episode)
        else:
            episodes_to_run.append(episode)

    if episodes_to_run:
        episode_args = [
            (
                attacked_lines,
                config.CALIBRATION_LINES_ATTACKED,
                chronic + ep,
                ep,
            )
            for ep in episodes_to_run
        ]

        run_parallel_jobs(
            episode_args,
            calibration_worker,
            max_workers=config.MAX_WORKERS_CALIBRATION,
            desc="Single Mode Episodes",
        )

    data_manager.set_attacked_lines(attacked_lines)
    return data_manager
