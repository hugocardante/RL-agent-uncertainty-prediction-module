from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from tqdm import tqdm

import config
from utils.data_structures import SimpleObservation, TrajectoryManager
from utils.environment import (
    get_warmup_cutoff_time,
    is_action_empty,
    needs_new_forecast,
)
from utils.global_utils import get_forecaster_model
from utils.model_wrapper import ModelWrapper
from utils.parallel import (
    TestEpisodeWorkerReturn,
    run_parallel_jobs,
    test_episode_worker,
)
from utils.stl_wrapper import STLWrapper


def run_test_episode(
    env: Environment,
    agent: BaseAgent,
    episode_num: int,
    predictors: dict[str, ModelWrapper],
    verifiers: dict[str, STLWrapper],
    chronic_id: int,
) -> TrajectoryManager:
    """
    runs a test episode with real-time conformal prediction and stl verificiation

    Args:
        env: Grid2Op environment
        agent: Grid2Op agent for decision making
        model_predict: Trained prediction model for load/generation forecasting
        episode_num: Integer episode number (starts at 1)
        predictors: Dictionary mapping model names to ModelWrapper objects
        verifiers: Dictionary mapping stl names to STLWrapper objects
        chronic_id: Integer chronic scenario identifier (starts at config.BASE_CHRONIC)

    Returns:
        TrajectoryManager containing all trajectories from the episode
    """
    print(f"Running Test Episode {episode_num}")

    n_steps = config.STEPS_TO_RUN

    # initialize the environment using a different seed
    # for each episode (it could be the same)
    # we also set the chronic id for each test episode, to
    # make sure we are testing the right chronic
    env.seed(config.BASE_ENV_SEED + chronic_id)
    env.set_id(chronic_id)
    obs = env.reset()

    forecaster = get_forecaster_model()

    # get the warmup cutoff time in the beginning of episode
    warmup_cutoff = get_warmup_cutoff_time(obs.get_time_stamp(), config.WARMUP_STEPS)

    observations = []
    trajectory_manager = TrajectoryManager(warmup_cutoff)
    last_forecasted_time = None
    expected_forecast_times = set()

    done, reward, step = False, 0, 0

    with tqdm(
        total=n_steps, desc=f"Test Ep{episode_num}", unit="step", leave=False
    ) as pb:
        while not done and step < n_steps:
            current_datetime = obs.get_time_stamp()
            simple_obs = SimpleObservation(obs)
            action = agent.act(obs, reward, done)
            is_action = not is_action_empty(action, env)
            real_rho_values = obs.rho.copy().tolist()

            # update trajectories with real data
            trajectory_manager.update_with_real_data(
                current_datetime, real_rho_values, simple_obs, predictors
            )

            force_forecast = is_action and config.PERFORM_FORECAST_ON_ACTION
            need_regular_forecast = needs_new_forecast(
                current_datetime,
                last_forecasted_time,
                expected_forecast_times,
            )
            if force_forecast or need_regular_forecast:
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
                        conf_predictors=predictors,
                        stl_verifiers=verifiers,
                    )
                    expected_forecast_times = new_timestamps
                    last_forecasted_time = max(agent_forecasts.keys())

            observations.append(simple_obs)
            if config.MAX_OBSERVATIONS is not None:
                if len(observations) > config.MAX_OBSERVATIONS:
                    observations.pop(0)

            # Advance the simulation
            obs, reward, done, _ = env.step(action)
            step += 1
            pb.update(1)

    return trajectory_manager


def run_test_episodes(
    n_test_episodes: int,
    predictors: dict[str, ModelWrapper],
    stl_verifiers: dict[str, STLWrapper],
    chronic_start: int,
) -> list[TestEpisodeWorkerReturn]:
    """
    Execute multiple test episodes with parallel execution and aggregate results.

    Args:
        n_test_episodes: Integer number of test episodes to run
        predictors: Dictionary mapping model names to ModelWrapper objects
        stl_verifiers: Optional dictionary mapping rule names to STLWrapper objects
        chronic_start: Integer starting chronic scenario ID

    Returns:
        List of TestEpisodeWorkerReturn (episode_num, trajectory_manager) tuples
    """

    enabled_predictors = {name: model for name, model in predictors.items()}

    print(
        f"Running {n_test_episodes} test episodes with {config.MAX_WORKERS_TESTING} workers"
    )

    # arguments for parallel execution
    test_args = [
        (
            ep + 1,  # Episode numbers start at 1
            enabled_predictors,
            chronic_start + ep,
            config.TESTING_LINES_ATTACKED,
            stl_verifiers,
        )
        for ep in range(n_test_episodes)
    ]

    # execute episodes in parallel
    # we can do this because we are sure the function returns
    # the list we are expecting..
    results: list[TestEpisodeWorkerReturn] = run_parallel_jobs(
        test_args,
        test_episode_worker,
        max_workers=config.MAX_WORKERS_TESTING,
        desc="Test Episodes",
    )

    return results
