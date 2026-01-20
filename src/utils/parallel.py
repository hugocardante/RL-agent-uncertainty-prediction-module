import gc
import multiprocessing as mp
import signal
import sys
from dataclasses import dataclass
from typing import Any, Callable

from tqdm import tqdm

import config
from utils.cache import DataManager
from utils.data_structures import TrajectoryManager
from utils.model_wrapper import ModelWrapper
from utils.stl_wrapper import STLWrapper


# The idea of these three classes is to have an easy return
# type that the functions can use without making it too complicated
@dataclass
class CalibrationWorkerReturn:
    episode_idx: int  # episode number
    status: str  # failed, success, cached


@dataclass
class ModelTrainingWorkerReturn:
    model_name: str
    trained_model: ModelWrapper | None
    success: bool


@dataclass
class TestEpisodeWorkerReturn:
    episode_num: int
    traj_man: TrajectoryManager | None


def setup_multiprocessing() -> None:
    """
    Configures multiprocessing settings if parallel execution is enabled
    Only sets spawn method if any parallel workers are configured
    """
    if any(
        [
            config.MAX_WORKERS_CALIBRATION > 1,
            config.MAX_WORKERS_MODELS > 1,
            config.MAX_WORKERS_TESTING > 1,
        ]
    ):
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing enabled with spawn method")

        signal.signal(signal.SIGINT, signal_handler)


def calibration_worker(
    args: tuple[list[int], list[str], int, int],
) -> CalibrationWorkerReturn:
    """
    Worker for calibration episodes (for both single and ensemble modes)

    Args:
        Tuple of (attacked_lines, lines_attacked_names, chronic_id, episode_idx)
            - attacked_lines: List of integer line numbers for caching
            - lines_attacked_names: List of line name strings for create_environment
            - chronic_id: Integer chronic scenario ID
            - episode_idx: Integer episode index (from 0 to config.CALIB_EPISODES)

    Returns:
        CalibrationWorkerReturn (episode_idx, status)
    """
    attacked_lines, lines_attacked_names, chronic_id, episode_idx = args

    print(f"Episode {episode_idx} for attacked lines {attacked_lines}")

    data_manager = DataManager()

    if data_manager.episode_exists(attacked_lines, episode_idx):
        # we only want to return previously ran episodes if this flag is not true
        if not config.IGNORE_CACHE_CALIBRATION:
            print(f"  Episode {episode_idx}: Loading from cache")
            return CalibrationWorkerReturn(episode_idx=episode_idx, status="cached")

    print(f"  Episode {episode_idx}: Running episode")

    try:
        from calibration import run_calibration_episode
        from utils.environment import create_environment

        env, agent = create_environment(config.BASE_ENV_SEED, lines_attacked_names)

        trajectory_manager = run_calibration_episode(
            env,
            agent,
            chronic_id,
        )

        trajectory_manager.filter_warmup()

        env.close()
        del env, agent
        gc.collect()

        # we save each episode after completion. when config.IGNORE_CACHE_CALIBRATION
        # is True, this will overwrite the previous episode (if there was any)
        data_manager.save_episode(attacked_lines, episode_idx, trajectory_manager)

        return CalibrationWorkerReturn(episode_idx=episode_idx, status="success")

    except Exception as e:
        print(f"  Episode {episode_idx}: Error - {e}")
        return CalibrationWorkerReturn(episode_idx=episode_idx, status="failed")


def model_training_worker(
    args: tuple[str, DataManager, int, float, int],
) -> ModelTrainingWorkerReturn:
    """
    Worker function for training the conformal prediction models in parallel

    Args:
        Tuple of (model_name, data_manager, n_lines, alpha, n_episodes)

    Returns:
        ModelTrainingWorkerReturn (model_name, trained_model, success_flag)
    """
    model_name, data_manager, n_lines, alpha, n_episodes = args

    try:
        from training import train_model_from_storage

        model = train_model_from_storage(
            model_name=model_name,
            data_manager=data_manager,
            n_lines=n_lines,
            alpha=alpha,
        )

        if model is not None:
            # we save the model after training
            print(f"  {model_name}: Training complete, saving to cache...")
            data_manager.save_model(model_name, n_episodes, model, alpha)
            return ModelTrainingWorkerReturn(
                model_name=model_name, trained_model=model, success=True
            )
        else:
            print(f"  {model_name}: failed to create")
            return ModelTrainingWorkerReturn(
                model_name=model_name, trained_model=None, success=False
            )

    except Exception as e:
        print(f"  {model_name}: error - {e}")
        import traceback

        traceback.print_exc()
        return ModelTrainingWorkerReturn(
            model_name=model_name, trained_model=None, success=False
        )


def test_episode_worker(
    args: tuple[
        int,
        dict[str, ModelWrapper],
        int,
        list[str],
        dict[str, STLWrapper],
    ],
) -> TestEpisodeWorkerReturn:
    """
    Worker function for parallel test episode execution

    Args:
        Tuple of (episode_num, predictors_data, chronic_id,
                  enabled_models, lines_attacked, stl_verifiers)

    Returns:
        TestEpisodeWorkerReturn (episode_num, trajectory_manager | None)
    """
    (
        episode_num,
        predictors,
        chronic_id,
        lines_attacked,
        stl_verifiers,
    ) = args

    try:
        from testing import run_test_episode
        from utils.environment import create_environment

        env, agent = create_environment(config.BASE_ENV_SEED, lines_attacked)

        trajectory_manager = run_test_episode(
            env,
            agent,
            episode_num,
            predictors,
            stl_verifiers,
            chronic_id,
        )

        trajectory_manager.filter_warmup()

        env.close()
        del env, agent, predictors, stl_verifiers
        gc.collect()

        print(f"  Test episode {episode_num}: completed")
        return TestEpisodeWorkerReturn(
            episode_num=episode_num, traj_man=trajectory_manager
        )

    except Exception as e:
        import traceback

        print(f"  Test episode {episode_num}: FAILED")
        print(f"     Error: {str(e)}")
        traceback.print_exc()

        return TestEpisodeWorkerReturn(episode_num=episode_num, traj_man=None)


def run_parallel_jobs(
    job_args_list: list[Any],
    worker_func: Callable[
        ...,
        CalibrationWorkerReturn | ModelTrainingWorkerReturn | TestEpisodeWorkerReturn,
    ],
    max_workers: int,
    desc: str,
) -> (
    list[CalibrationWorkerReturn]
    | list[ModelTrainingWorkerReturn]
    | list[TestEpisodeWorkerReturn]
):
    """
    Executes parallel jobs with progress bar: This is just
    a general function that can call any of the workers above

    Args:
        job_args_list: List of argument tuples to pass to each worker function
        worker_func: Callable function to execute for each job (one of the worker functions above)
        max_workers: Integer for maximum parallel workers
        desc: String description for the tqdm bar

    Returns:
        List of tuples depending on the worker function:
            -> For test_episode_worker, returns List of tuples of (episode_num, trajectory_manager, success)
            -> For model_training_worker, return List of tuples of (model_name, trained_model, success_flag)

    """
    if not job_args_list:
        return []

    print(f"Running {len(job_args_list)} {desc} jobs with {max_workers} workers")

    results = []
    successful = 0

    with mp.Pool(processes=max_workers, maxtasksperchild=1) as pool:
        with tqdm(total=len(job_args_list), desc=desc, unit="job") as pbar:
            for result in pool.imap_unordered(worker_func, job_args_list):
                try:
                    successful += 1
                    results.append(result)
                except Exception as e:
                    print(f"Job failed: {e}")

                pbar.update(1)

    gc.collect()
    print(f"Parallel jobs completed: {successful}/{len(job_args_list)} successful")
    return results


# I dont know if this is the best way to do this, took it from here
# https://stackoverflow.com/questions/1112343/how-do-i-capture-sigint-in-python
def signal_handler(sig, frame):
    """
    Catches the ctrl+c, so that we can terminate all processes that
    may still be active to avoid memory leaks in case of early
    termination
    """
    print("Terminating all processes...")

    for process in mp.active_children():
        process.terminate()

    sys.exit(1)
