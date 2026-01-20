from utils.global_utils import ensure_dir, get_env_details, ignore_warnings

ignore_warnings()

import os

# WARNING: This environment variable is important, do not remove!
# this prevents OpenMP from using multiple threads per worker
# which can cause issues with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"

import time

import config
from analysis import generate_csvs
from calibration import run_ensemble_calibration, run_single_calibration
from plotting_script import start_all_plots
from testing import run_test_episodes
from training import run_model_training, run_stl_training
from utils.model_wrapper import get_enabled_models
from utils.parallel import setup_multiprocessing
from utils.stl_wrapper import get_enabled_stl_rules


def run_conformal_simulation(
    alpha: float,
    output_dir: str,
) -> None:
    """
    executes the pipeline for the conformal models / stl

    Args:
        alpha: Float significance level for conformal prediction
        output_dir: Directory to save results
    """

    # Setup output directory
    ensure_dir(output_dir)
    ensure_dir(config.CALIBRATION_CACHE_DIR)

    # Get environment info and validate configuration
    n_lines, line_names = get_env_details()
    enabled_models = get_enabled_models()
    enabled_stl_rules = get_enabled_stl_rules()
    if not enabled_models and not enabled_stl_rules:
        print("DEBUG: No conformal models or stl rules enabled..")
        return

    # Phase 1: Calibration (starts at BASE_CHRONIC + 1)
    chronic = config.BASE_CHRONIC + 1
    if config.ENSEMBLE_MODE:
        calib_data = run_ensemble_calibration(chronic)
    else:
        calib_data = run_single_calibration(chronic)
    chronic += config.CALIB_EPISODES

    # Phase 2: Train conformal models
    predictors = {}
    if enabled_models:
        predictors = run_model_training(calib_data, n_lines, alpha)

    # Phase 2.5: Train STL rules
    stl_verifiers = {}
    if enabled_stl_rules:
        stl_verifiers = run_stl_training(calib_data, n_lines, alpha)

    # Phase 3: Testing
    results = run_test_episodes(
        config.TEST_EPISODES,
        predictors,
        stl_verifiers,
        chronic,
    )

    # Phase 4: Analysis (saves csvs)
    generate_csvs(
        alpha=alpha,
        results=results,
        line_names=line_names,
        output_dir=output_dir,
        stl_verifiers=stl_verifiers,
        conf_predictors=predictors,
    )


def run_all_alphas(all_alphas: list[float]) -> None:
    """
    At the moment, we accept to run the simulation with
    multiple alpha values. It will run multiple simulations,
    and aggregate all the results.


    Args:
        all_alphas: List of floats each representing one alpha for which we want to run the simulation
    """
    start_time = time.time()

    for alpha in all_alphas:
        print(f"now running alpha={alpha}")

        # we create alpha_* folder (it's important not to change this name because
        # it is used for aggregation)
        output_dir = os.path.join(config.OUTPUT_DIR, f"alpha_{alpha}")

        run_conformal_simulation(
            alpha=alpha,
            output_dir=output_dir,
        )

    # starts all plots if they are to be generated during simulation
    if config.AUTO_GEN_PLOTS:
        start_all_plots(config.OUTPUT_DIR)

    total_time = (time.time() - start_time) / 60
    print(f"Simulation is finished. Took: {total_time:.2f} minutes")


if __name__ == "__main__":
    setup_multiprocessing()

    alphas: list[float] = (
        config.ALPHA if isinstance(config.ALPHA, list) else [config.ALPHA]
    )
    run_all_alphas(alphas)
