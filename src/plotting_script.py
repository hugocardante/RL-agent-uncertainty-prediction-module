"""
Main entry point for generating plot visualizations.
Can be used by other parts of the framework or as a standalone script.

Usage:
    python plotting_script.py <RESULTS_DIR> <OUTPUT_DIR>
    python plotting_script.py <RESULTS_DIR>

<RESULTS_DIR> can be the root folder containing all csvs,
or isolated folders like "aggregated_csvs" or "episode_1"
"""

import argparse
import glob
import os

from plotting.plotting_config import PlottingTask
from plotting.start_plots import (
    collect_aggregated_tasks,
    collect_tasks_for_folder,
    run_all_plotting_tasks,
    start_multi_alpha_plots,
)
from utils.parallel import setup_multiprocessing

# Directory and file naming constants that were set
# during the analyze phase (see analyze.py)
# DO NOT CHANGE these constants without changing
# in the appropriate analyze.py
CSVS_DIR = "csvs"
PLOTS_DIR = "plots"
EPISODE_PREFIX = "episode_"
ALPHA_PREFIX = "alpha_"
AGGREGATED_CSVS_DIR = "aggregated_csvs"
AGGREGATED_PLOTS_DIR = "aggregated_plots"
CONFIG_CSV = "config.csv"

# Output filenames (we can change the filenames
# by changing the value in this (key, val) dict)
OUTPUTS = {
    "comparison": "comparison_plots.pdf",
    "timeseries": "all_models.pdf",
    "safety": "cp_plot_like_stl.pdf",
    "horizon": "horizon_plots.pdf",
    "stl": "stl.pdf",
    "stl_classification": "classification/stl_classification.pdf",
    "conformal_classification": "classification/conformal_classification.pdf",
    "aggregated_comparison": "aggregated_comparison.pdf",
    "aggregated_stl": "stl_classification_aggregated.pdf",
    "aggregated_conformal": "conformal_classification_aggregated.pdf",
    "multi_alpha_plots": "multi_alpha_plots.pdf",
}


def _get_path(folder: str, filename: str, must_exist: bool = False) -> str | None:
    """
    Gets the full path for a file, optionally checking if it exists

    Args:
        folder: base directory
        filename: name of the file
        must_exist: if True, returns None when file doesn't exist

    Returns:
        Full path string, or None if must_exist=True and file not found
    """
    path = os.path.join(folder, filename)
    if must_exist and not os.path.exists(path):
        return None
    return path


def _get_alpha_dirs(base_dir: str) -> list[str]:
    """
    Gets the sorted list of alpha_* directories

    Args:
        base_dir: base directory

    Returns:
        A list with the name of the alpha folders ([alpha_0.1, ...])
    """
    pattern = os.path.join(base_dir, f"{ALPHA_PREFIX}*")
    return sorted(glob.glob(pattern))


def _collect_tasks_for_alpha(alpha_dir: str, output_base: str) -> list[PlottingTask]:
    """
    Collects all plotting tasks for a single alpha folder

    Collects tasks for every episode inside and aggregated plots
    from the aggregated_csvs folder

    Args:
        alpha_dir: path to alpha_* directory
        output_base: base output directory for plots

    Returns:
        List of task tuples (name, func, args, kwargs)
    """
    tasks = []

    episode_pattern = os.path.join(alpha_dir, f"{EPISODE_PREFIX}*")
    episode_dirs = sorted(glob.glob(episode_pattern))

    # collect tasks for individual episodes
    for episode_dir in episode_dirs:
        episode_name = os.path.basename(episode_dir)
        csv_dir = os.path.join(episode_dir, CSVS_DIR)
        plots_dir = os.path.join(output_base, episode_name, PLOTS_DIR)

        if os.path.exists(csv_dir):
            tasks.extend(collect_tasks_for_folder(csv_dir, plots_dir, OUTPUTS))

    # collect tasks for aggregated results
    agg_csv_dir = os.path.join(alpha_dir, AGGREGATED_CSVS_DIR)
    agg_plots_dir = os.path.join(output_base, AGGREGATED_PLOTS_DIR)

    if os.path.exists(agg_csv_dir):
        config_csv_path = _get_path(agg_csv_dir, CONFIG_CSV, must_exist=True)
        tasks.extend(
            collect_aggregated_tasks(
                agg_csv_dir, agg_plots_dir, OUTPUTS, config_csv_path
            )
        )

    return tasks


def start_all_plots(base_dir: str, output_base: str | None = None) -> None:
    """
    Generates all the plots for the given directory

    Detects the input structure and processes accordingly:
        - root containing alpha_* folders: process all + multi-alpha comparison
        - single alpha_* folder: process that alpha only
        - single episode folder: process single episode
        - aggregated csvs folder: process only aggregated plots

    Args:
        base_dir: input directory containing csv data
        output_base: output directory for plots (defaults to base_dir)
    """
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # remove trailing slash, e.g., "a/b/" -> "a/b"
    base_dir = base_dir.rstrip(os.sep)
    if output_base is None:
        output_base = base_dir

    base_name = os.path.basename(base_dir)
    tasks = []

    # case 1: csvs folder
    if base_name == CSVS_DIR:
        if output_base == base_dir:
            plots_out = os.path.join(os.path.dirname(base_dir), PLOTS_DIR)
        else:
            plots_out = output_base
        tasks.extend(collect_tasks_for_folder(base_dir, plots_out, OUTPUTS))
        run_all_plotting_tasks(tasks)
        return

    # case 2: episode_* folder
    if base_name.startswith(EPISODE_PREFIX):
        csv_dir = os.path.join(base_dir, CSVS_DIR)
        if output_base == base_dir:
            plots_out = os.path.join(base_dir, PLOTS_DIR)
        else:
            plots_out = output_base

        if os.path.exists(csv_dir):
            tasks.extend(collect_tasks_for_folder(csv_dir, plots_out, OUTPUTS))
        run_all_plotting_tasks(tasks)
        return

    # case 3: aggregated_csvs folder
    if base_name == AGGREGATED_CSVS_DIR:
        config_csv_path = _get_path(base_dir, CONFIG_CSV, must_exist=True)
        tasks.extend(
            collect_aggregated_tasks(base_dir, output_base, OUTPUTS, config_csv_path)
        )
        run_all_plotting_tasks(tasks)
        return

    # case 4: alpha_* folder
    if base_name.startswith(ALPHA_PREFIX):
        tasks.extend(_collect_tasks_for_alpha(base_dir, output_base))
        run_all_plotting_tasks(tasks)
        return

    # case 5: root folder containing the alpha_* directories
    alpha_dirs = _get_alpha_dirs(base_dir)

    if alpha_dirs:
        # collect tasks from all alpha folders
        for alpha_dir in alpha_dirs:
            alpha_name = os.path.basename(alpha_dir)
            alpha_output = os.path.join(output_base, alpha_name)
            tasks.extend(_collect_tasks_for_alpha(alpha_dir, alpha_output))

        # run all tasks in parallel
        run_all_plotting_tasks(tasks)

        # multi-alpha plots run after all other plots are done
        # because they need the aggregated data from all alphas
        # at the moment only plotting the multi alpha classficiation plots
        # if we have more than one alpha, but it could be plotted with just one
        if len(alpha_dirs) > 1:
            multi_alpha_output = os.path.join(output_base, OUTPUTS["multi_alpha_plots"])
            start_multi_alpha_plots(base_dir, multi_alpha_output)
        return

    # just in case it is invalid
    print(f"wrong path is being passed, or invalid folder: {base_dir}")


def main():
    setup_multiprocessing()

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
    )
    args = parser.parse_args()

    start_all_plots(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
