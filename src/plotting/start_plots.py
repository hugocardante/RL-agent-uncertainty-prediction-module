import gc
import glob
import multiprocessing as mp
import os

import pandas as pd

from plotting.classification_plots import classification_plots
from plotting.comparison_plots import (
    aggregated_comparison_plots,
    comparison_plots,
    horizon_plots,
)
from plotting.conf_plots_like_stl import (
    conf_plots_like_stl,
    individual_conf_plots_like_stl,
)
from plotting.conformal_plots import all_models_plots, individual_model_plots
from plotting.data import (
    get_line_names,
    get_model_names,
    get_stl_rule_names,
    load_comparison_data,
    load_config,
    load_csv,
    load_timeseries_data,
)
from plotting.multi_alpha_plots import multi_alpha_plots
from plotting.plotting_config import (
    MAX_WORKERS,
    MODELS_TO_PLOT,
    PLOT_ONLY_MULTI_ALPHA,
    RULES_TO_PLOT,
    PlottingTask,
)
from plotting.stl_plots import individual_stl_plots, stl_plots
from utils.global_utils import ensure_dir


def _run_plotting_task(task: PlottingTask) -> tuple[str, str | None]:
    """
    Worker function that executes a single plotting task

    Args:
        task: Tuple of (name, func, args, kwargs)

    Returns:
        Tuple of (name, error_message or None)
    """
    try:
        task.func(*task.args, **task.kwargs)
        return task.name, None
    except Exception as e:
        return task.name, str(e)


def run_all_plotting_tasks(tasks: list[PlottingTask]):
    """
    Runs all plotting tasks in parallel (using multiprocessing.Pool)

    This is the main entry point for parallel plotting

    Args:
        tasks: list of tuples (name, func, args, kwargs)
               - name: task name for logging
               - func: function to call
               - args: positional arguments for func
               - kwargs: keyword arguments for func
    """
    if not tasks:
        return

    print(f"\nPlotting {len(tasks)} plots...")

    completed = 0

    with mp.Pool(processes=MAX_WORKERS, maxtasksperchild=10) as pool:
        results = pool.imap_unordered(_run_plotting_task, tasks)

        for name, error in results:
            completed += 1
            if error:
                print(f"  [{completed}/{len(tasks)}] ERROR {name}: {error}")
            else:
                print(f"  [{completed}/{len(tasks)}] {name}")

    gc.collect()


def _ensure_dir_with_ext(path: str):
    """
    Creates directory for a path if it doesn't exist

    Handles both directory paths and file paths (extracts parent dir from files)

    Args:
        path: directory path or file path
    """
    # if path has an extension (like .pdf), get the parent dir
    # otherwise treat the whole path as a directory
    if "." in os.path.basename(path):
        dir_path = os.path.dirname(path)
    else:
        dir_path = path

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def _get_path(folder: str, filename: str, check_exists: bool = False) -> str | None:
    """
    Joins folder and filename, optionally checking if file exists

    Args:
        folder: base folder path
        filename: filename to join
        check_exists: if True, return None if file doesn't exist

    Returns:
        Full path string, or None if check_exists=True and file doesn't exist
    """

    path = os.path.join(folder, filename)
    if check_exists and not os.path.exists(path):
        return None
    return path


def collect_tasks_for_folder(
    csv_folder: str, plots_folder: str, outputs: dict[str, str]
) -> list[PlottingTask]:
    """
    Collects all plotting tasks for a single episode folder

    Does not execute tasks, just returns them for later parallel execution.

    Args:
        csv_folder: path to folder containing csv files
        plots_folder: path to folder where plots will be saved
        outputs: dict mapping output type to filename (e.g., {"comparison": "comparison.pdf"})

    Returns:
        List of task tuples (name, func, args, kwargs)
    """
    tasks = []

    # skip if we only want multi-alpha plots (see PLOT_ONLY_MULTI_ALPHA)
    if PLOT_ONLY_MULTI_ALPHA:
        return tasks

    if not os.path.exists(csv_folder):
        print(f"csv folder not found: {csv_folder}")
        return tasks

    ensure_dir(plots_folder)

    # load config to get coverage targets, line names, model names, etc.
    config_path = _get_path(csv_folder, "config.csv", check_exists=False)
    config = load_config(config_path)
    if config is None:
        return tasks

    target_coverage = float(config["target_coverage"])
    threshold = float(config["rho_safety_threshold"])
    line_names = get_line_names(config)
    model_names = get_model_names(config)
    stl_rule_names = get_stl_rule_names(config)

    # load data files
    comp_csv = _get_path(csv_folder, "conformal_data.csv", check_exists=True)
    comp_df = load_comparison_data(comp_csv)

    ts_csv = _get_path(csv_folder, "timeseries.csv", check_exists=True)
    ts_df = load_timeseries_data(ts_csv) if ts_csv else None

    horizon_csv = _get_path(csv_folder, "conformal_data_horizon.csv", check_exists=True)
    horizon_df = load_csv(horizon_csv)

    # helper to add a task
    def add_task(name, func, *args, **kwargs):
        tasks.append(PlottingTask(name, func, args, kwargs))

    # comparison plots (coverage, width, action_inf_coverage by power line)
    if not comp_df.empty:
        output_path = os.path.join(plots_folder, outputs["comparison"])
        add_task(
            "Comparison Plots",
            comparison_plots,
            comp_df,
            model_names,
            target_coverage,
            output_path,
        )

    # timeseries plots (all models, individual) and conformal plots like stl safety
    if ts_df is not None:
        # all models grid
        output_path = os.path.join(plots_folder, outputs["timeseries"])
        add_task(
            "All models",
            all_models_plots,
            ts_df,
            comp_df,
            model_names,
            line_names,
            output_path,
        )

        # conformal safety plots (like STL plots for comparison)
        output_path = os.path.join(plots_folder, outputs["safety"])
        add_task(
            "Conformal plots like STL",
            conf_plots_like_stl,
            ts_df,
            model_names,
            line_names,
            threshold,
            output_path,
        )

        # individual model plots (one pdf per model)
        model_dir = os.path.join(plots_folder, "individual_models")
        _ensure_dir_with_ext(model_dir)
        safety_dir = os.path.join(plots_folder, "individual_cp_plots_like_stl")
        _ensure_dir_with_ext(safety_dir)

        for model in model_names:
            # replace spaces with underscores because of filenames (when necessary)
            safe_name = model.replace(" ", "_")
            add_task(
                f"Individual: {model}",
                individual_model_plots,
                ts_df,
                comp_df,
                model,
                line_names,
                os.path.join(model_dir, f"{safe_name}_cp.pdf"),
            )
            add_task(
                f"Individual conf like STL: {model}",
                individual_conf_plots_like_stl,
                ts_df,
                model,
                line_names,
                threshold,
                os.path.join(safety_dir, f"{safe_name}_experiment.pdf"),
            )

    # horizon analysis (coverage and width by forecast horizon)
    if horizon_df is not None:
        output_path = os.path.join(plots_folder, outputs["horizon"])
        add_task(
            "Horizon",
            horizon_plots,
            horizon_df,
            model_names,
            target_coverage,
            output_path,
        )

    # STL (Signal Temporal Logic) plots
    if ts_df is not None and stl_rule_names:
        # all rules grid
        output_path = os.path.join(plots_folder, outputs["stl"])
        add_task(
            "STL",
            stl_plots,
            ts_df,
            stl_rule_names,
            line_names,
            threshold,
            output_path,
        )

        # individual STL rule plots (one pdf per rule)
        stl_dir = os.path.join(plots_folder, "individual_stl")
        _ensure_dir_with_ext(stl_dir)
        for rule in stl_rule_names:
            add_task(
                f"STL: {rule}",
                individual_stl_plots,
                ts_df,
                rule,
                line_names,
                threshold,
                os.path.join(stl_dir, f"{rule}.pdf"),
            )

    # classification plots (F1, false alarms, overlooked violations) for both STL and Conf
    stl_class_csv = _get_path(csv_folder, "stl_classification.csv", check_exists=True)
    if stl_class_csv:
        output_path = os.path.join(plots_folder, outputs["stl_classification"])
        _ensure_dir_with_ext(output_path)
        add_task(
            "STL classification plots",
            classification_plots,
            stl_class_csv,
            output_path,
            name_column="rule_name",
        )

    conf_class_csv = _get_path(
        csv_folder, "conformal_classification.csv", check_exists=True
    )
    if conf_class_csv:
        output_path = os.path.join(plots_folder, outputs["conformal_classification"])
        _ensure_dir_with_ext(output_path)
        add_task(
            "Conformal classification plots",
            classification_plots,
            conf_class_csv,
            output_path,
            name_column="model_name",
        )

    return tasks


def collect_aggregated_tasks(
    csv_folder: str,
    plots_folder: str,
    outputs: dict[str, str],
    config_csv_path: str | None,
) -> list[PlottingTask]:
    """
    Collects plotting tasks for aggregated csv folder

    Args:
        csv_folder: path to folder containing aggregated csv files
        plots_folder: path to folder where plots will be saved
        outputs: dict mapping output type to filename
        config_csv_path: path to the config csv file

    Returns:
        List of task tuples (name, func, args, kwargs)
    """
    tasks = []

    # skip if we only want multi-alpha plots (see PLOT_ONLY_MULTI_ALPHA)
    if PLOT_ONLY_MULTI_ALPHA:
        return tasks

    if not os.path.exists(csv_folder):
        print(f"Aggregated csv folder not found: {csv_folder}")
        return tasks

    ensure_dir(plots_folder)

    def add_task(name, func, *args, **kwargs):
        tasks.append(PlottingTask(name, func, args, kwargs))

    # aggregated comparison (coverage, width with error bars)
    comp_csv = _get_path(csv_folder, "aggregated_comparison.csv", check_exists=True)

    # we load the config first to get coverage targets and line names
    config_path = _get_path(csv_folder, "config.csv", check_exists=False)
    config = load_config(config_path)
    model_names = get_model_names(config)

    if comp_csv:
        output_path = os.path.join(plots_folder, outputs["aggregated_comparison"])
        add_task(
            "Aggregated Comparison",
            aggregated_comparison_plots,
            comp_csv,
            model_names,
            config_csv_path,
            output_path,
        )

    # aggregated classification (STL)
    stl_csv = _get_path(
        csv_folder, "stl_classification_aggregated.csv", check_exists=True
    )
    if stl_csv:
        output_path = os.path.join(plots_folder, outputs["aggregated_stl"])
        _ensure_dir_with_ext(output_path)
        add_task(
            "Agg STL Class",
            classification_plots,
            stl_csv,
            output_path,
            name_column="rule_name",
        )

    # aggregated classification (Conformal)
    conf_csv = _get_path(
        csv_folder, "conformal_classification_aggregated.csv", check_exists=True
    )
    if conf_csv:
        output_path = os.path.join(plots_folder, outputs["aggregated_conformal"])
        _ensure_dir_with_ext(output_path)
        add_task(
            "Agg Conf Class",
            classification_plots,
            conf_csv,
            output_path,
            name_column="model_name",
        )

    return tasks


def _collect_multi_alpha_data(base_dir: str):
    """
    Collects STL and conformal results from multiple alpha_* directories

    Scans for folders named alpha_0.05, alpha_0.10, etc. and combines
    their classification results into single dataframes for multi-alpha plotting

    Args:
        base_dir: base directory containing alpha_* subdirectories

    Returns:
        Tuple of (stl_df, conformal_df), either can be None if no data found
    """
    # find all folders with the pattern "alpha_*" as name
    alpha_pattern = os.path.join(base_dir, "alpha_*")
    alpha_dirs = sorted(glob.glob(alpha_pattern))

    stl_dfs, conformal_dfs = [], []

    for alpha_dir in alpha_dirs:
        # extract alpha value from the folder name (e.g., "alpha_0.05" -> 0.05)
        alpha_val = float(os.path.basename(alpha_dir).split("_")[-1])

        agg_dir = os.path.join(alpha_dir, "aggregated_csvs")

        # load STL classification data
        stl_path = _get_path(
            agg_dir, "stl_classification_aggregated.csv", check_exists=True
        )
        if stl_path:
            df = load_csv(stl_path)
            if df is not None:
                # filter rules if RULES_TO_PLOT is set
                if RULES_TO_PLOT is not None:
                    df = df[df["rule_name"].isin(RULES_TO_PLOT)]

                # add alpha column so we can distinguish between different runs
                df["alpha"] = alpha_val
                stl_dfs.append(df)

        # load conformal classification data
        conf_path = _get_path(
            agg_dir, "conformal_classification_aggregated.csv", check_exists=True
        )
        if conf_path:
            df = load_csv(conf_path)
            if df is not None:
                # filter models if MODELS_TO_PLOT is set
                if MODELS_TO_PLOT is not None:
                    df = df[df["model_name"].isin(MODELS_TO_PLOT)]

                df["alpha"] = alpha_val
                conformal_dfs.append(df)

    # concatenate all dataframes, or return None if empty
    stl_result = pd.concat(stl_dfs, ignore_index=True) if stl_dfs else None
    conf_result = pd.concat(conformal_dfs, ignore_index=True) if conformal_dfs else None

    return stl_result, conf_result


def start_multi_alpha_plots(base_dir: str, output_path: str):
    """
    Generates comparison plots across multiple alpha values

    Creates ROC-like plots showing the trade-off between false alarms
    and missed violations at different alpha (significance) levels.

    Args:
        base_dir: base directory containing alpha_* subdirectories
        output_path: path to save the output PDF
    """
    print("\nGenerating Multi-Alpha Comparison Plots...")

    stl_df, conformal_df = _collect_multi_alpha_data(base_dir)

    # convert empty dataframes to None
    if stl_df is not None and stl_df.empty:
        stl_df = None

    if conformal_df is not None and conformal_df.empty:
        conformal_df = None

    if stl_df is None and conformal_df is None:
        print("No valid data found for multi-alpha plots")
        return

    _ensure_dir_with_ext(output_path)
    multi_alpha_plots(stl_df, conformal_df, output_path)
