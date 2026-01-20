import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from plotting.data import load_config, load_csv
from plotting.plotting_config import (
    FONTS,
    LAYOUT,
    MODELS_TO_PLOT,
    add_model_legend,
    apply_axis_style,
    get_color,
    get_display_name,
    save_fig,
)


def comparison_plots(
    comparison_df: pd.DataFrame,
    model_names: list[str],
    target_coverage: float,
    output_path: str,
):
    """
    Generates comparison bar charts pdf

    Pages: One per metric (coverage, width, action_inf_coverage)

    Args:
        comparison_df: dataframe with comparison metrics
        model_names: list of model names to plot
        target_coverage: target coverage percentage for the reference line
        output_path: path to save the output pdf
    """
    if comparison_df.empty:
        return

    # each tuple is (column_name, display_label)
    metrics = [
        ("coverage", "Coverage (%)"),
        ("width", "Average Width"),
        ("action_inf_coverage", "Action-influenced Coverage (%)"),
    ]

    with PdfPages(output_path) as pdf:
        for metric, ylabel in metrics:
            if metric not in comparison_df.columns:
                continue

            # action_inf_coverage may be all NaN if no action-influenced periods exist
            if metric == "action_inf_coverage" and comparison_df[metric].isna().all():
                continue

            fig, ax = plt.subplots(figsize=(15, 6))
            fig.suptitle(f"{ylabel} by Power Line", fontsize=FONTS.page_title_fontsize)

            _draw_grouped_bars(
                ax, comparison_df, metric, model_names, target_coverage, ylabel
            )

            add_model_legend(fig, model_names)
            save_fig(pdf, fig, rect=(0.0, LAYOUT.legend_space, 1.0, 1.0))


def horizon_plots(
    horizon_df: pd.DataFrame,
    model_names: list[str],
    target_coverage: float,
    output_path: str,
):
    """
    Generates horizon analysis pdf

    Pages:
        1. Coverage by forecast horizon
        2. Width by forecast horizon

    Args:
        horizon_df: dataframe with horizon metrics
        model_names: list of model names to plot
        target_coverage: target coverage percentage for the reference line
        output_path: path to save the output pdf
    """
    if horizon_df.empty:
        return

    with PdfPages(output_path) as pdf:
        # [page 1] coverage by forecast horizon
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.suptitle("Coverage by Forecast Horizon", fontsize=FONTS.page_title_fontsize)

        _draw_horizon_line(
            ax, horizon_df, "coverage", model_names, target_coverage, "Coverage (%)"
        )

        add_model_legend(fig, model_names)
        save_fig(pdf, fig, rect=(0, LAYOUT.legend_space, 1, 1))

        # [page 2] width by forecast horizon
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.suptitle(
            "Interval Width by Forecast Horizon", fontsize=FONTS.page_title_fontsize
        )

        _draw_horizon_line(ax, horizon_df, "width", model_names, None, "Average Width")

        add_model_legend(fig, model_names)
        save_fig(pdf, fig, rect=(0, LAYOUT.legend_space, 1, 1))


def aggregated_comparison_plots(
    comparison_csv: str,
    model_names: list[str],
    config_csv: str,
    output_path: str,
):
    """
    Generates aggregated comparison bar charts pdf

    Same as comparison_plots but with error bars from std columns.

    Args:
        comparison_csv: path to the aggregated comparison csv
        model_names: list of model names to plot
        config_csv: path to the config csv
        output_path: path to save the output pdf
    """

    df = load_csv(comparison_csv)
    if df is None:
        return

    # filter out models (see MODELS_TO_PLOT)
    if MODELS_TO_PLOT is not None:
        df = df[df["model_name"].isin(MODELS_TO_PLOT)]

        # if filter removed everything we dont plot anything and just return
        if df.empty:
            return

    config = load_config(config_csv)
    target_coverage = float(config.get("target_coverage", 90))

    # get n_episodes for the title, or "?" if not available
    n_episodes = df["n_episodes"].iloc[0] if "n_episodes" in df.columns else "?"

    # each tuple is (column_name, display_label)
    metrics = [
        ("coverage_mean", "Coverage (%)"),
        ("width_mean", "Average Width"),
        ("action_inf_coverage_mean", "Action-influenced Coverage (%)"),
    ]

    with PdfPages(output_path) as pdf:
        for metric, ylabel in metrics:
            # skip if metric column doesn't exist or is all NaN
            if metric not in df.columns or df[metric].isna().all():
                continue

            fig, ax = plt.subplots(figsize=(15, 6))
            fig.suptitle(
                f"Comparison: {ylabel} (averaged across {n_episodes} episodes)",
                fontsize=FONTS.page_title_fontsize,
            )

            _draw_grouped_bars(
                ax,
                df,
                metric,
                model_names,
                target_coverage,
                ylabel,
                show_error_bars=True,
            )

            add_model_legend(fig, model_names)
            save_fig(pdf, fig, rect=(0, LAYOUT.legend_space, 1, 1))


def _draw_grouped_bars(
    ax,
    df: pd.DataFrame,
    metric: str,
    models: list[str],
    target_line: float,
    ylabel: str,
    show_error_bars: bool = False,
):
    """
    Draws grouped bar chart comparing a metric across models and lines

    Args:
        ax: matplotlib axis to draw on
        df: dataframe with the metric data
        metric: column name of the metric to plot
        models: list of model names to include
        target_line: value for the horizontal target line (e.g., 90% coverage)
        ylabel: label for the y-axis
        show_error_bars: if True, show error bars from std columns
    """
    # pivot creates a matrix where rows=lines, cols=models
    # to make sure that every model aligns with the same line idx
    pivot = df.pivot(index="line_index", columns="model_name", values=metric)
    line_indices = pivot.index.tolist()

    # prepare the std data (for the aggregated case)
    # e.g., "coverage_mean" -> "coverage_std"
    std_col = metric.replace("_mean", "_std")
    has_std = show_error_bars and std_col in df.columns
    pivot_std = pd.DataFrame()
    if has_std:
        pivot_std = df.pivot(index="line_index", columns="model_name", values=std_col)

    # calculate bar positions
    n_models = len(models)
    bar_width = 0.8 / n_models
    # x positions are the center of each line group (0, 1, 2, ...)
    x = np.arange(len(line_indices))

    # bars for each model
    for i, model in enumerate(models):
        if model not in pivot.columns:
            continue

        # offset: spread bars left/right of center
        # (i - n_models / 2 + 0.5) shifts index 0 to far left, last index to far right
        pos = x + (i - n_models / 2 + 0.5) * bar_width

        # get std values if available
        yerr = None
        if has_std and model in pivot_std.columns:
            yerr = pivot_std[model]

        ax.bar(
            pos,
            pivot[model],
            bar_width,
            yerr=yerr,
            label=get_display_name(model),
            color=get_color(model),
            edgecolor="black",
            lw=0.5,
            capsize=3 if yerr is not None else 0,
            alpha=0.9,
        )

    # target line (e.g. 90% coverage)
    if target_line is not None:
        ax.axhline(
            target_line,
            color="r",
            ls="--",
            lw=2,
            zorder=10,
            label=f"Target ({target_line}%)",
        )

    apply_axis_style(
        ax, xlabel="Power Line", ylabel=ylabel, rotate_xticks=True, grid=True
    )

    # x-ticks to correspond to the line indices
    ax.set_xticks(x)
    ax.set_xticklabels([f"Line {i}" for i in line_indices])

    # y-axis limits based on metric type
    if "coverage" in metric:
        ax.set_ylim(65, 100)
    elif "width" in metric:
        ax.set_ylim(0, 0.4)


def _draw_horizon_line(
    ax,
    df: pd.DataFrame,
    metric: str,
    models: list[str],
    target_line: float | None,
    ylabel: str,
):
    """
    Draws line plot showing metric by forecast horizon

    Args:
        ax: matplotlib axis to draw on
        df: dataframe with horizon metrics
        metric: column name of the metric to plot
        models: list of model names to include
        target_line: value for the horizontal target line, or None
        ylabel: label for the y-axis
    """
    # pivot creates a matrix where rows=horizon (1, 2, ..., 12), cols=model_name
    pivot_mean = df.pivot(index="horizon", columns="model_name", values=metric)

    # plot for each model (one line)
    for model in models:
        # skip if this model isn't in the pivot table
        if model not in pivot_mean.columns:
            continue

        # values for this paritcular model
        y_values = pivot_mean[model]

        ax.plot(
            pivot_mean.index,
            y_values,
            marker="o",
            markersize=8,
            linewidth=2,
            color=get_color(model),
            label=get_display_name(model),
        )

    if target_line is not None:
        ax.axhline(
            target_line, color="r", ls="--", lw=2, label=f"Target ({target_line}%)"
        )

    apply_axis_style(
        ax, xlabel="Forecast Horizon (steps ahead)", ylabel=ylabel, grid=True
    )

    # y-axis limits for metric type
    if "coverage" in metric:
        ax.set_ylim(75, 100)
    elif "width" in metric:
        ax.set_ylim(0.07, 0.22)
