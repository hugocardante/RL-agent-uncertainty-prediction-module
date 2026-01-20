import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from plotting.data import get_line_data, get_model_stats
from plotting.plotting_config import (
    FONTS,
    apply_axis_style,
    calculate_grid,
    get_color,
    get_display_name,
    save_fig,
)


def all_models_plots(
    timeseries_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_names: list[str],
    line_names: list[str],
    output_path: str,
):
    """
    Generates grid of all models for each power line

    Structure:
        -> one page per power line
        -> each page contains a grid of subplots (one subplot per model)

    Args:
        timeseries_df: dataframe with timeseries data
        comparison_df: dataframe with comparison metrics (coverage, width)
        model_names: list of model names to plot
        line_names: list of power line names
        output_path: path to save the output pdf
    """
    n_models = len(model_names)
    rows, cols = calculate_grid(n_models)

    with PdfPages(output_path) as pdf:
        for line_idx, line_name in enumerate(line_names):
            line_data = get_line_data(timeseries_df, line_idx)
            if line_data.empty:
                continue

            # setup the grid
            fig, axes = plt.subplots(
                nrows=rows, ncols=cols, figsize=(11 * cols, 7 * rows), squeeze=False
            )
            fig.suptitle(
                f"Line {line_idx}: {line_name}", fontsize=FONTS.page_title_fontsize
            )

            # flatten axes matrix to 1D so it is easier to itertate
            axes_flat = axes.flatten()

            # plot each model in its own subplot
            for ax, model in zip(axes_flat, model_names):
                stats = get_model_stats(comparison_df, model, line_idx)
                _draw_conformal_interval(ax, line_data, model, stats)

            # hide unused subplots (when n_models < rows*cols)
            for ax in axes_flat[n_models:]:
                ax.set_visible(False)

            # shared legend - we extract handles/labels from the first subplot
            # since all subplots share the same legend items
            handles, labels = axes_flat[0].get_legend_handles_labels()
            if handles:
                # just in case to remove duplicates (by converting to dict and back)
                by_label = dict(zip(labels, handles))
                axes_flat[0].legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="best",
                    fontsize=FONTS.legend_item_size,
                )

            save_fig(pdf, fig)


def individual_model_plots(
    timeseries_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_name: str,
    line_names: list[str],
    output_path: str,
):
    """
    Generates detailed plots for a single model

    Structure:
        -> one page per power line
        -> full-size plot for just the selected model

    Args:
        timeseries_df: dataframe with timeseries data
        comparison_df: dataframe with comparison metrics (coverage, width)
        model_name: name of the model to plot
        line_names: list of power line names
        output_path: path to save the output pdf
    """
    with PdfPages(output_path) as pdf:
        for line_idx, line_name in enumerate(line_names):
            line_data = get_line_data(timeseries_df, line_idx)
            if line_data.empty:
                continue

            # get stats to include in the title (e.g., "Coverage: 90.5%, Width: 0.123")
            stats = get_model_stats(comparison_df, model_name, line_idx)
            stats_text = ""
            if stats:
                stats_text = (
                    f"\nCoverage: {stats['coverage']:.1f}%, Width: {stats['width']:.3f}"
                )
                # only add action coverage if it exists and is not NaN
                if stats["action_inf_coverage"] is not None and not np.isnan(
                    stats["action_inf_coverage"]
                ):
                    stats_text += f", Action-influenced Coverage: {stats['action_inf_coverage']:.1f}%"

            fig, ax = plt.subplots(figsize=(16, 10))
            fig.suptitle(
                f"{get_display_name(model_name)} - Line {line_idx}: {line_name}{stats_text}",
                fontsize=FONTS.page_title_fontsize,
            )

            # disable internal title since we put all info in the page title
            _draw_conformal_interval(ax, line_data, model_name, stats, show_title=False)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize=FONTS.legend_item_size)

            save_fig(pdf, fig)


def _draw_conformal_interval(
    ax: Axes,
    line_data: pd.DataFrame | pd.Series,
    model_name: str,
    stats: dict[str, float | None] | None = None,
    show_title: bool = True,
):
    """
    Draw prediction interval with forecast, actual, and coverage markers.

    It has:
        - Blue line: Forecasted rho values
        - Black line: Actual rho values
        - Shaded area: Prediction interval (model color, see plotting_config.COLORS)
        - Orange shaded area: Prediction interval during action-influenced periods
        - Red 'x': Points where actual value fell outside the interval
        - Horizontal lines: Safety thresholds (0.90 and 0.95)

    Args:
        ax: matplotlib axis to draw on
        line_data: dataframe with timeseries data for a specific line
        model_name: name of the model
        stats: dict with coverage and width stats, or None
        show_title: if True, show the model name and stats as subplot title
    """
    color = get_color(model_name)
    upper_col = f"{model_name}_upper_bound"
    lower_col = f"{model_name}_lower_bound"
    covered_col = f"{model_name}_covered"

    # safety check for missing columns
    if upper_col not in line_data.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # subplot title with stats (for the all_models grid)
    title = None
    if show_title:
        if stats:
            # title = f"{get_display_name(model_name)}\nCoverage: {stats['coverage']:.1f}%\nWidth: {stats['width']:.3f}\nAction-influenced Coverage: {stats['action_inf_coverage']:.1f}%"
            title = f"{get_display_name(model_name)}\nCoverage: {stats['coverage']:.1f}%\nWidth: {stats['width']:.3f}"
        else:
            title = get_display_name(model_name)

    # forecast vs actual lines
    ax.plot(
        line_data["datetime"],
        line_data["forecast_rho"],
        "b-",
        marker="o",
        markersize=3,
        label="Forecast",
    )
    ax.plot(
        line_data["datetime"],
        line_data["true_rho"],
        "k-",
        marker="x",
        markersize=4,
        label="Actual",
    )

    # get action_influenced_flag column, default to False
    action_mask = line_data.get(
        "action_influenced_flag", pd.Series(False, index=line_data.index)
    )
    action_mask = action_mask.fillna(False).astype(bool).to_numpy()

    # draw prediction intervals
    if lower_col in line_data.columns:
        # standard interval (not action-influenced)
        ax.fill_between(
            line_data["datetime"],
            line_data[lower_col],
            line_data[upper_col],
            where=list(~action_mask),
            alpha=0.3,
            color=color,
            label="Prediction Interval",
        )
        # action-inluenced interval
        # if the agent intervened (action_influenced_flag is true), we color the interval yellow
        # originally it was yellow, but it is was too ugly
        ax.fill_between(
            line_data["datetime"],
            line_data[lower_col],
            line_data[upper_col],
            where=list(action_mask),
            alpha=0.4,
            color="orange",
            edgecolor="darkorange",
            label="Agent Action",
        )

    # mark uncovered points in red (actual value outside prediction interval)
    if covered_col in line_data.columns:
        covered = line_data[covered_col].fillna(True).astype(bool)
        uncovered = line_data[~covered]
        if not uncovered.empty:
            ax.plot(
                uncovered["datetime"],
                uncovered["true_rho"],
                "rx",
                markersize=5,
                ls="none",
                label="Outside Bounds",
            )

    # safety threshold lines at the top of the figures
    ax.axhline(0.9, color="orange", ls="--", label="High (rho = 0.90)")
    ax.axhline(0.95, color="red", ls="--", label="Limit (rho = 0.95)")

    apply_axis_style(ax, title=title, ylabel="Rho", time_axis=True)

    # how much we want to see vertically (we can change top)
    ax.set_ylim(bottom=0, top=1.1)
