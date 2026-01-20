import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from plotting.data import get_line_data
from plotting.plotting_config import (
    FONTS,
    apply_axis_style,
    calculate_grid,
    get_color,
    get_display_name,
    save_fig,
)


def conf_plots_like_stl(
    timeseries_df: pd.DataFrame,
    model_names: list[str],
    line_names: list[str],
    threshold: float,
    output_path: str,
):
    """
    Generates safety scatter plots for each line across all models

    Structure:
        -> one page per power line
        -> each page contains a grid of subplots (one subplot per model)

    Args:
        timeseries_df: dataframe with timeseries data
        model_names: list of model names to plot
        line_names: list of power line names
        threshold: safety threshold value (e.g., 0.95)
        output_path: path to save the output pdf
    """
    n_models = len(model_names)
    rows, cols = calculate_grid(n_models)

    with PdfPages(output_path) as pdf:
        for line_idx, line_name in enumerate(line_names):
            # data for this line
            line_data = get_line_data(timeseries_df, line_idx)
            if line_data.empty:
                continue

            # grid setup
            fig, axes = plt.subplots(
                nrows=rows, ncols=cols, figsize=(11 * cols, 7 * rows), squeeze=False
            )
            fig.suptitle(
                f"Safety Analysis - Line {line_idx}: {line_name}",
                fontsize=FONTS.page_title_fontsize,
            )

            # we flatten the 2D matrix of axes into a 1D list to iterate more easily
            axes_flat = axes.flatten()

            # each model gets its own subplot
            for ax, model in zip(axes_flat, model_names):
                _draw_conf_safety_scatter(ax, line_data, model, threshold)

            # hide any unused subplots (when n_models < rows * cols)
            for ax in axes_flat[n_models:]:
                ax.set_visible(False)

            save_fig(pdf, fig)


def individual_conf_plots_like_stl(
    timeseries_df: pd.DataFrame,
    model_name: str,
    line_names: list[str],
    threshold: float,
    output_path: str,
):
    """
    Generates safety plots for a single model

    Structure:
        - one page per power line
        - just one plot (full size) for the selected model

    Args:
        timeseries_df: dataframe with timeseries data
        model_name: name of the model to plot
        line_names: list of power line names
        threshold: safety threshold value
        output_path: path to save the output pdf
    """
    with PdfPages(output_path) as pdf:
        for line_idx, line_name in enumerate(line_names):
            line_data = get_line_data(timeseries_df, line_idx)
            if line_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(16, 10))
            fig.suptitle(
                f"Safety: {get_display_name(model_name)} - Line {line_idx}: {line_name}",
                fontsize=FONTS.page_title_fontsize,
            )

            # disable the internal title because for individual plots we use the page title
            _draw_conf_safety_scatter(
                ax, line_data, model_name, threshold, show_title=False
            )

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize=FONTS.legend_item_size)

            save_fig(pdf, fig)


def _draw_conf_safety_scatter(
    ax: Axes,
    line_data: pd.DataFrame | pd.Series,
    model_name: str,
    threshold: float,
    show_title: bool = True,
):
    """
    Draws scatter plot colored by predicted safety status

    Points are colored green if the model predicted safe, red if predicted unsafe
    Also shows the forecast line, actual line, prediction interval, and safety threshold

    Args:
        ax: matplotlib axis to draw on
        line_data: dataframe with timeseries data for a specific line
        model_name: name of the model
        threshold: safety threshold value (e.g., 0.95)
        show_title: if True, show the model name as subplot title
    """
    color = get_color(model_name)
    upper_col = f"{model_name}_upper_bound"
    lower_col = f"{model_name}_lower_bound"
    predicted_safe_col = f"{model_name}_predicted_safe"

    # check if column exists before plotting
    if predicted_safe_col not in line_data.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    title = get_display_name(model_name) if show_title else None

    # data by predicted safety
    safe_data = line_data[line_data[predicted_safe_col] == True]
    unsafe_data = line_data[line_data[predicted_safe_col] == False]

    # we plot the actual points, but we paint them
    # in green to show that they were predicted safe
    # or in red to show that they were predicted unsafe
    if not safe_data.empty:
        ax.scatter(
            safe_data["datetime"],
            safe_data["true_rho"],
            c="green",
            alpha=0.3,
            s=50,
            label="Safe",
        )
    if not unsafe_data.empty:
        ax.scatter(
            unsafe_data["datetime"],
            unsafe_data["true_rho"],
            c="red",
            alpha=0.5,
            s=50,
            label="Unsafe",
        )

    # forecast and actual lines
    ax.plot(
        line_data["datetime"],
        line_data["forecast_rho"],
        "b-",
        alpha=0.5,
        label="Forecast",
    )
    ax.plot(
        line_data["datetime"], line_data["true_rho"], "k-", alpha=0.3, label="Actual"
    )

    # conformal interval, this is not necessary
    # but helps to see visually that these are the conformal
    # plots, with the purpose of comparing them to the STL plots
    if lower_col in line_data.columns and upper_col in line_data.columns:
        ax.fill_between(
            line_data["datetime"],
            line_data[lower_col],
            line_data[upper_col],
            alpha=0.2,
            color=color,
            label="Prediction Interval",
        )

    # rho safety threshold line
    ax.axhline(threshold, color="red", ls="--", lw=2, label="Safety Threshold")

    apply_axis_style(ax, title=title, ylabel="Rho", time_axis=True)
