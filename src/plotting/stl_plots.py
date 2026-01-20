import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from plotting.data import get_line_data
from plotting.plotting_config import (
    FONTS,
    apply_axis_style,
    calculate_grid,
    get_display_name,
    save_fig,
)


def stl_plots(
    timeseries_df: pd.DataFrame,
    rules: list[str],
    line_names: list[str],
    threshold: float,
    output_path: str,
):
    """
    Generates STL verification plots for each line across all rules

    Structure:
        -> one page per power line
        -> each page contains a grid of subplots (one subplot per rule)

    Args:
        timeseries_df: dataframe with timeseries data
        rules: list of STL rule names to plot
        line_names: list of power line names
        threshold: safety threshold value (e.g., 0.95)
        output_path: path to save the output PDF
    """
    if timeseries_df is None or timeseries_df.empty:
        return

    if not rules:
        print("No STL rules found in timeseries data")
        return

    n_rules = len(rules)
    rows, cols = calculate_grid(n_rules)

    with PdfPages(output_path) as pdf:
        for line_idx, line_name in enumerate(line_names):
            line_data = get_line_data(timeseries_df, line_idx)
            if line_data.empty:
                continue

            fig, axes = plt.subplots(
                rows, cols, figsize=(11 * cols, 7 * rows), squeeze=False
            )
            fig.suptitle(
                f"STL Verification - Line {line_idx}: {line_name}",
                fontsize=FONTS.page_title_fontsize,
            )

            # flatten the axes array for easier iteration
            axes_flat = axes.flatten()

            # each rule is one subplot
            for ax, rule in zip(axes_flat, rules):
                _draw_stl_safety(ax, line_data, rule, threshold)

            # hide unused subplots (when n_rules < rows*cols)
            for ax in axes_flat[n_rules:]:
                ax.set_visible(False)

            save_fig(pdf, fig)


def individual_stl_plots(
    timeseries_df: pd.DataFrame,
    rule_name: str,
    line_names: list[str],
    threshold: float,
    output_path: str,
):
    """
    Generates detailed plots for a single STL rule

    Structure:
        -> one page per power line
        -> full-size plot for just the selected rule

    Args:
        timeseries_df: dataframe with timeseries data
        rule_name: name of the STL rule to plot
        line_names: list of power line names
        threshold: safety threshold value (e.g., 0.95)
        output_path: path to save the output PDF
    """
    predicted_safe_col = f"{rule_name}_predicted_safe"

    # check if this column exists
    if predicted_safe_col not in timeseries_df.columns:
        print(f"No safety column found for rule: {rule_name}")
        return

    with PdfPages(output_path) as pdf:
        for line_idx, line_name in enumerate(line_names):
            line_data = get_line_data(timeseries_df, line_idx)
            if line_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(16, 10))
            fig.suptitle(
                f"{get_display_name(rule_name)} - Line {line_idx}: {line_name}",
                fontsize=FONTS.page_title_fontsize,
            )

            # disable internal title since we use the page title
            _draw_stl_safety(ax, line_data, rule_name, threshold, show_title=False)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="best", fontsize=FONTS.legend_item_size)

            save_fig(pdf, fig)


def _draw_stl_safety(
    ax: Axes,
    line_data: pd.DataFrame | pd.Series,
    rule_name: str,
    threshold: float,
    show_title: bool = True,
) -> None:
    """
    Draws STL verification plot with safety colors

    Points are colored green if the rule predicted safe, red if predicted unsafe.
    Also shows the forecast line, actual line, and safety threshold.

    Args:
        ax: matplotlib axis to draw on
        line_data: dataframe with timeseries data for a specific line
        rule_name: name of the STL rule
        threshold: safety threshold value (e.g., 0.95)
        show_title: if True, show the rule name as subplot title
    """
    predicted_safe_col = f"{rule_name}_predicted_safe"

    # check if required column exists
    if predicted_safe_col not in line_data.columns:
        ax.text(
            0.5, 0.5, "No STL data", ha="center", va="center", transform=ax.transAxes
        )
        return

    title = get_display_name(rule_name) if show_title else None

    # split data by predicted safety status (directly from dataframe)
    safe_data = line_data[line_data[predicted_safe_col] == True]
    unsafe_data = line_data[line_data[predicted_safe_col] == False]

    # scatter the actual rho points, colored by predicted safety
    # green = rule predicted this trajectory would be safe
    # red = rule predicted this trajectory would be unsafe
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
        color="blue",
        linestyle="-",
        alpha=0.5,
        label="Forecast",
    )
    ax.plot(
        line_data["datetime"],
        line_data["true_rho"],
        color="black",
        linestyle="-",
        alpha=0.3,
        label="Actual",
    )

    # safety threshold line
    ax.axhline(threshold, color="red", ls="--", lw=2, label="Safety Threshold")

    apply_axis_style(ax, title=title, ylabel="Rho", time_axis=True)
