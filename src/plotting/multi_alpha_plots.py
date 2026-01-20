from typing import TypedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from plotting.plotting_config import (
    FONTS,
    apply_axis_style,
    calculate_grid,
    get_color,
    get_display_name,
    save_fig,
)


class SourceDict(TypedDict):
    """
    This class is just so it is easier to pass the type to the functions
    The idea is that we have a dataframe for either the conformal models or stl rules
    and a few more string attributes, so it is easier to iterate in the functions
    """

    df: pd.DataFrame
    col: str
    type: str
    prefix: str
    marker: str
    linestyle: str


def multi_alpha_plots(
    stl_df: pd.DataFrame | None,
    conformal_df: pd.DataFrame | None,
    output_path: str,
):
    """
    Generates multi-alpha comparison pdf

    Pages:
        1. Metrics vs Alpha (F1, False Alarms, Overlooked Violations)
        2. Confusion Matrix vs Alpha (TP, FP, TN, FN)
        3. Combined ROC (all methods on one plot)
        4. Individual ROC (grid of subplots, one per method)

    Args:
        stl_df: dataframe with STL classification metrics across alphas, or None
        conformal_df: dataframe with conformal classification metrics across alphas, or None
        output_path: path to save the output pdf
    """
    # we put the stl dataframe and the conformal dataframe in sources,
    # so that we can iterate it as one thing
    sources: list[SourceDict] = []
    if stl_df is not None:
        sources.append(
            SourceDict(
                df=stl_df,
                col="rule_name",
                type="rule",
                prefix="STL: ",
                marker="o",
                linestyle="-",
            )
        )
    if conformal_df is not None:
        sources.append(
            SourceDict(
                df=conformal_df,
                col="model_name",
                type="model",
                prefix="Conf: ",
                marker="s",
                linestyle="--",
            )
        )

    if not sources:
        return

    # number of episodes from any dataframe (they should all have the same)
    n_episodes = sources[0]["df"]["n_episodes"].iloc[0]

    # calculate global axis limits so all ROC plots share the same scale
    # this makes visual comparison easier
    roc_xlim, roc_ylim = _calculate_global_limits(sources)

    # generate each of the four pages
    with PdfPages(output_path) as pdf:
        _plot_metrics_vs_alpha(pdf, sources, n_episodes)
        _plot_confusion_matrix_vs_alpha(pdf, sources, n_episodes)
        _plot_combined_roc(pdf, sources, n_episodes, roc_xlim, roc_ylim)
        _plot_individual_roc(pdf, sources, n_episodes, roc_xlim, roc_ylim)


def _calculate_global_limits(
    sources: list[SourceDict],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Calculates x and y axis limits for all ROC plots

    This is so that the combined ROC-like plot and all individual ROC subplots
    share the exact same scale, which makes it easier to compare them visually

    Logic:
    1. Aggregates 'miss' (x) and 'fa' (y) data from all sources
    2. Finds the global min and max values.
    3. Adds a padding of 15% (or minimum 0.05) to sides so points aren't cut off
    Args:
        sources: list of source dictionaries with dataframes

    Returns:
        Tuple of ((x_min, x_max), (y_min, y_max))
    """
    if not sources:
        return (0.0, 1.0), (0.0, 1.0)

    # aggregate miss (x) and fa (y) data from all sources
    all_miss = pd.concat([s["df"]["miss"] for s in sources])
    all_fa = pd.concat([s["df"]["fa"] for s in sources])

    def get_padded_range(series: pd.Series) -> tuple[float, float]:
        """Adds padding so points aren't cut off at edges"""
        d_min, d_max = series.min(), series.max()
        span = d_max - d_min
        # 15% of the data span as padding, or at least 0.05
        padding = max(span * 0.15, 0.05)
        return max(0, d_min - padding), d_max + padding

    # the limits for x and y (range)
    xlim = get_padded_range(all_miss)
    ylim = get_padded_range(all_fa)

    return xlim, ylim


def _draw_metric_line(
    ax: Axes,
    df: pd.DataFrame,
    metric: str,
    name_col: str,
    prefix: str,
    marker: str,
    ls: str,
):
    """
    Draws line plot of metric vs alpha

    Args:
        ax: matplotlib axis to draw on
        df: dataframe with metrics
        metric: column name of the metric to plot
        name_col: column name for grouping ("rule_name" or "model_name")
        prefix: label prefix ("STL: " or "Conf: ")
        marker: marker style for the line
        ls: line style
    """
    # draw one line for every unique rule/model
    for name in df[name_col].unique():
        # filter data for this specific rule/model
        data = df[df[name_col] == name]

        color = get_color(name)
        label = f"{prefix}{get_display_name(name)}"

        ax.plot(
            data["alpha"],
            data[metric],
            marker=marker,
            linestyle=ls,
            color=color,
            label=label,
            linewidth=2.5,
            markersize=8,
            alpha=0.8,
        )


def _draw_roc_curve(
    ax: Axes,
    df: pd.DataFrame,
    name_col: str,
    prefix: str,
    marker: str,
    ls: str,
):
    """
    Draws ROC-like curve (but with miss vs fa) with alpha value annotations

    Args:
        ax: matplotlib axis to draw on
        df: dataframe with metrics
        name_col: column name for grouping ("rule_name" or "model_name")
        prefix: label prefix ("STL: " or "Conf: ")
        marker: marker style for the line
        ls: line style
    """
    # draw one curve per unique rule/model
    for name in df[name_col].unique():
        data = df[df[name_col] == name]

        color = get_color(name)
        label = f"{prefix}{get_display_name(name)}"

        # plot the main line with markers
        ax.plot(
            data["miss"],
            data["fa"],
            marker=marker,
            linestyle=ls,
            color=color,
            label=label,
            linewidth=2.5,
            markersize=10,
            alpha=0.8,
        )

        # annotate the roc-like plots with the labels (which is each alpha value)
        for _, row in data.iterrows():
            ax.annotate(
                f'{row["alpha"]:.2f}',
                (row["miss"], row["fa"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=FONTS.annotation_size,
                alpha=0.7,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.2,
                    edgecolor="none",
                ),
            )


def _draw_individual_roc_subplot(
    ax: Axes,
    method_data: pd.DataFrame,
    color: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    """
    Draws a single ROC curve on a dedicated subplot

    Args:
        ax: matplotlib axis to draw on
        method_data: dataframe with metrics for a single method
        color: line color
        title: subplot title
        xlim: x-axis limits (min, max)
        ylim: y-axis limits (min, max)
    """
    ax.plot(
        method_data["miss"],
        method_data["fa"],
        marker="o",
        linestyle="-",
        color=color,
        linewidth=2.5,
        markersize=10,
        alpha=0.8,
    )

    # annotation offset (so that it is a little to the side)
    x_offset = (xlim[1] - xlim[0]) * 0.02
    y_offset = (ylim[1] - ylim[0]) * 0.02

    # annotate each point with its alpha value
    for _, row in method_data.iterrows():
        ax.annotate(
            f'{row["alpha"]:.2f}',
            (row["miss"], row["fa"]),
            xytext=(x_offset * 10, y_offset * 10),
            textcoords="offset points",
            fontsize=FONTS.annotation_size,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                alpha=0.9,
            ),
        )

    apply_axis_style(
        ax, title=title, xlabel="Overlooked Violations", ylabel="False Alarms"
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _plot_metrics_vs_alpha(pdf, sources: list[SourceDict], n_episodes: int):
    """
    Plots page 1: F1, False Alarms, and Overlooked Violations vs Alpha

    Args:
        pdf: PdfPages object to save to
        sources: list of source dictionaries with dataframes
        n_episodes: number of episodes for the title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Metrics Comparison ({n_episodes} episodes)",
        fontsize=FONTS.page_title_fontsize,
    )

    # each tuple is (column_name, display_label)
    metrics = [
        ("f1", "F1 Score"),
        ("fa", "False Alarms"),
        ("miss", "Overlooked Violations"),
    ]

    # axes[0] -> left plot, axes[1] -> middle plot, axes[2] -> right plot
    for ax, (metric, ylabel) in zip(axes, metrics):
        # draw lines for both STL and conformal sources
        for source in sources:
            _draw_metric_line(
                ax,
                source["df"],
                metric,
                source["col"],
                source["prefix"],
                source["marker"],
                source["linestyle"],
            )

        apply_axis_style(ax, title=ylabel, xlabel="Alpha", ylabel=ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=FONTS.legend_item_size, loc="best")

    save_fig(pdf, fig)


def _plot_confusion_matrix_vs_alpha(pdf, sources: list[SourceDict], n_episodes: int):
    """
    Plots page 2: Confusion Matrix elements vs Alpha

    Args:
        pdf: PdfPages object to save to
        sources: list of source dictionaries with dataframes
        n_episodes: number of episodes for the title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Confusion Matrix ({n_episodes} episodes)",
        fontsize=FONTS.page_title_fontsize,
    )

    # each tuple is (column_name, display_label)
    conf_metrics = [
        ("TP", "True Positives"),
        ("FP", "False Positives"),
        ("TN", "True Negatives"),
        ("FN", "False Negatives"),
    ]

    # we flatten the axes so we obtain [ax_top_left, ax_top_right, ax_bot_left, ax_bot_right]
    # and zip it after [(ax_top_left, "TP"), (ax_top_right, "FP"), (ax_bot_left, "TN"), (ax_bot_right, FN)]
    for ax, (metric, ylabel) in zip(axes.flatten(), conf_metrics):
        for source in sources:
            _draw_metric_line(
                ax,
                source["df"],
                metric,
                source["col"],
                source["prefix"],
                source["marker"],
                source["linestyle"],
            )

        apply_axis_style(ax, title=ylabel, xlabel="Alpha", ylabel=ylabel)
        ax.legend(fontsize=FONTS.legend_item_size, loc="best")
        # format y-axis tick labels with thousands separator (e.g., 1000 -> "1,000")
        # FuncFormatter takes a function(value, position) -> string
        # we ignore position (_) and just format the value
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    save_fig(pdf, fig)


def _plot_combined_roc(
    pdf,
    sources: list[SourceDict],
    n_episodes: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    """
    Plots page 3: Combined ROC curves (all methods on one plot)

    Args:
        pdf: PdfPages object to save to
        sources: list of source dictionaries with dataframes
        n_episodes: number of episodes for the title
        xlim: x-axis limits (min, max)
        ylim: y-axis limits (min, max)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(
        f"ROC Curve ({n_episodes} episodes)", fontsize=FONTS.page_title_fontsize
    )

    # draw both STL and conformal curves on the same axis
    for source in sources:
        _draw_roc_curve(
            ax,
            source["df"],
            source["col"],
            source["prefix"],
            source["marker"],
            source["linestyle"],
        )

    apply_axis_style(ax, xlabel="Overlooked Violations", ylabel="False Alarms")
    # use global limits so this plot matches the individual ROC plots
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=FONTS.legend_item_size, loc="best")

    save_fig(pdf, fig)


def _plot_individual_roc(
    pdf,
    sources: list[SourceDict],
    n_episodes: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    """
    Plots page 4: Individual ROC curves (grid of subplots)

    Args:
        pdf: PdfPages object to save to
        sources: list of source dictionaries with dataframes
        n_episodes: number of episodes for the title
        xlim: x-axis limits (min, max)
        ylim: y-axis limits (min, max)
    """
    # FIX: When I have the time, I should fix the order, which at the moment
    # seems random (order that the plots appear in the last page)

    # collect all individual methods (STL and conf) into one list
    individual_plots = []
    for source in sources:
        df = source["df"]
        col = source["col"]
        for name in df[col].unique():
            individual_plots.append(
                {
                    "data": df[df[col] == name],
                    "title": f"{source['prefix']}{get_display_name(name)}",
                    "color": get_color(name),
                }
            )

    if not individual_plots:
        return

    # setup grid based on how many total methods we have
    # this is fine for now but maybe I'll change it later
    rows, cols = calculate_grid(len(individual_plots), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), squeeze=False)
    fig.suptitle(
        f"ROC Curves - Individual ({n_episodes} episodes)",
        fontsize=FONTS.page_title_fontsize,
    )

    # flatten axes and plot each method on its own subplot
    axes_flat = axes.flatten()
    for ax, item in zip(axes_flat, individual_plots):
        _draw_individual_roc_subplot(
            ax, item["data"], item["color"], item["title"], xlim, ylim
        )

    # hide any remaining empty subplots in the grid
    for ax in axes_flat[len(individual_plots) :]:
        ax.set_visible(False)

    save_fig(pdf, fig)
