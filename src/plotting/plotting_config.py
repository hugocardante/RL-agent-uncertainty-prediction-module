import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# WARNING: This environment variable is important, do not remove!
# this prevents OpenMP from using multiple threads per worker
# which can cause issues with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"


# number of parallel workers for plotting
# mp.cpu_count() uses all available cores
MAX_WORKERS = mp.cpu_count()

# filter which models to plot (None = plot all)
# set to a list like ["vanilla", "knn_norm"] to plot only specific models
MODELS_TO_PLOT: list[str] | None = None

# filter which STL rules to plot (None = plot all)
# similar to MODELS_TO_PLOT but for STL rules
RULES_TO_PLOT: list[str] | None = None

# if True, only generate multi_alpha_plots and skip all other plots
PLOT_ONLY_MULTI_ALPHA = False


class PlottingTask(NamedTuple):
    """Task tuple for parallel plotting execution"""

    name: str
    func: Callable[..., None]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class FontConfig:
    """
    Font configuration for all plot elements

    The goal is to keep font sizes and weights consistent for all the plots
    Individual plots can still override these values if needed
    The weight can be "normal" or "bold", or check docs for other options
    """

    # page title (suptitle)
    page_title_fontsize: int = 16
    page_title_weight: str = "normal"

    # subplot (axes) titles
    subplot_title_fontsize: int = 14
    subplot_title_fontweight: str = "normal"

    # axis labels (e.g., "Coverage (%)", "Power Line")
    axis_label_fontsize: int = 16
    axis_label_fontweight: str = "normal"

    # tick labels
    tick_label_size: int = 10

    # legend
    legend_title_size: int = 16
    legend_title_fontweight: str = "normal"
    legend_item_size: int = 14
    legend_item_fontweight: str = "normal"

    # annotations / small text
    annotation_size: int = 9
    annotation_fontweight: str = "normal"


@dataclass
class LayoutConfig:
    """Layout and spacing configuration for figures"""

    title_space: float = 0
    # vertical space reserved at bottom for legend
    legend_space: float = 0.13
    # y position of legend bottom edge
    legend_bottom_y: float = 0
    # number of columns in legend
    legend_ncols: int = 4


# global config instances
FONTS = FontConfig()
LAYOUT = LayoutConfig()


# color mapping for models and STL rules
# keys must match the names used in config.py (check csv files)
COLORS = {
    "vanilla": "green",
    "vanilla_rule": "#2ca02c",
    "knn_norm": "blue",
    "knn_norm_rule": "#ff7f0e",
    "knn_norm_with_aci": "darkolivegreen",
    "aci": "#9467bd",
}


# display names for models (prettier labels in plots)
DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "knn_norm": "k-NN (normalised)",
    "aci": "ACI",
    "knn_norm_with_aci": "k-NN (normalised) + ACI rule",
    "vanilla_rule": "(STL) Vanilla rule",
    "knn_norm_rule": "(STL) k-NN (norm) rule",
}


def get_display_name(name: str) -> str:
    """
    Gets the display name for a model

    Args:
        model_name/rule_name: internal model name (e.g., "knn_norm")

    Returns:
        Display name if defined, otherwise converts underscores to spaces
        and title-cases (e.g., "knn_norm" -> "Knn Norm")
    """
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def get_color(name: str) -> str:
    """
    Gets color for a model or rule

    Args:
        name: model or rule name

    Returns:
        Color string if defined, otherwise defaults to 'grey'
    """
    return COLORS.get(name, "grey")


def calculate_grid(n_subplots: int, max_cols: int = 4) -> tuple[int, int]:
    """
    Calculates grid dimensions for n subplots

    Tries to create a roughly square grid, but limits columns to max_cols.

    Args:
        n_subplots: number of subplots to fit
        max_cols: maximum number of columns allowed

    Returns:
        Tuple of (rows, cols)
    """
    if n_subplots <= 0:
        return 1, 1
    # try to make it kinda square, but cap at max_cols
    cols = min(math.ceil(math.sqrt(n_subplots)), max_cols)
    # calculate rows needed to fit all subplots
    rows = math.ceil(n_subplots / cols)
    return rows, cols


def apply_axis_style(
    ax: Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    rotate_xticks: bool = False,
    time_axis: bool = False,
):
    """
    Applies consistent styling to an axis

    Centralizes axis styling to ensure all plots have the same look.

    Args:
        ax: matplotlib axis to style
        title: subplot title, or None
        xlabel: x-axis label, or None
        ylabel: y-axis label, or None
        grid: if True, show grid lines
        rotate_xticks: if True, rotate x-axis tick labels 45 degrees
        time_axis: if True, format x-axis for time data (HH:MM format)
    """
    if title:
        ax.set_title(
            title,
            fontsize=FONTS.subplot_title_fontsize,
            fontweight=FONTS.subplot_title_fontweight,
        )

    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=FONTS.axis_label_fontsize,
            fontweight=FONTS.axis_label_fontweight,
        )

    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=FONTS.axis_label_fontsize,
            fontweight=FONTS.axis_label_fontweight,
        )

    ax.tick_params(axis="both", labelsize=FONTS.tick_label_size)

    if rotate_xticks:
        # rotate labels and align right so they don't overlap
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if grid:
        ax.grid(alpha=0.3, linestyle="--")

    if time_axis:
        # format x-axis as hours:minutes
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def add_model_legend(fig, model_names: list[str]):
    """
    Adds a legend showing all models at bottom of figure

    Args:
        fig: matplotlib figure to add legend to
        model_names: list of model names to include in legend
    """
    # create colored patches for each model
    handles = [Patch(facecolor=get_color(m), edgecolor="black") for m in model_names]
    labels = [get_display_name(m) for m in model_names]

    fig.legend(
        handles,
        labels,
        title="Models",
        loc="lower center",
        bbox_to_anchor=(0.5, LAYOUT.legend_bottom_y),
        ncol=min(LAYOUT.legend_ncols, len(handles)),
        frameon=True,
        fontsize=FONTS.legend_item_size,
        title_fontsize=FONTS.legend_title_size,
    )


def save_fig(
    pdf: PdfPages, fig: Figure, rect: tuple[float, float, float, float] | None = None
):
    """
    Saves figure to pdf and closes it

    Args:
        pdf: PdfPages object to save to
        fig: matplotlib figure to save
        rect: optional rect for tight_layout [left, bottom, right, top]
              used to leave space for legends outside the plot area
    """
    if rect is not None:
        plt.tight_layout(rect=rect)
    else:
        plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)
