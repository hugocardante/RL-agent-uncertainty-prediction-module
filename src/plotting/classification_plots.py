import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from plotting.data import load_csv
from plotting.plotting_config import (
    FONTS,
    MODELS_TO_PLOT,
    RULES_TO_PLOT,
    apply_axis_style,
    calculate_grid,
    get_color,
    get_display_name,
    save_fig,
)


def classification_plots(csv_path: str, output_path: str, name_column: str):
    """
    Generates classification metrics pdf.

    Pages:
        1. Overall Performance (F1, False Alarms, Overlooked Violations)
        2. Confusion Matrices (one per model/rule)
        3. ROC-like plot (overlooked violations vs false alarms)

    Args:
        csv_path: path to the classification csv file
        output_path: path to save the output pdf
        name_column: column name to group by ("rule_name" or "model_name")
    """
    df = load_csv(csv_path)
    if df is None:
        return

    # (see RULES_TO_PLOT and MODELS_TO_PLOT)
    if name_column == "rule_name" and RULES_TO_PLOT is not None:
        df = df[df["rule_name"].isin(RULES_TO_PLOT)]

    if name_column == "model_name" and MODELS_TO_PLOT is not None:
        df = df[df["model_name"].isin(MODELS_TO_PLOT)]

    df = df.dropna(subset=["miss", "fa", "f1"])
    # This should not happen, unless something goes wrong in
    # the analysis phase..
    if df.empty:
        return

    label = "STL" if name_column == "rule_name" else "Conformal"

    # For the aggregated plots we put the number of episodes too
    number_episodes = df["n_episodes"].iloc[0] if "n_episodes" in df.columns else None
    title_suffix = f" ({number_episodes}) " if number_episodes else ""

    with PdfPages(output_path) as pdf:
        # [page 1] metrics bars
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        fig.suptitle(
            f"{label} Overall Performance{title_suffix}",
            fontsize=FONTS.page_title_fontsize,
        )

        _draw_metric_bars(axes[0], df, "f1", name_column, "F1 Score")
        _draw_metric_bars(axes[1], df, "fa", name_column, "False Alarms")
        _draw_metric_bars(axes[2], df, "miss", name_column, "Overlooked Violations")

        save_fig(pdf, fig)

        # [page 2] confusion grid (shows tp, tn, fp, fn)
        n_items = len(df)
        rows, cols = calculate_grid(n_items, max_cols=3)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle(
            f"{label} Confusion Matrices{title_suffix}",
            fontsize=FONTS.page_title_fontsize,
        )

        # so we can loop even if we only have one item
        axes_flat = axes.flatten() if n_items > 1 else [axes]

        for ax, (_, row) in zip(axes_flat, df.iterrows()):
            _draw_confusion_table(ax, row, name_column)

        for ax in axes_flat[n_items:]:
            ax.set_visible(False)

        save_fig(pdf, fig)

        # [page 3] Roc-like plot (but with overlooked violations vs false alarms)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle(
            f"ROC: Overlooked Violations vs False Alarms{title_suffix}",
            fontsize=FONTS.page_title_fontsize,
        )

        _draw_roc_scatter(ax, df, name_column)
        save_fig(pdf, fig)


def _draw_metric_bars(
    ax: Axes, df: pd.DataFrame, metric: str, name_column: str, ylabel: str
):
    """
    Draws bar chart comparing metrics (f1, miss, fa)

    Args:
        ax: matplotlib axis to draw on
        df: dataframe with classification metrics
        metric: column name of the metric to plot
        name_column: column name to group by ("rule_name" or "model_name")
        ylabel: label for the y-axis
    """
    # set the name_column (e.g. "model_name") as index, then extract the metric column
    # this gives us a Series where the index is the model/rule name and values are the metric
    values = df.set_index(name_column)[metric]

    labels = [get_display_name(name) for name in values.index]
    colors = [get_color(name) for name in values.index]

    # draw one bar per model/rule
    # range(len(values)) gives x positions (0, 1, 2, ...)
    # values.values gives the bar heights (the metric values)
    bars = ax.bar(
        range(len(values)), np.array(values), color=colors, alpha=0.7, edgecolor="black"
    )

    # axis style
    title = f"{ylabel} by {'Rule' if name_column == 'rule_name' else 'Model'}"
    apply_axis_style(ax, title=title, ylabel=ylabel, rotate_xticks=True, grid=False)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # values on top of bars (annotations)
    for bar, val in zip(bars, values.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONTS.annotation_size,
        )


def _draw_confusion_table(ax: Axes, row: pd.Series, name_column: str):
    """
    Draws a confusion matrix table for a single model/rule

    Args:
        ax: matplotlib axis to draw on
        row: pandas Series with TP, TN, FP, FN values
        name_column: column name to get the title from
    """
    ax.axis("off")
    title = get_display_name(row[name_column])

    ax.set_title(
        title,
        fontsize=FONTS.subplot_title_fontsize,
        fontweight=FONTS.subplot_title_fontweight,
    )

    table_data = [
        ["", "Classified as Positive", "Classified as Negative"],
        ["Positive", f"{row['TP']:.0f}", f"{row['FN']:.0f}"],
        ["Negative", f"{row['FP']:.0f}", f"{row['TN']:.0f}"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(FONTS.tick_label_size)
    table.scale(1.2, 2)


def _draw_roc_scatter(ax: Axes, df: pd.DataFrame, name_column: str):
    """
    Draws ROC-like scatter plot (overlooked violations vs false alarms)

    Args:
        ax: matplotlib axis to draw on
        df: dataframe with classification metrics
        name_column: column name to group by ("rule_name" or "model_name")
    """
    for _, row in df.iterrows():
        color = get_color(row[name_column])
        label = get_display_name(row[name_column])

        ax.scatter(row["miss"], row["fa"], s=300, alpha=0.7, color=color, zorder=3)
        ax.annotate(
            label,
            (row["miss"], row["fa"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=FONTS.annotation_size,
        )

    apply_axis_style(ax, xlabel="Overlooked Violations", ylabel="False Alarms")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
