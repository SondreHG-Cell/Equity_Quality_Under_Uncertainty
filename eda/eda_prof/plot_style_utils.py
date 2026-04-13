
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import display

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

PLOT_EXPORT_DIR = Path("plot_exports")
PLOT_EXPORT_DIR.mkdir(exist_ok=True)

THEME = {
    "figure.facecolor": "#f5f5f7",
    "axes.facecolor": "#ebedf2",
    "axes.edgecolor": "#c9ced8",
    "axes.labelcolor": "#2f3b4a",
    "axes.titlecolor": "#1f2937",
    "axes.grid": True,
    "grid.color": "#b8c0cc",
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.color": "#435165",
    "ytick.color": "#435165",
    "legend.frameon": False,
    "figure.autolayout": False,
}

PALETTE = {
    "blue": "#4C78A8",
    "teal": "#54A6A6",
    "green": "#72B7B2",
    "orange": "#F2A65A",
    "red": "#D95F5F",
    "gold": "#E9C46A",
    "navy": "#2A4E6E",
    "slate": "#7B8BA3",
}

def apply_plot_theme():
    plt.rcParams.update(THEME)
    if HAS_SEABORN:
        sns.set_theme(style="darkgrid", rc=THEME)

def fmt_int(x, pos=None):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def add_bar_labels(ax, fmt="{:,.0f}", pad_frac=0.01):
    ymax = max([p.get_height() for p in ax.patches], default=0)
    offset = ymax * pad_frac if ymax > 0 else 0.1
    for patch in ax.patches:
        value = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#253243",
            fontweight="semibold",
        )

def add_total_text(ax, total, label="Total"):
    ax.text(
        0.995,
        -0.18,
        f"{label}: {total:,.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#4b5563",
    )

def style_axis(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))

def save_plot_data(name, df):
    out = PLOT_EXPORT_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out

def plot_annotated_bar(series, title, xlabel, ylabel, color, top_n=None, rotation=0):
    s = series.copy()
    if top_n is not None:
        s = s.head(top_n)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(s.index.astype(str), s.values, color=color, edgecolor="white", linewidth=1.2)
    add_bar_labels(ax)
    add_total_text(ax, float(s.sum()))
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticklabels(s.index.astype(str), rotation=rotation, ha="right" if rotation else "center")
    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_ratio_distribution(series, low_threshold, high_threshold, extreme_low, extreme_high):
    data = series.dropna().clip(lower=0, upper=3)
    fig, ax = plt.subplots(figsize=(11, 5))
    if HAS_SEABORN:
        sns.histplot(data, bins=60, kde=True, color=PALETTE["blue"], edgecolor="white", alpha=0.90, ax=ax)
    else:
        ax.hist(data, bins=60, color=PALETTE["blue"], edgecolor="white", alpha=0.90)
    for value, color, label in [
        (low_threshold, PALETTE["red"], "Low threshold"),
        (high_threshold, PALETTE["orange"], "High threshold"),
        (extreme_low, PALETTE["navy"], "Extreme low"),
        (extreme_high, PALETTE["teal"], "Extreme high"),
    ]:
        ax.axvline(value, linestyle="--", linewidth=2, color=color, label=f"{label}: {value:.2f}")
    style_axis(
        ax,
        title="Distribution of cost ratio",
        xlabel="(COGS + XSGA_COMPONENTS) / REVT   [clipped to 0–3 for display]",
        ylabel="Number of observations",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
    return fig, ax

def display_summary_table(flag_summary):
    styled = (
        flag_summary.style
        .hide(axis="index")
        .format({"value": "{:,.0f}"})
        .set_properties(**{"text-align": "left"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#d9dee7"), ("color", "#1f2937"), ("font-weight", "bold")]},
            {"selector": "td", "props": [("padding", "6px 10px")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("font-family", "DejaVu Sans")]},
        ])
    )
    display(styled)

apply_plot_theme()
