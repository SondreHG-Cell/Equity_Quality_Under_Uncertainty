from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

METHOD_COLS = {
    "Observed Quality": "theta_obs",
    "Latent Quality": "theta_post_mean",
    "Conservative Quality": "theta_conservative",
}
METHOD_COLORS = {
    "Observed Quality": "#173a5e",
    "Latent Quality": "#2c73ad",
    "Conservative Quality": "#58a7df",
}
ACCENT = "#8ec5e8"
TEXT = "#173a5e"
PANEL_BG = "#f6f9fc"
GRID = "#9ebfda"
EXCHANGE_MAP = {
    "CO": "Copenhagen",
    "HE": "Helsinki",
    "OL": "Oslo",
    "ST": "Stockholm",
    "IC": "Iceland",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate appendix figures that describe how accounting uncertainty changes firm quality."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "current_res",
        help="Main result directory containing latent_prof_model outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/appendix_quality_uncertainty.",
    )
    parser.add_argument(
        "--manual-figure-dir",
        type=Path,
        default=PROJECT_ROOT / "manual_review" / "Figures" / "quality_uncertainty_appendix",
        help="Optional figure mirror for thesis copy/paste workflows.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--copy-to-manual", action="store_true", default=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def exchange_from_ticker(ticker: str) -> str:
    suffix = str(ticker).split(".")[-1]
    return EXCHANGE_MAP.get(suffix, suffix)


def clean_filename(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace("--", "_")
        .replace("__", "_")
    )


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def apply_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="--", alpha=0.65)
    ax.grid(axis="x", color="#c8d9e8", linewidth=0.45, linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a7bfd5")
    ax.spines["bottom"].set_color("#a7bfd5")
    ax.tick_params(colors="#20384f", labelsize=9)


def save_fig(fig: plt.Figure, fig_dir: Path, name: str, dpi: int) -> list[Path]:
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png, pdf]


def quantile_agg(series: pd.Series, q: float) -> float:
    return float(series.quantile(q))


def load_main_data(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "latent_prof_model" / "latent_prof_firm_year.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing latent quality file: {path}")
    df = pd.read_csv(path)
    df["Exchange"] = df["Ticker"].map(exchange_from_ticker)
    df["Sector"] = df["Sector"].fillna("Unknown")
    df["Industry"] = df["Industry"].fillna("Unknown")
    df["latent_adjustment"] = df["theta_post_mean"] - df["theta_obs"]
    df["conservative_adjustment"] = df["theta_conservative"] - df["theta_obs"]
    df["abs_latent_adjustment"] = df["latent_adjustment"].abs()
    df["abs_conservative_adjustment"] = df["conservative_adjustment"].abs()
    df["latent_ci_lower"] = df["theta_post_mean"] - 1.96 * df["theta_post_sd"]
    df["latent_ci_upper"] = df["theta_post_mean"] + 1.96 * df["theta_post_sd"]
    df["latent_ci_width"] = df["latent_ci_upper"] - df["latent_ci_lower"]
    df["annual_median_sigma"] = df.groupby("FormationYear")["sigma_acc"].transform("median")
    df["sigma_to_year_median"] = df["sigma_acc"] / df["annual_median_sigma"]
    df["uncertainty_weight"] = 1.0 - df["lambda_i"]
    return df


def make_long_quality(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Ticker", "FormationYear", "Exchange", "Sector", "sigma_acc", "lambda_i"]
    frames = []
    for method, col in METHOD_COLS.items():
        sub = df[cols + [col]].copy()
        sub["Method"] = method
        sub["Quality"] = sub[col]
        frames.append(sub.drop(columns=[col]))
    return pd.concat(frames, ignore_index=True)


def summarize_quality_levels(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.groupby(["FormationYear", "Method"], as_index=False)
        .agg(
            mean_quality=("Quality", "mean"),
            median_quality=("Quality", "median"),
            std_quality=("Quality", "std"),
            p05_quality=("Quality", lambda s: quantile_agg(s, 0.05)),
            p25_quality=("Quality", lambda s: quantile_agg(s, 0.25)),
            p75_quality=("Quality", lambda s: quantile_agg(s, 0.75)),
            p95_quality=("Quality", lambda s: quantile_agg(s, 0.95)),
            n_firm_years=("Quality", "size"),
        )
        .assign(
            iqr_quality=lambda x: x["p75_quality"] - x["p25_quality"],
            p95_p05_spread=lambda x: x["p95_quality"] - x["p05_quality"],
        )
    )


def plot_quality_levels(summary: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, metric, title in zip(
        axes,
        ["mean_quality", "median_quality"],
        ["Panel A: Mean quality", "Panel B: Median quality"],
    ):
        apply_axis_style(ax)
        for method in METHOD_COLS:
            sub = summary.loc[summary["Method"].eq(method)].sort_values("FormationYear")
            ax.plot(
                sub["FormationYear"],
                sub[metric],
                label=method,
                color=METHOD_COLORS[method],
                linewidth=2.25,
            )
        ax.set_title(title, color=TEXT, fontweight="bold", fontsize=10)
        ax.set_xlabel("Formation year", color=TEXT)
        ax.set_ylabel("Quality signal", color=TEXT)
    fig.suptitle("Quality before and after uncertainty adjustment", color=TEXT, fontweight="bold", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.subplots_adjust(top=0.78, wspace=0.25)
    return save_fig(fig, fig_dir, "quality_levels_mean_median", dpi)


def plot_quality_spread(summary: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    fig.patch.set_facecolor("white")
    for ax, metric, title in zip(
        axes,
        ["std_quality", "p95_p05_spread"],
        ["Panel A: Cross-sectional standard deviation", "Panel B: 5th-95th percentile spread"],
    ):
        apply_axis_style(ax)
        for method in METHOD_COLS:
            sub = summary.loc[summary["Method"].eq(method)].sort_values("FormationYear")
            ax.plot(sub["FormationYear"], sub[metric], color=METHOD_COLORS[method], label=method, linewidth=2.25)
        ax.set_title(title, color=TEXT, fontweight="bold", fontsize=10)
        ax.set_xlabel("Formation year", color=TEXT)
        ax.set_ylabel("Quality spread", color=TEXT)
    fig.suptitle("Quality spread before and after adjustment", color=TEXT, fontweight="bold", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.subplots_adjust(top=0.78, wspace=0.25)
    return save_fig(fig, fig_dir, "quality_spread_before_after", dpi)


def plot_adjustment_distribution(df: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    for ax in axes:
        apply_axis_style(ax)

    q_low, q_high = df[["latent_adjustment", "conservative_adjustment"]].stack().quantile([0.01, 0.99])
    bins = np.linspace(q_low, q_high, 70)
    axes[0].hist(
        df["latent_adjustment"].clip(q_low, q_high),
        bins=bins,
        color=METHOD_COLORS["Latent Quality"],
        alpha=0.75,
        label="Latent - Observed",
    )
    axes[0].hist(
        df["conservative_adjustment"].clip(q_low, q_high),
        bins=bins,
        color=METHOD_COLORS["Conservative Quality"],
        alpha=0.55,
        label="Conservative - Observed",
    )
    axes[0].axvline(0, color=TEXT, linestyle="--", linewidth=1.0)
    axes[0].set_title("Panel A: Signed adjustment", color=TEXT, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Change in quality signal", color=TEXT)
    axes[0].set_ylabel("Firm-years", color=TEXT)
    axes[0].legend(frameon=False)

    abs_low, abs_high = df[["abs_latent_adjustment", "abs_conservative_adjustment"]].stack().quantile([0.0, 0.99])
    bins = np.linspace(abs_low, abs_high, 70)
    axes[1].hist(
        df["abs_latent_adjustment"].clip(abs_low, abs_high),
        bins=bins,
        color=METHOD_COLORS["Latent Quality"],
        alpha=0.75,
        label="Latent",
    )
    axes[1].hist(
        df["abs_conservative_adjustment"].clip(abs_low, abs_high),
        bins=bins,
        color=METHOD_COLORS["Conservative Quality"],
        alpha=0.55,
        label="Conservative",
    )
    axes[1].set_title("Panel B: Absolute adjustment", color=TEXT, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Absolute change in quality signal", color=TEXT)
    axes[1].set_ylabel("Firm-years", color=TEXT)
    axes[1].legend(frameon=False)

    fig.suptitle("Distribution of firm-year quality adjustments", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    return save_fig(fig, fig_dir, "quality_adjustment_distribution", dpi)


def plot_observed_vs_adjusted(df: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    plot_df = df.copy()
    x_low, x_high = plot_df["theta_obs"].quantile([0.01, 0.99])
    y_low, y_high = plot_df[["theta_post_mean", "theta_conservative"]].stack().quantile([0.01, 0.99])
    low = min(x_low, y_low)
    high = max(x_high, y_high)
    sample = plot_df.sample(n=min(5000, len(plot_df)), random_state=20260518)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, col, title, color in [
        (axes[0], "theta_post_mean", "Panel A: Latent Quality", METHOD_COLORS["Latent Quality"]),
        (axes[1], "theta_conservative", "Panel B: Conservative Quality", METHOD_COLORS["Conservative Quality"]),
    ]:
        apply_axis_style(ax)
        ax.scatter(
            sample["theta_obs"].clip(low, high),
            sample[col].clip(low, high),
            c=sample["sigma_to_year_median"].clip(0, sample["sigma_to_year_median"].quantile(0.99)),
            cmap=LinearSegmentedColormap.from_list("blues_custom", ["#cfe5f6", color, "#173a5e"]),
            s=12,
            alpha=0.45,
            edgecolors="none",
        )
        ax.plot([low, high], [low, high], color=TEXT, linestyle="--", linewidth=1.1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(title, color=TEXT, fontweight="bold", fontsize=10)
        ax.set_xlabel("Observed Quality", color=TEXT)
    axes[0].set_ylabel("Adjusted quality", color=TEXT)
    fig.suptitle("Observed quality versus uncertainty-adjusted quality", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.84, wspace=0.12)
    return save_fig(fig, fig_dir, "observed_vs_adjusted_quality", dpi)


def summarize_group_impact(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, as_index=False)
        .agg(
            n_firm_years=("Ticker", "size"),
            n_firms=("Ticker", "nunique"),
            mean_sigma=("sigma_acc", "mean"),
            median_sigma=("sigma_acc", "median"),
            mean_lambda=("lambda_i", "mean"),
            mean_uncertainty_weight=("uncertainty_weight", "mean"),
            mean_abs_latent_adjustment=("abs_latent_adjustment", "mean"),
            mean_abs_conservative_adjustment=("abs_conservative_adjustment", "mean"),
            mean_latent_adjustment=("latent_adjustment", "mean"),
            mean_conservative_adjustment=("conservative_adjustment", "mean"),
        )
        .sort_values("mean_abs_latent_adjustment", ascending=False)
    )


def plot_group_impact(summary: pd.DataFrame, group_col: str, fig_dir: Path, dpi: int) -> list[Path]:
    if group_col == "Sector":
        plot_df = summary.loc[summary["n_firm_years"].ge(30)].head(12).iloc[::-1]
        figsize = (12, 6.6)
    else:
        plot_df = summary.iloc[::-1]
        figsize = (12, 4.8)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("white")
    y = np.arange(len(plot_df))
    for ax in axes:
        apply_axis_style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6, linestyle="--", alpha=0.65)
        ax.grid(axis="y", visible=False)

    axes[0].barh(y - 0.18, plot_df["mean_abs_latent_adjustment"], height=0.36, color=METHOD_COLORS["Latent Quality"], label="Latent")
    axes[0].barh(
        y + 0.18,
        plot_df["mean_abs_conservative_adjustment"],
        height=0.36,
        color=METHOD_COLORS["Conservative Quality"],
        label="Conservative",
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df[group_col])
    axes[0].set_title("Panel A: Mean absolute quality adjustment", color=TEXT, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Absolute adjustment", color=TEXT)
    axes[0].legend(frameon=False)

    axes[1].barh(y, plot_df["mean_sigma"], color=ACCENT)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_title("Panel B: Mean accounting uncertainty", color=TEXT, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Mean $\\sigma_i$", color=TEXT)

    fig.suptitle(f"Impact of accounting uncertainty by {group_col.lower()}", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.84, wspace=0.18)
    return save_fig(fig, fig_dir, f"quality_uncertainty_impact_by_{group_col.lower()}", dpi)


def summarize_sigma_by_year(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("FormationYear", as_index=False)
        .agg(
            mean_sigma=("sigma_acc", "mean"),
            median_sigma=("sigma_acc", "median"),
            p05_sigma=("sigma_acc", lambda s: quantile_agg(s, 0.05)),
            p25_sigma=("sigma_acc", lambda s: quantile_agg(s, 0.25)),
            p75_sigma=("sigma_acc", lambda s: quantile_agg(s, 0.75)),
            p95_sigma=("sigma_acc", lambda s: quantile_agg(s, 0.95)),
            mean_lambda=("lambda_i", "mean"),
            mean_uncertainty_weight=("uncertainty_weight", "mean"),
            n_firm_years=("sigma_acc", "size"),
        )
    )


def plot_sigma_evolution(summary: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor("white")
    apply_axis_style(ax)
    x = summary["FormationYear"].to_numpy()
    ax.fill_between(x, summary["p05_sigma"].to_numpy(), summary["p95_sigma"].to_numpy(), color=ACCENT, alpha=0.35, label="5th-95th percentile")
    ax.plot(x, summary["mean_sigma"], color=METHOD_COLORS["Observed Quality"], linewidth=2.25, label="Mean $\\sigma_i$")
    ax.plot(x, summary["median_sigma"], color=METHOD_COLORS["Latent Quality"], linewidth=2.25, label="Median $\\sigma_i$")
    ax.set_title("Evolution of accounting uncertainty", color=TEXT, fontweight="bold", fontsize=13)
    ax.set_xlabel("Formation year", color=TEXT)
    ax.set_ylabel("$\\sigma_i$", color=TEXT)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(bottom=0.22)
    return save_fig(fig, fig_dir, "sigma_evolution", dpi)


def plot_sigma_distribution(df: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    for ax in axes:
        apply_axis_style(ax)
    q99 = df["sigma_acc"].quantile(0.99)
    axes[0].hist(df["sigma_acc"].clip(upper=q99), bins=70, color=METHOD_COLORS["Latent Quality"], alpha=0.82)
    axes[0].axvline(df["sigma_acc"].median(), color=TEXT, linestyle="--", linewidth=1.2, label="Median")
    axes[0].axvline(df["sigma_acc"].mean(), color="#58a7df", linestyle="-", linewidth=1.4, label="Mean")
    axes[0].set_title("Panel A: Pooled firm-year distribution", color=TEXT, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("$\\sigma_i$ (99th percentile clipped)", color=TEXT)
    axes[0].set_ylabel("Firm-years", color=TEXT)
    axes[0].legend(frameon=False)

    years = sorted(df["FormationYear"].unique())
    data = [df.loc[df["FormationYear"].eq(year), "sigma_acc"].clip(upper=q99).dropna() for year in years]
    axes[1].boxplot(
        data,
        labels=years,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT, "linewidth": 1.2},
        boxprops={"facecolor": ACCENT, "edgecolor": "#4b86b5", "alpha": 0.75},
        whiskerprops={"color": "#4b86b5"},
        capprops={"color": "#4b86b5"},
    )
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Panel B: Distribution by formation year", color=TEXT, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Formation year", color=TEXT)
    axes[1].set_ylabel("$\\sigma_i$", color=TEXT)
    fig.suptitle("Distribution of accounting uncertainty", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    return save_fig(fig, fig_dir, "sigma_distribution", dpi)


def summarize_adjustment_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sigma_decile"] = pd.qcut(out["sigma_acc"], 10, labels=False, duplicates="drop") + 1
    return (
        out.groupby("sigma_decile", as_index=False)
        .agg(
            min_sigma=("sigma_acc", "min"),
            max_sigma=("sigma_acc", "max"),
            mean_sigma=("sigma_acc", "mean"),
            mean_lambda=("lambda_i", "mean"),
            mean_uncertainty_weight=("uncertainty_weight", "mean"),
            mean_abs_latent_adjustment=("abs_latent_adjustment", "mean"),
            mean_abs_conservative_adjustment=("abs_conservative_adjustment", "mean"),
            mean_latent_ci_width=("latent_ci_width", "mean"),
            n_firm_years=("Ticker", "size"),
        )
    )


def plot_adjustment_by_sigma_decile(summary: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        apply_axis_style(ax)
    x = summary["sigma_decile"].to_numpy()
    axes[0].plot(x, summary["mean_abs_latent_adjustment"], marker="o", linewidth=2.2, color=METHOD_COLORS["Latent Quality"], label="Latent")
    axes[0].plot(
        x,
        summary["mean_abs_conservative_adjustment"],
        marker="o",
        linewidth=2.2,
        color=METHOD_COLORS["Conservative Quality"],
        label="Conservative",
    )
    axes[0].set_title("Panel A: Mean absolute adjustment", color=TEXT, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("$\\sigma_i$ decile", color=TEXT)
    axes[0].set_ylabel("Absolute quality adjustment", color=TEXT)
    axes[0].legend(frameon=False)

    axes[1].plot(x, summary["mean_lambda"], marker="o", linewidth=2.2, color=METHOD_COLORS["Observed Quality"], label="$\\lambda_i$")
    axes[1].plot(x, summary["mean_uncertainty_weight"], marker="o", linewidth=2.2, color=ACCENT, label="$1-\\lambda_i$")
    axes[1].set_title("Panel B: Shrinkage and uncertainty weight", color=TEXT, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("$\\sigma_i$ decile", color=TEXT)
    axes[1].set_ylabel("Mean weight", color=TEXT)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[1].legend(frameon=False)
    fig.suptitle("How adjustment strength varies with accounting uncertainty", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    return save_fig(fig, fig_dir, "adjustment_by_sigma_decile", dpi)


def summarize_sigma_persistence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sigma = df[["Ticker", "FormationYear", "sigma_acc"]].dropna().sort_values(["Ticker", "FormationYear"])
    sigma["sigma_lag1"] = sigma.groupby("Ticker")["sigma_acc"].shift(1)
    lag1_pairs = sigma.dropna(subset=["sigma_lag1"]).copy()

    rows = []
    for lag in range(1, 6):
        temp = sigma[["Ticker", "FormationYear", "sigma_acc"]].copy()
        temp[f"sigma_lag{lag}"] = temp.groupby("Ticker")["sigma_acc"].shift(lag)
        valid = temp.dropna(subset=[f"sigma_lag{lag}"])
        rows.append(
            {
                "lag": lag,
                "n_pairs": len(valid),
                "pearson_corr": valid["sigma_acc"].corr(valid[f"sigma_lag{lag}"]),
                "spearman_corr": valid["sigma_acc"].corr(valid[f"sigma_lag{lag}"], method="spearman"),
            }
        )
    return lag1_pairs, pd.DataFrame(rows)


def plot_sigma_persistence(lag1_pairs: pd.DataFrame, persistence: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    for ax in axes:
        apply_axis_style(ax)

    q99 = lag1_pairs[["sigma_acc", "sigma_lag1"]].stack().quantile(0.99)
    x = lag1_pairs["sigma_lag1"].clip(upper=q99)
    y = lag1_pairs["sigma_acc"].clip(upper=q99)
    axes[0].hexbin(x, y, gridsize=42, cmap=LinearSegmentedColormap.from_list("density", ["#d7ecfa", "#58a7df", "#173a5e"]), mincnt=1)
    axes[0].plot([0, q99], [0, q99], color=TEXT, linestyle="--", linewidth=1.0)
    axes[0].set_title("Panel A: $\\sigma_i$ against lagged $\\sigma_i$", color=TEXT, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("$\\sigma_{i,t-1}$", color=TEXT)
    axes[0].set_ylabel("$\\sigma_{i,t}$", color=TEXT)

    axes[1].bar(persistence["lag"], persistence["pearson_corr"], color=METHOD_COLORS["Latent Quality"], alpha=0.85, label="Pearson")
    axes[1].plot(persistence["lag"], persistence["spearman_corr"], color=TEXT, marker="o", linewidth=2.0, label="Spearman")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(persistence["lag"])
    axes[1].set_title("Panel B: Persistence by lag", color=TEXT, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Lag in years", color=TEXT)
    axes[1].set_ylabel("Correlation", color=TEXT)
    axes[1].legend(frameon=False)
    fig.suptitle("Persistence in accounting uncertainty", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    return save_fig(fig, fig_dir, "sigma_persistence", dpi)


def plot_latent_interval_width(df: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    summary = (
        df.groupby("FormationYear", as_index=False)
        .agg(
            mean_width=("latent_ci_width", "mean"),
            median_width=("latent_ci_width", "median"),
            p25_width=("latent_ci_width", lambda s: quantile_agg(s, 0.25)),
            p75_width=("latent_ci_width", lambda s: quantile_agg(s, 0.75)),
        )
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor("white")
    apply_axis_style(ax)
    x = summary["FormationYear"].to_numpy()
    ax.fill_between(x, summary["p25_width"].to_numpy(), summary["p75_width"].to_numpy(), color=ACCENT, alpha=0.35, label="Interquartile range")
    ax.plot(x, summary["mean_width"], color=METHOD_COLORS["Observed Quality"], linewidth=2.25, label="Mean interval width")
    ax.plot(x, summary["median_width"], color=METHOD_COLORS["Latent Quality"], linewidth=2.25, label="Median interval width")
    ax.set_title("Approximate 95\\% interval width around Latent Quality", color=TEXT, fontweight="bold", fontsize=13)
    ax.set_xlabel("Formation year", color=TEXT)
    ax.set_ylabel("Width of $\\theta_{i,t}^{latent} \\pm 1.96\\,sd$", color=TEXT)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(bottom=0.22)
    return save_fig(fig, fig_dir, "latent_quality_interval_width", dpi)


def transition_matrices(run_dir: Path, table_dir: Path) -> pd.DataFrame:
    path = run_dir / "portfolio_formation" / "portfolio_assignments_wide.csv"
    assignments = pd.read_csv(path)
    rows = []
    observed_col = "Method1_ObservedQuality_Portfolio"
    for method, col in [
        ("Latent Quality", "Method2_LatentQuality_Portfolio"),
        ("Conservative Quality", "Method3_ConservativeQuality_Portfolio"),
    ]:
        for observed in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            mask = assignments[observed_col].eq(observed)
            denom = int(mask.sum())
            for adjusted in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                count = int((mask & assignments[col].eq(adjusted)).sum())
                rows.append(
                    {
                        "Method": method,
                        "ObservedPortfolio": observed,
                        "AdjustedPortfolio": adjusted,
                        "FirmYears": count,
                        "ShareObservedPortfolio": count / denom if denom else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "portfolio_transition_matrices.csv", index=False)
    return out


def plot_transition_heatmaps(transitions: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    cmap = LinearSegmentedColormap.from_list("blues_heat", ["#f6f9fc", "#8ec5e8", "#2c73ad", "#173a5e"])
    for ax, method in zip(axes, ["Latent Quality", "Conservative Quality"]):
        apply_axis_style(ax)
        matrix = (
            transitions.loc[transitions["Method"].eq(method)]
            .pivot(index="ObservedPortfolio", columns="AdjustedPortfolio", values="ShareObservedPortfolio")
            .loc[["Q1", "Q2", "Q3", "Q4", "Q5"], ["Q1", "Q2", "Q3", "Q4", "Q5"]]
        )
        image = ax.imshow(matrix.values, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(5), labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_yticks(range(5), labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_xlabel("Adjusted portfolio", color=TEXT)
        ax.set_ylabel("Observed portfolio", color=TEXT)
        ax.set_title(method, color=TEXT, fontweight="bold", fontsize=10)
        for i in range(5):
            for j in range(5):
                value = matrix.values[i, j]
                label_color = "white" if value > 0.55 else TEXT
                ax.text(j, i, f"{100 * value:.0f}%", ha="center", va="center", color=label_color, fontsize=8)
    fig.suptitle("Portfolio transitions after uncertainty adjustment", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.80, wspace=0.30, right=0.86)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.45])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Share of observed portfolio", color=TEXT)
    cbar.ax.tick_params(colors=TEXT)
    return save_fig(fig, fig_dir, "portfolio_transition_heatmaps", dpi)


def plot_probabilistic_tail_probabilities(df: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    for ax in axes:
        apply_axis_style(ax)

    axes[0].hist(df["p_q5"], bins=60, color=METHOD_COLORS["Latent Quality"], alpha=0.75, label="$P(Q5)$")
    axes[0].hist(df["p_q1"], bins=60, color=METHOD_COLORS["Conservative Quality"], alpha=0.55, label="$P(Q1)$")
    axes[0].set_title("Panel A: Tail-probability distribution", color=TEXT, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Probability", color=TEXT)
    axes[0].set_ylabel("Firm-years", color=TEXT)
    axes[0].legend(frameon=False)

    sample = df.sample(n=min(5000, len(df)), random_state=20260518)
    q_low, q_high = sample["theta_obs"].quantile([0.01, 0.99])
    axes[1].scatter(sample["theta_obs"].clip(q_low, q_high), sample["p_q5"], s=12, color=METHOD_COLORS["Latent Quality"], alpha=0.35, label="$P(Q5)$")
    axes[1].scatter(sample["theta_obs"].clip(q_low, q_high), sample["p_q1"], s=12, color=METHOD_COLORS["Conservative Quality"], alpha=0.35, label="$P(Q1)$")
    axes[1].set_title("Panel B: Tail probability by observed quality", color=TEXT, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Observed Quality", color=TEXT)
    axes[1].set_ylabel("Probability", color=TEXT)
    axes[1].legend(frameon=False)
    fig.suptitle("Probabilistic Quality tail probabilities", color=TEXT, fontweight="bold", fontsize=13)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    return save_fig(fig, fig_dir, "probabilistic_tail_probabilities", dpi)


def write_latex_snippet(figures: list[Path], output_dir: Path) -> None:
    captions = {
        "quality_levels_mean_median": (
            "Quality before and after accounting-uncertainty adjustment",
            "The figure compares the annual mean and median of Observed Quality, Latent Quality, and Conservative Quality.",
        ),
        "quality_spread_before_after": (
            "Quality spread before and after adjustment",
            "The figure reports the cross-sectional standard deviation and 5th--95th percentile spread of quality signals by formation year.",
        ),
        "quality_adjustment_distribution": (
            "Distribution of firm-year quality adjustments",
            "The figure shows the signed and absolute changes in firm-level quality induced by Latent Quality and Conservative Quality relative to Observed Quality.",
        ),
        "observed_vs_adjusted_quality": (
            "Observed quality versus adjusted quality",
            "The figure compares Observed Quality with Latent Quality and Conservative Quality. Points are shaded by firm-year accounting uncertainty relative to the annual median.",
        ),
        "quality_uncertainty_impact_by_exchange": (
            "Impact of accounting uncertainty by exchange",
            "The figure reports the mean absolute quality adjustment and mean accounting uncertainty by listing exchange.",
        ),
        "quality_uncertainty_impact_by_sector": (
            "Impact of accounting uncertainty by sector",
            "The figure reports the mean absolute quality adjustment and mean accounting uncertainty by sector.",
        ),
        "sigma_evolution": (
            "Evolution of accounting uncertainty",
            "The figure reports the mean, median, and 5th--95th percentile range of firm-level accounting uncertainty by formation year.",
        ),
        "sigma_distribution": (
            "Distribution of accounting uncertainty",
            "The figure reports the pooled and annual distributions of firm-level accounting uncertainty.",
        ),
        "adjustment_by_sigma_decile": (
            "Adjustment strength by accounting-uncertainty decile",
            "The figure shows how quality adjustments and shrinkage weights vary across deciles of accounting uncertainty.",
        ),
        "sigma_persistence": (
            "Persistence in accounting uncertainty",
            "The figure compares firm-level accounting uncertainty with its lagged value and reports persistence correlations by lag.",
        ),
        "latent_quality_interval_width": (
            "Approximate interval width around Latent Quality",
            "The figure reports the annual width of the approximate 95\\% interval around the Latent Quality signal.",
        ),
        "portfolio_transition_heatmaps": (
            "Portfolio transitions after uncertainty adjustment",
            "The figure shows how Latent Quality and Conservative Quality move firms across quality portfolios relative to Observed Quality.",
        ),
        "probabilistic_tail_probabilities": (
            "Probabilistic Quality tail probabilities",
            "The figure shows the distribution of Q1 and Q5 assignment probabilities used by the Probabilistic Quality method.",
        ),
    }
    lines = [
        "% Auto-generated by modelling/shared/generate_quality_uncertainty_appendix_figures.py",
        "% Copy the relevant figure blocks into the appendix.",
        "",
    ]
    seen: set[str] = set()
    for path in figures:
        if path.suffix != ".png":
            continue
        stem = path.stem
        if stem in seen:
            continue
        seen.add(stem)
        short, caption = captions.get(stem, (stem.replace("_", " ").title(), ""))
        lines.extend(
            [
                "\\begin{figure}[H]",
                "    \\centering",
                f"    \\includegraphics[width=0.95\\linewidth]{{Figures/quality_uncertainty_appendix/{stem}.png}}",
                f"    \\caption[{short}]{{{caption}}}",
                f"    \\label{{fig:{stem}}}",
                "\\end{figure}",
                "",
            ]
        )
    (output_dir / "appendix_quality_uncertainty_figures.tex").write_text("\n".join(lines))


def copy_figures(figures: list[Path], manual_dir: Path) -> None:
    manual_dir.mkdir(parents=True, exist_ok=True)
    for path in figures:
        shutil.copy2(path, manual_dir / path.name)


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else run_dir / "appendix_quality_uncertainty"
    fig_dir, table_dir = ensure_dirs(output_dir)

    df = load_main_data(run_dir)
    long_quality = make_long_quality(df)
    quality_summary = summarize_quality_levels(long_quality)
    exchange_summary = summarize_group_impact(df, "Exchange")
    sector_summary = summarize_group_impact(df, "Sector")
    sigma_year_summary = summarize_sigma_by_year(df)
    decile_summary = summarize_adjustment_deciles(df)
    lag1_pairs, persistence = summarize_sigma_persistence(df)
    transitions = transition_matrices(run_dir, table_dir)

    df.to_csv(table_dir / "quality_uncertainty_firm_year_inputs.csv", index=False)
    quality_summary.to_csv(table_dir / "quality_levels_and_spread_by_year.csv", index=False)
    exchange_summary.to_csv(table_dir / "quality_uncertainty_impact_by_exchange.csv", index=False)
    sector_summary.to_csv(table_dir / "quality_uncertainty_impact_by_sector.csv", index=False)
    sigma_year_summary.to_csv(table_dir / "sigma_summary_by_year.csv", index=False)
    decile_summary.to_csv(table_dir / "quality_adjustment_by_sigma_decile.csv", index=False)
    lag1_pairs.to_csv(table_dir / "sigma_lag1_pairs.csv", index=False)
    persistence.to_csv(table_dir / "sigma_persistence_correlations.csv", index=False)

    figures: list[Path] = []
    figures += plot_quality_levels(quality_summary, fig_dir, args.dpi)
    figures += plot_quality_spread(quality_summary, fig_dir, args.dpi)
    figures += plot_adjustment_distribution(df, fig_dir, args.dpi)
    figures += plot_observed_vs_adjusted(df, fig_dir, args.dpi)
    figures += plot_group_impact(exchange_summary, "Exchange", fig_dir, args.dpi)
    figures += plot_group_impact(sector_summary, "Sector", fig_dir, args.dpi)
    figures += plot_sigma_evolution(sigma_year_summary, fig_dir, args.dpi)
    figures += plot_sigma_distribution(df, fig_dir, args.dpi)
    figures += plot_adjustment_by_sigma_decile(decile_summary, fig_dir, args.dpi)
    figures += plot_sigma_persistence(lag1_pairs, persistence, fig_dir, args.dpi)
    figures += plot_latent_interval_width(df, fig_dir, args.dpi)
    figures += plot_transition_heatmaps(transitions, fig_dir, args.dpi)
    figures += plot_probabilistic_tail_probabilities(df, fig_dir, args.dpi)

    write_latex_snippet(figures, output_dir)
    if args.copy_to_manual:
        copy_figures(figures, resolve(args.manual_figure_dir))

    print(f"Created {len([p for p in figures if p.suffix == '.png'])} appendix figures.")
    print(f"Figure directory: {fig_dir}")
    print(f"Table directory:  {table_dir}")
    print(f"LaTeX snippet:    {output_dir / 'appendix_quality_uncertainty_figures.tex'}")
    if args.copy_to_manual:
        print(f"Copied figures to {resolve(args.manual_figure_dir)}")


if __name__ == "__main__":
    main()
