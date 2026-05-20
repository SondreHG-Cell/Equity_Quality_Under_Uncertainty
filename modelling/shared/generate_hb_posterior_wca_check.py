from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

TEXT = "#173a5e"
TITLE = "#000000"
PANEL_BG = "#f6f9fc"
GRID = "#9ebfda"
OBSERVED = "#173a5e"
POST_MEAN = "#58a7df"
POST_DRAW = "#2c73ad"
INTERVAL = "#9ecae1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate posterior WCA versus actual WCA checks for the HB accrual model."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "current_res",
        help="Main result directory containing uncertainty_model outputs.",
    )
    parser.add_argument(
        "--portfolio-year",
        type=int,
        default=2015,
        help="Portfolio year whose HB estimation window is plotted.",
    )
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--sample-size", type=int, default=2500)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/uncertainty_model/posterior_wca_check.",
    )
    parser.add_argument(
        "--manual-figure-dir",
        type=Path,
        default=PROJECT_ROOT / "manual_review" / "Figures" / "hb_posterior_wca",
        help="Optional figure mirror for thesis copy/paste workflows.",
    )
    parser.add_argument("--copy-to-manual", action="store_true", default=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def apply_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="--", alpha=0.65)
    ax.grid(axis="x", color="#c8d9e8", linewidth=0.45, linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a7bfd5")
    ax.spines["bottom"].set_color("#a7bfd5")
    ax.tick_params(colors="#20384f", labelsize=8)


def save_fig(fig: plt.Figure, fig_dir: Path, name: str, dpi: int) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = fig_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(path)
    plt.close(fig)
    return paths


def load_posterior_wca(run_dir: Path, portfolio_year: int, seed: int) -> pd.DataFrame:
    uncertainty_dir = run_dir / "uncertainty_model"
    expected = pd.read_csv(uncertainty_dir / "expected_accruals_summary.csv")
    sigma = pd.read_csv(uncertainty_dir / "sigma_posteriors_summary.csv")

    df = expected.loc[expected["Year"].eq(portfolio_year)].copy()
    if df.empty:
        raise ValueError(f"No expected accrual posterior summaries found for Year={portfolio_year}")

    sigma_cols = ["Year", "Ticker", "sigma_mean", "sigma_median", "sigma_q05", "sigma_q95"]
    df = df.merge(sigma[sigma_cols], on=["Year", "Ticker"], how="left", validate="many_to_one")
    missing_sigma = int(df["sigma_mean"].isna().sum())
    if missing_sigma:
        raise ValueError(f"{missing_sigma} rows could not be matched to sigma posterior summaries")

    rng = np.random.default_rng(seed)
    mu_draw = rng.normal(
        loc=df["expected_wca_mean"].to_numpy(float),
        scale=df["expected_wca_std"].clip(lower=1e-8).to_numpy(float),
    )

    # The full posterior predictive draws are not stored. This reconstructs
    # the predictive WCA distribution from the saved posterior mean uncertainty
    # and firm-level residual standard deviation summaries.
    pred_sd = np.sqrt(
        np.square(df["expected_wca_std"].to_numpy(float))
        + np.square(df["sigma_mean"].to_numpy(float))
    )
    pred_draw = rng.normal(loc=mu_draw, scale=np.clip(pred_sd, 1e-8, None))

    out = df.copy()
    out["posterior_expected_wca_draw"] = mu_draw
    out["posterior_predictive_wca_draw"] = pred_draw
    out["posterior_predictive_sd_approx"] = pred_sd
    out["posterior_predictive_q05_approx"] = out["expected_wca_mean"] - 1.645 * pred_sd
    out["posterior_predictive_q95_approx"] = out["expected_wca_mean"] + 1.645 * pred_sd
    out["actual_inside_predictive_90_approx"] = (
        out["observed_wca_scaled"].between(
            out["posterior_predictive_q05_approx"],
            out["posterior_predictive_q95_approx"],
        )
    )
    return out


def axis_limits(df: pd.DataFrame, cols: list[str], q: float = 0.995, minimum: float = 0.35) -> tuple[float, float]:
    values = df[cols].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    lim = max(float(np.quantile(np.abs(values), q)), minimum)
    return -lim, lim


def sample_for_plot(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    return df.sample(sample_size, random_state=seed).copy()


def make_main_figure(df: pd.DataFrame, portfolio_year: int, fig_dir: Path, dpi: int, sample_size: int, seed: int) -> list[Path]:
    plot_df = sample_for_plot(df, sample_size, seed)
    lim = axis_limits(
        plot_df,
        ["observed_wca_scaled", "expected_wca_mean", "posterior_predictive_wca_draw"],
        q=0.995,
        minimum=0.35,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    fig.patch.set_facecolor("white")
    for ax in axes:
        apply_axis_style(ax)

    axes[0].scatter(
        plot_df["observed_wca_scaled"],
        plot_df["expected_wca_mean"],
        s=10,
        alpha=0.42,
        color=POST_MEAN,
        edgecolors="none",
    )
    axes[0].plot(lim, lim, color=TEXT, linestyle="--", linewidth=1.0, alpha=0.75)
    axes[0].set_xlim(*lim)
    axes[0].set_ylim(*lim)
    axes[0].set_title("Panel A: Posterior mean", color=TEXT, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Actual WCA scaled", color=TEXT)
    axes[0].set_ylabel("Posterior WCA", color=TEXT)

    axes[1].scatter(
        plot_df["observed_wca_scaled"],
        plot_df["posterior_predictive_wca_draw"],
        s=10,
        alpha=0.34,
        color=POST_DRAW,
        edgecolors="none",
    )
    axes[1].plot(lim, lim, color=TEXT, linestyle="--", linewidth=1.0, alpha=0.75)
    axes[1].set_xlim(*lim)
    axes[1].set_ylim(*lim)
    axes[1].set_title("Panel B: Posterior predictive draw", color=TEXT, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Actual WCA scaled", color=TEXT)
    axes[1].set_ylabel("Posterior WCA", color=TEXT)

    hist_lim = axis_limits(
        plot_df,
        ["observed_wca_scaled", "posterior_predictive_wca_draw"],
        q=0.99,
        minimum=0.35,
    )
    bins = np.linspace(hist_lim[0], hist_lim[1], 70)
    axes[2].hist(
        plot_df["observed_wca_scaled"].clip(*hist_lim),
        bins=bins,
        density=True,
        alpha=0.62,
        color=OBSERVED,
        label="Actual WCA",
    )
    axes[2].hist(
        plot_df["posterior_predictive_wca_draw"].clip(*hist_lim),
        bins=bins,
        density=True,
        alpha=0.52,
        color=INTERVAL,
        label="Posterior WCA",
    )
    axes[2].set_title("Panel C: Distribution", color=TEXT, fontweight="bold", fontsize=10)
    axes[2].set_xlabel("WCA scaled", color=TEXT)
    axes[2].set_ylabel("Density", color=TEXT)
    axes[2].legend(frameon=False, fontsize=8, labelcolor=TEXT)

    fig.suptitle(
        f"Posterior WCA compared with actual WCA, {portfolio_year} window",
        color=TITLE,
        fontweight="bold",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.82, wspace=0.27)
    return save_fig(fig, fig_dir, f"hb_posterior_wca_vs_actual_{portfolio_year}", dpi)


def make_interval_figure(df: pd.DataFrame, portfolio_year: int, fig_dir: Path, dpi: int, sample_size: int, seed: int) -> list[Path]:
    plot_df = sample_for_plot(df, sample_size, seed).copy()
    plot_df["abs_error"] = (plot_df["observed_wca_scaled"] - plot_df["expected_wca_mean"]).abs()
    plot_df = plot_df.sort_values("observed_wca_scaled").reset_index(drop=True)
    x = np.arange(len(plot_df))
    lim = axis_limits(
        plot_df,
        ["observed_wca_scaled", "expected_wca_mean", "posterior_predictive_q05_approx", "posterior_predictive_q95_approx"],
        q=0.995,
        minimum=0.35,
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    fig.patch.set_facecolor("white")
    apply_axis_style(ax)
    ax.fill_between(
        x,
        plot_df["posterior_predictive_q05_approx"].to_numpy(float),
        plot_df["posterior_predictive_q95_approx"].to_numpy(float),
        color=INTERVAL,
        alpha=0.45,
        label="Approx. 90% posterior WCA interval",
    )
    ax.scatter(
        x,
        plot_df["observed_wca_scaled"],
        s=8,
        alpha=0.45,
        color=OBSERVED,
        edgecolors="none",
        label="Actual WCA",
    )
    ax.plot(x, plot_df["expected_wca_mean"], color=POST_DRAW, linewidth=1.4, label="Posterior mean")
    ax.set_ylim(*lim)
    ax.set_xlabel("Firm-year observations sorted by actual WCA", color=TEXT)
    ax.set_ylabel("WCA scaled", color=TEXT)
    fig.suptitle(
        f"Posterior WCA intervals and actual WCA, {portfolio_year} window",
        color=TITLE,
        fontweight="bold",
        fontsize=13,
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=8, labelcolor=TEXT)
    fig.subplots_adjust(top=0.78, left=0.10, right=0.98, bottom=0.17)
    return save_fig(fig, fig_dir, f"hb_posterior_wca_interval_{portfolio_year}", dpi)


def make_column_figure(df: pd.DataFrame, portfolio_year: int, fig_dir: Path, dpi: int, sample_size: int, seed: int) -> list[Path]:
    plot_df = sample_for_plot(df, sample_size, seed)
    lim = axis_limits(
        plot_df,
        ["observed_wca_scaled", "posterior_predictive_wca_draw"],
        q=0.995,
        minimum=0.35,
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    fig.patch.set_facecolor("white")
    apply_axis_style(ax)
    ax.scatter(
        plot_df["observed_wca_scaled"],
        plot_df["posterior_predictive_wca_draw"],
        s=8,
        alpha=0.34,
        color=POST_DRAW,
        edgecolors="none",
    )
    ax.plot(lim, lim, color=TEXT, linestyle="--", linewidth=0.8, alpha=0.75)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Actual WCA scaled", color=TEXT, fontsize=8)
    ax.set_ylabel("Posterior WCA", color=TEXT, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"Posterior vs actual WCA, {portfolio_year}", color=TITLE, fontweight="bold", fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.19)
    return save_fig(fig, fig_dir, f"hb_posterior_wca_vs_actual_{portfolio_year}_column", dpi)


def write_summary(df: pd.DataFrame, table_dir: Path, portfolio_year: int) -> Path:
    actual = df["observed_wca_scaled"].to_numpy(float)
    mu = df["expected_wca_mean"].to_numpy(float)
    pred = df["posterior_predictive_wca_draw"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "portfolio_year": portfolio_year,
                "n_window_rows": len(df),
                "n_unique_firms": df["Ticker"].nunique(),
                "actual_wca_mean": actual.mean(),
                "actual_wca_sd": actual.std(ddof=1),
                "posterior_mean_wca_mean": mu.mean(),
                "posterior_mean_wca_sd": mu.std(ddof=1),
                "posterior_predictive_draw_mean": pred.mean(),
                "posterior_predictive_draw_sd": pred.std(ddof=1),
                "corr_actual_posterior_mean": np.corrcoef(actual, mu)[0, 1],
                "corr_actual_posterior_predictive_draw": np.corrcoef(actual, pred)[0, 1],
                "rmse_actual_posterior_mean": np.sqrt(np.mean((actual - mu) ** 2)),
                "share_actual_inside_predictive_90_approx": df["actual_inside_predictive_90_approx"].mean(),
            }
        ]
    )
    path = table_dir / f"hb_posterior_wca_check_summary_{portfolio_year}.csv"
    summary.to_csv(path, index=False)
    return path


def write_latex_snippet(output_dir: Path, portfolio_year: int) -> None:
    main_stem = f"hb_posterior_wca_vs_actual_{portfolio_year}"
    interval_stem = f"hb_posterior_wca_interval_{portfolio_year}"
    lines = [
        "% Auto-generated by modelling/shared/generate_hb_posterior_wca_check.py",
        "",
        "\\begin{figure}[H]",
        "    \\centering",
        f"    \\includegraphics[width=0.95\\linewidth]{{Figures/hb_posterior_wca/{main_stem}.png}}",
        f"    \\caption[Posterior WCA compared with actual WCA]{{Posterior WCA compared with actual WCA for the {portfolio_year} estimation window. Panel A plots actual scaled WCA against the posterior mean expected WCA, Panel B plots actual scaled WCA against one posterior predictive draw, and Panel C compares the distributions. The dashed line in Panels A and B is the 45-degree line.}}",
        f"    \\label{{fig:{main_stem}}}",
        "\\end{figure}",
        "",
        "\\begin{figure}[H]",
        "    \\centering",
        f"    \\includegraphics[width=0.95\\linewidth]{{Figures/hb_posterior_wca/{interval_stem}.png}}",
        f"    \\caption[Posterior WCA intervals and actual WCA]{{Approximate posterior WCA intervals and actual WCA for the {portfolio_year} estimation window. Observations are sorted by actual scaled WCA.}}",
        f"    \\label{{fig:{interval_stem}}}",
        "\\end{figure}",
        "",
        "% Compact version for two-column layouts:",
        "% \\begin{figure}[t]",
        "%     \\centering",
        f"%     \\includegraphics[width=\\columnwidth]{{Figures/hb_posterior_wca/{main_stem}_column.png}}",
        f"%     \\caption[Posterior WCA compared with actual WCA]{{Posterior WCA compared with actual WCA for the {portfolio_year} estimation window.}}",
        f"%     \\label{{fig:{main_stem}_column}}",
        "% \\end{figure}",
        "",
    ]
    (output_dir / "hb_posterior_wca_check_figures.tex").write_text("\n".join(lines))


def copy_figures(figures: list[Path], manual_dir: Path) -> None:
    manual_dir.mkdir(parents=True, exist_ok=True)
    for path in figures:
        shutil.copy2(path, manual_dir / path.name)


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else run_dir / "uncertainty_model" / "posterior_wca_check"
    fig_dir, table_dir = ensure_dirs(output_dir)

    df = load_posterior_wca(run_dir, args.portfolio_year, args.seed)
    summary_path = write_summary(df, table_dir, args.portfolio_year)

    figures: list[Path] = []
    figures += make_main_figure(df, args.portfolio_year, fig_dir, args.dpi, args.sample_size, args.seed)
    figures += make_interval_figure(df, args.portfolio_year, fig_dir, args.dpi, args.sample_size, args.seed)
    figures += make_column_figure(df, args.portfolio_year, fig_dir, args.dpi, args.sample_size, args.seed)
    write_latex_snippet(output_dir, args.portfolio_year)
    if args.copy_to_manual:
        copy_figures(figures, resolve(args.manual_figure_dir))

    print(f"Created posterior WCA figures in {fig_dir}")
    print(f"Created summary table: {summary_path}")
    print(f"LaTeX snippet: {output_dir / 'hb_posterior_wca_check_figures.tex'}")
    if args.copy_to_manual:
        print(f"Copied figures to {resolve(args.manual_figure_dir)}")


if __name__ == "__main__":
    main()
