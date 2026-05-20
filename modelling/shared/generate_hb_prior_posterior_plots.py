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
PRIOR = "#9ecae1"
POST = "#173a5e"
POST_ALT = "#2c73ad"
ACCENT = "#58a7df"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prior and posterior plots for the HB accounting uncertainty model."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "current_res",
        help="Main result directory containing uncertainty_model outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/uncertainty_model/prior_posterior_plots.",
    )
    parser.add_argument(
        "--manual-figure-dir",
        type=Path,
        default=PROJECT_ROOT / "manual_review" / "Figures" / "hb_prior_posterior",
        help="Optional figure mirror for thesis copy/paste workflows.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--n-prior-draws", type=int, default=500_000)
    parser.add_argument(
        "--n-posterior-draws",
        type=int,
        default=800,
        help="Number of sigma posterior draw columns to load from sigma_posteriors_full.parquet.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    ax.tick_params(colors="#20384f", labelsize=9)


def add_top_legend(fig: plt.Figure, handles, labels, ncol: int, y: float = 0.93) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        fontsize=9,
        labelcolor=TEXT,
    )


def save_fig(fig: plt.Figure, fig_dir: Path, name: str, dpi: int) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = fig_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def simulate_priors(n_draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma_0 = np.abs(rng.normal(0.0, 0.05, n_draws))
    sigma_sector_raw = np.abs(rng.normal(0.0, 1.0, n_draws))
    sigma_firm_raw = np.abs(rng.normal(0.0, 1.0, n_draws))
    sigma_firm = sigma_0 * sigma_sector_raw * sigma_firm_raw
    nu = 2.0 + rng.exponential(scale=10.0, size=n_draws)
    sigma_firm_sd = sigma_firm * np.sqrt(nu / (nu - 2.0))

    return pd.DataFrame(
        {
            "mu_0": rng.normal(0.0, 0.1, n_draws),
            "hierarchical_scale": np.abs(rng.normal(0.0, 0.05, n_draws)),
            "beta": rng.normal(0.0, 0.3, n_draws),
            "nu": nu,
            "sigma_firm": sigma_firm,
            "sigma_firm_sd": sigma_firm_sd,
        }
    )


def load_posteriors(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "uncertainty_model" / "sigma_posteriors_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing posterior summary: {path}")
    df = pd.read_csv(path)
    required = {
        "Year",
        "Ticker",
        "sigma_mean",
        "sigma_median",
        "sigma_q05",
        "sigma_q95",
        "sigma_scale_mean",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Posterior summary missing required columns: {sorted(missing)}")
    return df


def get_parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        import fastparquet

        return list(fastparquet.ParquetFile(str(path)).columns)


def evenly_spaced_draw_columns(draw_cols: list[str], n_draws: int | None) -> list[str]:
    draw_cols = sorted(draw_cols, key=lambda c: int(c.split("_", 1)[1]))
    if n_draws is None or n_draws <= 0 or n_draws >= len(draw_cols):
        return draw_cols
    idx = np.linspace(0, len(draw_cols) - 1, n_draws, dtype=int)
    return [draw_cols[i] for i in idx]


def load_full_posterior_draws(
    run_dir: Path,
    n_draws: int | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    path = run_dir / "uncertainty_model" / "sigma_posteriors_full.parquet"
    if not path.exists():
        return None, []

    columns = get_parquet_columns(path)
    draw_cols = [c for c in columns if c.startswith("draw_")]
    selected_draws = evenly_spaced_draw_columns(draw_cols, n_draws)
    selected_cols = ["Year", "Ticker", "firm_idx"] + selected_draws

    try:
        draws = pd.read_parquet(path, columns=selected_cols)
    except Exception as exc:
        print(f"pyarrow could not read full posterior parquet ({exc}); trying fastparquet.")
        draws = pd.read_parquet(path, engine="fastparquet", columns=selected_cols)

    draws["Year"] = pd.to_numeric(draws["Year"], errors="coerce").astype("Int64")
    return draws, selected_draws


def flatten_draw_values(draws: pd.DataFrame, draw_cols: list[str]) -> np.ndarray:
    values = draws[draw_cols].to_numpy(dtype=float, copy=True).ravel()
    values = values[np.isfinite(values)]
    return values


def plot_model_priors(prior: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
    fig.suptitle("HB Model Priors", fontsize=15, fontweight="bold", color=TITLE, y=0.985)

    panels = [
        ("mu_0", r"$\mu_0 \sim \mathcal{N}(0, 0.1)$", r"$\mu_0$", (-0.35, 0.35)),
        (
            "hierarchical_scale",
            r"$\omega,\tau,\sigma_0 \sim \mathrm{HalfNormal}(0.05)$",
            "Scale parameter",
            (0.0, 0.18),
        ),
        ("beta", r"$\beta_k \sim \mathcal{N}(0, 0.3)$", r"$\beta_k$", (-1.0, 1.0)),
        ("nu", r"$\nu = 2 + \mathrm{Exponential}(10)$", r"$\nu$", (2.0, 55.0)),
    ]

    for ax, (col, title, xlabel, xlim) in zip(axes.ravel(), panels):
        apply_axis_style(ax)
        data = prior[col].to_numpy()
        data = data[np.isfinite(data)]
        data = data[(data >= xlim[0]) & (data <= xlim[1])]
        ax.hist(data, bins=70, density=True, color=PRIOR, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlabel(xlabel, color=TEXT)
        ax.set_ylabel("Prior density", color=TEXT)
        ax.set_xlim(*xlim)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, fig_dir, "hb_model_prior_distributions", dpi)


def plot_sigma_prior_posterior(
    prior: pd.DataFrame,
    post: pd.DataFrame,
    posterior_draws: pd.DataFrame | None,
    draw_cols: list[str],
    fig_dir: Path,
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    fig.suptitle("Prior and Posterior Distributions of Firm-Level Accounting Uncertainty", fontsize=14, fontweight="bold", color=TITLE, y=0.98)

    posterior_means = post["sigma_mean"].to_numpy(dtype=float)
    posterior_values = (
        flatten_draw_values(posterior_draws, draw_cols)
        if posterior_draws is not None and draw_cols
        else posterior_means
    )
    specs = [
        ("Posterior means", posterior_means, "Posterior mean $\\sigma_i$"),
        ("Posterior draws", posterior_values, "Posterior draws of $\\sigma_i$"),
    ]

    handles = []
    labels = []
    prior_values = prior["sigma_firm_sd"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    for ax, (title, post_values, xlabel) in zip(axes, specs):
        apply_axis_style(ax)
        post_values = post_values[np.isfinite(post_values)]
        upper = max(np.quantile(prior_values, 0.995), np.quantile(post_values, 0.995))
        prior_clip = prior_values[(prior_values >= 0) & (prior_values <= upper)]
        post_clip = post_values[(post_values >= 0) & (post_values <= upper)]
        bins = np.linspace(0, upper, 80)
        h1 = ax.hist(prior_clip, bins=bins, density=True, color=PRIOR, alpha=0.72, edgecolor="white", label="Implied prior")[2][0]
        h2 = ax.hist(post_clip, bins=bins, density=True, color=POST, alpha=0.72, edgecolor="white", label="Posterior")[2][0]
        ax.axvline(np.median(prior_values), color=PRIOR, linewidth=2.1, linestyle="--")
        ax.axvline(np.median(post_values), color=POST, linewidth=2.1, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT)
        ax.set_xlabel(xlabel, color=TEXT)
        ax.set_ylabel("Density", color=TEXT)
        ax.set_xlim(0, upper)
        if not handles:
            handles = [h1, h2]
            labels = ["Implied prior", "Posterior"]

    add_top_legend(fig, handles, labels, ncol=2, y=0.91)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return save_fig(fig, fig_dir, "hb_sigma_prior_posterior_distribution", dpi)


def summarize_posterior_by_year(
    post: pd.DataFrame,
    posterior_draws: pd.DataFrame | None,
    draw_cols: list[str],
) -> pd.DataFrame:
    if posterior_draws is None or not draw_cols:
        return (
            post.groupby("Year")
            .agg(
                mean_sigma=("sigma_mean", "mean"),
                median_sigma=("sigma_mean", "median"),
                p05_sigma=("sigma_mean", lambda s: s.quantile(0.05)),
                p95_sigma=("sigma_mean", lambda s: s.quantile(0.95)),
                n_firms=("Ticker", "nunique"),
            )
            .reset_index()
        )

    rows = []
    for year, sub in posterior_draws.groupby("Year"):
        values = flatten_draw_values(sub, draw_cols)
        rows.append(
            {
                "Year": int(year),
                "mean_sigma": float(np.mean(values)),
                "median_sigma": float(np.median(values)),
                "p05_sigma": float(np.quantile(values, 0.05)),
                "p95_sigma": float(np.quantile(values, 0.95)),
                "n_firms": int(sub["Ticker"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("Year")


def plot_sigma_posterior_by_year(
    post: pd.DataFrame,
    posterior_draws: pd.DataFrame | None,
    draw_cols: list[str],
    fig_dir: Path,
    dpi: int,
) -> list[Path]:
    summary = summarize_posterior_by_year(post, posterior_draws, draw_cols)

    fig, ax = plt.subplots(figsize=(11.8, 5.1))
    apply_axis_style(ax)
    fig.suptitle("Posterior Accounting Uncertainty by Formation Year", fontsize=14, fontweight="bold", color=TITLE, y=0.98)
    x = summary["Year"].to_numpy()
    ax.fill_between(x, summary["p05_sigma"], summary["p95_sigma"], color=ACCENT, alpha=0.32, label="5th-95th percentile of posterior draws")
    line_mean, = ax.plot(x, summary["mean_sigma"], color=POST, linewidth=2.4, label="Mean posterior $\\sigma_i$")
    line_med, = ax.plot(x, summary["median_sigma"], color=POST_ALT, linewidth=2.4, label="Median posterior $\\sigma_i$")
    ax.set_xlabel("Formation year", color=TEXT)
    ax.set_ylabel(r"$\sigma_i$", color=TEXT)
    ax.set_xlim(x.min(), x.max())
    add_top_legend(fig, [line_mean, line_med], ["Mean posterior $\\sigma_i$", "Median posterior $\\sigma_i$"], ncol=2, y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return save_fig(fig, fig_dir, "hb_sigma_posterior_by_year", dpi)


def plot_posterior_interval_width(post: pd.DataFrame, fig_dir: Path, dpi: int) -> list[Path]:
    df = post.copy()
    df["ci_width"] = df["sigma_q95"] - df["sigma_q05"]
    rng = np.random.default_rng(42)
    if len(df) > 3500:
        df = df.sample(3500, random_state=42)
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    apply_axis_style(ax)
    fig.suptitle("Posterior Interval Width and Estimated Accounting Uncertainty", fontsize=14, fontweight="bold", color=TITLE, y=0.98)
    jitter = rng.normal(0, 0.001, len(df))
    sc = ax.scatter(
        df["sigma_mean"],
        df["ci_width"] + jitter,
        s=12,
        alpha=0.35,
        color=POST_ALT,
        edgecolor="none",
    )
    ax.set_xlabel(r"Posterior mean $\sigma_i$", color=TEXT)
    ax.set_ylabel("90% posterior interval width", color=TEXT)
    ax.set_xlim(0, post["sigma_mean"].quantile(0.995) * 1.05)
    ax.set_ylim(0, post.assign(ci_width=post["sigma_q95"] - post["sigma_q05"])["ci_width"].quantile(0.995) * 1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, fig_dir, "hb_sigma_posterior_interval_width", dpi)


def plot_posterior_examples(
    post: pd.DataFrame,
    posterior_draws: pd.DataFrame | None,
    draw_cols: list[str],
    fig_dir: Path,
    dpi: int,
) -> list[Path]:
    latest_year = int(post["Year"].max())
    latest = post.loc[post["Year"].eq(latest_year)].copy()
    latest = latest.sort_values("sigma_mean").reset_index(drop=True)
    if len(latest) < 5:
        return []

    positions = [0.05, 0.25, 0.50, 0.75, 0.95]
    idx = [min(len(latest) - 1, max(0, int(round(p * (len(latest) - 1))))) for p in positions]
    examples = latest.iloc[idx].copy()

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    apply_axis_style(ax)
    fig.suptitle(f"Selected Firm-Level Sigma Posteriors, {latest_year}", fontsize=14, fontweight="bold", color=TITLE, y=0.98)

    if posterior_draws is None or not draw_cols:
        labels = [f"{row.Ticker}" for row in examples.itertuples()]
        y = np.arange(len(examples))
        xerr = np.vstack(
            [
                examples["sigma_median"].to_numpy() - examples["sigma_q05"].to_numpy(),
                examples["sigma_q95"].to_numpy() - examples["sigma_median"].to_numpy(),
            ]
        )
        ax.errorbar(
            examples["sigma_median"],
            y,
            xerr=xerr,
            fmt="o",
            markersize=6,
            color=POST,
            ecolor=ACCENT,
            elinewidth=4,
            capsize=5,
            label="Posterior median and 90% interval",
        )
        ax.scatter(examples["sigma_mean"], y, color="#c44536", s=28, zorder=4, label="Posterior mean")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=TEXT)
        ax.set_ylabel("Ticker", color=TEXT)
        ax.invert_yaxis()
    else:
        merged = examples[["Year", "Ticker"]].merge(
            posterior_draws,
            on=["Year", "Ticker"],
            how="left",
            validate="1:1",
        )
        max_x = np.nanquantile(merged[draw_cols].to_numpy(dtype=float), 0.995)
        bins = np.linspace(0, max_x, 90)
        colors = ["#173a5e", "#2c73ad", "#58a7df", "#8ec5e8", "#c44536"]
        handles = []
        labels = []
        for i, row in merged.iterrows():
            values = row[draw_cols].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            values = values[(values >= 0) & (values <= max_x)]
            hist, edges = np.histogram(values, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            line, = ax.plot(
                centers,
                hist,
                color=colors[i % len(colors)],
                linewidth=2.2,
                label=f"{row['Ticker']} ($\\bar{{\\sigma}}$={np.mean(values):.3f})",
            )
            handles.append(line)
            labels.append(line.get_label())
        add_top_legend(fig, handles, labels, ncol=3, y=0.90)
        ax.set_ylabel("Posterior density", color=TEXT)
        ax.set_xlim(0, max_x)

    ax.set_xlabel(r"$\sigma_i$", color=TEXT)
    if posterior_draws is None or not draw_cols:
        add_top_legend(fig, ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], ncol=2, y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return save_fig(fig, fig_dir, "hb_sigma_posterior_examples", dpi)


def write_summary_tables(
    prior: pd.DataFrame,
    post: pd.DataFrame,
    posterior_draws: pd.DataFrame | None,
    draw_cols: list[str],
    table_dir: Path,
) -> None:
    summary_rows = []
    pairs = [
        ("sigma_firm_scale", [("prior", prior["sigma_firm"]), ("posterior_mean", post["sigma_scale_mean"])]),
        ("sigma_firm_residual_sd", [("prior", prior["sigma_firm_sd"]), ("posterior_mean", post["sigma_mean"])]),
    ]
    if posterior_draws is not None and draw_cols:
        pairs[1][1].append(("posterior_draws", pd.Series(flatten_draw_values(posterior_draws, draw_cols))))
    probs = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    for label, sources in pairs:
        for source, values in sources:
            values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
            row = {
                "quantity": label,
                "source": source,
                "mean": values.mean(),
                "std": values.std(),
            }
            for p in probs:
                row[f"q{int(p * 100):02d}"] = values.quantile(p)
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(table_dir / "hb_prior_posterior_sigma_summary.csv", index=False)


def write_latex_snippet(figures: list[Path], output_dir: Path) -> None:
    names = []
    for path in figures:
        if path.suffix == ".png":
            names.append(path.stem)
    names = list(dict.fromkeys(names))

    captions = {
        "hb_model_prior_distributions": "Prior distributions used in the hierarchical Bayesian accrual model.",
        "hb_sigma_prior_posterior_distribution": "Implied prior and posterior-draw distributions of firm-level accounting uncertainty.",
        "hb_sigma_posterior_by_year": "Posterior accounting uncertainty by formation year.",
        "hb_sigma_posterior_interval_width": "Posterior interval width as a function of estimated accounting uncertainty.",
        "hb_sigma_posterior_examples": "Selected firm-level posterior distributions for accounting uncertainty.",
    }
    labels = {
        name: "fig:" + name.replace("_", "-")
        for name in names
    }

    lines = [
        "% Auto-generated by modelling/shared/generate_hb_prior_posterior_plots.py",
        "% Copy figure files from Figures/hb_prior_posterior/ into your thesis figure folder.",
        "",
    ]
    for name in names:
        lines.extend(
            [
                "\\begin{figure}[H]",
                "    \\centering",
                f"    \\includegraphics[width=0.95\\linewidth]{{Figures/hb_prior_posterior/{name}.png}}",
                f"    \\caption{{{captions.get(name, name.replace('_', ' ').title())}}}",
                f"    \\label{{{labels[name]}}}",
                "\\end{figure}",
                "",
            ]
        )
    (output_dir / "hb_prior_posterior_figures.tex").write_text("\n".join(lines), encoding="utf-8")


def copy_figures(figures: list[Path], manual_dir: Path) -> None:
    manual_dir.mkdir(parents=True, exist_ok=True)
    for path in figures:
        shutil.copy2(path, manual_dir / path.name)


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else run_dir / "uncertainty_model" / "prior_posterior_plots"
    fig_dir, table_dir = ensure_dirs(output_dir)

    posterior = load_posteriors(run_dir)
    posterior_draws, draw_cols = load_full_posterior_draws(
        run_dir=run_dir,
        n_draws=args.n_posterior_draws,
    )
    if posterior_draws is not None:
        print(
            f"Loaded {len(draw_cols)} posterior draw columns from "
            "sigma_posteriors_full.parquet."
        )
    prior = simulate_priors(n_draws=args.n_prior_draws, seed=args.seed)

    figures: list[Path] = []
    figures.extend(plot_model_priors(prior, fig_dir, args.dpi))
    figures.extend(plot_sigma_prior_posterior(prior, posterior, posterior_draws, draw_cols, fig_dir, args.dpi))
    figures.extend(plot_sigma_posterior_by_year(posterior, posterior_draws, draw_cols, fig_dir, args.dpi))
    figures.extend(plot_posterior_interval_width(posterior, fig_dir, args.dpi))
    figures.extend(plot_posterior_examples(posterior, posterior_draws, draw_cols, fig_dir, args.dpi))

    write_summary_tables(prior, posterior, posterior_draws, draw_cols, table_dir)
    write_latex_snippet(figures, output_dir)

    if args.copy_to_manual:
        copy_figures(figures, resolve(args.manual_figure_dir))

    print(f"Created {len([p for p in figures if p.suffix == '.png'])} prior/posterior figures.")
    print(f"Figure directory: {fig_dir}")
    print(f"Table directory: {table_dir}")
    print(f"LaTeX snippet: {output_dir / 'hb_prior_posterior_figures.tex'}")
    if args.copy_to_manual:
        print(f"Copied figures to {resolve(args.manual_figure_dir)}")


if __name__ == "__main__":
    main()
