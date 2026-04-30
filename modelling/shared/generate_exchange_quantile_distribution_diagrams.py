from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_exchange_split_risk_adjusted_table_data as exchange_split
import generate_risk_adjusted_table_data as vw
from helper_functions import find_project_root, resolve_path


PORTFOLIO_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5"]
EXCHANGE_ORDER_ALL = ["Copenhagen", "Helsinki", "Oslo", "Stockholm", "Iceland"]
EXCHANGE_COLORS = {
    "Copenhagen": "#4C78A8",
    "Helsinki": "#F58518",
    "Oslo": "#54A24B",
    "Stockholm": "#B279A2",
    "Iceland": "#E45756",
}
PORTFOLIO_COLORS = {
    "Q1": "#4C78A8",
    "Q2": "#72B7B2",
    "Q3": "#EECF6D",
    "Q4": "#F2A65A",
    "Q5": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create diagrams and CSVs showing exchange composition across yearly "
            "Q1-Q5 portfolio assignments."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional results run directory, e.g. results/current_res.",
    )
    parser.add_argument(
        "--assignments-csv",
        type=Path,
        default=None,
        help="Optional portfolio_assignments_long.csv source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "<portfolio_evaluation_dir>/exchange_quantile_distribution_diagrams."
        ),
    )
    parser.add_argument(
        "--exclude-iceland",
        action="store_true",
        help="Exclude Iceland from the distribution diagrams.",
    )
    return parser.parse_args()


def resolve_cli_path(path: Path | None, project_root: Path) -> Path | None:
    if path is None:
        return None
    return resolve_path(path, project_root)


def method_slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def infer_variant_name(portfolio_eval_dir: Path) -> str | None:
    if portfolio_eval_dir.parent.name == "portfolio_evaluation":
        return portfolio_eval_dir.name
    return None


def choose_assignment_source(
    run_dir: Path,
    portfolio_eval_dir: Path,
    requested_source: Path | None,
) -> Path:
    searched: list[Path] = []

    if requested_source is not None:
        if requested_source.exists():
            return requested_source
        raise FileNotFoundError(f"Requested assignments CSV does not exist: {requested_source}")

    variant = infer_variant_name(portfolio_eval_dir)
    if variant is not None:
        candidate = run_dir / "portfolio_formation" / variant / "portfolio_assignments_long.csv"
        searched.append(candidate)
        if candidate.exists():
            return candidate

    preferred = run_dir / "portfolio_formation" / "HB" / "portfolio_assignments_long.csv"
    searched.append(preferred)
    if preferred.exists():
        return preferred

    globbed = sorted((run_dir / "portfolio_formation").glob("*/portfolio_assignments_long.csv"))
    searched.extend(globbed)
    if globbed:
        return globbed[0]

    raise FileNotFoundError(
        "Could not locate portfolio_assignments_long.csv.\nSearched:\n"
        + "\n".join(str(p) for p in searched)
    )


def exchange_order(include_iceland: bool) -> list[str]:
    exchanges = EXCHANGE_ORDER_ALL.copy()
    if not include_iceland:
        exchanges = [exchange for exchange in exchanges if exchange != "Iceland"]
    return exchanges


def load_assignments(path: Path, include_iceland: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["Ticker", "FormationYear", "Method", "PortfolioNum", "Portfolio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required assignment columns: {missing}")

    out = df[required].copy()
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out["FormationYear"] = pd.to_numeric(out["FormationYear"], errors="coerce")
    out["PortfolioNum"] = pd.to_numeric(out["PortfolioNum"], errors="coerce")
    out["Method"] = out["Method"].astype(str).str.strip()
    out["Portfolio"] = out["Portfolio"].astype(str).str.strip()
    out = out.dropna(subset=["Ticker", "FormationYear", "Method", "PortfolioNum", "Portfolio"])
    out["FormationYear"] = out["FormationYear"].astype(int)
    out["PortfolioNum"] = out["PortfolioNum"].astype(int)

    out = exchange_split.add_exchange_labels(out, path)
    if not include_iceland:
        out = out.loc[out["Exchange"] != "Iceland"].copy()

    out = out.loc[out["Method"].isin(vw.METHODS) & out["Portfolio"].isin(PORTFOLIO_ORDER)].copy()
    if out.empty:
        raise ValueError(f"{path} has no usable rows after method/portfolio/exchange filtering.")

    out["Portfolio"] = pd.Categorical(out["Portfolio"], categories=PORTFOLIO_ORDER, ordered=True)
    out["Exchange"] = pd.Categorical(
        out["Exchange"],
        categories=exchange_order(include_iceland),
        ordered=True,
    )
    return out.sort_values(["FormationYear", "Method", "Portfolio", "Exchange", "Ticker"]).reset_index(drop=True)


def complete_counts(
    counts: pd.DataFrame,
    index_cols: list[str],
    exchange_col: str,
    exchanges: list[str],
) -> pd.DataFrame:
    levels = [sorted(counts[col].dropna().unique()) for col in index_cols]
    index = pd.MultiIndex.from_product(
        levels + [exchanges],
        names=index_cols + [exchange_col],
    )
    return (
        counts.set_index(index_cols + [exchange_col])
        .reindex(index, fill_value=0)
        .reset_index()
    )


def build_quantile_distribution(assignments: pd.DataFrame, exchanges: list[str]) -> pd.DataFrame:
    counts = (
        assignments.groupby(["FormationYear", "Method", "Portfolio", "Exchange"], observed=False)
        .agg(n_firms=("Ticker", "nunique"))
        .reset_index()
    )
    counts = complete_counts(
        counts=counts,
        index_cols=["FormationYear", "Method", "Portfolio"],
        exchange_col="Exchange",
        exchanges=exchanges,
    )
    totals = counts.groupby(["FormationYear", "Method", "Portfolio"])["n_firms"].transform("sum")
    counts["share_of_quantile"] = counts["n_firms"] / totals.where(totals != 0)
    return counts


def build_year_distribution(assignments: pd.DataFrame, exchanges: list[str]) -> pd.DataFrame:
    firm_year = assignments[["Ticker", "FormationYear", "Exchange"]].drop_duplicates()
    counts = (
        firm_year.groupby(["FormationYear", "Exchange"], observed=False)
        .agg(n_firms=("Ticker", "nunique"))
        .reset_index()
    )
    counts = complete_counts(
        counts=counts,
        index_cols=["FormationYear"],
        exchange_col="Exchange",
        exchanges=exchanges,
    )
    totals = counts.groupby("FormationYear")["n_firms"].transform("sum")
    counts["share_of_year"] = counts["n_firms"] / totals.where(totals != 0)
    return counts


def build_exchange_placement_distribution(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        assignments.groupby(["FormationYear", "Method", "Exchange", "Portfolio"], observed=False)
        .agg(n_firms=("Ticker", "nunique"))
        .reset_index()
    )
    totals = counts.groupby(["FormationYear", "Method", "Exchange"], observed=False)["n_firms"].transform("sum")
    counts["share_of_exchange"] = counts["n_firms"] / totals.where(totals != 0)
    return counts


def save_distribution_csvs(
    output_dir: Path,
    quantile_distribution: pd.DataFrame,
    year_distribution: pd.DataFrame,
    placement_distribution: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "exchange_quantile_distribution": output_dir / "exchange_quantile_distribution.csv",
        "exchange_year_total_distribution": output_dir / "exchange_year_total_distribution.csv",
        "exchange_placement_by_quantile_distribution": output_dir
        / "exchange_placement_by_quantile_distribution.csv",
    }
    quantile_distribution.to_csv(outputs["exchange_quantile_distribution"], index=False)
    year_distribution.to_csv(outputs["exchange_year_total_distribution"], index=False)
    placement_distribution.to_csv(outputs["exchange_placement_by_quantile_distribution"], index=False)
    return outputs


def stacked_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    stack_col: str,
    value_col: str,
    stack_order: list[str],
    colors: dict[str, str],
) -> None:
    years = sorted(data[x_col].dropna().unique())
    bottom = pd.Series(0.0, index=years)
    for label in stack_order:
        values = (
            data.loc[data[stack_col] == label]
            .set_index(x_col)[value_col]
            .reindex(years, fill_value=0.0)
        )
        ax.bar(
            years,
            values.values,
            bottom=bottom.values,
            width=0.82,
            label=label,
            color=colors.get(label),
            edgecolor="white",
            linewidth=0.25,
        )
        bottom = bottom.add(values, fill_value=0.0)


def save_quantile_composition_plots(
    quantile_distribution: pd.DataFrame,
    output_dir: Path,
    exchanges: list[str],
) -> dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for method in vw.METHODS:
        method_data = quantile_distribution.loc[quantile_distribution["Method"] == method].copy()
        if method_data.empty:
            continue

        fig, axes = plt.subplots(
            nrows=len(PORTFOLIO_ORDER),
            ncols=1,
            figsize=(14, 11.5),
            sharex=True,
            sharey=True,
        )
        for ax, portfolio in zip(axes, PORTFOLIO_ORDER):
            sub = method_data.loc[method_data["Portfolio"] == portfolio]
            stacked_bar(
                ax=ax,
                data=sub,
                x_col="FormationYear",
                stack_col="Exchange",
                value_col="share_of_quantile",
                stack_order=exchanges,
                colors=EXCHANGE_COLORS,
            )
            ax.set_title(f"{portfolio}: exchange share within quantile", loc="left", fontsize=10)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Formation year")
        fig.suptitle(f"Exchange Composition Within Each Quantile ({method})", y=0.995, fontsize=14)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(exchanges), frameon=False)
        fig.tight_layout(rect=[0, 0.04, 1, 0.975])

        path = plot_dir / f"exchange_composition_by_quantile_{method_slug(method)}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs[method] = path

    return outputs


def save_total_distribution_plot(
    year_distribution: pd.DataFrame,
    output_dir: Path,
    exchanges: list[str],
) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5.6))
    stacked_bar(
        ax=ax,
        data=year_distribution,
        x_col="FormationYear",
        stack_col="Exchange",
        value_col="share_of_year",
        stack_order=exchanges,
        colors=EXCHANGE_COLORS,
    )
    ax.set_title("Total Firm-Year Exchange Distribution")
    ax.set_xlabel("Formation year")
    ax.set_ylabel("Share of firms")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(exchanges), frameon=False)
    fig.tight_layout()

    path = plot_dir / "exchange_total_distribution_by_year.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_exchange_placement_plots(
    placement_distribution: pd.DataFrame,
    output_dir: Path,
    exchanges: list[str],
) -> dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for method in vw.METHODS:
        method_data = placement_distribution.loc[placement_distribution["Method"] == method].copy()
        if method_data.empty:
            continue

        fig, axes = plt.subplots(
            nrows=len(exchanges),
            ncols=1,
            figsize=(14, 9.5),
            sharex=True,
            sharey=True,
        )
        if len(exchanges) == 1:
            axes = [axes]

        for ax, exchange in zip(axes, exchanges):
            sub = method_data.loc[method_data["Exchange"] == exchange]
            stacked_bar(
                ax=ax,
                data=sub,
                x_col="FormationYear",
                stack_col="Portfolio",
                value_col="share_of_exchange",
                stack_order=PORTFOLIO_ORDER,
                colors=PORTFOLIO_COLORS,
            )
            ax.set_title(f"{exchange}: quantile placement of local firms", loc="left", fontsize=10)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Formation year")
        fig.suptitle(f"Where Each Exchange's Firms Are Placed ({method})", y=0.995, fontsize=14)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(PORTFOLIO_ORDER), frameon=False)
        fig.tight_layout(rect=[0, 0.04, 1, 0.975])

        path = plot_dir / f"exchange_firm_placement_by_quantile_{method_slug(method)}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs[method] = path

    return outputs


def print_identification(
    run_dir: Path,
    portfolio_eval_dir: Path,
    assignments_csv: Path,
    output_dir: Path,
    include_iceland: bool,
) -> None:
    print("\nIdentified inputs")
    print(f"  run_dir: {run_dir}")
    print(f"  portfolio_evaluation_dir used for defaults: {portfolio_eval_dir}")
    print(f"  portfolio assignment source: {assignments_csv}")
    print(f"  exchange source: ticker suffix mapping {exchange_split.EXCHANGE_CODE_LABELS}")
    print(f"  methods included: {vw.METHODS}")
    print(f"  Iceland included: {include_iceland}")
    print(f"  output_dir: {output_dir}")


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    run_dir = vw.choose_run_dir(project_root, resolve_cli_path(args.run_dir, project_root))
    portfolio_eval_dir = vw.portfolio_eval_dir_for_run(run_dir)
    if portfolio_eval_dir is None:
        portfolio_eval_dir = run_dir / "portfolio_evaluation"

    assignments_csv = choose_assignment_source(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        requested_source=resolve_cli_path(args.assignments_csv, project_root),
    )
    output_dir = resolve_cli_path(args.output_dir, project_root)
    if output_dir is None:
        suffix = "excluding_iceland" if args.exclude_iceland else "all_exchanges"
        output_dir = portfolio_eval_dir / f"exchange_quantile_distribution_diagrams_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    include_iceland = not args.exclude_iceland
    exchanges = exchange_order(include_iceland=include_iceland)

    print_identification(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        assignments_csv=assignments_csv,
        output_dir=output_dir,
        include_iceland=include_iceland,
    )

    assignments = load_assignments(assignments_csv, include_iceland=include_iceland)
    quantile_distribution = build_quantile_distribution(assignments, exchanges)
    year_distribution = build_year_distribution(assignments, exchanges)
    placement_distribution = build_exchange_placement_distribution(assignments)

    csv_outputs = save_distribution_csvs(
        output_dir=output_dir,
        quantile_distribution=quantile_distribution,
        year_distribution=year_distribution,
        placement_distribution=placement_distribution,
    )
    quantile_plots = save_quantile_composition_plots(
        quantile_distribution=quantile_distribution,
        output_dir=output_dir,
        exchanges=exchanges,
    )
    total_plot = save_total_distribution_plot(
        year_distribution=year_distribution,
        output_dir=output_dir,
        exchanges=exchanges,
    )
    placement_plots = save_exchange_placement_plots(
        placement_distribution=placement_distribution,
        output_dir=output_dir,
        exchanges=exchanges,
    )

    print("\nCreated CSV files")
    row_counts = {
        "exchange_quantile_distribution": len(quantile_distribution),
        "exchange_year_total_distribution": len(year_distribution),
        "exchange_placement_by_quantile_distribution": len(placement_distribution),
    }
    for key, path in csv_outputs.items():
        print(f"  {path} ({row_counts[key]} rows)")

    print("\nCreated diagram files")
    for path in quantile_plots.values():
        print(f"  {path}")
    print(f"  {total_plot}")
    for path in placement_plots.values():
        print(f"  {path}")

    print("\nSummary")
    print(f"  firm-year assignment rows used: {len(assignments)}")
    print(f"  exchanges shown: {exchanges}")
    print(f"  formation years: {assignments['FormationYear'].min()}-{assignments['FormationYear'].max()}")


if __name__ == "__main__":
    main()
