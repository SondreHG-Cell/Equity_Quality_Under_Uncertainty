from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

METHODS = [
    "Method1_ObservedQuality",
    "Method2_LatentQuality",
    "Method3_ConservativeQuality",
    "Method4_ProbabilisticQuality",
]

METHOD_LABELS = {
    "Method1_ObservedQuality": "Observed Quality",
    "Method2_LatentQuality": "Latent Quality",
    "Method3_ConservativeQuality": "Conservative Quality",
    "Method4_ProbabilisticQuality": "Probabilistic Quality",
}

BLUE_PALETTE = {
    "Method1_ObservedQuality": "#173a5e",
    "Method2_LatentQuality": "#2c73ad",
    "Method3_ConservativeQuality": "#58a7df",
    "Method4_ProbabilisticQuality": "#8ec5e8",
}

STRATEGY_ORDER = ["LongShort", "Q5", "Q4"]
STRATEGY_PANEL_LABELS = {
    "LongShort": "Long-short strategy",
    "Q5": "Long strategy",
    "Q4": "Q4 strategy",
}

DEFAULT_ROOTS = [
    PROJECT_ROOT / "results" / "current_res",
    PROJECT_ROOT / "results" / "cur_res_ols",
    PROJECT_ROOT / "results" / "OLS_res",
    PROJECT_ROOT / "results" / "cur_res_analyst_cfo",
    PROJECT_ROOT / "results" / "res_analyst_cfo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis-style blue cumulative return figures from existing monthly return outputs."
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Root folders to search for monthly_portfolio_returns_used*.csv files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="PNG output resolution.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save PDF copies.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_monthly_return_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        root = resolve(root)
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("monthly_portfolio_returns_used*.csv")
            if "both_specs" not in path.name
        )
    return sorted(set(files))


def cumulative_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Return"] = pd.to_numeric(out["Return"], errors="coerce")
    out = out.dropna(subset=["Date", "Method", "PortfolioStrategy", "Return"])
    out = out.sort_values(["PortfolioStrategy", "Method", "Date"])
    out["CumulativeReturn"] = (
        out.groupby(["PortfolioStrategy", "Method"])["Return"].transform(lambda s: (1.0 + s).cumprod() - 1.0)
    )
    return out.reset_index(drop=True)


def strategy_sequence(cumulative: pd.DataFrame) -> list[str]:
    present = set(cumulative["PortfolioStrategy"].dropna().astype(str))
    ordered = [strategy for strategy in STRATEGY_ORDER if strategy in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered[:2]


def clean_suffix(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def apply_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#f6f9fc")
    ax.axhline(0.0, color="#3f79b7", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.grid(axis="y", color="#9ebfda", linewidth=0.6, linestyle="--", alpha=0.65)
    ax.grid(axis="x", color="#c8d9e8", linewidth=0.45, linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a7bfd5")
    ax.spines["bottom"].set_color("#a7bfd5")
    ax.tick_params(colors="#20384f", labelsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Cumulative return (%)", color="#173a5e", fontsize=11)


def plot_combined(cumulative: pd.DataFrame, output_path: Path, title_suffix: str | None, dpi: int, save_pdf: bool) -> None:
    strategies = strategy_sequence(cumulative)
    if not strategies:
        return

    fig, axes = plt.subplots(1, len(strategies), figsize=(12.0, 5.8), sharex=False)
    if len(strategies) == 1:
        axes = np.array([axes])

    fig.patch.set_facecolor("white")
    handles = []
    labels = []

    for ax, strategy in zip(axes, strategies):
        sub = cumulative.loc[cumulative["PortfolioStrategy"].eq(strategy)].copy()
        apply_axis_style(ax)

        for method in METHODS:
            method_sub = sub.loc[sub["Method"].eq(method)].sort_values("Date")
            if method_sub.empty:
                continue
            (line,) = ax.plot(
                method_sub["Date"],
                method_sub["CumulativeReturn"],
                color=BLUE_PALETTE[method],
                linewidth=2.25,
                label=METHOD_LABELS[method],
            )
            if METHOD_LABELS[method] not in labels:
                handles.append(line)
                labels.append(METHOD_LABELS[method])

        y = sub["CumulativeReturn"].dropna()
        if not y.empty:
            ymin = min(float(y.min()), 0.0)
            ymax = max(float(y.max()), 0.0)
            padding = max((ymax - ymin) * 0.08, 0.03)
            ax.set_ylim(ymin - padding, ymax + padding)

        ax.set_xlabel(STRATEGY_PANEL_LABELS.get(strategy, strategy), color="#173a5e", fontsize=12, fontweight="bold")

    title = "Cumulative returns" if not title_suffix else f"Cumulative returns: {title_suffix}"
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#173a5e", y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.925))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.78, wspace=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def output_stem(path: Path, group_col: str | None, group_value: str | None) -> str:
    if group_col == "SizeGroup" and group_value:
        return f"cumulative_returns_combined_{clean_suffix(group_value)}"
    if group_col == "Exchange" and group_value:
        return f"cumulative_returns_combined_{clean_suffix(group_value)}"
    return "cumulative_returns_combined"


def title_suffix_for_group(group_col: str | None, group_value: str | None) -> str | None:
    if group_col == "SizeGroup" and group_value:
        return group_value.replace("Cap", "-cap")
    if group_col == "Exchange" and group_value:
        return group_value
    return None


def process_file(path: Path, dpi: int, save_pdf: bool) -> list[Path]:
    monthly = pd.read_csv(path)
    if "Date" not in monthly.columns or "Return" not in monthly.columns:
        return []

    if "SizeGroup" in monthly.columns:
        group_col = "SizeGroup"
    elif "Exchange" in monthly.columns:
        group_col = "Exchange"
    else:
        group_col = None
    plot_dir = path.parent / "plots"
    outputs: list[Path] = []

    if group_col is None:
        cumulative = cumulative_returns(monthly)
        out = plot_dir / "cumulative_returns_combined.png"
        plot_combined(cumulative, out, None, dpi=dpi, save_pdf=save_pdf)
        outputs.append(out)
        return outputs

    for group_value, sub in monthly.groupby(group_col, sort=True):
        cumulative = cumulative_returns(sub.drop(columns=[group_col]))
        out = plot_dir / f"{output_stem(path, group_col, str(group_value))}.png"
        plot_combined(cumulative, out, title_suffix_for_group(group_col, str(group_value)), dpi=dpi, save_pdf=save_pdf)
        outputs.append(out)
    return outputs


def main() -> None:
    args = parse_args()
    files = find_monthly_return_files(args.roots)
    if not files:
        raise FileNotFoundError("No monthly_portfolio_returns_used*.csv files found.")

    outputs: list[Path] = []
    for path in files:
        outputs.extend(process_file(path, dpi=args.dpi, save_pdf=args.pdf))

    print(f"Created {len(outputs)} cumulative return figures.")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
