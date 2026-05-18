from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    "Method1_ObservedQuality",
    "Method2_LatentQuality",
    "Method3_ConservativeQuality",
    "Method4_ProbabilisticQuality",
]

METHOD_DISPLAY_LABELS = {
    "Method1_ObservedQuality": "Observed Quality",
    "Method2_LatentQuality": "Latent Quality",
    "Method3_ConservativeQuality": "Conservative Quality",
    "Method4_ProbabilisticQuality": "Probabilistic Quality",
}

BASE_METHOD = "Method1_ObservedQuality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired bootstrap Sharpe-ratio difference tests from existing monthly strategy returns."
    )
    parser.add_argument(
        "--returns-csv",
        type=Path,
        required=True,
        help="Path to monthly_portfolio_returns_used.csv.",
    )
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=None,
        help="Optional factor CSV containing RF. Needed if testing Q5 with excess returns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the same folder as returns-csv.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=6,
        help="Expected block length for monthly stationary bootstrap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def parse_factor_date(series: pd.Series) -> pd.Series:
    """
    Robust monthly date parser.

    Handles dates like:
    - 2010-07-31
    - 201007
    - 2010-07
    """
    s = series.astype(str).str.strip()

    parsed = pd.to_datetime(s, errors="coerce")

    mask = parsed.isna() & s.str.fullmatch(r"\d{6}", na=False)
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(s.loc[mask] + "01", format="%Y%m%d", errors="coerce")

    return parsed.dt.to_period("M").dt.to_timestamp("M")


def load_rf(factors_csv: Path | None) -> pd.Series | None:
    if factors_csv is None:
        return None

    factors = pd.read_csv(factors_csv)

    date_col = None
    for candidate in ["Date", "date", "Month", "month", "YYYYMM", "yyyymm"]:
        if candidate in factors.columns:
            date_col = candidate
            break

    if date_col is None:
        date_col = factors.columns[0]

    rf_col = None
    for candidate in ["RF", "rf", "Rf", "risk_free", "RiskFree"]:
        if candidate in factors.columns:
            rf_col = candidate
            break

    if rf_col is None:
        raise ValueError(
            f"Could not find RF column in {factors_csv}. Available columns: {list(factors.columns)}"
        )

    out = factors[[date_col, rf_col]].copy()
    out["Date"] = parse_factor_date(out[date_col])
    out["RF"] = pd.to_numeric(out[rf_col], errors="coerce")
    out = out.dropna(subset=["Date", "RF"]).drop_duplicates("Date").set_index("Date")

    # Heuristic: Kenneth French factors are often in percent, e.g. 0.35 means 0.35%.
    # Your portfolio returns are decimals, e.g. 0.01 means 1%.
    if out["RF"].abs().median() > 0.01:
        out["RF"] = out["RF"] / 100.0

    return out["RF"].sort_index()


def load_strategy_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["Date", "Method", "PortfolioStrategy", "Return"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[required].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    df["Method"] = df["Method"].astype(str).str.strip()
    df["PortfolioStrategy"] = df["PortfolioStrategy"].astype(str).str.strip()
    df["Return"] = pd.to_numeric(df["Return"], errors="coerce")
    df = df.dropna(subset=["Date", "Method", "PortfolioStrategy", "Return"])

    return df


def annualized_sharpe(excess_returns: pd.Series, periods_per_year: int = 12) -> float:
    x = pd.to_numeric(excess_returns, errors="coerce").dropna()

    if len(x) < 2:
        return np.nan

    vol = x.std(ddof=1)

    if pd.isna(vol) or vol <= 0:
        return np.nan

    return float(np.sqrt(periods_per_year) * x.mean() / vol)


def stationary_bootstrap_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    p = 1.0 / block_length
    idx = np.empty(n, dtype=int)

    idx[0] = rng.integers(0, n)

    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n

    return idx


def sharpe_difference_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    strategy: str,
    rf: pd.Series | None,
    n_bootstrap: int,
    block_length: int,
    seed: int,
) -> dict:
    """
    Paired stationary bootstrap test of Sharpe(A) - Sharpe(B).

    LongShort is treated as self-financing excess return.
    Q5 subtracts RF if RF is supplied.
    """
    a = pd.to_numeric(returns_a, errors="coerce")
    b = pd.to_numeric(returns_b, errors="coerce")

    a.index = pd.to_datetime(a.index).to_period("M").to_timestamp("M")
    b.index = pd.to_datetime(b.index).to_period("M").to_timestamp("M")

    paired = pd.concat({"a": a, "b": b}, axis=1).dropna()

    if strategy == "LongShort":
        excess = paired.copy()
    else:
        if rf is None:
            # This allows Q5 to run, but it will be raw-return Sharpe.
            # Prefer passing factors-csv for excess-return Sharpe.
            excess = paired.copy()
        else:
            rf_use = rf.copy()
            rf_use.index = pd.to_datetime(rf_use.index).to_period("M").to_timestamp("M")
            excess = paired.subtract(rf_use.reindex(paired.index), axis=0).dropna()

    n = len(excess)
    if n < 24:
        raise ValueError(f"Too few observations for {strategy}: {n}")

    sr_a = annualized_sharpe(excess["a"])
    sr_b = annualized_sharpe(excess["b"])
    obs_diff = sr_a - sr_b

    values = excess[["a", "b"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, block_length, rng)
        sample = values[idx, :]

        boot_diffs[i] = (
            annualized_sharpe(pd.Series(sample[:, 0]))
            - annualized_sharpe(pd.Series(sample[:, 1]))
        )

    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    # Two-sided centered bootstrap p-value.
    centered_boot = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(np.abs(centered_boot) >= abs(obs_diff)))

    return {
        "sharpe_method": sr_a,
        "sharpe_base": sr_b,
        "sharpe_difference": obs_diff,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "p_value": p_value,
        "n_obs": int(n),
        "n_bootstrap": int(n_bootstrap),
        "block_length": int(block_length),
    }


def make_wide(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    sub = df.loc[df["PortfolioStrategy"] == strategy].copy()

    wide = sub.pivot_table(
        index="Date",
        columns="Method",
        values="Return",
        aggfunc="first",
    ).sort_index()

    missing = [m for m in METHODS if m not in wide.columns]
    if missing:
        raise ValueError(f"{strategy} is missing methods: {missing}. Available: {list(wide.columns)}")

    return wide[METHODS]


def run_tests(
    df: pd.DataFrame,
    rf: pd.Series | None,
    n_bootstrap: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    rows = []

    for strategy in ["LongShort", "Q5"]:
        wide = make_wide(df, strategy)

        for method in METHODS:
            if method == BASE_METHOD:
                continue

            result = sharpe_difference_test(
                returns_a=wide[method],
                returns_b=wide[BASE_METHOD],
                strategy=strategy,
                rf=rf,
                n_bootstrap=n_bootstrap,
                block_length=block_length,
                seed=seed,
            )

            method_label = METHOD_DISPLAY_LABELS.get(method, method)
            base_label = METHOD_DISPLAY_LABELS.get(BASE_METHOD, BASE_METHOD)

            rows.append(
                {
                    "PortfolioStrategy": strategy,
                    "Comparison": f"{method_label} minus {base_label}",
                    "Method": method,
                    "BaseMethod": BASE_METHOD,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def add_significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def save_latex_table(results: pd.DataFrame, path: Path, strategy: str) -> None:
    sub = results.loc[results["PortfolioStrategy"] == strategy].copy()

    table = pd.DataFrame()
    table["Comparison"] = sub["Comparison"]
    table["Sharpe method"] = sub["sharpe_method"].map(lambda x: f"{x:.2f}")
    table["Sharpe baseline"] = sub["sharpe_base"].map(lambda x: f"{x:.2f}")
    table["Difference"] = sub.apply(
        lambda r: f"{r['sharpe_difference']:.2f}{add_significance_stars(r['p_value'])}",
        axis=1,
    )
    table["95\\% CI"] = sub.apply(
        lambda r: f"[{r['ci_95_lower']:.2f}, {r['ci_95_upper']:.2f}]",
        axis=1,
    )
    table["p-value"] = sub["p_value"].map(lambda x: f"{x:.3f}")

    caption = (
        "Bootstrap tests of Sharpe-ratio differences for long--short strategies"
        if strategy == "LongShort"
        else "Bootstrap tests of Sharpe-ratio differences for pure Q5 strategies"
    )

    latex = table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=f"tab:sharpe_diff_{strategy.lower()}",
    )

    path.write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.returns_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = load_strategy_returns(args.returns_csv)
    rf = load_rf(args.factors_csv)

    results = run_tests(
        df=returns,
        rf=rf,
        n_bootstrap=args.n_bootstrap,
        block_length=args.block_length,
        seed=args.seed,
    )

    all_path = output_dir / "table_sharpe_differences.csv"
    ls_path = output_dir / "table_ls_sharpe_differences.csv"
    q5_path = output_dir / "table_q5_sharpe_differences.csv"

    results.to_csv(all_path, index=False)
    results.loc[results["PortfolioStrategy"] == "LongShort"].to_csv(ls_path, index=False)
    results.loc[results["PortfolioStrategy"] == "Q5"].to_csv(q5_path, index=False)

    save_latex_table(results, output_dir / "table_ls_sharpe_differences.tex", "LongShort")
    save_latex_table(results, output_dir / "table_q5_sharpe_differences.tex", "Q5")

    print("\nCreated Sharpe-ratio difference outputs")
    print(f"  {all_path}")
    print(f"  {ls_path}")
    print(f"  {q5_path}")
    print(f"  {output_dir / 'table_ls_sharpe_differences.tex'}")
    print(f"  {output_dir / 'table_q5_sharpe_differences.tex'}")

    if rf is None:
        print(
            "\nNote: No factors CSV was supplied, so Q5 Sharpe ratios were computed from raw returns. "
            "For thesis consistency, pass --factors-csv so Q5 uses excess returns."
        )


if __name__ == "__main__":
    main()