from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

OBSERVED_METHOD = "Method1_ObservedQuality"
CONSERVATIVE_METHOD = "Method3_ConservativeQuality"

MAIN_METHODS = {
    "Latent": "Method2_LatentQuality",
    "Conservative": "Method3_ConservativeQuality",
    "Probabilistic": "Method4_ProbabilisticQuality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate studentized Ledoit-Wolf Sharpe ratio difference tests for "
            "the main sorting methods and Conservative Quality gamma robustness."
        )
    )
    parser.add_argument(
        "--portfolio-evaluation-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "current_res" / "portfolio_evaluation",
        help="Directory containing thesis risk-adjusted table output folders.",
    )
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "extraction_static" / "factor_data.csv",
        help="CSV containing Date and RF columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<portfolio-evaluation-dir>/thesis_risk_adjusted_tables_conservative_gamma_robustness_ucits_5_10_40."
        ),
    )
    parser.add_argument(
        "--main-table-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the main UCITS 5/10/40 monthly_portfolio_returns_used.csv. "
            "Defaults to <portfolio-evaluation-dir>/thesis_risk_adjusted_tables_ucits_5_10_40."
        ),
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        default=["0.10", "0.15", "0.20", "0.30"],
        help="Gamma values to include for Conservative Quality robustness.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=6,
        help="Circular block bootstrap block length. Default is 6 months.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10_000,
        help="Number of bootstrap replications. Default is 10,000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20_260_518,
        help="Random seed for bootstrap draws.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def gamma_label_for_dir(gamma: str) -> str:
    return f"{float(gamma):.2f}".replace(".", "_")


def gamma_table_dirs(portfolio_evaluation_dir: Path, main_table_dir: Path, gammas: list[str]) -> dict[str, Path]:
    dirs = {}
    for gamma in gammas:
        if f"{float(gamma):.2f}" == "0.10":
            dirs["0.10"] = main_table_dir
        else:
            label = gamma_label_for_dir(gamma)
            dirs[f"{float(gamma):.2f}"] = (
                portfolio_evaluation_dir
                / f"thesis_risk_adjusted_tables_conservative_gamma_{label}_ucits_5_10_40"
            )
    return dirs


def load_monthly_returns(table_dir: Path) -> pd.DataFrame:
    path = table_dir / "monthly_portfolio_returns_used.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly returns file: {path}")
    return pd.read_csv(path, parse_dates=["Date"])


def strategy_excess_returns(
    monthly_returns: pd.DataFrame,
    rf: pd.Series,
    strategy: str,
    method: str,
) -> pd.Series:
    sub = monthly_returns[
        monthly_returns["PortfolioStrategy"].eq(strategy)
        & monthly_returns["Method"].eq(method)
    ].copy()
    if sub.empty:
        raise ValueError(f"No returns for strategy={strategy}, method={method}")

    returns = sub.sort_values("Date").set_index("Date")["Return"].astype(float)

    if strategy == "LongShort":
        return returns.dropna()

    return returns.subtract(rf.reindex(returns.index), fill_value=np.nan).dropna()


def theta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([x.mean(), y.mean(), (x * x).mean(), (y * y).mean()], dtype=float)


def monthly_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    return float(x.mean() / sd)


def delta_from_theta(values: np.ndarray) -> float:
    mean_i, mean_n, second_i, second_n = values
    var_i = second_i - mean_i * mean_i
    var_n = second_n - mean_n * mean_n
    return float(mean_i / np.sqrt(var_i) - mean_n / np.sqrt(var_n))


def grad_delta(values: np.ndarray) -> np.ndarray:
    mean_i, mean_n, second_i, second_n = values
    var_i = second_i - mean_i * mean_i
    var_n = second_n - mean_n * mean_n
    var_i_32 = var_i**1.5
    var_n_32 = var_n**1.5
    return np.array(
        [
            second_i / var_i_32,
            -second_n / var_n_32,
            -mean_i / (2.0 * var_i_32),
            mean_n / (2.0 * var_n_32),
        ],
        dtype=float,
    )


def centered_moment_matrix(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray | None = None,
) -> np.ndarray:
    if values is None:
        values = theta(x, y)
    mean_i, mean_n, second_i, second_n = values
    return np.column_stack(
        [
            x - mean_i,
            y - mean_n,
            x * x - second_i,
            y * y - second_n,
        ]
    )


def hac_bartlett_se(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Original-sample standard error for the Sharpe difference.

    This uses Eq. (5) in Ledoit and Wolf (2008) with a Bartlett HAC covariance
    estimate. The existing thesis table is reproduced by using lag equal to the
    block length.
    """
    n_obs = len(x)
    values = theta(x, y)
    moments = centered_moment_matrix(x, y, values)

    psi = (moments.T @ moments) / n_obs
    for j in range(1, lag + 1):
        weight = 1.0 - j / (lag + 1.0)
        covariance = (moments[j:].T @ moments[:-j]) / n_obs
        psi += weight * (covariance + covariance.T)

    psi *= n_obs / (n_obs - 4.0)
    variance = float(grad_delta(values) @ psi @ grad_delta(values) / n_obs)
    return float(np.sqrt(max(variance, 0.0)))


def block_bootstrap_se(x: np.ndarray, y: np.ndarray, block_length: int) -> float:
    """
    Bootstrap-world standard error using the block dependence structure.

    This follows the time-series bootstrap standard error described in
    Ledoit and Wolf (2008), Section 3.2.2.
    """
    n_obs = len(x)
    values = theta(x, y)
    moments = centered_moment_matrix(x, y, values)

    n_blocks = n_obs // block_length
    if n_blocks < 1:
        return np.nan

    usable = moments[: n_blocks * block_length]
    block_sums = usable.reshape(n_blocks, block_length, 4).sum(axis=1) / np.sqrt(block_length)
    psi = (block_sums.T @ block_sums) / n_blocks

    variance = float(grad_delta(values) @ psi @ grad_delta(values) / n_obs)
    return float(np.sqrt(max(variance, 0.0)))


def circular_block_indices(
    n_obs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_blocks = int(np.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs, size=n_blocks)
    indices = (starts[:, None] + np.arange(block_length)) % n_obs
    return indices.ravel()[:n_obs]


def ledoit_wolf_test(
    method_returns: pd.Series,
    observed_returns: pd.Series,
    *,
    rng: np.random.Generator,
    block_length: int,
    n_boot: int,
) -> dict:
    """
    Studentized circular block-bootstrap Sharpe ratio difference test.

    The p-value follows Ledoit and Wolf (2008), Remark 3.2:
    compare |D_hat| / s(D_hat) with the bootstrap distribution of
    |D_hat* - D_hat| / s(D_hat*).
    """
    data = pd.concat(
        [method_returns.rename("method"), observed_returns.rename("observed")],
        axis=1,
    ).dropna()

    method = data["method"].to_numpy(dtype=float)
    observed = data["observed"].to_numpy(dtype=float)
    n_obs = len(data)

    method_sharpe_monthly = monthly_sharpe(method)
    observed_sharpe_monthly = monthly_sharpe(observed)
    delta_hat = delta_from_theta(theta(method, observed))
    se_hat = hac_bartlett_se(method, observed, lag=block_length)
    studentized_stat = abs(delta_hat) / se_hat

    d_star = np.empty(n_boot, dtype=float)
    valid = 0
    for _ in range(n_boot):
        indices = circular_block_indices(n_obs, block_length, rng)
        method_boot = method[indices]
        observed_boot = observed[indices]
        delta_star = delta_from_theta(theta(method_boot, observed_boot))
        se_star = block_bootstrap_se(method_boot, observed_boot, block_length=block_length)

        if np.isfinite(se_star) and se_star > 0:
            d_star[valid] = abs(delta_star - delta_hat) / se_star
            valid += 1

    d_star = d_star[:valid]
    bootstrap_quantile = float(np.quantile(d_star, 0.95))
    p_value = float((np.sum(d_star >= studentized_stat) + 1.0) / (len(d_star) + 1.0))
    annualizer = np.sqrt(12.0)

    return {
        "method_sharpe": method_sharpe_monthly * annualizer,
        "observed_sharpe": observed_sharpe_monthly * annualizer,
        "difference": delta_hat * annualizer,
        "ci_lower": (delta_hat - bootstrap_quantile * se_hat) * annualizer,
        "ci_upper": (delta_hat + bootstrap_quantile * se_hat) * annualizer,
        "p_value": p_value,
        "n_obs": int(n_obs),
        "n_boot": int(n_boot),
        "valid_boot": int(len(d_star)),
        "block_length": int(block_length),
        "se_monthly": se_hat,
        "studentized_stat": studentized_stat,
        "studentized_quantile_95": bootstrap_quantile,
    }


def run_main_method_tests(
    main_table_dir: Path,
    rf: pd.Series,
    rng: np.random.Generator,
    block_length: int,
    n_boot: int,
) -> pd.DataFrame:
    monthly_returns = load_monthly_returns(main_table_dir)
    rows = []

    for strategy in ["LongShort", "Q5"]:
        observed = strategy_excess_returns(monthly_returns, rf, strategy, OBSERVED_METHOD)
        for comparison, method in MAIN_METHODS.items():
            result = ledoit_wolf_test(
                strategy_excess_returns(monthly_returns, rf, strategy, method),
                observed,
                rng=rng,
                block_length=block_length,
                n_boot=n_boot,
            )
            rows.append(
                {
                    "Comparison": comparison,
                    "Method": method,
                    "PortfolioStrategy": strategy,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def run_gamma_tests(
    gamma_dirs: dict[str, Path],
    rf: pd.Series,
    rng: np.random.Generator,
    block_length: int,
    n_boot: int,
) -> pd.DataFrame:
    rows = []

    for gamma, table_dir in gamma_dirs.items():
        monthly_returns = load_monthly_returns(table_dir)
        for strategy in ["LongShort", "Q5"]:
            observed = strategy_excess_returns(monthly_returns, rf, strategy, OBSERVED_METHOD)
            result = ledoit_wolf_test(
                strategy_excess_returns(monthly_returns, rf, strategy, CONSERVATIVE_METHOD),
                observed,
                rng=rng,
                block_length=block_length,
                n_boot=n_boot,
            )
            rows.append(
                {
                    "Gamma": gamma,
                    "Comparison": rf"$\gamma={gamma}$",
                    "Method": CONSERVATIVE_METHOD,
                    "PortfolioStrategy": strategy,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def write_latex_table(
    path: Path,
    rows: pd.DataFrame,
    caption: str,
    label: str,
    comparison_col: str,
) -> None:
    def fmt(value: float) -> str:
        return f"{value:.3f}"

    def strategy_title(strategy: str) -> str:
        return "Long--short strategy" if strategy == "LongShort" else "Long strategy"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{threeparttable}",
    ]

    for i, strategy in enumerate(["LongShort", "Q5"]):
        panel = "A" if i == 0 else "B"
        lines.extend(
            [
                rf"{{\centering\textbf{{Panel {panel}: {strategy_title(strategy)}}}\par}}",
                r"\vspace{0.4em}",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc}",
                r"\toprule",
                r"Comparison & Method Sharpe & Observed Sharpe & Difference & \multicolumn{2}{c}{95\% CI} & $p$-value \\",
                r"\cmidrule(lr){5-6}",
                r"& & & & Lower & Upper & \\",
                r"\midrule",
            ]
        )

        sub = rows[rows["PortfolioStrategy"].eq(strategy)]
        for _, row in sub.iterrows():
            lines.append(
                f"{row[comparison_col]} & "
                f"{fmt(row['method_sharpe'])} & "
                f"{fmt(row['observed_sharpe'])} & "
                f"{fmt(row['difference'])} & "
                f"{fmt(row['ci_lower'])} & "
                f"{fmt(row['ci_upper'])} & "
                f"{fmt(row['p_value'])} \\\\"
            )

        lines.extend([r"\bottomrule", r"\end{tabular*}"])
        if i == 0:
            lines.append(r"\vspace{1em}")

    lines.extend(
        [
            r"\begin{tablenotes}[flushleft]",
            r"\scriptsize",
            (
                r"\item Note: The table reports pairwise Sharpe ratio differences "
                r"relative to the Observed Quality baseline, estimated using the "
                r"studentized circular block bootstrap of \textcite{ledoit2008} "
                r"with block length 6 and 10,000 replications. Sharpe ratios are "
                r"computed from annualised excess returns. For the long strategy, "
                r"excess returns are measured relative to the risk-free rate; the "
                r"long--short strategy is treated as self-financing. Confidence "
                r"intervals are reported at the 95\% level. $p$-values are two-sided "
                r"and computed from the studentized bootstrap distribution."
            ),
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    portfolio_evaluation_dir = resolve_path(args.portfolio_evaluation_dir)
    factors_csv = resolve_path(args.factors_csv)
    main_table_dir = (
        resolve_path(args.main_table_dir)
        if args.main_table_dir is not None
        else portfolio_evaluation_dir / "thesis_risk_adjusted_tables_ucits_5_10_40"
    )
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else portfolio_evaluation_dir
        / "thesis_risk_adjusted_tables_conservative_gamma_robustness_ucits_5_10_40"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_data = pd.read_csv(factors_csv, parse_dates=["Date"]).set_index("Date")
    rf = factor_data["RF"].astype(float)

    main_rng = np.random.default_rng(args.seed)
    main_results = run_main_method_tests(
        main_table_dir=main_table_dir,
        rf=rf,
        rng=main_rng,
        block_length=args.block_length,
        n_boot=args.n_boot,
    )
    main_results.to_csv(output_dir / "main_methods_sharpe_difference_tests_ledoit_wolf.csv", index=False)
    write_latex_table(
        output_dir / "main_methods_sharpe_difference_tests_ledoit_wolf.tex",
        main_results,
        "Ledoit--Wolf Sharpe ratio difference tests",
        "tab:main_methods_sharpe_difference_tests_ledoit_wolf",
        "Comparison",
    )

    gamma_rng = np.random.default_rng(args.seed)
    gamma_results = run_gamma_tests(
        gamma_dirs=gamma_table_dirs(portfolio_evaluation_dir, main_table_dir, args.gammas),
        rf=rf,
        rng=gamma_rng,
        block_length=args.block_length,
        n_boot=args.n_boot,
    )
    gamma_results.to_csv(
        output_dir / "conservative_gamma_sharpe_difference_tests_ledoit_wolf.csv",
        index=False,
    )
    write_latex_table(
        output_dir / "conservative_gamma_sharpe_difference_tests_ledoit_wolf.tex",
        gamma_results,
        "Ledoit--Wolf Sharpe ratio difference tests: Conservative Quality gamma robustness",
        "tab:conservative_gamma_sharpe_difference_tests_ledoit_wolf",
        "Comparison",
    )

    print("Created Ledoit-Wolf Sharpe ratio difference outputs:")
    for file_name in [
        "main_methods_sharpe_difference_tests_ledoit_wolf.csv",
        "main_methods_sharpe_difference_tests_ledoit_wolf.tex",
        "conservative_gamma_sharpe_difference_tests_ledoit_wolf.csv",
        "conservative_gamma_sharpe_difference_tests_ledoit_wolf.tex",
    ]:
        print(output_dir / file_name)


if __name__ == "__main__":
    main()
