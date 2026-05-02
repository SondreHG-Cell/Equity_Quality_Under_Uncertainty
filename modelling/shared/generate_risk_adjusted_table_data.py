from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helper_functions import find_project_root, load_factor_data, parse_month_series, resolve_path
from step5_evaluation import alpha_differences, risk_adjusted_performance


METHODS = [
    "Method1_ObservedQuality",
    "Method2_LatentQuality",
    "Method3_ConservativeQuality",
    "Method4_ProbabilisticQuality",
]

MODEL_LABELS = {
    "CAPM": "CAPM",
    "FF3": "FF3",
    "Carhart": "FF3+MOM",
    "FF5": "FF5",
    "FF5_MOM": "FF5+MOM",
}
INTERNAL_MODELS = list(MODEL_LABELS.keys())
FACTOR_COLUMNS = {
    "CAPM": ["MKT"],
    "FF3": ["MKT", "SMB", "HML"],
    "Carhart": ["MKT", "SMB", "HML", "MOM"],
    "FF5": ["MKT", "SMB", "HML", "RMW", "CMA"],
    "FF5_MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
}

COMPARISON_LABELS = {
    "Method2_LatentQuality vs Method1_ObservedQuality": (
        "Method2_LatentQuality minus Method1_ObservedQuality"
    ),
    "Method3_ConservativeQuality vs Method1_ObservedQuality": (
        "Method3_ConservativeQuality minus Method1_ObservedQuality"
    ),
    "Method4_ProbabilisticQuality vs Method1_ObservedQuality": (
        "Method4_ProbabilisticQuality minus Method1_ObservedQuality"
    ),
}
COMPARISONS = list(COMPARISON_LABELS.values())

METHOD_COLORS = {
    "Method1_ObservedQuality": "#4C78A8",
    "Method2_LatentQuality": "#F2A65A",
    "Method3_ConservativeQuality": "#72B7B2",
    "Method4_ProbabilisticQuality": "#B279A2",
}

METHOD_DISPLAY_LABELS = {
    "Method1_ObservedQuality": "Observed Quality",
    "Method2_LatentQuality": "Latent Quality",
    "Method3_ConservativeQuality": "Conservative Quality",
    "Method4_ProbabilisticQuality": "Probabilistic Quality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis table data for risk-adjusted performance using the "
            "repo's monthly holdings/factor outputs and existing HAC regression helpers."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional results run directory, e.g. results/current_res.",
    )
    parser.add_argument(
        "--portfolio-source",
        type=Path,
        default=None,
        help="Optional portfolio source CSV. Prefer monthly_holdings.csv with WeightedReturn.",
    )
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=None,
        help="Optional factor returns CSV. Defaults to run_config.json or results/extraction_static/factor_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <run-dir>/portfolio_evaluation/thesis_risk_adjusted_tables.",
    )
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=12,
        help="Newey-West/HAC lags. Default is 12.",
    )
    return parser.parse_args()


def resolve_cli_path(path: Path | None, project_root: Path) -> Path | None:
    if path is None:
        return None
    return resolve_path(path, project_root)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def complete_portfolio_eval_dirs(project_root: Path) -> tuple[list[Path], list[Path]]:
    direct = sorted((project_root / "results").glob("*/portfolio_evaluation"))
    nested = sorted(p for p in (project_root / "results").glob("*/portfolio_evaluation/*") if p.is_dir())
    candidates = direct + nested
    complete = [
        p
        for p in candidates
        if (p / "monthly_holdings.csv").exists() or (p / "monthly_portfolio_returns.csv").exists()
    ]
    return candidates, complete


def portfolio_eval_dir_for_run(run_dir: Path) -> Path | None:
    if (run_dir / "monthly_holdings.csv").exists() or (run_dir / "monthly_portfolio_returns.csv").exists():
        return run_dir

    direct = run_dir / "portfolio_evaluation"
    if (direct / "monthly_holdings.csv").exists() or (direct / "monthly_portfolio_returns.csv").exists():
        return direct

    if direct.exists():
        nested = [
            p
            for p in sorted(direct.iterdir())
            if p.is_dir()
            and ((p / "monthly_holdings.csv").exists() or (p / "monthly_portfolio_returns.csv").exists())
        ]
        for preferred_name in ["HB", "baseline"]:
            for path in nested:
                if path.name == preferred_name:
                    return path
        if nested:
            return max(nested, key=lambda p: p.stat().st_mtime)

    return None


def run_dir_from_portfolio_eval_dir(portfolio_eval_dir: Path) -> Path:
    if portfolio_eval_dir.name == "portfolio_evaluation":
        return portfolio_eval_dir.parent
    if portfolio_eval_dir.parent.name == "portfolio_evaluation":
        return portfolio_eval_dir.parent.parent
    return portfolio_eval_dir


def choose_run_dir(project_root: Path, requested_run_dir: Path | None) -> Path:
    searched_eval_dirs, complete_eval_dirs = complete_portfolio_eval_dirs(project_root)

    if requested_run_dir is not None:
        run_dir = requested_run_dir
        if portfolio_eval_dir_for_run(run_dir) is not None:
            return run_dir
        raise FileNotFoundError(
            "The requested run directory does not contain a usable portfolio_evaluation source.\n"
            f"Requested: {run_dir}\n"
            "Searched portfolio_evaluation directories:\n"
            + "\n".join(str(p) for p in searched_eval_dirs)
        )

    preferred = project_root / "results" / "current_res"
    if portfolio_eval_dir_for_run(preferred) is not None:
        return preferred

    if complete_eval_dirs:
        return run_dir_from_portfolio_eval_dir(max(complete_eval_dirs, key=lambda p: p.stat().st_mtime))

    raise FileNotFoundError(
        "Could not locate a portfolio_evaluation directory with monthly holdings or portfolio returns.\n"
        "Searched portfolio_evaluation directories:\n"
        + "\n".join(str(p) for p in searched_eval_dirs)
    )


def choose_portfolio_source(run_dir: Path, requested_source: Path | None) -> tuple[Path, str, Path]:
    portfolio_eval_dir = portfolio_eval_dir_for_run(run_dir)
    if portfolio_eval_dir is None:
        portfolio_eval_dir = run_dir / "portfolio_evaluation"

    searched = [
        portfolio_eval_dir / "monthly_holdings.csv",
        portfolio_eval_dir / "monthly_portfolio_returns.csv",
    ]

    if requested_source is not None:
        if not requested_source.exists():
            raise FileNotFoundError(
                "Requested portfolio source does not exist.\n"
                f"Requested: {requested_source}\n"
                "Default paths checked:\n"
                + "\n".join(str(p) for p in searched)
            )
        return requested_source, classify_portfolio_source(requested_source), requested_source.parent

    holdings = searched[0]
    if holdings.exists():
        return holdings, "monthly_holdings_weighted_constituents", portfolio_eval_dir

    wide_returns = searched[1]
    if wide_returns.exists():
        return wide_returns, "monthly_portfolio_returns_wide", portfolio_eval_dir

    raise FileNotFoundError(
        "Could not locate a portfolio source file.\n"
        "Searched:\n" + "\n".join(str(p) for p in searched)
    )


def classify_portfolio_source(path: Path) -> str:
    sample = pd.read_csv(path, nrows=5)
    weighted_cols = {"Method", "Portfolio", "Date", "WeightedReturn"}
    if weighted_cols.issubset(sample.columns):
        return "monthly_holdings_weighted_constituents"
    return "monthly_portfolio_returns_wide"


def choose_factor_csv(
    project_root: Path,
    run_dir: Path,
    requested_factors: Path | None,
) -> Path:
    searched: list[Path] = []

    if requested_factors is not None:
        if requested_factors.exists():
            return requested_factors
        raise FileNotFoundError(f"Requested factors CSV does not exist: {requested_factors}")

    run_config = read_json_if_exists(run_dir / "run_config.json")
    if "factors_csv" in run_config:
        candidate = resolve_path(run_config["factors_csv"], project_root)
        searched.append(candidate)
        if candidate.exists():
            return candidate

    default_factor = project_root / "results" / "extraction_static" / "factor_data.csv"
    searched.append(default_factor)
    if default_factor.exists():
        return default_factor

    globbed = sorted((project_root / "results").glob("**/factor_data.csv"))
    searched.extend(globbed)
    for path in globbed:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate monthly factor returns.\n"
        "Searched:\n" + "\n".join(str(p) for p in searched)
    )


def load_returns_from_weighted_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["Method", "Portfolio", "Date", "WeightedReturn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required weighted-holdings columns: {missing}")

    out = df[required].copy()
    out["Date"] = parse_month_series(out["Date"])
    out["Method"] = out["Method"].astype(str).str.strip()
    out["Portfolio"] = out["Portfolio"].astype(str).str.strip()
    out["WeightedReturn"] = pd.to_numeric(out["WeightedReturn"], errors="coerce")
    out = out.dropna(subset=["Date", "Method", "Portfolio", "WeightedReturn"])

    monthly = (
        out.groupby(["Method", "Portfolio", "Date"], as_index=False)["WeightedReturn"]
        .sum()
        .rename(columns={"WeightedReturn": "Return"})
        .sort_values(["Method", "Portfolio", "Date"])
        .reset_index(drop=True)
    )
    return monthly


def load_returns_from_wide_csv(path: Path) -> pd.DataFrame:
    try:
        wide = pd.read_csv(path, header=[0, 1], index_col=0)
    except Exception as exc:
        raise ValueError(
            f"Could not read {path} as monthly portfolio returns with two header rows."
        ) from exc

    if not isinstance(wide.columns, pd.MultiIndex):
        raise ValueError(f"{path} does not have Method/Portfolio MultiIndex columns.")

    wide.index = parse_month_series(pd.Series(wide.index, index=wide.index)).values
    wide = wide[wide.index.notna()].sort_index()

    try:
        stacked = wide.stack([0, 1], future_stack=True)
    except TypeError:
        stacked = wide.stack([0, 1])

    long_df = stacked.rename("Return").reset_index()
    long_df.columns = ["Date", "Method", "Portfolio", "Return"]
    long_df["Return"] = pd.to_numeric(long_df["Return"], errors="coerce")
    return long_df.dropna(subset=["Date", "Method", "Portfolio", "Return"]).reset_index(drop=True)


def load_monthly_portfolio_returns(path: Path, source_type: str) -> pd.DataFrame:
    if source_type == "monthly_holdings_weighted_constituents":
        return load_returns_from_weighted_holdings(path)
    return load_returns_from_wide_csv(path)


def validate_methods_and_portfolios(
    monthly_returns: pd.DataFrame,
    methods: list[str] | None = None,
) -> None:
    if methods is None:
        methods = METHODS

    available_methods = sorted(monthly_returns["Method"].dropna().unique())
    missing_methods = [m for m in methods if m not in available_methods]
    if missing_methods:
        raise ValueError(
            "Monthly portfolio returns are missing required sorting methods.\n"
            f"Missing: {missing_methods}\n"
            f"Available: {available_methods}"
        )

    missing = []
    for method in methods:
        portfolios = set(monthly_returns.loc[monthly_returns["Method"] == method, "Portfolio"])
        for portfolio in ["Q1", "Q5"]:
            if portfolio not in portfolios:
                missing.append(f"{method}/{portfolio}")
    if missing:
        raise ValueError("Missing required Q1/Q5 portfolios: " + ", ".join(missing))


def build_strategy_returns(
    monthly_returns: pd.DataFrame,
    factors: pd.DataFrame,
    methods: list[str] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.DataFrame]:
    if methods is None:
        methods = METHODS

    validate_methods_and_portfolios(monthly_returns, methods=methods)

    q5_returns: dict[str, pd.Series] = {}
    ls_returns: dict[str, pd.Series] = {}
    used_rows = []

    for method in methods:
        sub = monthly_returns.loc[monthly_returns["Method"] == method]
        q1 = (
            sub.loc[sub["Portfolio"] == "Q1", ["Date", "Return"]]
            .drop_duplicates("Date")
            .set_index("Date")["Return"]
            .sort_index()
        )
        q5 = (
            sub.loc[sub["Portfolio"] == "Q5", ["Date", "Return"]]
            .drop_duplicates("Date")
            .set_index("Date")["Return"]
            .sort_index()
        )

        aligned = pd.concat({"Q1": q1, "Q5": q5}, axis=1).dropna()
        aligned = aligned.loc[aligned.index.intersection(factors.index)].sort_index()
        if aligned.empty:
            raise ValueError(f"No factor-aligned Q1/Q5 observations for {method}.")

        q5_series = aligned["Q5"].rename(method)
        ls_series = (aligned["Q5"] - aligned["Q1"]).rename(method)
        q5_returns[method] = q5_series
        ls_returns[method] = ls_series

        used_rows.append(
            pd.DataFrame(
                {
                    "Date": q5_series.index,
                    "Method": method,
                    "PortfolioStrategy": "Q5",
                    "Return": q5_series.values,
                }
            )
        )
        used_rows.append(
            pd.DataFrame(
                {
                    "Date": ls_series.index,
                    "Method": method,
                    "PortfolioStrategy": "LongShort",
                    "Return": ls_series.values,
                }
            )
        )

    used = pd.concat(used_rows, ignore_index=True)
    used["Date"] = pd.to_datetime(used["Date"]).dt.strftime("%Y-%m-%d")
    return q5_returns, ls_returns, used


def run_level_regressions(
    strategy_returns: dict[str, pd.Series],
    factors: pd.DataFrame,
    rf: pd.Series,
    strategy_label: str,
    nw_lags: int,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    if methods is None:
        methods = METHODS

    rows = []
    for method in methods:
        for internal_model in INTERNAL_MODELS:
            res = risk_adjusted_performance(
                portfolio_returns=strategy_returns[method],
                factors=factors,
                rf=rf,
                models=[internal_model],
                lags=nw_lags,
            ).iloc[0]

            alpha_monthly = float(res["alpha"]) if pd.notna(res["alpha"]) else np.nan
            rows.append(
                {
                    "PortfolioStrategy": strategy_label,
                    "FactorModel": MODEL_LABELS[internal_model],
                    "Method": method,
                    "alpha_annualized": alpha_monthly * 12,
                    "alpha_monthly": alpha_monthly,
                    "t_stat": res["t_stat"],
                    "p_value": res["p_value"],
                    "r_squared": res["r_squared"],
                    "n_obs": int(res["n_obs"]),
                }
            )

    return pd.DataFrame(rows)


def run_alpha_difference_tests(
    strategy_returns: dict[str, pd.Series],
    factors: pd.DataFrame,
    rf: pd.Series,
    strategy_label: str,
    nw_lags: int,
    methods: list[str] | None = None,
    base_method: str = "Method1_ObservedQuality",
) -> pd.DataFrame:
    if methods is None:
        methods = METHODS

    frames = []
    for internal_model in INTERNAL_MODELS:
        res = alpha_differences(
            ls_returns={method: strategy_returns[method] for method in methods},
            factors=factors,
            rf=rf,
            model=internal_model,
            lags=nw_lags,
            base_method=base_method,
        )
        res["Comparison"] = res["Comparison"].replace(COMPARISON_LABELS)
        res = res.loc[res["Comparison"].isin(COMPARISONS)].copy()
        res["PortfolioStrategy"] = strategy_label
        res["FactorModel"] = MODEL_LABELS[internal_model]
        res["alpha_difference_annualized"] = res["alpha_difference"] * 12
        res = res.rename(columns={"alpha_difference": "alpha_difference_monthly"})
        frames.append(res)

    out = pd.concat(frames, ignore_index=True)
    out["Comparison"] = pd.Categorical(out["Comparison"], categories=COMPARISONS, ordered=True)
    out["FactorModel"] = pd.Categorical(
        out["FactorModel"],
        categories=list(MODEL_LABELS.values()),
        ordered=True,
    )
    return (
        out.sort_values(["FactorModel", "Comparison"])
        .reset_index(drop=True)
        [
            [
                "PortfolioStrategy",
                "FactorModel",
                "Comparison",
                "alpha_difference_annualized",
                "alpha_difference_monthly",
                "t_stat",
                "p_value",
            ]
        ]
    )


def run_grs_tests(
    strategy_returns: dict[str, pd.Series],
    factors: pd.DataFrame,
    rf: pd.Series,
    strategy_label: str,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classic Gibbons-Ross-Shanken joint alpha test across sorting methods.

    Pass the regular RF series for long-only portfolios. For self-financing
    long-short spreads, pass a zero RF series so the spread is not adjusted twice.
    """
    if methods is None:
        methods = METHODS

    missing_methods = [method for method in methods if method not in strategy_returns]
    if missing_methods:
        raise ValueError(f"Missing strategy returns for GRS methods: {missing_methods}")

    test_rows = []
    alpha_rows = []
    rf = rf.copy()
    rf.index = pd.to_datetime(rf.index)

    for internal_model in INTERNAL_MODELS:
        factor_cols = FACTOR_COLUMNS[internal_model]
        missing_factors = [col for col in factor_cols if col not in factors.columns]
        if missing_factors:
            raise ValueError(f"Factors missing columns for {internal_model}: {missing_factors}")

        factor_frame = factors[factor_cols].copy()
        factor_frame.index = pd.to_datetime(factor_frame.index)
        factor_frame = factor_frame.apply(pd.to_numeric, errors="coerce").dropna()

        excess_by_method = {}
        for method in methods:
            returns = strategy_returns[method].copy()
            returns.index = pd.to_datetime(returns.index)
            returns = pd.to_numeric(returns, errors="coerce")
            excess_by_method[method] = returns.subtract(rf.reindex(returns.index), fill_value=np.nan)

        excess = pd.concat(excess_by_method, axis=1).dropna()
        idx = factor_frame.index.intersection(excess.index).sort_values()
        factor_frame = factor_frame.loc[idx]
        excess = excess.loc[idx, methods]

        T = int(len(idx))
        N = int(len(methods))
        K = int(len(factor_cols))
        if N < 2:
            raise ValueError("The GRS test requires at least two test assets.")
        if T <= N + K:
            raise ValueError(
                f"Not enough observations for the GRS test: T={T}, N={N}, K={K}."
            )

        F = factor_frame.to_numpy(dtype=float)
        Y = excess.to_numpy(dtype=float)
        X = np.column_stack([np.ones(T), F])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        alpha_vec = beta[0, :]
        residuals = Y - X @ beta

        sigma = residuals.T @ residuals / T
        mu_f = F.mean(axis=0)
        f_demeaned = F - mu_f
        omega = f_demeaned.T @ f_demeaned / T

        sigma_inv = np.linalg.pinv(sigma)
        omega_inv = np.linalg.pinv(omega)
        kappa = 1.0 + float(mu_f @ omega_inv @ mu_f)

        grs_f = (
            (T / N)
            * ((T - N - K) / (T - K - 1))
            * float(alpha_vec @ sigma_inv @ alpha_vec)
            / kappa
        )
        p_value = 1.0 - stats.f.cdf(grs_f, dfn=N, dfd=T - N - K)

        factor_label = MODEL_LABELS[internal_model]
        test_rows.append(
            {
                "PortfolioStrategy": strategy_label,
                "FactorModel": factor_label,
                "grs_f_stat": float(grs_f),
                "p_value": float(p_value),
                "reject_h0_5pct": bool(p_value < 0.05),
                "n_obs": T,
                "n_test_assets": N,
                "n_factors": K,
                "test_assets": ";".join(methods),
            }
        )
        for method, alpha_monthly in zip(methods, alpha_vec):
            alpha_rows.append(
                {
                    "PortfolioStrategy": strategy_label,
                    "FactorModel": factor_label,
                    "Method": method,
                    "MethodLabel": METHOD_DISPLAY_LABELS.get(method, method),
                    "alpha_annualized": float(alpha_monthly * 12.0),
                    "alpha_monthly": float(alpha_monthly),
                }
            )

    grs_tests = pd.DataFrame(test_rows)
    grs_alpha_components = pd.DataFrame(alpha_rows)

    factor_order = pd.CategoricalDtype(list(MODEL_LABELS.values()), ordered=True)
    grs_tests["FactorModel"] = grs_tests["FactorModel"].astype(factor_order)
    grs_alpha_components["FactorModel"] = grs_alpha_components["FactorModel"].astype(factor_order)
    method_order = pd.CategoricalDtype(methods, ordered=True)
    grs_alpha_components["Method"] = grs_alpha_components["Method"].astype(method_order)

    return (
        grs_tests.sort_values(["PortfolioStrategy", "FactorModel"]).reset_index(drop=True),
        grs_alpha_components.sort_values(["PortfolioStrategy", "FactorModel", "Method"]).reset_index(drop=True),
    )


def build_preview(levels: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    level_wide = levels.pivot_table(
        index=["PortfolioStrategy", "FactorModel"],
        columns="Method",
        values=["alpha_annualized", "t_stat", "p_value"],
        aggfunc="first",
    )
    level_wide.columns = [f"{method}_{metric}" for metric, method in level_wide.columns]
    level_wide = level_wide.reset_index()

    diff_wide = differences.pivot_table(
        index=["PortfolioStrategy", "FactorModel"],
        columns="Comparison",
        values=["alpha_difference_annualized", "t_stat", "p_value"],
        aggfunc="first",
        observed=False,
    )
    diff_wide.columns = [
        f"{comparison}_{metric}".replace(" ", "_").replace("+", "plus")
        for metric, comparison in diff_wide.columns
    ]
    diff_wide = diff_wide.reset_index()

    return level_wide.merge(diff_wide, on=["PortfolioStrategy", "FactorModel"], how="left")


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def compute_raw_performance_table(
    monthly_used: pd.DataFrame,
    rf: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build table-ready unadjusted performance metrics from the exact monthly
    strategy returns used in regressions.

    Q5 is long-only, so excess returns subtract RF. LongShort is self-financing,
    so the strategy return is already an excess return.
    """
    data = monthly_used.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Return"] = pd.to_numeric(data["Return"], errors="coerce")
    data = data.dropna(subset=["Date", "Method", "PortfolioStrategy", "Return"])

    if rf is not None:
        rf_aligned = rf.copy()
        rf_aligned.index = pd.to_datetime(rf_aligned.index)
    else:
        rf_aligned = pd.Series(dtype=float)

    rows = []
    for (strategy, method), sub in data.groupby(["PortfolioStrategy", "Method"], sort=True):
        sub = sub.sort_values("Date")
        returns = sub.set_index("Date")["Return"].astype(float)

        if strategy == "LongShort":
            excess = returns.copy()
        elif not rf_aligned.empty:
            excess = returns.subtract(rf_aligned.reindex(returns.index), fill_value=np.nan)
        else:
            excess = returns.copy()

        returns = returns.dropna()
        excess = excess.dropna()

        if returns.empty:
            continue

        annualized_return = float(returns.mean() * 12.0)
        annualized_excess_return = float(excess.mean() * 12.0) if not excess.empty else np.nan
        volatility_ann = float(returns.std(ddof=1) * np.sqrt(12.0)) if len(returns) > 1 else np.nan
        excess_volatility_ann = (
            float(excess.std(ddof=1) * np.sqrt(12.0)) if len(excess) > 1 else np.nan
        )
        sharpe_ratio = (
            annualized_excess_return / excess_volatility_ann
            if pd.notna(excess_volatility_ann) and excess_volatility_ann > 0
            else np.nan
        )

        rows.append(
            {
                "PortfolioStrategy": strategy,
                "Method": method,
                "MethodLabel": METHOD_DISPLAY_LABELS.get(method, method),
                "annualized_return": annualized_return,
                "annualized_excess_return": annualized_excess_return,
                "volatility_ann": volatility_ann,
                "excess_volatility_ann": excess_volatility_ann,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown(returns),
                "n_obs": int(len(returns)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    strategy_order = pd.CategoricalDtype(["Q5", "LongShort"], ordered=True)
    method_order = pd.CategoricalDtype(METHODS, ordered=True)
    out["PortfolioStrategy"] = out["PortfolioStrategy"].astype(strategy_order)
    out["Method"] = out["Method"].astype(method_order)
    return out.sort_values(["PortfolioStrategy", "Method"]).reset_index(drop=True)


def build_cumulative_returns(monthly_used: pd.DataFrame) -> pd.DataFrame:
    cumulative = monthly_used.copy()
    cumulative["Date"] = pd.to_datetime(cumulative["Date"], errors="coerce")
    cumulative["Return"] = pd.to_numeric(cumulative["Return"], errors="coerce")
    cumulative = cumulative.dropna(subset=["Date", "Method", "PortfolioStrategy", "Return"])
    cumulative = cumulative.sort_values(["PortfolioStrategy", "Method", "Date"])
    cumulative["CumulativeReturn"] = (
        cumulative.groupby(["PortfolioStrategy", "Method"])["Return"]
        .transform(lambda s: (1.0 + s).cumprod() - 1.0)
    )
    return cumulative.reset_index(drop=True)


def save_cumulative_return_plots(monthly_used: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cumulative = build_cumulative_returns(monthly_used)
    outputs: dict[str, Path] = {}

    strategy_titles = {
        "LongShort": "Cumulative Returns: Long-Short Q5 - Q1",
        "Q5": "Cumulative Returns: Pure Q5",
    }

    for strategy, title in strategy_titles.items():
        sub = cumulative.loc[cumulative["PortfolioStrategy"] == strategy].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.axhline(0.0, color="#2f3b4a", linewidth=0.9, linestyle="--", alpha=0.75)

        for method in METHODS:
            method_sub = sub.loc[sub["Method"] == method].sort_values("Date")
            if method_sub.empty:
                continue
            ax.plot(
                method_sub["Date"],
                method_sub["CumulativeReturn"],
                label=method,
                color=METHOD_COLORS.get(method),
                linewidth=2.1,
            )

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

        filename = f"cumulative_returns_{strategy.lower()}.png"
        path = plot_dir / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs[f"cumulative_returns_{strategy.lower()}"] = path

    return outputs


def save_outputs(
    output_dir: Path,
    ls_levels: pd.DataFrame,
    ls_diffs: pd.DataFrame,
    q5_levels: pd.DataFrame,
    q5_diffs: pd.DataFrame,
    monthly_used: pd.DataFrame,
    preview: pd.DataFrame,
    rf: pd.Series | None = None,
    grs_tests: pd.DataFrame | None = None,
    grs_alpha_components: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_performance = compute_raw_performance_table(monthly_used=monthly_used, rf=rf)

    outputs = {
        "table_ls_alpha_levels": output_dir / "table_ls_alpha_levels.csv",
        "table_ls_alpha_differences": output_dir / "table_ls_alpha_differences.csv",
        "table_q5_alpha_levels": output_dir / "table_q5_alpha_levels.csv",
        "table_q5_alpha_differences": output_dir / "table_q5_alpha_differences.csv",
        "table_raw_performance": output_dir / "table_raw_performance.csv",
        "table_ls_raw_performance": output_dir / "table_ls_raw_performance.csv",
        "table_q5_raw_performance": output_dir / "table_q5_raw_performance.csv",
        "monthly_portfolio_returns_used": output_dir / "monthly_portfolio_returns_used.csv",
        "risk_adjusted_table_preview": output_dir / "risk_adjusted_table_preview.csv",
    }
    if grs_tests is not None:
        outputs.update(
            {
                "table_grs_tests": output_dir / "table_grs_tests.csv",
                "table_ls_grs_tests": output_dir / "table_ls_grs_tests.csv",
                "table_q5_grs_tests": output_dir / "table_q5_grs_tests.csv",
            }
        )
    if grs_alpha_components is not None:
        outputs["table_grs_alpha_components"] = output_dir / "table_grs_alpha_components.csv"

    ls_levels.to_csv(outputs["table_ls_alpha_levels"], index=False)
    ls_diffs.to_csv(outputs["table_ls_alpha_differences"], index=False)
    q5_levels.to_csv(outputs["table_q5_alpha_levels"], index=False)
    q5_diffs.to_csv(outputs["table_q5_alpha_differences"], index=False)
    raw_performance.to_csv(outputs["table_raw_performance"], index=False)
    raw_performance.loc[
        raw_performance["PortfolioStrategy"].astype(str) == "LongShort"
    ].to_csv(outputs["table_ls_raw_performance"], index=False)
    raw_performance.loc[
        raw_performance["PortfolioStrategy"].astype(str) == "Q5"
    ].to_csv(outputs["table_q5_raw_performance"], index=False)
    monthly_used.to_csv(outputs["monthly_portfolio_returns_used"], index=False)
    preview.to_csv(outputs["risk_adjusted_table_preview"], index=False)
    if grs_tests is not None:
        grs_tests.to_csv(outputs["table_grs_tests"], index=False)
        grs_tests.loc[
            grs_tests["PortfolioStrategy"].astype(str) == "LongShort"
        ].to_csv(outputs["table_ls_grs_tests"], index=False)
        grs_tests.loc[
            grs_tests["PortfolioStrategy"].astype(str) == "Q5"
        ].to_csv(outputs["table_q5_grs_tests"], index=False)
    if grs_alpha_components is not None:
        grs_alpha_components.to_csv(outputs["table_grs_alpha_components"], index=False)

    return outputs


def assert_expected_shapes(
    ls_levels: pd.DataFrame,
    ls_diffs: pd.DataFrame,
    q5_levels: pd.DataFrame,
    q5_diffs: pd.DataFrame,
    methods: list[str] | None = None,
) -> None:
    if methods is None:
        methods = METHODS

    n_level_rows = len(INTERNAL_MODELS) * len(methods)
    n_difference_rows = len(INTERNAL_MODELS) * (len(methods) - 1)
    expected = {
        "Long-short alpha levels": (ls_levels, n_level_rows),
        "Long-short alpha differences": (ls_diffs, n_difference_rows),
        "Q5 alpha levels": (q5_levels, n_level_rows),
        "Q5 alpha differences": (q5_diffs, n_difference_rows),
    }
    bad = [f"{name}: expected {n}, got {len(df)}" for name, (df, n) in expected.items() if len(df) != n]
    if bad:
        raise RuntimeError("Unexpected output row counts:\n" + "\n".join(bad))


def print_identification(
    run_dir: Path,
    portfolio_eval_dir: Path,
    portfolio_source: Path,
    source_type: str,
    factors_csv: Path,
    output_dir: Path,
    nw_lags: int,
) -> None:
    print("\nIdentified inputs and reused helpers")
    print(f"  run_dir: {run_dir}")
    print(f"  portfolio_evaluation_dir: {portfolio_eval_dir}")
    print(f"  portfolio constituent / weighted-return data: {portfolio_source}")
    print(f"  portfolio source type: {source_type}")
    print(f"  monthly factor returns: {factors_csv}")
    print("  risk-adjusted regression helper: modelling/shared/step5_evaluation.py::risk_adjusted_performance")
    print("  Newey-West/HAC helper: modelling/shared/step5_evaluation.py::_ols_newey_west_full")
    print("  alpha-difference helper: modelling/shared/step5_evaluation.py::alpha_differences")
    print("  factor loader helper: modelling/shared/helper_functions.py::load_factor_data")
    print(f"  HAC lags: {nw_lags}")
    print(f"  output_dir: {output_dir}")


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    run_dir = choose_run_dir(project_root, resolve_cli_path(args.run_dir, project_root))
    portfolio_source, source_type, portfolio_eval_dir = choose_portfolio_source(
        run_dir=run_dir,
        requested_source=resolve_cli_path(args.portfolio_source, project_root),
    )
    factors_csv = choose_factor_csv(
        project_root=project_root,
        run_dir=run_dir,
        requested_factors=resolve_cli_path(args.factors_csv, project_root),
    )
    output_dir = resolve_cli_path(args.output_dir, project_root)
    if output_dir is None:
        output_dir = portfolio_eval_dir / "thesis_risk_adjusted_tables"

    print_identification(
        run_dir=run_dir,
        portfolio_eval_dir=portfolio_eval_dir,
        portfolio_source=portfolio_source,
        source_type=source_type,
        factors_csv=factors_csv,
        output_dir=output_dir,
        nw_lags=args.nw_lags,
    )

    monthly_returns = load_monthly_portfolio_returns(portfolio_source, source_type)
    factors = load_factor_data(factors_csv)
    rf = factors["RF"].copy()
    zero_rf = pd.Series(0.0, index=rf.index, name="RF")

    q5_returns, ls_returns, monthly_used = build_strategy_returns(monthly_returns, factors)

    ls_levels = run_level_regressions(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
        nw_lags=args.nw_lags,
    )
    q5_levels = run_level_regressions(
        strategy_returns=q5_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q5",
        nw_lags=args.nw_lags,
    )
    ls_diffs = run_alpha_difference_tests(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
        nw_lags=args.nw_lags,
    )
    q5_diffs = run_alpha_difference_tests(
        strategy_returns=q5_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q5",
        nw_lags=args.nw_lags,
    )
    ls_grs, ls_grs_alpha = run_grs_tests(
        strategy_returns=ls_returns,
        factors=factors,
        rf=zero_rf,
        strategy_label="LongShort",
    )
    q5_grs, q5_grs_alpha = run_grs_tests(
        strategy_returns=q5_returns,
        factors=factors,
        rf=rf,
        strategy_label="Q5",
    )
    grs_tests = pd.concat([ls_grs, q5_grs], ignore_index=True)
    grs_alpha_components = pd.concat([ls_grs_alpha, q5_grs_alpha], ignore_index=True)

    assert_expected_shapes(ls_levels, ls_diffs, q5_levels, q5_diffs)
    preview = build_preview(
        levels=pd.concat([ls_levels, q5_levels], ignore_index=True),
        differences=pd.concat([ls_diffs, q5_diffs], ignore_index=True),
    )
    outputs = save_outputs(
        output_dir=output_dir,
        ls_levels=ls_levels,
        ls_diffs=ls_diffs,
        q5_levels=q5_levels,
        q5_diffs=q5_diffs,
        monthly_used=monthly_used,
        preview=preview,
        rf=rf,
        grs_tests=grs_tests,
        grs_alpha_components=grs_alpha_components,
    )
    plot_outputs = save_cumulative_return_plots(monthly_used=monthly_used, output_dir=output_dir)

    print("\nCreated CSV files")
    row_counts = {
        "table_ls_alpha_levels": len(ls_levels),
        "table_ls_alpha_differences": len(ls_diffs),
        "table_q5_alpha_levels": len(q5_levels),
        "table_q5_alpha_differences": len(q5_diffs),
        "table_grs_tests": len(grs_tests),
        "table_ls_grs_tests": len(ls_grs),
        "table_q5_grs_tests": len(q5_grs),
        "table_grs_alpha_components": len(grs_alpha_components),
        "monthly_portfolio_returns_used": len(monthly_used),
        "risk_adjusted_table_preview": len(preview),
    }
    for key, path in outputs.items():
        n_rows = row_counts.get(key)
        if n_rows is None and path.suffix.lower() == ".csv":
            n_rows = len(pd.read_csv(path))
        if n_rows is None:
            print(f"  {path}")
        else:
            print(f"  {path} ({n_rows} rows)")

    print("\nCreated plot files")
    for path in plot_outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
