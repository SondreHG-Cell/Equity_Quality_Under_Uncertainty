# run_main.py

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

from uncertainty_model import run_uncertainty_model
from latent_prof_model import DEFAULT_GAMMA as DEFAULT_LATENT_GAMMA
from latent_prof_model import run_latent_prof_model
from portfolio_formation import run_portfolio_formation
from portfolio_evaluation import run_portfolio_evaluation


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

@dataclass
class RunConfig:
    extracted_input_csv: str = "results/extraction_static/prepared_step2_input.csv"
    returns_csv: str = "data/processed_data_lseg/all_stock_prices_nok.csv"
    market_cap_csv: str = "data/processed_data_lseg/historical_market_cap_nok.csv"
    factors_csv: str = "results/extraction_static/factor_data.csv"
    results_root: str = "results"
    run_name: Optional[str] = None
    uncertainty_method: str = "OLS"
    n_portfolios: int = 5
    nw_lags: int = 12
    save_intermediate: bool = True
    
    # HB CFO handling
    hb_cfo_lead_mode: str = "none"   # legacy: "best_external" or "none"
    hb_cfo_t1_source: Optional[str] = None  # realized, analyst, hybrid, external, or none
    hb_use_analyst_cfo_forecast: bool = False
    hb_analyst_cfo_forecast_csv: Optional[str] = None
    hb_run_model_specification: str = "baseline"  # baseline, analyst_cfo, or both

    # OLS uncertainty settings aligned with the HB rolling-window design.
    ols_min_obs_per_year: int = 20
    ols_rolling_window: int = 5
    ols_min_periods_start: int = 4
    ols_sigma_history_start_year: int = 2004

    # Step 3 latent PROF settings
    latent_gamma: float = DEFAULT_LATENT_GAMMA
    latent_use_full_propagation: bool = False
    latent_n_sigma_draws: Optional[int] = None
    latent_checkpoint_every_draws: int = 25


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def find_project_root() -> Path:
    """
    Find the project root as the first parent containing a 'data' folder.
    """
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()

    for p in [here] + list(here.parents):
        if (p / "data").exists():
            return p

    raise FileNotFoundError("Could not find project root containing a 'data' folder.")


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    """
    Resolve a path relative to project_root unless already absolute.
    """
    p = Path(path_like)
    if p.is_absolute():
        return p
    return project_root / p


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(results_root: Path, run_name: Optional[str] = None) -> Path:
    stamp = now_timestamp()
    if run_name:
        run_dir = results_root / f"{stamp}_{run_name}"
    else:
        run_dir = results_root / f"{stamp}_run"

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_step_dirs(run_dir: Path) -> Dict[str, Path]:
    step_dirs = {
        "uncertainty_model": run_dir / "uncertainty_model",
        "latent_prof_model": run_dir / "latent_prof_model",
        "portfolio_formation": run_dir / "portfolio_formation",
        "portfolio_evaluation": run_dir / "portfolio_evaluation",
    }
    for p in step_dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return step_dirs


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_file_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def save_run_summary(
    summary_path: Path,
    config: RunConfig,
    run_dir: Path,
    status: str,
    durations: Dict[str, float],
    outputs: Dict[str, Any],
    error: Optional[str] = None,
) -> None:
    payload = {
        "status": status,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
        "config": asdict(config),
        "durations_seconds": durations,
        "outputs": outputs,
        "error": error,
    }
    write_json(summary_path, payload)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_pipeline(config: RunConfig) -> Path:
    project_root = find_project_root()

    results_root = resolve_path(config.results_root, project_root)
    results_root.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(results_root=results_root, run_name=config.run_name)
    step_dirs = make_step_dirs(run_dir)

    log_path = run_dir / "run_log.txt"
    summary_path = run_dir / "run_summary.json"
    config_path = run_dir / "run_config.json"

    write_json(config_path, asdict(config))

    durations: Dict[str, float] = {}
    outputs: Dict[str, Any] = {}

    append_log(log_path, f"Created run directory: {run_dir}")

    try:
        # ------------------------------------------
        # Validate inputs
        # ------------------------------------------
        extracted_input_csv = ensure_file_exists(
            resolve_path(config.extracted_input_csv, project_root),
            "Extracted input CSV",
        )
        returns_csv = ensure_file_exists(
            resolve_path(config.returns_csv, project_root),
            "Returns CSV",
        )
        market_cap_csv = ensure_file_exists(
            resolve_path(config.market_cap_csv, project_root),
            "Market cap CSV",
        )
        factors_csv = ensure_file_exists(
            resolve_path(config.factors_csv, project_root),
            "Factors CSV",
        )

        outputs["resolved_inputs"] = {
            "project_root": str(project_root),
            "extracted_input_csv": str(extracted_input_csv),
            "returns_csv": str(returns_csv),
            "market_cap_csv": str(market_cap_csv),
            "factors_csv": str(factors_csv),
        }

        append_log(log_path, "Input validation completed.")

        # ------------------------------------------
        # Step 1: uncertainty_model
        # ------------------------------------------
        append_log(log_path, "Starting Step 1: uncertainty_model")
        t0 = perf_counter()

        uncertainty_kwargs = {}

        if config.uncertainty_method.upper() == "HB":
            uncertainty_kwargs.update(
                {
                    "cfo_lead_mode": config.hb_cfo_lead_mode,
                    "cfo_t1_source": config.hb_cfo_t1_source,
                    "use_analyst_cfo_forecast": config.hb_use_analyst_cfo_forecast,
                    "analyst_cfo_forecast_csv": config.hb_analyst_cfo_forecast_csv,
                    "run_model_specification": config.hb_run_model_specification,
                }
            )
        elif config.uncertainty_method.upper() == "OLS":
            uncertainty_kwargs.update(
                {
                    "min_obs_per_year": config.ols_min_obs_per_year,
                    "rolling_window": config.ols_rolling_window,
                    "min_periods_start": config.ols_min_periods_start,
                    "sigma_history_start_year": config.ols_sigma_history_start_year,
                }
            )

        uncertainty_result = run_uncertainty_model(
            input_csv=extracted_input_csv,
            output_dir=step_dirs["uncertainty_model"],
            method=config.uncertainty_method,
            **uncertainty_kwargs,
        )

        durations["uncertainty_model"] = perf_counter() - t0
        uncertainty_csv = Path(uncertainty_result["firm_year_csv"])

        hb_full_posterior_parquet = None
        if config.uncertainty_method.upper() == "HB":
            hb_full_posterior_parquet = uncertainty_result.get("full_posterior_parquet")

        outputs["uncertainty_model"] = uncertainty_result

        append_log(log_path, f"Finished Step 1 in {durations['uncertainty_model']:.2f}s")

        # ------------------------------------------
        # Step 2: latent_prof_model
        # ------------------------------------------
        append_log(log_path, "Starting Step 2: latent_prof_model")
        t0 = perf_counter()

        if config.latent_use_full_propagation:
            if config.uncertainty_method.upper() != "HB":
                raise ValueError("Full propagation in Step 3 requires uncertainty_method='HB'.")
            if hb_full_posterior_parquet is None:
                raise ValueError(
                    "HB full propagation requested, but Step 2 did not return full_posterior_parquet."
                )

        latent_results = run_latent_prof_model(
            input_csv=uncertainty_csv,
            output_dir=step_dirs["latent_prof_model"],
            uncertainty_method=config.uncertainty_method,
            gamma=config.latent_gamma,
            use_full_propagation=config.latent_use_full_propagation,
            hb_full_posterior_parquet=hb_full_posterior_parquet,
            n_sigma_draws=config.latent_n_sigma_draws,
            checkpoint_every_draws=config.latent_checkpoint_every_draws,
        )

        durations["latent_prof_model"] = perf_counter() - t0
        outputs["latent_prof_model"] = latent_results

        append_log(log_path, f"Finished Step 2 in {durations['latent_prof_model']:.2f}s")
        append_log(log_path, f"Latent firm-year output: {latent_results['firm_year_csv']}")

        # ------------------------------------------
        # Step 3: portfolio_formation
        # ------------------------------------------
        append_log(log_path, "Starting Step 3: portfolio_formation")
        t0 = perf_counter()

        portfolio_result = run_portfolio_formation(
            input_csv=latent_results["firm_year_csv"],
            output_dir=step_dirs["portfolio_formation"],
            n_portfolios=config.n_portfolios,
        )

        durations["portfolio_formation"] = perf_counter() - t0
        outputs["portfolio_formation"] = portfolio_result
        append_log(log_path, f"Finished Step 3 in {durations['portfolio_formation']:.2f}s")

        # ------------------------------------------
        # Step 4: portfolio_evaluation
        # ------------------------------------------
        append_log(log_path, "Starting Step 4: portfolio_evaluation")
        t0 = perf_counter()

        portfolio_long_csv = Path(portfolio_result["portfolio_assignments_long_csv"])

        evaluation_result = run_portfolio_evaluation(
            assignments_csv=portfolio_long_csv,
            stock_prices_csv=returns_csv,
            market_cap_csv=market_cap_csv,
            factors_csv=factors_csv,
            output_dir=step_dirs["portfolio_evaluation"],
            n_portfolios=config.n_portfolios,
            nw_lags=config.nw_lags,
        )

        durations["portfolio_evaluation"] = perf_counter() - t0
        outputs["portfolio_evaluation"] = evaluation_result
        append_log(log_path, f"Finished Step 4 in {durations['portfolio_evaluation']:.2f}s")

        # ------------------------------------------
        # Done
        # ------------------------------------------
        durations["total"] = sum(durations.values())

        save_run_summary(
            summary_path=summary_path,
            config=config,
            run_dir=run_dir,
            status="success",
            durations=durations,
            outputs=outputs,
            error=None,
        )

        append_log(log_path, f"Run completed successfully in {durations['total']:.2f}s")
        append_log(log_path, f"Results stored in: {run_dir}")

        return run_dir

    except Exception as e:
        error_text = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        durations["total"] = sum(durations.values())

        save_run_summary(
            summary_path=summary_path,
            config=config,
            run_dir=run_dir,
            status="failed",
            durations=durations,
            outputs=outputs,
            error=error_text,
        )

        append_log(log_path, "Run failed.")
        append_log(log_path, error_text)

        raise


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full post-extraction pipeline: "
            "uncertainty_model -> latent_prof_model -> portfolio_formation -> portfolio_evaluation"
        )
    )

    parser.add_argument(
        "--extracted_input_csv",
        type=str,
        default=RunConfig.extracted_input_csv,
        help="Path to prepared_step2_input.csv.",
    )
    parser.add_argument(
        "--returns_csv",
        type=str,
        default=RunConfig.returns_csv,
        help="CSV containing stock prices / returns used in portfolio evaluation.",
    )
    parser.add_argument(
        "--market_cap_csv",
        type=str,
        default=RunConfig.market_cap_csv,
        help="CSV containing monthly market cap data used for monthly value-weighting.",
    )
    parser.add_argument(
        "--factors_csv",
        type=str,
        default=RunConfig.factors_csv,
        help="CSV containing factor returns.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=RunConfig.results_root,
        help="Root folder for storing timestamped run folders.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional custom suffix for the run folder.",
    )
    parser.add_argument(
        "--uncertainty_method",
        type=str,
        default=RunConfig.uncertainty_method,
        choices=["HB", "OLS"],
        help="Which uncertainty model to run.",
    )
    parser.add_argument(
        "--hb_cfo_lead_mode",
        type=str,
        default=RunConfig.hb_cfo_lead_mode,
        choices=["best_external", "none"],
        help="Legacy HB CFO_{t+1} handling when --hb_cfo_t1_source is not set.",
    )
    parser.add_argument(
        "--hb_cfo_t1_source",
        type=str,
        default=RunConfig.hb_cfo_t1_source,
        choices=["realized", "realised", "analyst", "analyst_cfo", "hybrid", "external", "none"],
        help=(
            "HB CFO_{t+1} source. 'hybrid' uses realized CFO_{t+1} for training rows "
            "and analyst forecasts for portfolio-year rows."
        ),
    )
    parser.add_argument(
        "--hb_use_analyst_cfo_forecast",
        action="store_true",
        default=RunConfig.hb_use_analyst_cfo_forecast,
        help="Use analyst CFO forecasts for HB CFO_{t+1}.",
    )
    parser.add_argument(
        "--hb_analyst_cfo_forecast_csv",
        type=str,
        default=RunConfig.hb_analyst_cfo_forecast_csv,
        help="Path to cfo_forecast_complete_cases_no_gaps_until_2024.csv.",
    )
    parser.add_argument(
        "--hb_run_model_specification",
        type=str,
        default=RunConfig.hb_run_model_specification,
        choices=["baseline", "analyst_cfo", "analystcfo", "both"],
        help="Run baseline HB, analyst-CFO HB, or both with comparison outputs.",
    )
    parser.add_argument(
        "--ols_min_obs_per_year",
        type=int,
        default=RunConfig.ols_min_obs_per_year,
        help="Minimum observations required to estimate an OLS year regression.",
    )
    parser.add_argument(
        "--ols_rolling_window",
        type=int,
        default=RunConfig.ols_rolling_window,
        help="Rolling residual window used for OLS sigma.",
    )
    parser.add_argument(
        "--ols_min_periods_start",
        type=int,
        default=RunConfig.ols_min_periods_start,
        help="Minimum residual observations required before OLS sigma is non-missing.",
    )
    parser.add_argument(
        "--ols_sigma_history_start_year",
        type=int,
        default=RunConfig.ols_sigma_history_start_year,
        help=(
            "First residual year allowed to enter OLS sigma history. "
            "Use 0 or a negative value to use all available history."
        ),
    )
    parser.add_argument(
        "--n_portfolios",
        type=int,
        default=RunConfig.n_portfolios,
        help="Number of portfolios to form.",
    )
    parser.add_argument(
        "--nw_lags",
        type=int,
        default=RunConfig.nw_lags,
        help="Newey-West lags for evaluation regressions.",
    )
    parser.add_argument(
        "--latent_gamma",
        type=float,
        default=RunConfig.latent_gamma,
        help=(
            "Penalty size for Method3_ConservativeQuality. "
            "theta_adj = theta_obs - gamma * (1 - lambda_i)."
        ),
    )
    parser.add_argument(
        "--latent_use_full_propagation",
        action="store_true",
        default=RunConfig.latent_use_full_propagation,
        help="If set, Step 3 uses HB full propagation from sigma_posteriors_full.parquet.",
    )
    parser.add_argument(
        "--latent_n_sigma_draws",
        type=int,
        default=None,
        help="Number of HB sigma draws to use in Step 3 full propagation. Default: all available.",
    )
    parser.add_argument(
        "--latent_checkpoint_every_draws",
        type=int,
        default=RunConfig.latent_checkpoint_every_draws,
        help="How often Step 3 full propagation reports progress.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RunConfig(
        extracted_input_csv=args.extracted_input_csv,
        returns_csv=args.returns_csv,
        market_cap_csv=args.market_cap_csv,
        factors_csv=args.factors_csv,
        results_root=args.results_root,
        run_name=args.run_name,
        uncertainty_method=args.uncertainty_method,
        n_portfolios=args.n_portfolios,
        nw_lags=args.nw_lags,
        hb_cfo_lead_mode=args.hb_cfo_lead_mode,
        hb_cfo_t1_source=args.hb_cfo_t1_source,
        hb_use_analyst_cfo_forecast=args.hb_use_analyst_cfo_forecast,
        hb_analyst_cfo_forecast_csv=args.hb_analyst_cfo_forecast_csv,
        hb_run_model_specification=args.hb_run_model_specification,
        ols_min_obs_per_year=args.ols_min_obs_per_year,
        ols_rolling_window=args.ols_rolling_window,
        ols_min_periods_start=args.ols_min_periods_start,
        ols_sigma_history_start_year=args.ols_sigma_history_start_year,
        latent_gamma=args.latent_gamma,
        latent_use_full_propagation=args.latent_use_full_propagation,
        latent_n_sigma_draws=args.latent_n_sigma_draws,
        latent_checkpoint_every_draws=args.latent_checkpoint_every_draws,
    )

    run_dir = run_pipeline(config)
    print(f"\nFinished. Results saved in:\n{run_dir}\n")


if __name__ == "__main__":
    main()
