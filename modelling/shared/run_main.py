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
    uncertainty_method: str = "HB"
    n_portfolios: int = 5
    nw_lags: int = 12
    save_intermediate: bool = True

    # HB CFO handling
    hb_cfo_lead_mode: str = "none"   # "best_external" or "none"

    # Step 3 full propagation settings
    latent_use_full_propagation: bool = True
    latent_n_sigma_draws: Optional[int] = None
    latent_checkpoint_every_draws: int = 50


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

        uncertainty_result = run_uncertainty_model(
            input_csv=extracted_input_csv,
            output_dir=step_dirs["uncertainty_model"],
            method=config.uncertainty_method,
            cfo_lead_mode=config.hb_cfo_lead_mode,
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
            use_full_propagation=config.latent_use_full_propagation,
            hb_full_posterior_parquet=hb_full_posterior_parquet,
            n_sigma_draws=config.latent_n_sigma_draws,
            checkpoint_every_draws=config.latent_checkpoint_every_draws,
        )

        durations["latent_prof_model"] = perf_counter() - t0
        outputs["latent_prof_model"] = latent_results

        append_log(log_path, f"Finished Step 2 in {durations['latent_prof_model']:.2f}s")
        append_log(log_path, f"Latent variants produced: {', '.join(latent_results.keys())}")

        # ------------------------------------------
        # Steps 3-4: portfolio_formation + portfolio_evaluation
        # for each latent variant
        # ------------------------------------------
        append_log(log_path, "Starting Steps 3-4 for all latent variants")

        portfolio_results: Dict[str, Any] = {}
        evaluation_results: Dict[str, Any] = {}
        variant_durations: Dict[str, Dict[str, float]] = {}

        t0_pf_total = perf_counter()

        for variant_name, latent_variant_result in latent_results.items():
            append_log(log_path, f"Starting downstream pipeline for variant: {variant_name}")

            variant_pf_dir = step_dirs["portfolio_formation"] / variant_name
            variant_eval_dir = step_dirs["portfolio_evaluation"] / variant_name

            # --------------------------------------
            # Step 3: portfolio_formation
            # --------------------------------------
            t_pf = perf_counter()

            portfolio_result = run_portfolio_formation(
                input_csv=latent_variant_result["firm_year_csv"],
                output_dir=variant_pf_dir,
                n_portfolios=config.n_portfolios,
            )

            pf_seconds = perf_counter() - t_pf
            portfolio_results[variant_name] = portfolio_result

            append_log(
                log_path,
                f"Finished Step 3 for {variant_name} in {pf_seconds:.2f}s"
            )

            # --------------------------------------
            # Step 4: portfolio_evaluation
            # --------------------------------------
            t_eval = perf_counter()

            portfolio_long_csv = Path(portfolio_result["portfolio_assignments_long_csv"])

            evaluation_result = run_portfolio_evaluation(
                assignments_csv=portfolio_long_csv,
                stock_prices_csv=returns_csv,
                market_cap_csv=market_cap_csv,
                factors_csv=factors_csv,
                output_dir=variant_eval_dir,
                n_portfolios=config.n_portfolios,
                nw_lags=config.nw_lags,
            )

            eval_seconds = perf_counter() - t_eval
            evaluation_results[variant_name] = evaluation_result

            variant_durations[variant_name] = {
                "portfolio_formation": pf_seconds,
                "portfolio_evaluation": eval_seconds,
                "downstream_total": pf_seconds + eval_seconds,
            }

            append_log(
                log_path,
                f"Finished Step 4 for {variant_name} in {eval_seconds:.2f}s"
            )

        downstream_total = perf_counter() - t0_pf_total
        durations["portfolio_formation_and_evaluation_total"] = downstream_total

        outputs["portfolio_formation"] = portfolio_results
        outputs["portfolio_evaluation"] = evaluation_results
        outputs["variant_durations_seconds"] = variant_durations

        append_log(
            log_path,
            f"Finished downstream steps for all variants in {downstream_total:.2f}s"
        )

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
        help="How HB Step 2 should handle CFO_{t+1}.",
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
        "--latent_use_full_propagation",
        action="store_true",
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
        default=25,
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
        latent_use_full_propagation=args.latent_use_full_propagation,
        latent_n_sigma_draws=args.latent_n_sigma_draws,
        latent_checkpoint_every_draws=args.latent_checkpoint_every_draws,
    )

    run_dir = run_pipeline(config)
    print(f"\nFinished. Results saved in:\n{run_dir}\n")


if __name__ == "__main__":
    main()