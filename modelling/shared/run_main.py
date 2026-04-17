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


# --------------------------------------------------
# ASSUMED IMPORTS
# --------------------------------------------------
# Adjust these imports to match your actual files/functions.

from uncertainty_model import run_uncertainty_model
from latent_prof_model import run_latent_prof_model
from portfolio_formation import run_portfolio_formation
from portfolio_evaluation import run_portfolio_evaluation


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

@dataclass
class RunConfig:
    extracted_input_csv: str
    returns_csv: str
    factors_csv: str
    results_root: str = "results"
    run_name: Optional[str] = None
    uncertainty_method: str = "HB"      # or "OLS"
    n_portfolios: int = 5
    nw_lags: int = 12
    save_intermediate: bool = True


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

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


def ensure_file_exists(path: str | Path, label: str) -> Path:
    path = Path(path)
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
    results_root = Path(config.results_root)
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
        extracted_input_csv = ensure_file_exists(config.extracted_input_csv, "Extracted input CSV")
        returns_csv = ensure_file_exists(config.returns_csv, "Returns CSV")
        factors_csv = ensure_file_exists(config.factors_csv, "Factors CSV")

        append_log(log_path, "Input validation completed.")

        # ------------------------------------------
        # Step 1: uncertainty_model
        # ------------------------------------------
        append_log(log_path, "Starting Step 1: uncertainty_model")
        t0 = perf_counter()

        # Assumed behavior:
        # - reads extracted_input_csv
        # - writes outputs to step_dirs["uncertainty_model"]
        # - returns either:
        #   A) a DataFrame + saved files, or
        #   B) a dict with output paths, or
        #   C) nothing, but writes a known output file
        #
        # We assume here it returns a dict containing:
        # {"firm_year_csv": "..."}
        uncertainty_result = run_uncertainty_model(
            input_csv=extracted_input_csv,
            output_dir=step_dirs["uncertainty_model"],
            method=config.uncertainty_method,
        )

        durations["uncertainty_model"] = perf_counter() - t0

        # Adjust this line if your function returns something else
        uncertainty_csv = Path(uncertainty_result["firm_year_csv"])

        outputs["uncertainty_model"] = {
            "output_dir": str(step_dirs["uncertainty_model"]),
            "firm_year_csv": str(uncertainty_csv),
        }

        append_log(log_path, f"Finished Step 1 in {durations['uncertainty_model']:.2f}s")

        # ------------------------------------------
        # Step 2: latent_prof_model
        # ------------------------------------------
        append_log(log_path, "Starting Step 2: latent_prof_model")
        t0 = perf_counter()

        # Assumed return:
        # {"firm_year_csv": "..."}
        latent_result = run_latent_prof_model(
            input_csv=uncertainty_csv,
            output_dir=step_dirs["latent_prof_model"],
            uncertainty_method=config.uncertainty_method,
        )

        durations["latent_prof_model"] = perf_counter() - t0

        # Adjust if needed
        latent_csv = Path(latent_result["firm_year_csv"])

        outputs["latent_prof_model"] = {
            "output_dir": str(step_dirs["latent_prof_model"]),
            "firm_year_csv": str(latent_csv),
        }

        append_log(log_path, f"Finished Step 2 in {durations['latent_prof_model']:.2f}s")

        # ------------------------------------------
        # Step 3: portfolio_formation
        # ------------------------------------------
        append_log(log_path, "Starting Step 3: portfolio_formation")
        t0 = perf_counter()

        # Assumed return from your portfolio_formation.py:
        # long_df, wide_df, summary_df
        # and that it saves:
        # - portfolio_assignments_long.csv
        # - portfolio_assignments_wide.csv
        # - portfolio_formation_summary.csv
        portfolio_result = run_portfolio_formation(
            input_csv=latent_csv,
            output_dir=step_dirs["portfolio_formation"],
            n_portfolios=config.n_portfolios,
        )

        portfolio_long_csv = Path(portfolio_result["portfolio_assignments_long_csv"])
        portfolio_wide_csv = Path(portfolio_result["portfolio_assignments_wide_csv"])
        portfolio_summary_csv = Path(portfolio_result["portfolio_summary_csv"])

        outputs["portfolio_formation"] = portfolio_result

        append_log(log_path, f"Finished Step 3 in {durations['portfolio_formation']:.2f}s")

        # ------------------------------------------
        # Step 4: portfolio_evaluation
        # ------------------------------------------
        append_log(log_path, "Starting Step 4: portfolio_evaluation")
        t0 = perf_counter()

        # Assumed behavior:
        # - merges portfolio assignments with returns
        # - evaluates raw and risk-adjusted performance
        # - saves results in portfolio_evaluation folder
        # - returns a dict with main output paths
        evaluation_result = run_portfolio_evaluation(
            assignments_csv=portfolio_long_csv,
            returns_csv=returns_csv,
            factors_csv=factors_csv,
            output_dir=step_dirs["portfolio_evaluation"],
            n_portfolios=config.n_portfolios,
            nw_lags=config.nw_lags,
        )

        durations["portfolio_evaluation"] = perf_counter() - t0

        outputs["portfolio_evaluation"] = {
            "output_dir": str(step_dirs["portfolio_evaluation"]),
            **evaluation_result,
        }

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
        description="Run the full post-extraction pipeline: uncertainty_model -> latent_prof_model -> portfolio_formation -> portfolio_evaluation"
    )

    parser.add_argument(
        "--extracted_input_csv",
        type=str,
        required=True,
        help="CSV produced by the extraction pipeline, ready for step 2.",
    )
    parser.add_argument(
        "--returns_csv",
        type=str,
        required=True,
        help="CSV containing stock return data used in portfolio evaluation.",
    )
    parser.add_argument(
        "--factors_csv",
        type=str,
        required=True,
        help="CSV containing factor returns (e.g. CAPM/FF/Carhart inputs).",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
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
        default="HB",
        choices=["HB", "OLS"],
        help="Which uncertainty model to run.",
    )
    parser.add_argument(
        "--n_portfolios",
        type=int,
        default=5,
        help="Number of portfolios to form.",
    )
    parser.add_argument(
        "--nw_lags",
        type=int,
        default=12,
        help="Newey-West lags for evaluation regressions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RunConfig(
        extracted_input_csv=args.extracted_input_csv,
        returns_csv=args.returns_csv,
        factors_csv=args.factors_csv,
        results_root=args.results_root,
        run_name=args.run_name,
        uncertainty_method=args.uncertainty_method,
        n_portfolios=args.n_portfolios,
        nw_lags=args.nw_lags,
    )

    run_dir = run_pipeline(config)
    print(f"\nFinished. Results saved in:\n{run_dir}\n")


if __name__ == "__main__":
    main()