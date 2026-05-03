# uncertainty_model.py

from __future__ import annotations

import argparse
from pathlib import Path

from uncertainty_model_hb import run_uncertainty_model_hb
from uncertainty_model_ols import run_uncertainty_model_ols


def run_uncertainty_model(
    input_csv: str | Path,
    output_dir: str | Path,
    method: str = "HB",
    **kwargs,
) -> dict:
    """
    Step 2 wrapper.

    Dispatches to the chosen uncertainty model implementation.
    """
    method = method.upper()

    if method == "HB":
        return run_uncertainty_model_hb(
            input_csv=input_csv,
            output_dir=output_dir,
            **kwargs,
        )

    if method == "OLS":
        return run_uncertainty_model_ols(
            input_csv=input_csv,
            output_dir=output_dir,
            **kwargs,
        )

    raise ValueError(
        f"Unknown uncertainty method: {method}. Use 'HB' or 'OLS'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 2 uncertainty model.")

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Prepared firm-year panel from the extraction step.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save Step 2 outputs.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="HB",
        choices=["HB", "OLS"],
        help="Which uncertainty model to run.",
    )

    # HB-specific passthrough arguments
    parser.add_argument("--model_name", type=str, default="ar1")
    parser.add_argument("--year_start", type=int, default=2009)
    parser.add_argument("--year_end", type=int, default=2025)
    parser.add_argument("--n_draws", type=int, default=2000)
    parser.add_argument("--n_tune", type=int, default=4000)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.95)
    parser.add_argument("--min_train_years", type=int, default=3)
    parser.add_argument("--max_train_years", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--no_full_posteriors", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--cfo_draws", type=int, default=1000)
    parser.add_argument("--cfo_tune", type=int, default=1500)
    parser.add_argument("--cfo_prediction_mode", type=str, default="mean", choices=["mean", "draw"])
    parser.add_argument(
        "--cfo_lead_mode",
        type=str,
        default="none",
        choices=["best_external", "none"],
        help=(
            "Legacy CFO_{t+1} handling when --cfo_t1_source is unset. "
            "Default 'none' matches the no-look-ahead main specification."
        ),
    )
    parser.add_argument(
        "--cfo_t1_source",
        type=str,
        default=None,
        choices=["realized", "realised", "analyst", "analyst_cfo", "hybrid", "external", "none"],
        help=(
            "Source for CFO_{t+1} in HB: none, analyst, hybrid, external, or explicit realized. "
            "Use realized only for diagnostics/backtests because it can create look-ahead bias "
            "in portfolio-year rows. Hybrid uses realized CFO_{t+1} in training rows and "
            "analyst forecasts in portfolio-year rows."
        ),
    )
    parser.add_argument(
        "--use_analyst_cfo_forecast",
        action="store_true",
        help="Use analyst CFO forecasts for CFO_{t+1}; defaults to hybrid source handling.",
    )
    parser.add_argument(
        "--analyst_cfo_forecast_csv",
        type=str,
        default=None,
        help="Path to analyst CFO forecast CSV.",
    )
    parser.add_argument(
        "--run_model_specification",
        type=str,
        default="baseline",
        choices=["baseline", "analyst_cfo", "analystcfo", "both"],
        help=(
            "Run baseline, analyst-CFO, or both HB specifications. 'analyst_cfo' defaults "
            "to hybrid CFO handling; 'both' compares analyst-CFO hybrid with a no-lead HB "
            "model matched to the analyst-CFO estimation sample."
        ),
    )

    # OLS-specific passthrough arguments
    parser.add_argument("--min_obs_per_year", type=int, default=20)
    parser.add_argument("--rolling_window", type=int, default=5)
    parser.add_argument("--min_periods_start", type=int, default=4)
    parser.add_argument(
        "--sigma_history_start_year",
        type=int,
        default=2004,
        help=(
            "First residual year allowed to enter OLS sigma history. "
            "Use 0 or a negative value to use all available history."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    kwargs = {}

    if args.method.upper() == "HB":
        kwargs = {
            "model_name": args.model_name,
            "year_start": args.year_start,
            "year_end": args.year_end,
            "n_draws": args.n_draws,
            "n_tune": args.n_tune,
            "n_chains": args.n_chains,
            "target_accept": args.target_accept,
            "min_train_years": args.min_train_years,
            "max_train_years": args.max_train_years,
            "random_seed": args.random_seed,
            "save_full_posteriors": not args.no_full_posteriors,
            "save_plots": not args.no_plots,
            "cfo_draws": args.cfo_draws,
            "cfo_tune": args.cfo_tune,
            "cfo_prediction_mode": args.cfo_prediction_mode,
            "cfo_lead_mode": args.cfo_lead_mode,
            "cfo_t1_source": args.cfo_t1_source,
            "use_analyst_cfo_forecast": args.use_analyst_cfo_forecast,
            "analyst_cfo_forecast_csv": args.analyst_cfo_forecast_csv,
            "run_model_specification": args.run_model_specification,
        }

    elif args.method.upper() == "OLS":
        kwargs = {
            "min_obs_per_year": args.min_obs_per_year,
            "rolling_window": args.rolling_window,
            "min_periods_start": args.min_periods_start,
            "sigma_history_start_year": args.sigma_history_start_year,
        }

    result = run_uncertainty_model(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        method=args.method,
        **kwargs,
    )

    print("\nSaved Step 2 outputs:")
    for key, value in result.items():
        print(f"  {key}: {value}")
