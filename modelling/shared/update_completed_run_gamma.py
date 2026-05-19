from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from portfolio_evaluation import run_portfolio_evaluation
from portfolio_formation import run_portfolio_formation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update Conservative Quality gamma in an existing completed run and "
            "regenerate downstream portfolio formation/evaluation outputs."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed run directory, e.g. results/current_res.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.40,
        help="New Conservative Quality gamma. Default: 0.40.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.06,
        help="Kappa value to record in configs. The latent posterior is not recomputed. Default: 0.06.",
    )
    parser.add_argument(
        "--n-portfolios",
        type=int,
        default=5,
        help="Number of quantile portfolios. Default: 5.",
    )
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=12,
        help="Newey-West lags for portfolio evaluation. Default: 12.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only update latent/config and portfolio formation, skipping monthly portfolio evaluation.",
    )
    return parser.parse_args()


def resolve(path: Path, root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def path_from_config(config: dict[str, Any], key: str, default: str) -> Path:
    return resolve(Path(config.get(key, default)))


def update_config_files(run_dir: Path, gamma: float, kappa: float) -> None:
    run_config_path = run_dir / "run_config.json"
    run_config = read_json(run_config_path)
    if run_config:
        run_config["latent_gamma"] = gamma
        run_config["latent_noise_share_of_prof_var"] = kappa
        write_json(run_config_path, run_config)

    run_summary_path = run_dir / "run_summary.json"
    run_summary = read_json(run_summary_path)
    if run_summary:
        config = run_summary.setdefault("config", {})
        config["latent_gamma"] = gamma
        config["latent_noise_share_of_prof_var"] = kappa
        write_json(run_summary_path, run_summary)

    latent_config_path = run_dir / "latent_prof_model" / "latent_prof_config.json"
    latent_config = read_json(latent_config_path)
    if latent_config:
        old_gamma = latent_config.get("gamma")
        latent_config["gamma"] = gamma
        latent_config["noise_share_of_prof_var"] = kappa
        if old_gamma != gamma:
            latent_config["gamma_update_note"] = (
                f"Updated downstream Conservative Quality outputs to gamma={gamma:g}; "
                "kappa unchanged at 0.06."
            )
        write_json(latent_config_path, latent_config)


def update_latent_file(run_dir: Path, gamma: float) -> None:
    latent_path = run_dir / "latent_prof_model" / "latent_prof_firm_year.csv"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_path}")

    config = read_json(run_dir / "latent_prof_model" / "latent_prof_config.json")
    old_gamma = float(config.get("gamma", 0.20) or 0.20)

    df = pd.read_csv(latent_path)
    required = {"theta_obs", "lambda_i"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{latent_path} is missing required columns: {sorted(missing)}")

    df["theta_conservative"] = df["theta_obs"].astype(float) - gamma * (1.0 - df["lambda_i"].astype(float))

    sd_col = "theta_conservative_sd_between"
    if sd_col in df.columns and old_gamma > 0:
        df[sd_col] = pd.to_numeric(df[sd_col], errors="coerce") * (gamma / old_gamma)

    df.to_csv(latent_path, index=False)


def regenerate_downstream(run_dir: Path, gamma: float, kappa: float, n_portfolios: int, nw_lags: int, skip_evaluation: bool) -> None:
    update_latent_file(run_dir=run_dir, gamma=gamma)
    update_config_files(run_dir=run_dir, gamma=gamma, kappa=kappa)

    formation = run_portfolio_formation(
        input_csv=run_dir / "latent_prof_model" / "latent_prof_firm_year.csv",
        output_dir=run_dir / "portfolio_formation",
        n_portfolios=n_portfolios,
    )

    if skip_evaluation:
        return

    config = read_json(run_dir / "run_config.json")
    run_portfolio_evaluation(
        assignments_csv=formation["portfolio_assignments_long_csv"],
        stock_prices_csv=path_from_config(
            config,
            "returns_csv",
            "data/processed_data_lseg/all_stock_prices_nok.csv",
        ),
        dividends_csv=path_from_config(
            config,
            "dividends_csv",
            "data/processed_data_lseg/dividends_monthly_nok.csv",
        ),
        market_cap_csv=path_from_config(
            config,
            "market_cap_csv",
            "data/processed_data_lseg/historical_market_cap_nok.csv",
        ),
        factors_csv=path_from_config(
            config,
            "factors_csv",
            "results/extraction_static/factor_data.csv",
        ),
        output_dir=run_dir / "portfolio_evaluation",
        n_portfolios=n_portfolios,
        nw_lags=nw_lags,
    )


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    regenerate_downstream(
        run_dir=run_dir,
        gamma=args.gamma,
        kappa=args.kappa,
        n_portfolios=args.n_portfolios,
        nw_lags=args.nw_lags,
        skip_evaluation=args.skip_evaluation,
    )
    print(f"Updated {run_dir} to gamma={args.gamma:g}, kappa={args.kappa:g}")


if __name__ == "__main__":
    main()
