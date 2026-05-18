from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from portfolio_evaluation import run_portfolio_evaluation
from portfolio_formation import run_portfolio_formation


DEFAULT_GAMMAS = [0.40, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Conservative Quality gamma robustness runs from an existing "
            "latent profitability output."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results/current_res",
        help="Run directory containing the main latent_prof_model output.",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=DEFAULT_GAMMAS,
        help="Gamma values to generate. Default: 0.40 0.50.",
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
        help="Newey-West lags for portfolio evaluation and table generation. Default: 12.",
    )
    return parser.parse_args()


def resolve(path: Path, root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else root / path


def gamma_label(gamma: float) -> str:
    return f"{gamma:.2f}".replace(".", "_")


def load_run_config(source_run_dir: Path) -> dict:
    config_path = source_run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def path_from_config(config: dict, key: str, default: str) -> Path:
    return resolve(Path(config.get(key, default)))


def write_gamma_latent_file(source_run_dir: Path, gamma: float, output_path: Path) -> None:
    source_path = source_run_dir / "latent_prof_model" / "latent_prof_firm_year.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source latent firm-year file: {source_path}")

    df = pd.read_csv(source_path)
    required = {"theta_obs", "lambda_i"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Source latent file is missing columns needed for gamma update: {sorted(missing)}")

    df["theta_conservative"] = df["theta_obs"].astype(float) - gamma * (1.0 - df["lambda_i"].astype(float))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_metadata(
    source_run_dir: Path,
    gamma: float,
    latent_path: Path,
    inputs_dir: Path,
    config: dict,
) -> None:
    metadata = {
        "source_run_dir": str(source_run_dir.resolve()),
        "source_latent_prof_firm_year": str((source_run_dir / "latent_prof_model" / "latent_prof_firm_year.csv").resolve()),
        "gamma_for_other_methods": config.get("latent_gamma"),
        "gamma": gamma,
        "conservative_quality_formula": "theta_obs - gamma * (1 - lambda_i)",
        "kappa_noise_share_of_prof_var": config.get("latent_noise_share_of_prof_var"),
        "latent_prof_firm_year": str(latent_path.resolve()),
    }
    with (inputs_dir / "conservative_gamma_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_table_generation(inputs_dir: Path, output_dir: Path, nw_lags: int) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_capped_weight_risk_adjusted_table_data.py"),
        "--run-dir",
        str(inputs_dir),
        "--output-dir",
        str(output_dir),
        "--nw-lags",
        str(nw_lags),
    ]
    subprocess.run(cmd, check=True)


def generate_gamma_run(source_run_dir: Path, gamma: float, n_portfolios: int, nw_lags: int) -> Path:
    config = load_run_config(source_run_dir)
    portfolio_eval_dir = source_run_dir / "portfolio_evaluation"
    output_dir = (
        portfolio_eval_dir
        / f"thesis_risk_adjusted_tables_conservative_gamma_{gamma_label(gamma)}_ucits_5_10_40"
    )
    inputs_dir = output_dir / "inputs"
    formation_dir = inputs_dir / "portfolio_formation"
    evaluation_dir = inputs_dir / "portfolio_evaluation"
    latent_path = inputs_dir / f"latent_prof_firm_year_conservative_gamma_{gamma_label(gamma)}.csv"

    write_gamma_latent_file(source_run_dir=source_run_dir, gamma=gamma, output_path=latent_path)
    write_metadata(
        source_run_dir=source_run_dir,
        gamma=gamma,
        latent_path=latent_path,
        inputs_dir=inputs_dir,
        config=config,
    )

    formation_result = run_portfolio_formation(
        input_csv=latent_path,
        output_dir=formation_dir,
        n_portfolios=n_portfolios,
    )
    run_portfolio_evaluation(
        assignments_csv=formation_result["portfolio_assignments_long_csv"],
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
        output_dir=evaluation_dir,
        n_portfolios=n_portfolios,
        nw_lags=nw_lags,
    )
    run_table_generation(inputs_dir=inputs_dir, output_dir=output_dir, nw_lags=nw_lags)
    return output_dir


def main() -> None:
    args = parse_args()
    source_run_dir = resolve(args.source_run_dir)

    created = []
    for gamma in args.gammas:
        out = generate_gamma_run(
            source_run_dir=source_run_dir,
            gamma=float(gamma),
            n_portfolios=args.n_portfolios,
            nw_lags=args.nw_lags,
        )
        created.append(out)
        print(f"Generated Conservative Quality gamma robustness run: gamma={gamma:.2f} -> {out}")

    print("\nCompleted gamma robustness generation.")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
