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

from latent_prof_model import run_latent_prof_model
from portfolio_evaluation import run_portfolio_evaluation
from portfolio_formation import run_portfolio_formation


DEFAULT_KAPPAS = [0.08, 0.06, 0.04]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Latent Quality kappa robustness runs from an existing "
            "uncertainty-model output."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results/current_res",
        help="Run directory containing uncertainty_model/uncertainty_firm_year.csv.",
    )
    parser.add_argument(
        "--kappas",
        nargs="+",
        type=float,
        default=DEFAULT_KAPPAS,
        help="Kappa values to generate. Default: 0.08 0.06 0.04.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.40,
        help="Conservative Quality gamma to use in each kappa run. Default: 0.40.",
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


def kappa_label(kappa: float) -> str:
    return f"{kappa:.2f}".replace(".", "_")


def load_run_config(source_run_dir: Path) -> dict:
    config_path = source_run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def path_from_config(config: dict, key: str, default: str) -> Path:
    return resolve(Path(config.get(key, default)))


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


def write_metadata(
    source_run_dir: Path,
    kappa: float,
    gamma: float,
    latent_path: Path,
    inputs_dir: Path,
    config: dict,
) -> None:
    payload = {
        "source_run_dir": str(source_run_dir.resolve()),
        "source_uncertainty_firm_year": str(
            (source_run_dir / "uncertainty_model" / "uncertainty_firm_year.csv").resolve()
        ),
        "gamma_for_other_methods": gamma,
        "kappa_noise_share_of_prof_var": kappa,
        "kappa_formula": "obs_var_i = kappa * var_obs_t * (sigma_i / median_sigma_t)^2",
        "latent_quality_formula": "theta_post_mean = lambda_i * theta_obs + (1 - lambda_i) * mu_t",
        "main_kappa_noise_share_of_prof_var": config.get("latent_noise_share_of_prof_var"),
        "latent_prof_firm_year": str(latent_path.resolve()),
    }
    with (inputs_dir / "latent_kappa_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_kappa_run(source_run_dir: Path, kappa: float, gamma: float, n_portfolios: int, nw_lags: int) -> Path:
    config = load_run_config(source_run_dir)
    uncertainty_csv = source_run_dir / "uncertainty_model" / "uncertainty_firm_year.csv"
    if not uncertainty_csv.exists():
        raise FileNotFoundError(f"Missing source uncertainty firm-year file: {uncertainty_csv}")

    portfolio_eval_dir = source_run_dir / "portfolio_evaluation"
    output_dir = portfolio_eval_dir / f"thesis_risk_adjusted_tables_latent_kappa_{kappa_label(kappa)}_ucits_5_10_40"
    inputs_dir = output_dir / "inputs"
    latent_work_dir = inputs_dir / "latent_prof_model"
    formation_dir = inputs_dir / "portfolio_formation"
    evaluation_dir = inputs_dir
    latent_path = inputs_dir / f"latent_prof_firm_year_latent_kappa_{kappa_label(kappa)}.csv"

    latent_result = run_latent_prof_model(
        input_csv=uncertainty_csv,
        output_dir=latent_work_dir,
        uncertainty_method=config.get("uncertainty_method", "HB"),
        gamma=gamma,
        noise_share_of_prof_var=kappa,
        use_full_propagation=False,
        hb_full_posterior_parquet=None,
        n_sigma_draws=None,
        checkpoint_every_draws=50,
    )
    generated_latent = Path(latent_result["firm_year_csv"])
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pd.read_csv(generated_latent).to_csv(latent_path, index=False)
    write_metadata(
        source_run_dir=source_run_dir,
        kappa=kappa,
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
    for kappa in args.kappas:
        out = generate_kappa_run(
            source_run_dir=source_run_dir,
            kappa=float(kappa),
            gamma=args.gamma,
            n_portfolios=args.n_portfolios,
            nw_lags=args.nw_lags,
        )
        created.append(out)
        print(f"Generated Latent Quality kappa robustness run: kappa={kappa:.2f} -> {out}")

    print("\nCompleted kappa robustness generation.")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
