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


DEFAULT_KAPPAS_P = [0.04, 0.06, 0.08, 0.10, 0.12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Probabilistic Quality kappa_P robustness runs while keeping "
            "Observed, Latent, and Conservative Quality at the main calibration."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results/current_res",
        help="Run directory containing the main latent_prof_model and uncertainty_model outputs.",
    )
    parser.add_argument(
        "--kappas-p",
        nargs="+",
        type=float,
        default=DEFAULT_KAPPAS_P,
        help="kappa_P values to generate. Default: 0.04 0.06 0.08 0.10 0.12.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.40,
        help="Gamma used when creating the temporary probability run. Default: 0.40.",
    )
    parser.add_argument(
        "--main-kappa",
        type=float,
        default=0.06,
        help="Main kappa used for Latent Quality and Conservative Quality. Default: 0.06.",
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
    kappa_p: float,
    main_kappa: float,
    gamma: float,
    latent_path: Path,
    inputs_dir: Path,
    config: dict,
) -> None:
    payload = {
        "source_run_dir": str(source_run_dir.resolve()),
        "source_main_latent_prof_firm_year": str(
            (source_run_dir / "latent_prof_model" / "latent_prof_firm_year.csv").resolve()
        ),
        "source_uncertainty_firm_year": str(
            (source_run_dir / "uncertainty_model" / "uncertainty_firm_year.csv").resolve()
        ),
        "main_kappa_for_latent_and_conservative": main_kappa,
        "kappa_p_for_probabilistic_quality": kappa_p,
        "gamma_for_conservative_quality": gamma,
        "probabilistic_quality_note": (
            "Only p_q5, p_median, and p_q1 are replaced with probabilities "
            "computed under kappa_P. The observed, latent, and conservative "
            "signals are kept at the main calibration."
        ),
        "kappa_p_formula": "obs_var_i = kappa_P * var_obs_t * (sigma_i / median_sigma_t)^2",
        "latent_prof_firm_year": str(latent_path.resolve()),
        "main_run_config_kappa": config.get("latent_noise_share_of_prof_var"),
        "main_run_config_gamma": config.get("latent_gamma"),
    }
    with (inputs_dir / "probabilistic_kappa_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def combine_main_signals_with_probability_run(
    main_latent_path: Path,
    probability_latent_path: Path,
    kappa_p: float,
    output_path: Path,
) -> None:
    main = pd.read_csv(main_latent_path)
    probability = pd.read_csv(probability_latent_path)
    keys = ["Ticker", "FormationYear"]

    missing_keys = [c for c in keys if c not in main.columns or c not in probability.columns]
    if missing_keys:
        raise ValueError(f"Missing merge keys in latent files: {missing_keys}")

    probability_cols = [
        "p_q5",
        "p_q5_sd_mc",
        "p_median",
        "p_median_sd_mc",
        "p_q1",
        "p_q1_sd_mc",
        "theta_post_sd",
        "obs_var_i",
        "tau2_t",
        "lambda_i",
    ]
    keep_cols = keys + [c for c in probability_cols if c in probability.columns]
    probability = probability[keep_cols].copy()

    merged = main.merge(
        probability,
        on=keys,
        how="left",
        suffixes=("", "_prob_kappa"),
        validate="1:1",
    )

    for col in ["p_q5", "p_q5_sd_mc", "p_median", "p_median_sd_mc", "p_q1", "p_q1_sd_mc"]:
        prob_col = f"{col}_prob_kappa"
        if prob_col in merged.columns:
            merged[col] = merged[prob_col]
            merged = merged.drop(columns=[prob_col])

    rename_map = {
        "theta_post_sd_prob_kappa": "prob_theta_post_sd",
        "obs_var_i_prob_kappa": "prob_obs_var_i",
        "tau2_t_prob_kappa": "prob_tau2_t",
        "lambda_i_prob_kappa": "prob_lambda_i",
    }
    merged = merged.rename(columns={k: v for k, v in rename_map.items() if k in merged.columns})
    merged["prob_kappa_p"] = kappa_p

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


def generate_probabilistic_kappa_run(
    source_run_dir: Path,
    kappa_p: float,
    main_kappa: float,
    gamma: float,
    n_portfolios: int,
    nw_lags: int,
) -> Path:
    config = load_run_config(source_run_dir)
    main_latent_path = source_run_dir / "latent_prof_model" / "latent_prof_firm_year.csv"
    uncertainty_csv = source_run_dir / "uncertainty_model" / "uncertainty_firm_year.csv"
    if not main_latent_path.exists():
        raise FileNotFoundError(f"Missing main latent firm-year file: {main_latent_path}")
    if not uncertainty_csv.exists():
        raise FileNotFoundError(f"Missing source uncertainty firm-year file: {uncertainty_csv}")

    portfolio_eval_dir = source_run_dir / "portfolio_evaluation"
    output_dir = portfolio_eval_dir / f"thesis_risk_adjusted_tables_probabilistic_kappa_{kappa_label(kappa_p)}_ucits_5_10_40"
    inputs_dir = output_dir / "inputs"
    probability_work_dir = inputs_dir / "probabilistic_kappa_latent_prof_model"
    formation_dir = inputs_dir / "portfolio_formation"
    evaluation_dir = inputs_dir
    latent_path = inputs_dir / f"latent_prof_firm_year_probabilistic_kappa_{kappa_label(kappa_p)}.csv"

    probability_result = run_latent_prof_model(
        input_csv=uncertainty_csv,
        output_dir=probability_work_dir,
        uncertainty_method=config.get("uncertainty_method", "HB"),
        gamma=gamma,
        noise_share_of_prof_var=kappa_p,
        use_full_propagation=False,
        hb_full_posterior_parquet=None,
        n_sigma_draws=None,
        checkpoint_every_draws=50,
    )
    combine_main_signals_with_probability_run(
        main_latent_path=main_latent_path,
        probability_latent_path=Path(probability_result["firm_year_csv"]),
        kappa_p=kappa_p,
        output_path=latent_path,
    )
    write_metadata(
        source_run_dir=source_run_dir,
        kappa_p=kappa_p,
        main_kappa=main_kappa,
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
    for kappa_p in args.kappas_p:
        out = generate_probabilistic_kappa_run(
            source_run_dir=source_run_dir,
            kappa_p=float(kappa_p),
            main_kappa=args.main_kappa,
            gamma=args.gamma,
            n_portfolios=args.n_portfolios,
            nw_lags=args.nw_lags,
        )
        created.append(out)
        print(f"Generated Probabilistic Quality kappa_P robustness run: kappa_P={kappa_p:.2f} -> {out}")

    print("\nCompleted probabilistic kappa_P robustness generation.")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
