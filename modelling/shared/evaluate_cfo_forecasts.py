# evaluate_cfo_forecasts.py
#
# Updated version with two additional models:
# - Blend_GlobalAR1: weighted blend of CFO_t and pooled AR(1) forecast
# - Delta_GlobalARX: predict Delta CFO = CFO_{t+1} - CFO_t, then add CFO_t back
#
# Put this in: modelling/shared/evaluate_cfo_forecasts.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import gammaln

from hb_shared_utils import compute_wca, build_regressors


# =============================================================================
# Paths / loading
# =============================================================================

def find_project_root() -> Path:
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
    for p in [here] + list(here.parents):
        if (p / "data").exists() and (p / "results").exists():
            return p
    raise FileNotFoundError("Could not find project root containing 'data' and 'results'.")


def load_prepared_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = compute_wca(df)
    df = build_regressors(df, include_lead=True)

    required = [
        "Ticker",
        "Year",
        "Sector",
        "CFO_scaled",
        "CFO_lag1_scaled",
        "CFO_lead1_scaled",
        "WCA_scaled",
        "dREV_scaled",
        "PPE_scaled",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after feature construction: {missing}")

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    for c in [
        "CFO_scaled",
        "CFO_lag1_scaled",
        "CFO_lead1_scaled",
        "WCA_scaled",
        "dREV_scaled",
        "PPE_scaled",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Ticker", "Sector", "Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Target for delta models
    df["DeltaCFO_lead1"] = df["CFO_lead1_scaled"] - df["CFO_scaled"]

    return df.sort_values(["Ticker", "Year"]).reset_index(drop=True)


# =============================================================================
# Model helpers
# =============================================================================

SECTOR_MODEL_SPECS: Dict[str, List[str]] = {
    "Sector_AR1": ["CFO_scaled"],
    "Sector_AR2": ["CFO_scaled", "CFO_lag1_scaled"],
    "Sector_ARX": ["CFO_scaled", "CFO_lag1_scaled", "WCA_scaled", "dREV_scaled", "PPE_scaled"],
}

GLOBAL_LEVEL_MODEL_SPECS: Dict[str, List[str]] = {
    "Global_AR1": ["CFO_scaled"],
}

GLOBAL_DELTA_MODEL_SPECS: Dict[str, List[str]] = {
    "Delta_GlobalARX": ["CFO_scaled", "CFO_lag1_scaled", "WCA_scaled", "dREV_scaled", "PPE_scaled"],
}

GLOBAL_STUDENT_T_MODEL_SPECS: Dict[str, List[str]] = {
    "Global_AR1_StudentT": ["CFO_scaled"],
}

SECTOR_STUDENT_T_MODEL_SPECS: Dict[str, List[str]] = {
    "Sector_AR1_StudentT": ["CFO_scaled"],
}


def fit_ols(train_df: pd.DataFrame, x_cols: List[str], y_col: str):
    sub = train_df.dropna(subset=x_cols + [y_col]).copy()
    if len(sub) == 0:
        return None

    X = sm.add_constant(sub[x_cols], has_constant="add")
    y = sub[y_col]

    try:
        return sm.OLS(y, X).fit()
    except Exception:
        return None


def predict_from_model(model, test_df: pd.DataFrame, x_cols: List[str]) -> pd.Series:
    out = pd.Series(np.nan, index=test_df.index, dtype=float)
    if model is None:
        return out

    valid = test_df[x_cols].notna().all(axis=1)
    if valid.any():
        X = sm.add_constant(test_df.loc[valid, x_cols], has_constant="add")
        out.loc[valid] = model.predict(X)

    return out


def fit_predict_sector_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    min_sector_obs: int = 20,
) -> pd.Series:
    pred = pd.Series(np.nan, index=test_df.index, dtype=float)
    global_model = fit_ols(train_df, x_cols=x_cols, y_col=y_col)

    for sector, test_sub in test_df.groupby("Sector"):
        sector_train = train_df[train_df["Sector"] == sector].dropna(subset=x_cols + [y_col]).copy()

        if len(sector_train) >= min_sector_obs:
            model = fit_ols(sector_train, x_cols=x_cols, y_col=y_col)
        else:
            model = global_model

        pred.loc[test_sub.index] = predict_from_model(model, test_sub, x_cols=x_cols)

    return pred


def fit_predict_global_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
) -> pd.Series:
    model = fit_ols(train_df, x_cols=x_cols, y_col=y_col)
    return predict_from_model(model, test_df, x_cols=x_cols)


def fit_blend_weight(
    train_df: pd.DataFrame,
    model_pred_train: pd.Series,
    baseline_pred_train: pd.Series,
    y_true_col: str = "CFO_lead1_scaled",
) -> float:
    sub = pd.DataFrame(
        {
            "y_true": train_df[y_true_col],
            "pred_model": model_pred_train,
            "pred_base": baseline_pred_train,
        }
    ).dropna()

    if len(sub) == 0:
        return 1.0

    # y = w*base + (1-w)*model + e
    # equivalently y - model = w*(base - model) + e
    y = (sub["y_true"] - sub["pred_model"]).values
    x = (sub["pred_base"] - sub["pred_model"]).values

    denom = np.sum(x ** 2)
    if denom <= 0:
        return 1.0

    w = float(np.sum(x * y) / denom)
    return float(np.clip(w, 0.0, 1.0))


class StudentTLinearModel(GenericLikelihoodModel):
    """
    Linear regression with Student-t errors.

    Params:
        beta[0:k]      : regression coefficients
        log_sigma      : sigma = exp(log_sigma) > 0
        log_nu_minus_2 : nu = 2 + exp(log_nu_minus_2) > 2
    """

    def __init__(self, endog, exog, **kwargs):
        super().__init__(endog=endog, exog=exog, **kwargs)

    def nloglikeobs(self, params):
        k = self.exog.shape[1]
        beta = params[:k]
        log_sigma = params[k]
        log_nu_minus_2 = params[k + 1]

        sigma = np.exp(log_sigma)
        nu = 2.0 + np.exp(log_nu_minus_2)

        mu = self.exog @ beta
        resid = self.endog - mu
        z2 = (resid / sigma) ** 2

        # log pdf of Student-t with location=mu, scale=sigma
        ll = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - np.log(sigma)
            - ((nu + 1.0) / 2.0) * np.log1p(z2 / nu)
        )
        return -ll


def fit_student_t_regression(
    train_df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
):
    sub = train_df.dropna(subset=x_cols + [y_col]).copy()
    if len(sub) == 0:
        return None

    X = sm.add_constant(sub[x_cols], has_constant="add")
    y = sub[y_col].astype(float).values

    # OLS start values
    try:
        ols = sm.OLS(y, X).fit()
        beta0 = np.asarray(ols.params, dtype=float)
        resid0 = y - X.values @ beta0
        sigma0 = float(np.std(resid0, ddof=X.shape[1]))
        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = 0.1
    except Exception:
        beta0 = np.zeros(X.shape[1], dtype=float)
        sigma0 = 0.1

    # Start nu around 8
    start_params = np.concatenate(
        [
            beta0,
            np.array([np.log(max(sigma0, 1e-6))]),
            np.array([np.log(8.0 - 2.0)]),
        ]
    )

    try:
        model = StudentTLinearModel(y, X.values)
        result = model.fit(
            start_params=start_params,
            disp=False,
            maxiter=500,
        )
        result._x_cols = list(X.columns)
        return result
    except Exception:
        return None


def predict_from_student_t_model(model, test_df: pd.DataFrame, x_cols: List[str]) -> pd.Series:
    out = pd.Series(np.nan, index=test_df.index, dtype=float)
    if model is None:
        return out

    valid = test_df[x_cols].notna().all(axis=1)
    if not valid.any():
        return out

    X = sm.add_constant(test_df.loc[valid, x_cols], has_constant="add")
    k = X.shape[1]
    beta = np.asarray(model.params[:k], dtype=float)
    out.loc[valid] = X.values @ beta
    return out


def fit_predict_global_student_t_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
) -> pd.Series:
    model = fit_student_t_regression(train_df, x_cols=x_cols, y_col=y_col)
    return predict_from_student_t_model(model, test_df, x_cols=x_cols)


def fit_predict_sector_student_t_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    min_sector_obs: int = 20,
) -> pd.Series:
    pred = pd.Series(np.nan, index=test_df.index, dtype=float)

    global_model = fit_student_t_regression(train_df, x_cols=x_cols, y_col=y_col)

    for sector, test_sub in test_df.groupby("Sector"):
        sector_train = train_df[train_df["Sector"] == sector].dropna(subset=x_cols + [y_col]).copy()

        if len(sector_train) >= min_sector_obs:
            model = fit_student_t_regression(sector_train, x_cols=x_cols, y_col=y_col)
            if model is None:
                model = global_model
        else:
            model = global_model

        pred.loc[test_sub.index] = predict_from_student_t_model(model, test_sub, x_cols=x_cols)

    return pred

# =============================================================================
# Backtest construction
# =============================================================================

def build_year_splits(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    min_train_years: int,
    max_train_years: int,
):
    for test_year in range(year_start, year_end + 1):
        train_min = test_year - max_train_years
        train_max = test_year - 1

        train_df = df[
            (df["Year"] >= train_min)
            & (df["Year"] <= train_max)
            & df["CFO_lead1_scaled"].notna()
        ].copy()

        test_df = df[
            (df["Year"] == test_year)
            & df["CFO_lead1_scaled"].notna()
        ].copy()

        train_years = sorted(train_df["Year"].unique().tolist())

        if len(train_years) < min_train_years:
            continue
        if test_df.empty:
            continue

        yield test_year, train_df, test_df


def run_backtest(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    min_train_years: int,
    max_train_years: int,
    min_sector_obs: int,
) -> pd.DataFrame:
    rows = []

    for test_year, train_df, test_df in build_year_splits(
        df=df,
        year_start=year_start,
        year_end=year_end,
        min_train_years=min_train_years,
        max_train_years=max_train_years,
    ):
        out = test_df[
            [
                "Ticker",
                "Sector",
                "Year",
                "CFO_scaled",
                "CFO_lag1_scaled",
                "WCA_scaled",
                "dREV_scaled",
                "PPE_scaled",
                "CFO_lead1_scaled",
                "DeltaCFO_lead1",
            ]
        ].copy()

        out = out.rename(columns={"CFO_lead1_scaled": "y_true"})
        out["train_year_min"] = int(train_df["Year"].min())
        out["train_year_max"] = int(train_df["Year"].max())
        out["n_train_rows"] = int(len(train_df))

        # ---------------------------------------------------------------------
        # Baseline
        # ---------------------------------------------------------------------
        out["pred_Baseline_CFOeqCurrent"] = out["CFO_scaled"]

        # ---------------------------------------------------------------------
        # Sector level models on the CFO level
        # ---------------------------------------------------------------------
        for model_name, x_cols in SECTOR_MODEL_SPECS.items():
            out[f"pred_{model_name}"] = fit_predict_sector_model(
                train_df=train_df,
                test_df=out,
                x_cols=x_cols,
                y_col="CFO_lead1_scaled",
                min_sector_obs=min_sector_obs,
            )

        # ---------------------------------------------------------------------
        # Global level model(s)
        # ---------------------------------------------------------------------
        global_level_preds_train = {}
        global_level_preds_test = {}

        for model_name, x_cols in GLOBAL_LEVEL_MODEL_SPECS.items():
            pred_train = fit_predict_global_model(
                train_df=train_df,
                test_df=train_df,
                x_cols=x_cols,
                y_col="CFO_lead1_scaled",
            )
            pred_test = fit_predict_global_model(
                train_df=train_df,
                test_df=out,
                x_cols=x_cols,
                y_col="CFO_lead1_scaled",
            )

            global_level_preds_train[model_name] = pred_train
            global_level_preds_test[model_name] = pred_test
            out[f"pred_{model_name}"] = pred_test


        # ---------------------------------------------------------------------
        # Global Student-t level model(s)
        # ---------------------------------------------------------------------
        for model_name, x_cols in GLOBAL_STUDENT_T_MODEL_SPECS.items():
            out[f"pred_{model_name}"] = fit_predict_global_student_t_model(
                train_df=train_df,
                test_df=out,
                x_cols=x_cols,
                y_col="CFO_lead1_scaled",
            )

        # ---------------------------------------------------------------------
        # Sector Student-t level model(s)
        # ---------------------------------------------------------------------
        for model_name, x_cols in SECTOR_STUDENT_T_MODEL_SPECS.items():
            out[f"pred_{model_name}"] = fit_predict_sector_student_t_model(
                train_df=train_df,
                test_df=out,
                x_cols=x_cols,
                y_col="CFO_lead1_scaled",
                min_sector_obs=min_sector_obs,
            )
        # ---------------------------------------------------------------------
        # Blend model: blend baseline with pooled Student-t AR(1) if available,
        # otherwise pooled OLS AR(1)
        # ---------------------------------------------------------------------
        if "pred_Global_AR1_StudentT" in out.columns:
            blend_source_train = fit_predict_global_student_t_model(
                train_df=train_df,
                test_df=train_df,
                x_cols=["CFO_scaled"],
                y_col="CFO_lead1_scaled",
            )
            blend_source_test = out["pred_Global_AR1_StudentT"]
        elif "Global_AR1" in global_level_preds_test:
            blend_source_train = global_level_preds_train["Global_AR1"]
            blend_source_test = global_level_preds_test["Global_AR1"]
        else:
            blend_source_train = None
            blend_source_test = None

        if blend_source_test is not None:
            blend_w = fit_blend_weight(
                train_df=train_df,
                model_pred_train=blend_source_train,
                baseline_pred_train=train_df["CFO_scaled"],
                y_true_col="CFO_lead1_scaled",
            )
            out["blend_weight_on_baseline"] = blend_w
            out["pred_Blend_GlobalAR1"] = (
                blend_w * out["CFO_scaled"]
                + (1.0 - blend_w) * blend_source_test
            )
        else:
            out["blend_weight_on_baseline"] = np.nan
            out["pred_Blend_GlobalAR1"] = np.nan

        # ---------------------------------------------------------------------
        # Delta model(s): predict Delta CFO, then add CFO_t back
        # ---------------------------------------------------------------------
        for model_name, x_cols in GLOBAL_DELTA_MODEL_SPECS.items():
            pred_delta = fit_predict_global_model(
                train_df=train_df,
                test_df=out,
                x_cols=x_cols,
                y_col="DeltaCFO_lead1",
            )
            out[f"pred_{model_name}"] = out["CFO_scaled"] + pred_delta

        rows.append(out)

        print(
            f"Test year {test_year}: "
            f"train_rows={len(train_df)}, test_rows={len(test_df)}, "
            f"train_years={sorted(train_df['Year'].unique().tolist())}"
        )

    if not rows:
        raise ValueError("No valid backtest years were produced.")

    return pd.concat(rows, ignore_index=True).sort_values(["Year", "Ticker"]).reset_index(drop=True)


# =============================================================================
# Evaluation
# =============================================================================

def add_error_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]

    for c in pred_cols:
        name = c.replace("pred_", "")
        err = pred_df["y_true"] - pred_df[c]
        pred_df[f"err_{name}"] = err
        pred_df[f"ae_{name}"] = err.abs()
        pred_df[f"se_{name}"] = err ** 2

    return pred_df


def compute_model_summary(pred_df: pd.DataFrame, baseline_name: str = "Baseline_CFOeqCurrent") -> pd.DataFrame:
    rows = []
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]

    base_rmse = np.sqrt(pred_df[f"se_{baseline_name}"].mean())
    base_mae = pred_df[f"ae_{baseline_name}"].mean()

    for c in pred_cols:
        name = c.replace("pred_", "")
        sub = pred_df[["y_true", c]].dropna().copy()
        err = sub["y_true"] - sub[c]

        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        corr = float(sub["y_true"].corr(sub[c])) if len(sub) > 1 else np.nan
        sign_acc = float(np.mean(np.sign(sub["y_true"]) == np.sign(sub[c])))

        rows.append(
            {
                "Model": name,
                "n_obs": int(len(sub)),
                "RMSE": rmse,
                "MAE": mae,
                "Bias": bias,
                "Corr": corr,
                "SignAccuracy": sign_acc,
                "RMSE_Improvement_vs_Baseline": float(1 - rmse / base_rmse) if base_rmse > 0 else np.nan,
                "MAE_Improvement_vs_Baseline": float(1 - mae / base_mae) if base_mae > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def compute_year_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    rows = []

    for year, sub in pred_df.groupby("Year", sort=True):
        for c in pred_cols:
            name = c.replace("pred_", "")
            tmp = sub[["y_true", c]].dropna().copy()
            if tmp.empty:
                continue
            err = tmp["y_true"] - tmp[c]
            rows.append(
                {
                    "Year": int(year),
                    "Model": name,
                    "n_obs": int(len(tmp)),
                    "RMSE": float(np.sqrt(np.mean(err ** 2))),
                    "MAE": float(np.mean(np.abs(err))),
                    "Bias": float(np.mean(err)),
                    "Corr": float(tmp["y_true"].corr(tmp[c])) if len(tmp) > 1 else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["Year", "RMSE"]).reset_index(drop=True)


def test_against_baseline(
    pred_df: pd.DataFrame,
    baseline_name: str = "Baseline_CFOeqCurrent",
) -> pd.DataFrame:
    rows = []
    candidate_models = [
        c.replace("pred_", "")
        for c in pred_df.columns
        if c.startswith("pred_") and c != f"pred_{baseline_name}"
    ]

    for model_name in candidate_models:
        sub = pred_df[
            [
                "Year",
                f"se_{baseline_name}",
                f"se_{model_name}",
                f"ae_{baseline_name}",
                f"ae_{model_name}",
            ]
        ].dropna().copy()

        for loss_type, base_col, model_col in [
            ("SE", f"se_{baseline_name}", f"se_{model_name}"),
            ("AE", f"ae_{baseline_name}", f"ae_{model_name}"),
        ]:
            d = sub[base_col] - sub[model_col]
            X = np.ones((len(d), 1))

            reg = sm.OLS(d, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["Year"]},
            )

            rows.append(
                {
                    "Comparison": f"{model_name} minus {baseline_name}",
                    "LossType": loss_type,
                    "avg_loss_reduction": float(reg.params[0]),
                    "t_stat": float(reg.tvalues[0]),
                    "p_value": float(reg.pvalues[0]),
                    "n_obs": int(len(sub)),
                    "n_years": int(sub["Year"].nunique()),
                }
            )

    return pd.DataFrame(rows).sort_values(["LossType", "p_value"]).reset_index(drop=True)


# =============================================================================
# Optional plots
# =============================================================================

def save_plots(pred_df: pd.DataFrame, model_summary: pd.DataFrame, year_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(model_summary["Model"], model_summary["RMSE"])
    ax.set_title("CFO forecast RMSE by model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "cfo_forecast_rmse_by_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pivot_rmse = year_summary.pivot(index="Year", columns="Model", values="RMSE").sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in pivot_rmse.columns:
        ax.plot(pivot_rmse.index, pivot_rmse[col], marker="o", label=col)
    ax.set_title("CFO forecast RMSE by year")
    ax.set_xlabel("Test year")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cfo_forecast_rmse_by_year.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def run_cfo_forecast_evaluation(
    input_csv: str | Path = "results/extraction_static/prepared_step2_input.csv",
    output_dir: str | Path = "results/cfo_forecast_evaluation",
    year_start: int = 2010,
    year_end: int = 2023,
    min_train_years: int = 3,
    max_train_years: int = 5,
    min_sector_obs: int = 20,
    make_plots: bool = True,
) -> dict:
    project_root = find_project_root()
    input_csv = (project_root / input_csv) if not Path(input_csv).is_absolute() else Path(input_csv)
    output_dir = (project_root / output_dir) if not Path(output_dir).is_absolute() else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_prepared_panel(input_csv)
    df = df[df["CFO_scaled"].notna()].copy()

    pred_df = run_backtest(
        df=df,
        year_start=year_start,
        year_end=year_end,
        min_train_years=min_train_years,
        max_train_years=max_train_years,
        min_sector_obs=min_sector_obs,
    )
    pred_df = add_error_columns(pred_df)

    model_summary = compute_model_summary(pred_df)
    year_summary = compute_year_summary(pred_df)
    loss_tests = test_against_baseline(pred_df)

    pred_path = output_dir / "cfo_forecast_predictions.csv"
    summary_path = output_dir / "cfo_forecast_model_summary.csv"
    year_path = output_dir / "cfo_forecast_year_summary.csv"
    test_path = output_dir / "cfo_forecast_loss_tests.csv"

    pred_df.to_csv(pred_path, index=False)
    model_summary.to_csv(summary_path, index=False)
    year_summary.to_csv(year_path, index=False)
    loss_tests.to_csv(test_path, index=False)

    if make_plots:
        save_plots(pred_df, model_summary, year_summary, output_dir)

    print("\nSaved:")
    print(f"  {pred_path}")
    print(f"  {summary_path}")
    print(f"  {year_path}")
    print(f"  {test_path}")

    print("\nModel summary:")
    print(model_summary)

    print("\nLoss tests vs baseline:")
    print(loss_tests)

    return {
        "predictions_csv": str(pred_path),
        "model_summary_csv": str(summary_path),
        "year_summary_csv": str(year_path),
        "loss_tests_csv": str(test_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate external CFO forecast models.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/extraction_static/prepared_step2_input.csv",
        help="Prepared firm-year panel from extraction step.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cfo_forecast_evaluation",
        help="Folder to store CFO forecast backtest outputs.",
    )
    parser.add_argument("--year_start", type=int, default=2010)
    parser.add_argument("--year_end", type=int, default=2023)
    parser.add_argument("--min_train_years", type=int, default=3)
    parser.add_argument("--max_train_years", type=int, default=5)
    parser.add_argument("--min_sector_obs", type=int, default=20)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_cfo_forecast_evaluation(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        year_start=args.year_start,
        year_end=args.year_end,
        min_train_years=args.min_train_years,
        max_train_years=args.max_train_years,
        min_sector_obs=args.min_sector_obs,
        make_plots=not args.no_plots,
    )