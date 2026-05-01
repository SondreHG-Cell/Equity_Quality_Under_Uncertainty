import numpy as np
import pandas as pd
from scipy import stats


FACTOR_MODELS = ["CAPM", "FF3", "Carhart", "FF5", "FF5_MOM"]


# =============================================================================
# 5.1 Raw Portfolio Performance
# =============================================================================

def raw_performance(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    excess = returns.subtract(rf, axis=0)
    results = {}

    for col in returns.columns:
        r = excess[col].dropna()
        mean_ann = r.mean() * 12
        vol_ann = r.std() * np.sqrt(12)
        sharpe = mean_ann / vol_ann if vol_ann > 0 else np.nan
        max_dd = _max_drawdown(returns[col].dropna())

        results[col] = {
            "mean_excess_ann": mean_ann,
            "volatility_ann": vol_ann,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

    out = pd.DataFrame(results).T
    out.index.names = ["Method", "Portfolio"] if isinstance(out.index, pd.MultiIndex) else [None]
    return out.reset_index()


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


# =============================================================================
# 5.2 Risk-Adjusted Performance
# =============================================================================

def risk_adjusted_performance(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    rf: pd.Series,
    models: list = None,
    lags: int = 12,
) -> pd.DataFrame:
    if models is None:
        models = FACTOR_MODELS

    excess_r = portfolio_returns - rf

    factor_sets = {
        "CAPM":    ["MKT"],
        "FF3":     ["MKT", "SMB", "HML"],
        "Carhart": ["MKT", "SMB", "HML", "MOM"],
        "FF5":     ["MKT", "SMB", "HML", "RMW", "CMA"],
        "FF5_MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
    }

    results = {}
    for model in models:
        cols = factor_sets[model]
        data = pd.concat([excess_r.rename("ret"), factors[cols]], axis=1).dropna()

        if data.empty:
            results[model] = {
                "alpha": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "n_obs": 0,
            }
            continue

        y = data["ret"].values
        X = np.column_stack([np.ones(len(data)), data[cols].values])

        reg = _ols_newey_west_full(y, X, lags=lags)

        results[model] = {
            "alpha": reg["beta"][0],
            "t_stat": reg["t_stat"][0],
            "p_value": reg["p_value"][0],
            "r_squared": reg["r_squared"],
            "n_obs": len(data),
        }

    return pd.DataFrame(results).T.reset_index(names="FactorModel")


def _ols_newey_west_full(y: np.ndarray, X: np.ndarray, lags: int = 12) -> dict:
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    y_var = np.var(y)
    r2 = np.nan if y_var == 0 else 1 - np.var(resid) / y_var

    cov = _newey_west_cov(X, resid, lags)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))

    return {
        "beta": beta,
        "cov": cov,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "r_squared": r2,
    }


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    n, k = X.shape
    scores = X * resid[:, None]
    S = scores.T @ scores / n

    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        Gl = scores[l:].T @ scores[:-l] / n
        S += w * (Gl + Gl.T)

    XtX_inv = np.linalg.inv(X.T @ X / n)
    return XtX_inv @ S @ XtX_inv / n


# =============================================================================
# 5.3 Alpha Differences
# =============================================================================

def alpha_differences(
    ls_returns: dict,
    factors: pd.DataFrame,
    rf: pd.Series,
    model: str = "FF5_MOM",
    lags: int = 12,
    base_method: str = "Method1_ObservedQuality",
) -> pd.DataFrame:
    factor_sets = {
        "CAPM":    ["MKT"],
        "FF3":     ["MKT", "SMB", "HML"],
        "Carhart": ["MKT", "SMB", "HML", "MOM"],
        "FF5":     ["MKT", "SMB", "HML", "RMW", "CMA"],
        "FF5_MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
    }

    cols = factor_sets[model]
    methods = list(ls_returns.keys())

    if base_method not in methods:
        raise ValueError(f"base_method='{base_method}' not found in ls_returns keys.")

    results = {}

    for method in methods:
        if method == base_method:
            continue

        r1 = (ls_returns[base_method] - rf).dropna()
        r2 = (ls_returns[method] - rf).dropna()
        idx = r1.index.intersection(r2.index).intersection(factors.index)

        if len(idx) == 0:
            results[f"{method} vs {base_method}"] = {
                "alpha_difference": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "n_obs": 0,
            }
            continue

        F = factors.loc[idx, cols].values
        y_base = r1.loc[idx].values
        y_comp = r2.loc[idx].values

        y_stack = np.concatenate([y_base, y_comp])
        d = np.concatenate([np.zeros(len(idx)), np.ones(len(idx))])
        F_stack = np.vstack([F, F])

        # const, method dummy, factors, dummy*factors
        X = np.column_stack([
            np.ones(len(y_stack)),
            d,
            F_stack,
            d[:, None] * F_stack,
        ])

        reg = _ols_newey_west_full(y_stack, X, lags=lags)

        # coefficient on dummy = alpha difference
        results[f"{method} vs {base_method}"] = {
            "alpha_difference": reg["beta"][1],
            "t_stat": reg["t_stat"][1],
            "p_value": reg["p_value"][1],
            "n_obs": len(idx),
        }

    return pd.DataFrame(results).T.reset_index(names="Comparison")


# =============================================================================
# 5.4 GRS Test
# =============================================================================

def grs_test(
    q5_returns: dict,
    factors: pd.DataFrame,
    rf: pd.Series,
    model: str = "FF5_MOM",
) -> dict:
    """
    Compute the Gibbons--Ross--Shanken (GRS) test for a given factor model.

    The test evaluates whether the intercepts from a set of time-series
    factor regressions are jointly equal to zero.

    Parameters
    ----------
    q5_returns : dict
        Dictionary of Q5 portfolio return series, one for each sorting method.
        Example: {"Method1_ObservedQuality": series, "Method2_LatentQuality": series, ...}

    factors : pd.DataFrame
        Monthly factor returns. Must contain the columns required by the
        selected factor model.

    rf : pd.Series
        Monthly risk-free rate.

    model : str
        Factor model to use. Supported values are:
        "CAPM", "FF3", "FF3_MOM", "FF5", and "FF5_MOM".

    Returns
    -------
    dict
        Dictionary containing the GRS statistic, p-value, estimated alphas,
        number of observations, number of portfolios, and number of factors.
    """

    factor_sets = {
        "CAPM":    ["MKT"],
        "FF3":     ["MKT", "SMB", "HML"],
        "FF3_MOM": ["MKT", "SMB", "HML", "MOM"],
        "FF5":     ["MKT", "SMB", "HML", "RMW", "CMA"],
        "FF5_MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
    }

    if model not in factor_sets:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Supported models are: {list(factor_sets.keys())}"
        )

    cols = factor_sets[model]
    methods = list(q5_returns.keys())
    N = len(methods)

    if N < 2:
        raise ValueError("The GRS test requires at least two test portfolios.")

    # Convert Q5 returns to excess returns.
    excess = {m: (q5_returns[m] - rf) for m in methods}

    # Align all portfolio returns, factors, and the risk-free rate.
    idx = factors.index
    for m in methods:
        idx = idx.intersection(excess[m].dropna().index)

    idx = idx.sort_values()

    F = factors.loc[idx, cols].dropna()
    idx = idx.intersection(F.index)

    for m in methods:
        idx = idx.intersection(excess[m].dropna().index)

    idx = idx.sort_values()

    F = factors.loc[idx, cols].values
    T, K = F.shape

    if T <= N + K:
        raise ValueError(
            f"Not enough observations for the GRS test: "
            f"T={T}, N={N}, K={K}."
        )

    F_demean = F - F.mean(axis=0)

    alphas = []
    residuals = []

    X = np.column_stack([np.ones(T), F])

    for m in methods:
        y = excess[m].loc[idx].values

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha = beta[0]
        resid = y - X @ beta

        alphas.append(alpha)
        residuals.append(resid)

    alpha_vec = np.array(alphas)
    resid_mat = np.column_stack(residuals)

    # Residual covariance matrix.
    Sigma = resid_mat.T @ resid_mat / T

    # Factor mean and covariance adjustment.
    mu_f = F.mean(axis=0)
    Omega = F_demean.T @ F_demean / T

    # Use pseudo-inverse for numerical stability, as Q5 portfolios across
    # methods can be highly correlated.
    Sigma_inv = np.linalg.pinv(Sigma)
    Omega_inv = np.linalg.pinv(Omega)

    kappa = 1 + mu_f @ Omega_inv @ mu_f

    grs_f = (
        (T / N)
        * ((T - N - K) / (T - K - 1))
        * (alpha_vec @ Sigma_inv @ alpha_vec)
        / kappa
    )

    p_val = 1 - stats.f.cdf(grs_f, dfn=N, dfd=T - N - K)

    return {
        "model": model,
        "grs_f_stat": float(grs_f),
        "p_value": float(p_val),
        "reject_h0_5pct": bool(p_val < 0.05),
        "alphas": dict(zip(methods, map(float, alphas))),
        "n_obs": int(T),
        "n_portfolios": int(N),
        "n_factors": int(K),
    }


def grs_tests_all_models(
    q5_returns: dict,
    factors: pd.DataFrame,
    rf: pd.Series,
) -> pd.DataFrame:
    """
    Run the GRS test separately for all factor specifications.

    This is used as a supplementary joint model adequacy check. It shows
    whether the joint pricing errors across Q5 portfolios decline as additional
    risk factors are included.

    Returns
    -------
    pd.DataFrame
        One row per factor model.
    """

    models = ["CAPM", "FF3", "FF3_MOM", "FF5", "FF5_MOM"]

    model_labels = {
        "CAPM": "CAPM",
        "FF3": "FF3",
        "FF3_MOM": "FF3+MOM",
        "FF5": "FF5",
        "FF5_MOM": "FF5+MOM",
    }

    rows = []

    for model in models:
        res = grs_test(
            q5_returns=q5_returns,
            factors=factors,
            rf=rf,
            model=model,
        )

        rows.append({
            "Model": model_labels[model],
            "GRS statistic": res["grs_f_stat"],
            "p-value": res["p_value"],
            "Reject H0": "Yes" if res["reject_h0_5pct"] else "No",
            "n_obs": res["n_obs"],
            "n_portfolios": res["n_portfolios"],
            "n_factors": res["n_factors"],
        })

    return pd.DataFrame(rows)
