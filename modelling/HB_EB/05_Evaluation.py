"""
Step 5 — Performance Evaluation
================================
Follows the methodology structure:
  5.1  Raw Portfolio Performance
  5.2  Risk-Adjusted Performance
  5.3  Alpha Differences
  5.4  GRS Test
  5.5  Probabilistic Model Evaluation (Calibration + Sharpness)
"""

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 5.1  Raw Portfolio Performance
# =============================================================================

def raw_performance(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """
    Compute descriptive statistics for Q1, Q5, and long-short (Q5-Q1)
    portfolios across all sorting methods.

    Parameters
    ----------
    returns : pd.DataFrame
        Monthly portfolio returns. Columns: MultiIndex (method, portfolio)
        e.g. ('method1', 'Q1'), ('method1', 'Q5'), ('method1', 'LS'), ...
    rf : pd.Series
        Monthly risk-free rate aligned to returns index.

    Returns
    -------
    pd.DataFrame
        Mean excess return (annualised), volatility (annualised),
        Sharpe ratio, and maximum drawdown for each portfolio.
    """
    excess = returns.subtract(rf, axis=0)
    results = {}

    for col in returns.columns:
        r = excess[col].dropna()
        mean_ann   = r.mean() * 12
        vol_ann    = r.std() * np.sqrt(12)
        sharpe     = mean_ann / vol_ann if vol_ann > 0 else np.nan
        max_dd     = _max_drawdown(returns[col].dropna())

        results[col] = {
            "mean_excess_ann": mean_ann,
            "volatility_ann":  vol_ann,
            "sharpe_ratio":    sharpe,
            "max_drawdown":    max_dd,
        }

    return pd.DataFrame(results).T


def _max_drawdown(returns: pd.Series) -> float:
    """Compute peak-to-trough maximum drawdown from a return series."""
    cumulative = (1 + returns).cumprod()
    peak       = cumulative.cummax()
    drawdown   = (cumulative - peak) / peak
    return drawdown.min()


# =============================================================================
# 5.2  Risk-Adjusted Performance
# =============================================================================

FACTOR_MODELS = ["CAPM", "FF3", "Carhart", "FF5", "FF5_MOM"]

def risk_adjusted_performance(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    rf: pd.Series,
    models: list = None,
    lags: int = 12,
) -> pd.DataFrame:
    """
    Estimate alpha for a single portfolio across factor model specifications.
    Factor models are estimated sequentially — CAPM through FF5+MOM — to
    assess sensitivity of alpha to different risk adjustments.

    Carhart and FF5+MOM are both included to assess whether the quality
    premium is robust to momentum adjustment: Carhart tests this within a
    three-factor structure, FF5+MOM after profitability and investment are
    already controlled for.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Monthly returns for the portfolio of interest.
    factors : pd.DataFrame
        Factor returns. Expected columns: MKT, SMB, HML, RMW, CMA, MOM.
        (MKT should already be excess return, i.e. Rm - Rf)
    rf : pd.Series
        Monthly risk-free rate.
    models : list, optional
        Subset of FACTOR_MODELS to estimate. Defaults to all.
    lags : int
        Number of Newey-West lags. Default 12.

    Returns
    -------
    pd.DataFrame
        Alpha, t-statistic, p-value, and R² for each factor model.
    """
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
        cols   = factor_sets[model]
        data   = pd.concat([excess_r, factors[cols]], axis=1).dropna()
        y      = data.iloc[:, 0].values
        X      = np.column_stack([np.ones(len(data)), data.iloc[:, 1:].values])

        alpha, t_stat, p_val, r2 = _ols_newey_west(y, X, lags=lags)
        results[model] = {
            "alpha":   alpha,
            "t_stat":  t_stat,
            "p_value": p_val,
            "r_squared": r2,
        }

    return pd.DataFrame(results).T


def _ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int = 12):
    """
    OLS with Newey-West standard errors.
    Returns alpha (intercept), its t-statistic, p-value, and R².
    """
    n, k   = X.shape
    beta   = np.linalg.lstsq(X, y, rcond=None)[0]
    resid  = y - X @ beta
    r2     = 1 - np.var(resid) / np.var(y)

    # Newey-West covariance
    S      = _newey_west_cov(X, resid, lags)
    se     = np.sqrt(np.diag(S))
    t_stat = beta[0] / se[0]
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - k))

    return beta[0], t_stat, p_val, r2


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance matrix."""
    n, k = X.shape
    scores = X * resid[:, None]
    S = scores.T @ scores / n

    for l in range(1, lags + 1):
        w   = 1 - l / (lags + 1)
        Gl  = scores[l:].T @ scores[:-l] / n
        S  += w * (Gl + Gl.T)

    XtX_inv = np.linalg.inv(X.T @ X / n)
    return XtX_inv @ S @ XtX_inv / n


# =============================================================================
# 5.3  Alpha Differences
# =============================================================================

def alpha_differences(
    ls_returns: dict,
    factors: pd.DataFrame,
    rf: pd.Series,
    model: str = "FF5",
    lags: int = 12,
) -> pd.DataFrame:
    """
    Test whether alpha(Method 2) > alpha(Method 1) and
                   alpha(Method 3) > alpha(Method 1).

    Implemented via a stacked regression framework: stack the long-short
    returns of all methods, add method dummies, and test the interaction
    of each dummy with the intercept.

    Parameters
    ----------
    ls_returns : dict
        Keys: method names (e.g. 'method1', 'method2', 'method3').
        Values: pd.Series of monthly long-short returns.
    factors : pd.DataFrame
        Factor returns for the chosen model.
    rf : pd.Series
        Monthly risk-free rate.
    model : str
        Factor model to use for the comparison. Default 'FF5'.
    lags : int
        Newey-West lags.

    Returns
    -------
    pd.DataFrame
        Alpha difference, t-statistic, and p-value for each pairwise test
        against Method 1.
    """
    factor_sets = {
        "CAPM":    ["MKT"],
        "FF3":     ["MKT", "SMB", "HML"],
        "Carhart": ["MKT", "SMB", "HML", "MOM"],
        "FF5":     ["MKT", "SMB", "HML", "RMW", "CMA"],
        "FF5_MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
    }
    cols    = factor_sets[model]
    methods = list(ls_returns.keys())
    base    = methods[0]

    results = {}
    for method in methods[1:]:
        r1 = (ls_returns[base]   - rf).dropna()
        r2 = (ls_returns[method] - rf).dropna()
        idx = r1.index.intersection(r2.index).intersection(factors.index)

        # Stacked regression: pool both series, add indicator for method 2/3
        y_stack = np.concatenate([r1.loc[idx].values, r2.loc[idx].values])
        F       = factors.loc[idx, cols].values
        ones    = np.ones(len(idx))
        zeros   = np.zeros(len(idx))

        # Columns: const, indicator, factors, indicator*factors (interaction)
        X_base  = np.column_stack([ones, zeros, F, zeros * F])
        X_comp  = np.column_stack([ones, ones,  F, F])
        X_stack = np.vstack([X_base, X_comp])

        alpha_diff, t_stat, p_val, _ = _ols_newey_west(y_stack, X_stack, lags)
        results[f"{method} vs {base}"] = {
            "alpha_difference": alpha_diff,
            "t_stat":           t_stat,
            "p_value":          p_val,
        }

    return pd.DataFrame(results).T


# =============================================================================
# 5.4  GRS Test
# =============================================================================

def grs_test(
    q5_returns: dict,
    factors: pd.DataFrame,
    rf: pd.Series,
    model: str = "FF5",
) -> dict:
    """
    Gibbons-Ross-Shanken (1989) test applied jointly to the Q5 portfolios
    across sorting methods.

    Tests H0: all intercepts are jointly zero. In this setting, rejection
    is the expected and desired outcome — it confirms that the portfolios
    generate returns the factor model cannot account for.

    Parameters
    ----------
    q5_returns : dict
        Keys: method names. Values: pd.Series of monthly Q5 returns.
    factors : pd.DataFrame
        Factor returns for the chosen model.
    rf : pd.Series
        Risk-free rate.
    model : str
        Factor model specification.

    Returns
    -------
    dict
        GRS F-statistic, p-value, and individual alphas with t-statistics.
    """
    factor_sets = {
        "CAPM":    ["MKT"],
        "FF3":     ["MKT", "SMB", "HML"],
        "Carhart": ["MKT", "SMB", "HML", "MOM"],
        "FF5":     ["MKT", "SMB", "HML", "RMW", "CMA"],
        "FF5_MOM": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
    }
    cols    = factor_sets[model]
    methods = list(q5_returns.keys())
    N       = len(methods)

    # Align all series to common index
    excess  = {m: (q5_returns[m] - rf) for m in methods}
    idx     = factors.index
    for m in methods:
        idx = idx.intersection(excess[m].dropna().index)

    F  = factors.loc[idx, cols].values
    T, K = F.shape
    F_demean = F - F.mean(axis=0)

    alphas, residuals = [], []
    for m in methods:
        y    = excess[m].loc[idx].values
        X    = np.column_stack([np.ones(T), F])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        alphas.append(beta[0])
        residuals.append(y - X @ beta)

    alpha_vec = np.array(alphas)
    resid_mat = np.column_stack(residuals)
    Sigma     = resid_mat.T @ resid_mat / T

    # GRS F-statistic
    Sigma_inv = np.linalg.inv(Sigma)
    mu_f      = F.mean(axis=0)
    Omega_inv = np.linalg.inv(F_demean.T @ F_demean / T)
    kappa     = 1 + mu_f @ Omega_inv @ mu_f

    grs_f = (T / N) * ((T - N - K) / (T - K - 1)) * \
            (alpha_vec @ Sigma_inv @ alpha_vec) / kappa

    p_val = 1 - stats.f.cdf(grs_f, dfn=N, dfd=T - N - K)

    return {
        "grs_f_stat": grs_f,
        "p_value":    p_val,
        "alphas":     dict(zip(methods, alphas)),
    }


'''
Here is a clear overview of what each function expects as input:

---

**step5_evaluation.py**

`raw_performance(returns, rf)`
- `returns` — DataFrame of monthly portfolio returns, columns as MultiIndex `(method, portfolio)` e.g. `('method1', 'Q5')`
- `rf` — Series of monthly risk-free rates, same index as returns

`risk_adjusted_performance(portfolio_returns, factors, rf, models, lags)`
- `portfolio_returns` — Series of monthly returns for one portfolio
- `factors` — DataFrame with columns `MKT, SMB, HML, RMW, CMA, MOM`
- `rf` — Series of monthly risk-free rates
- `models` — optional list, defaults to all five specifications
- `lags` — integer, defaults to 12

`alpha_differences(ls_returns, factors, rf, model, lags)`
- `ls_returns` — dict of Series, keys are method names e.g. `{'method1': series, 'method2': series, 'method3': series}`
- `factors` — same as above
- `rf` — same as above
- `model` — string specifying which factor model to use, defaults to `'FF5'`

`grs_test(q5_returns, factors, rf, model)`
- `q5_returns` — dict of Series, keys are method names, values are Q5 portfolio returns
- `factors` — same as above
- `rf` — same as above
- `model` — string, defaults to `'FF5'`

---

**step5_probabilistic.py**

All four functions share the same two core inputs:

- `y_true` — numpy array, binary: `1` if the firm realised in the top quintile of returns, `0` otherwise
- `y_prob` — numpy array, predicted probability `P(theta* in Q5)` from Method 3

The combined wrapper `probabilistic_evaluation(y_true, y_prob, n_bins, plot)` takes the same two plus optional `n_bins` for the calibration plot and a `plot` boolean.

---

**What you need to prepare before calling these:**

The two things that require the most upstream work are `factors` and `y_true`. The factors come directly from the Ken French Data Library and just need to be aligned to your return index. `y_true` requires you to define what "realised in Q5" means — typically whether a firm's return over the holding period lands in the top quintile of the cross-section, which you construct from your Step 4 output.

'''