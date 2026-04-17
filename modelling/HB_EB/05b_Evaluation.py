"""
Step 5 — Probabilistic Model Evaluation (Method 3)
====================================================
Evaluates whether P(theta* in Q5) is a reliable and informative
probability estimate, assessed in terms of:

  5.5.1  Calibration  — do predicted probabilities match realised frequencies?
  5.5.2  Sharpness    — does the model produce confident predictions?

All evaluation is conducted out-of-sample.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    log_loss,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# 5.5.1  Calibration
# =============================================================================

def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline: bool = True,
) -> dict:
    """
    Compute the Brier score for predicted probabilities, benchmarked
    against a naïve baseline (uniform probability equal to base rate).

    Lower Brier score = better calibration.

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcome: 1 if firm realised in top quintile, 0 otherwise.
    y_prob : np.ndarray
        Predicted probability P(theta* in Q5) from Method 3.
    baseline : bool
        Whether to also compute the naïve baseline Brier score.

    Returns
    -------
    dict
        Brier score for model and, optionally, naïve baseline.
    """
    bs_model = brier_score_loss(y_true, y_prob)
    result   = {"brier_score_model": bs_model}

    if baseline:
        base_rate    = y_true.mean()
        bs_baseline  = brier_score_loss(y_true, np.full_like(y_prob, base_rate))
        result["brier_score_baseline"] = bs_baseline
        result["brier_skill_score"]    = 1 - bs_model / bs_baseline

    return result


def calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    ax=None,
    title: str = "Calibration Plot — Method 3",
) -> plt.Axes:
    """
    Plot predicted probabilities against realised frequencies.
    A perfectly calibrated model falls on the diagonal.

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcomes.
    y_prob : np.ndarray
        Predicted probabilities P(theta* in Q5).
    n_bins : int
        Number of probability bins.
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    bins       = np.linspace(0, 1, n_bins + 1)
    bin_ids    = np.digitize(y_prob, bins[:-1]) - 1
    bin_ids    = np.clip(bin_ids, 0, n_bins - 1)

    bin_means, bin_fracs = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_means.append(y_prob[mask].mean())
            bin_fracs.append(y_true[mask].mean())

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            linewidth=1, label="Perfect calibration")
    ax.scatter(bin_means, bin_fracs, s=60, color="steelblue",
               label="Model", zorder=3)
    ax.plot(bin_means, bin_fracs, color="steelblue", linewidth=1.5)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Realised frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# 5.5.2  Sharpness
# =============================================================================

def sharpness_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Evaluate the concentration of predicted probabilities — whether the
    model produces confident predictions rather than collapsing toward
    the base rate.

    Metrics
    -------
    - AUC  : area under the ROC curve. Higher = better discrimination.
    - Log score : mean log-likelihood of predictions. Less negative = better.
    - Prob std  : standard deviation of predicted probabilities.
                  Higher = sharper (more confident) predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcomes.
    y_prob : np.ndarray
        Predicted probabilities P(theta* in Q5).

    Returns
    -------
    dict
        AUC, log score, and probability standard deviation.
    """
    auc       = roc_auc_score(y_true, y_prob)
    log_score = -log_loss(y_true, y_prob)     # negated: higher = better
    prob_std  = y_prob.std()

    return {
        "auc":        auc,
        "log_score":  log_score,
        "prob_std":   prob_std,
    }


def sharpness_plot(
    y_prob: np.ndarray,
    ax=None,
    title: str = "Sharpness — Distribution of Predicted Probabilities",
) -> plt.Axes:
    """
    Plot the distribution of predicted probabilities.
    A sharp model has a bimodal or spread distribution;
    an unsharp model clusters near the base rate.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities P(theta* in Q5).
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(y_prob, bins=20, color="steelblue", edgecolor="white",
            alpha=0.85, density=True)
    ax.axvline(y_prob.mean(), color="firebrick", linestyle="--",
               linewidth=1.5, label=f"Mean = {y_prob.mean():.2f}")
    ax.axvline(0.2, color="grey", linestyle=":", linewidth=1.2,
               label="Base rate (0.20)")

    ax.set_xlabel("Predicted probability P(theta* in Q5)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Combined evaluation — calls both subsections
# =============================================================================

def probabilistic_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    plot: bool = True,
) -> dict:
    """
    Full out-of-sample probabilistic evaluation of Method 3.
    Runs calibration and sharpness assessments and optionally
    produces both diagnostic plots.

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcome: 1 if firm realised in top quintile, 0 otherwise.
    y_prob : np.ndarray
        Predicted probability P(theta* in Q5) from Method 3.
    n_bins : int
        Number of bins for calibration plot.
    plot : bool
        Whether to display diagnostic plots.

    Returns
    -------
    dict
        Combined calibration and sharpness metrics.
    """
    calibration = brier_score(y_true, y_prob, baseline=True)
    sharpness   = sharpness_metrics(y_true, y_prob)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        calibration_plot(y_true, y_prob, n_bins=n_bins, ax=axes[0])
        sharpness_plot(y_prob, ax=axes[1])
        plt.tight_layout()
        plt.show()

    return {**calibration, **sharpness}