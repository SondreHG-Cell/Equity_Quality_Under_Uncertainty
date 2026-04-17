import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import matplotlib.pyplot as plt


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline: bool = True,
) -> dict:
    bs_model = brier_score_loss(y_true, y_prob)
    result = {"brier_score_model": bs_model}

    if baseline:
        base_rate = y_true.mean()
        bs_baseline = brier_score_loss(y_true, np.full(len(y_prob), base_rate))
        result["brier_score_baseline"] = bs_baseline
        result["brier_skill_score"] = 1 - bs_model / bs_baseline if bs_baseline != 0 else np.nan

    return result


def calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    ax=None,
    title: str = "Calibration Plot — Method 3",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[:-1]) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_means, bin_fracs = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_means.append(y_prob[mask].mean())
            bin_fracs.append(y_true[mask].mean())

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
    ax.scatter(bin_means, bin_fracs, s=60, color="steelblue", label="Model", zorder=3)
    ax.plot(bin_means, bin_fracs, color="steelblue", linewidth=1.5)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Realised frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def sharpness_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    out = {"prob_std": float(np.std(y_prob))}

    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = np.nan

    try:
        out["log_score"] = -log_loss(y_true, y_prob)
    except ValueError:
        out["log_score"] = np.nan

    return out


def sharpness_plot(
    y_prob: np.ndarray,
    ax=None,
    title: str = "Sharpness — Distribution of Predicted Probabilities",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(y_prob, bins=20, color="steelblue", edgecolor="white", alpha=0.85, density=True)
    ax.axvline(y_prob.mean(), color="firebrick", linestyle="--", linewidth=1.5, label=f"Mean = {y_prob.mean():.2f}")
    ax.axvline(0.2, color="grey", linestyle=":", linewidth=1.2, label="Base rate (0.20)")

    ax.set_xlabel("Predicted probability P(theta* in Q5)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def probabilistic_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    calibration = brier_score(y_true, y_prob, baseline=True)
    sharpness = sharpness_metrics(y_true, y_prob)
    return {**calibration, **sharpness}