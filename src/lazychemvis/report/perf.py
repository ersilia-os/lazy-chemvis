"""
Surrogate performance summaries.

The quality thresholds live here so the log summary and the HTML report cannot
disagree about whether a surrogate is "good".
"""

from ..helpers.logger import get_logger

logger = get_logger(__name__)

# R² mean → (label, css class)
_R2_BANDS = [
    (0.95, "EXCELLENT", "good"),
    (0.90, "VERY GOOD", "good"),
    (0.85, "GOOD", "warn"),
    (0.75, "ACCEPTABLE", "warn"),
]

# Mean Euclidean coordinate error → (label, css class). The reference spaces are all
# scaled to [-1, 1], so these are absolute distances in that space.
_EUCLIDEAN_BANDS = [
    (0.05, "EXCELLENT", "good"),
    (0.10, "VERY GOOD", "good"),
    (0.15, "GOOD", "warn"),
]


def r2_verdict(r2_mean):
    """Return ``(label, css_class)`` for a mean R²."""
    for threshold, label, css in _R2_BANDS:
        if r2_mean > threshold:
            return label, css
    return "NEEDS IMPROVEMENT", "bad"


def euclidean_verdict(euclidean_mean):
    """Return ``(label, css_class)`` for a mean Euclidean coordinate error."""
    for threshold, label, css in _EUCLIDEAN_BANDS:
        if euclidean_mean < threshold:
            return label, css
    return "MODERATE", "warn"


def summarise(metrics):
    """
    Reduce a metrics dict to the values the log and the HTML both render.

    Returns
    -------
    dict or None
        Keys: cv_folds, r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std,
        euclidean_mean, euclidean_std, r2_per_fold, euclidean_per_fold, quality,
        quality_class, accuracy, accuracy_class, spread_low, spread_high, stable.
    """
    if not metrics:
        return None
    quality, quality_class = r2_verdict(metrics["r2_mean"])
    accuracy, accuracy_class = euclidean_verdict(metrics["euclidean_mean"])
    out = dict(metrics)
    out.update(
        {
            "quality": quality,
            "quality_class": quality_class,
            "accuracy": accuracy,
            "accuracy_class": accuracy_class,
            # Mean ± 1 std. Deliberately *not* labelled a 95% confidence interval:
            # ±1 std over 5 folds is roughly a 68% spread, and calling it 95% would
            # overstate the precision of the estimate.
            "spread_low": metrics["r2_mean"] - metrics["r2_std"],
            "spread_high": metrics["r2_mean"] + metrics["r2_std"],
            "stable": metrics["r2_std"] <= 0.05,
        }
    )
    return out


def log_cv_results(metrics: dict) -> None:
    """
    Record a cross-validation summary in the log.

    The console already shows R² and Euclidean error on the step's completion line
    (see ``Pipeline._metric_rows``), so the full breakdown goes to the log file only;
    high fold-to-fold variance is still raised as a warning.

    Parameters
    ----------
    metrics : dict
        The dict produced by a surrogate's CV loop.
    """
    m = summarise(metrics)
    if m is None:
        return

    logger.info(
        f"Cross-validation ({m['cv_folds']} folds, scored on held-out folds) — "
        f"R² {m['r2_mean']:.4f} ± {m['r2_std']:.4f} | "
        f"RMSE {m['rmse_mean']:.4f} ± {m['rmse_std']:.4f} | "
        f"MAE {m['mae_mean']:.4f} ± {m['mae_std']:.4f} | "
        f"Euclidean {m['euclidean_mean']:.4f} ± {m['euclidean_std']:.4f} | "
        f"quality {m['quality']}, accuracy {m['accuracy']}"
    )
    for i in range(m["cv_folds"]):
        logger.debug(
            f"  fold {i + 1}: R² {m['r2_per_fold'][i]:.4f}, "
            f"Euclidean {m['euclidean_per_fold'][i]:.4f}"
        )

    if not m["stable"]:
        logger.warning(f"High variance across folds (R² std = {m['r2_std']:.4f}).")
