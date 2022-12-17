"""Model calibration routines for binary _classification_ models."""

import functools

import numpy
from numpy import typing as npt

from shaprank import ensure


def calibrate_threshold_at_target_alert_rate(
    y_pred_score: npt.ArrayLike,
    target: float,
) -> float:
    """
    Given the calibration data, evaluates that threshold achieving the requested target alert-rate.

    Parameters:
        y_pred_score: The array of scores predicted by the model for the calibration data set
        target      : The target alert rate

    Returns:
        The desired model threshold
    """
    _, target = ensure.binary_classification_metric(  # type: ignore[assignment]
        "alert_rate", target, allow_none_target=False, allow_extremes=False
    )

    y_pred_score = numpy.array(y_pred_score)

    scores, counts = numpy.unique(y_pred_score, return_counts=True)

    idx_sort = numpy.argsort(-scores)
    traj_scores = scores[idx_sort]
    traj_counts = counts[idx_sort]
    traj_r_blocked = numpy.cumsum(traj_counts) / len(y_pred_score)

    idx_feasible = numpy.argwhere(traj_r_blocked <= target)
    if not idx_feasible.size:
        return -numpy.Inf

    return float(numpy.min(traj_scores[idx_feasible].ravel()))


def calibrate_threshold_at_target_fpr(
    y_true_class: npt.ArrayLike,
    y_pred_score: npt.ArrayLike,
    target: float,
) -> float:
    """
    Given the calibration data, evaluates that threshold achieving the requested target FPR.

    Parameters:
        y_true_class: The array of true labels
        y_pred_score: The array of scores predicted by the model for the calibration data set
        target      : The target FPR

    Returns:
        The desired model threshold
    """
    _, target = ensure.binary_classification_metric(  # type: ignore[assignment]
        "fpr", target, allow_none_target=False, allow_extremes=False
    )

    y_true_class = numpy.array(y_true_class)
    y_pred_score = numpy.array(y_pred_score)

    n_negatives = numpy.sum(y_true_class < 1)
    if not n_negatives:
        raise ValueError("No negative instances")

    idx_sort = numpy.argsort(y_pred_score)
    traj_scores = y_pred_score[idx_sort]
    traj_fpr = numpy.cumsum(y_true_class[idx_sort[::-1]] < 1)[::-1] / n_negatives
    idx_feasible = numpy.argwhere(traj_fpr <= target)
    if not idx_feasible.size:
        return +numpy.Inf
    return float(numpy.min(traj_scores[idx_feasible].ravel()))


def calibrate_threshold_at_target_recall(
    y_true_class: npt.ArrayLike,
    y_pred_score: npt.ArrayLike,
    target: float,
) -> float:
    """
    Given the calibration data, evaluates that threshold achieving the requested target recall.

    Parameters:
        y_true_class: The array of true labels
        y_pred_score: The array of scores predicted by the model for the calibration data set
        target      : The target recall

    Returns:
        The desired model threshold
    """
    _, target = ensure.binary_classification_metric(  # type: ignore[assignment]
        "recall", target, allow_none_target=False, allow_extremes=False
    )

    y_true_class = numpy.array(y_true_class)
    y_pred_score = numpy.array(y_pred_score)
    n_positive = numpy.sum(y_true_class)
    idx_sort = numpy.argsort(-y_pred_score)
    traj_scores = y_pred_score[idx_sort]
    traj_recall = numpy.cumsum(y_true_class[idx_sort]) / n_positive
    idx_feasible = numpy.argwhere(traj_recall >= target)
    return numpy.max(traj_scores[idx_feasible])


def calibrate_threshold_at_target_precision(
    y_true_class: npt.ArrayLike,
    y_pred_score: npt.ArrayLike,
    target: float,
) -> float:
    """
    Given the calibration data, evaluates that threshold achieving the requested target precision.

    Parameters:
        y_true_class: The array of true labels
        y_pred_score: The array of scores predicted by the model for the calibration data set
        target      : The target precision

    Returns:
        The desired model threshold
    """
    _, target = ensure.binary_classification_metric(  # type: ignore[assignment]
        "precision", target, allow_none_target=False, allow_extremes=False
    )

    y_true_class = numpy.array(y_true_class)
    y_pred_score = numpy.array(y_pred_score)

    n_positives = numpy.sum(y_true_class > 0)
    if not n_positives:
        raise ValueError("No positive instances")

    idx_sort = numpy.argsort(y_pred_score)
    traj_scores = y_pred_score[idx_sort]
    traj_precision = numpy.cumsum(y_true_class[idx_sort[::-1]] > 0)[::-1] / n_positives
    idx_feasible = numpy.argwhere(traj_precision >= target)
    return float(numpy.max(traj_scores[idx_feasible].ravel()))


def calibrate_threshold(
    y_true_class: npt.ArrayLike, y_pred_score: npt.ArrayLike, metric: str, metric_target: float
) -> float:
    """
    Given the calibration data, evaluates that threshold achieving the requested target value for
    a specific metric.

    Parameters:
        y_true_class : The array of true labels
        y_pred_score : The array of scores predicted by the model for the calibration data set
        metric       : The relevant calibration metric
        metric_target: The target value for the calibration metric

    Returns:
        The desired model threshold
    """
    if metric == "alert_rate":
        fn = functools.partial(
            calibrate_threshold_at_target_alert_rate, y_pred_score, metric_target
        )
    elif metric == "fpr":
        fn = functools.partial(
            calibrate_threshold_at_target_fpr, y_true_class, y_pred_score, metric_target
        )

    elif metric in ("recall", "tpr"):
        fn = functools.partial(
            calibrate_threshold_at_target_recall, y_true_class, y_pred_score, metric_target
        )
    elif metric == "precision":
        fn = functools.partial(
            calibrate_threshold_at_target_precision, y_true_class, y_pred_score, metric_target
        )

    else:
        raise ValueError(f"Unhandled metric `{metric}`")

    return fn()
