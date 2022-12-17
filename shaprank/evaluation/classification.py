"""Common routines for the evaluation of calibrated _classification_ models."""

import functools
from collections.abc import Iterable

import numpy
import sklearn.metrics  # type: ignore[import]
from numpy import typing as npt


def evaluate_alert_rate(y_pred_class: npt.ArrayLike) -> float:
    """
    Evaluates the alert rate given the calibrated output of a classification model.

    Parameters:
        y_pred_class: The array of predicted classes

    Returns:
        The effective alert rate
    """
    y_pred_class = numpy.array(y_pred_class) > 0
    return numpy.array(y_pred_class).mean()


def evaluate_fpr(y_true_class: npt.ArrayLike, y_pred_class: npt.ArrayLike) -> float:
    """
    Evaluates the FPR given the target and calibrated outputs of a classification model.

    Parameters:
        y_true_class: The array of true labels
        y_pred_class: The array of predicted classes

    Returns:
        The effective FPR
    """
    y_true_class = numpy.array(y_true_class) > 0
    y_pred_class = numpy.array(y_pred_class) > 0
    tn, fp, _, _ = sklearn.metrics.confusion_matrix(
        y_true_class, y_pred_class, labels=[False, True]
    ).flatten()
    n_negatives = fp + tn
    if not n_negatives:
        return 0
    return fp / n_negatives


def evaluate_recall(y_true_class: npt.ArrayLike, y_pred_class: npt.ArrayLike) -> float:
    """
    Evaluates the recall given the target and calibrated outputs of a classification model.

    Parameters:
        y_true_class: The array of true labels
        y_pred_class: The array of predicted classes

    Returns:
        The effective recall
    """
    y_true_class = numpy.array(y_true_class) > 0
    y_pred_class = numpy.array(y_pred_class) > 0
    _, _, fn, tp = sklearn.metrics.confusion_matrix(
        y_true_class, y_pred_class, labels=[False, True]
    ).flatten()
    n_positives = tp + fn
    if not n_positives:
        return 0
    return tp / n_positives


def evaluate_precision(y_true_class: npt.ArrayLike, y_pred_class: npt.ArrayLike) -> float:
    """
    Evaluates the precision given the target and calibrated outputs of a classification model.

    Parameters:
        y_true_class: The array of true labels
        y_pred_class: The array of predicted classes

    Returns:
        The effective precision
    """
    y_true_class = numpy.array(y_true_class) > 0
    y_pred_class = numpy.array(y_pred_class) > 0
    _, fp, _, tp = sklearn.metrics.confusion_matrix(
        y_true_class, y_pred_class, labels=[False, True]
    ).flatten()
    n_alerts = tp + fp
    if not n_alerts:
        return 1
    return tp / n_alerts


def evaluate_metric(
    y_true_class: npt.ArrayLike,
    y_pred_score: npt.ArrayLike,
    y_pred_class: npt.ArrayLike,
    metric: str,
) -> float:
    """
    Evaluates the specified classification metric given the target and calibrated outputs of
    a classification model.

    Parameters:
        y_true_class: The array of true labels
        y_pred_score: The array of predicted scores
        y_pred_class: The array of predicted classes

    Returns:
        The effective value of the specified metric
    """
    if metric == "alert_rate":
        fn = functools.partial(evaluate_alert_rate, y_pred_class)
    elif metric == "fpr":
        fn = functools.partial(evaluate_fpr, y_true_class, y_pred_class)
    elif metric in ("recall", "tpr"):
        fn = functools.partial(evaluate_recall, y_true_class, y_pred_class)
    elif metric == "precision":
        fn = functools.partial(evaluate_precision, y_true_class, y_pred_class)
    else:
        raise ValueError(f"Unhandled metric `{metric}`")

    return fn()


def evaluate_calibrated_metrics(
    y_true_class: npt.ArrayLike,
    y_pred_score: npt.ArrayLike,
    threshold: float,
    metrics_requested: Iterable[str],
) -> dict[str, float]:
    """
    Evaluates the specified classification metrics given the target and calibrated outputs of
    a classification model.

    Parameters:
        y_true_class     : The array of true labels
        y_pred_score     : The array of predicted scores
        threshold        : The calibrated model threshold
        requested_metrics: The set of metrics to be evaluated

    Returns:
        The dictionary of (metric-name, effective-value) pairs for each of the requested metrics
    """
    y_true_class = numpy.array(y_true_class)
    y_pred_score = numpy.array(y_pred_score)
    y_pred_class = y_pred_score >= threshold

    metrics = {}
    for mtr in metrics_requested:
        metrics[mtr] = evaluate_metric(y_true_class, y_pred_score, y_pred_class, mtr)
    return metrics
