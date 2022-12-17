"""`Quick start` wrappers of common ShapRank functionality."""

from collections.abc import Sequence

import pandas

from shaprank import criteria
from shaprank.engine import ShapRank, ShapRankResult

__all__ = [
    "rank_classifier_features",
    "rank_regressor_features",
]


def rank_classifier_features(
    df_shap: pandas.DataFrame,
    c_inputs: Sequence[str],
    c_output: str,
    calib_metric: str,
    calib_metric_target: float,
    eval_metric: str,
    calib_metric_penalty: float,
    verbose: bool = False,
    show_progress: bool = False,
) -> ShapRankResult:
    """
    Uses ShapRank to rank the features of a binary classification model.

    Parameters:
        df_shap             : A frame containing the SHAP values for all model inputs and,
                              additionally, the raw classification target
        c_inputs            : The list of column name corresponding to the model features
        c_output            : The column name for the raw target
        calib_metric        : The relevant calibration metric
        calib_metric_target : The target value for which the model threshold calibration runs
        eval_metric         : The metric by which the performance of the calibrated model is
                              evaluated
        calib_metric_penalty: The penalty multiplier applied to compensate for violations of the
                              `calib_metric_target`
        verbose             : If True, verbose output is produced
        show_progress       : If True, log additional details during the greedy search iterations

    Returns:
        result              : The feature ranking produced by ShapRank's greedy search algorithm
    """
    criterion = criteria.classification.SoftConstrainedClassificationMetric(
        c_output, calib_metric, calib_metric_target, eval_metric, calib_metric_penalty
    )
    gre = ShapRank(c_inputs, c_output, criterion)
    result = gre.rank(df_shap, verbose=verbose, show_progress=show_progress)
    if verbose:
        result.log_summary()
    return result


def rank_regressor_features(
    df_shap: pandas.DataFrame,
    c_inputs: Sequence[str],
    c_output: str,
    eval_metric: str = "rmse",
    verbose: bool = False,
    show_progress: bool = False,
) -> ShapRankResult:
    """
    Uses ShapRank to rank the features of a scalar regression model.

    Parameters:
        df_shap             : A frame containing the SHAP values for all model inputs and,
                              additionally, the raw classification target
        c_inputs            : The list of column name corresponding to the model features
        c_output            : The column name for the raw target
        eval_metric         : The metric by which the performance of the model is evaluated
        verbose             : If True, verbose output is produced
        show_progress       : If True, log additional details during the greedy search iterations

    Returns:
        result              : The feature ranking produced by ShapRank's greedy search algorithm
    """
    criterion = criteria.regression.RegressionError(eval_metric)
    gre = ShapRank(c_inputs, c_output, criterion)
    result = gre.rank(df_shap, verbose=verbose, show_progress=show_progress)
    if verbose:
        result.log_summary()
    return result
