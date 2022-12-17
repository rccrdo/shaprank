"""Common routines for the evaluation of scalar _regression_ models."""

import numpy
import numpy.typing as npt


def eval_error_metric(
    x_target: npt.ArrayLike,
    x_pred: npt.ArrayLike,
    error_metric: str,
) -> float:
    """
    Evaluates the specified regression metric given target and predicted values in \\R.

    Parameters:
        x_target    : The scalar regression target
        x_pred      : The array of model predictions
        error_metric: The metric to evaluate

    Returns:
        The effective value of the requested metric
    """
    x_target = numpy.array(x_target)
    x_pred = numpy.array(x_pred)
    x_err = x_pred - x_target

    if error_metric == "mae":
        x_err = numpy.abs(x_err)
    elif error_metric in ("mse", "rmse"):
        x_err = numpy.power(x_err, 2)
    else:
        raise ValueError(f"Unknown error_metric `{error_metric}`")

    x_me = x_err.mean(axis=0)

    if error_metric == "rmse":
        return numpy.sqrt(x_me)

    return x_me
