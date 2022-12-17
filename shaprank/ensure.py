"""Routines to facilitate defensive input checking at the level of ShapRank's user-facing APIs."""

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union

import numpy


def numeric(
    x: Optional[Union[int, float]], allow_none: bool = False, allow_nan: bool = False
) -> Optional[Union[int, float]]:
    """
    Helper to run a number of routine checks on a number-like object

    Parameters:
        x         : The numeric-like obj
        allow_none: If False, None values for `x` will raise an exception
        allow_nan : If False, NaN values for `x` will raise an exception

    Returns:
        The original object `x` if no exception was raised
    """
    if x is None:
        if not allow_none:
            raise ValueError(f"Received a None object but {allow_none}")
        return None

    if not isinstance(x, (int, float)):
        raise ValueError(f"Received non-numeric {type=}")

    if numpy.isnan(x) and not allow_nan:
        raise ValueError(f"Received an NaN float but {allow_nan=}")
    return x


def string(
    obj: Optional[str], allow_none: bool = False, allow_empty: bool = False
) -> Optional[str]:
    """
    Helper to run a number of routine checks on a string-like object

    Parameters:
        obj        : The string-like obj
        allow_none : If False, None values for `obj` will raise an exception
        allow_empty: If False, empty / zero-length values for `obj` will raise an exception

    Returns:
        The original object `obj` if no exception was raised
    """
    if obj is None:
        if not allow_none:
            raise ValueError(f"Received a None object but {allow_none}")
        return None

    if not isinstance(obj, str):
        raise ValueError("Object is not a string")

    if not obj and not allow_empty:
        raise ValueError(f"Received an empty string but {allow_empty=}")

    return obj


def iterable(
    obj: Optional[Sequence[Any]],
    allow_none: bool = False,
    allow_empty: bool = False,
    allow_duplicates: bool = False,
    allow_none_values: bool = False,
    allow_empty_values: bool = False,
    check_value_types: bool = False,
    allowed_value_types: Optional[Sequence[type]] = None,
) -> Optional[Sequence[Any]]:
    """
    Helper to run routine checks on an iterable object of heterogenous values

    Parameters:
        x                  : The iterable obj
        allow_none         : If False, None values for `obj` will raise an exception
        allow_empty        : If False, empty / zero-length `obj` will raise an exception
        allow_duplicates   : If False, any duplicate values in `obj` will raise an exception
        allow_none_values  : If False, Iterables `obj` containing None values will raise an
                             exception
        allow_empty_values : If False, Iterables `obj` containing empty / zero-length values will
                             raise an exception
        check_value_types  : If True, any values in `obj` with type not in `allowed_value_types`
                             will raise an exception
        allowed_value_types: See `check_value_types`

    Returns:
        The original object `obj` if no exception was raised
    """
    if obj is None:
        if not allow_none:
            raise ValueError(f"Received a None object but {allow_none}")
        return None

    if not isinstance(obj, Sequence):
        raise ValueError("Object is not a Sequence")

    n_obj = len(obj)
    if not n_obj:
        if not allow_empty:
            raise ValueError(f"Received an empty Sequence but {allow_empty=}")
        return obj

    if not allow_duplicates:
        if n_obj != len(set(obj)):
            raise ValueError(f"Received Sequence with duplicates but {allow_duplicates=}")

    if not allow_none_values and any(v is None for v in obj):
        raise ValueError(f"Received Sequence with None values but {allow_none_values=}")

    if not allow_empty_values and any(isinstance(v, Iterable) and not v for v in obj):
        raise ValueError(f"Received Sequence with empty values but {allow_empty_values=}")

    if check_value_types:
        if not allowed_value_types:
            raise ValueError(
                f"Received None or empty allowed_value_types but {check_value_types=}"
            )

        if any(not isinstance(t, type) for t in allowed_value_types):
            raise ValueError("Argument allowed_value_types should contain only type objects")
        allowed_types: set[type] = set(allowed_value_types)
        if allow_none_values:
            allowed_types.add(type(None))
        if not all(type(v) in allowed_types for v in obj):
            raise ValueError(f"Sequence values have types outside of {allowed_types=}")

    return obj


def feature(
    name: str,
) -> str:
    ret = string(
        name,
        allow_none=False,
        allow_empty=False,
    )
    """
    Helper to run a number of routine checks on a feature name

    Parameters:
        name       : The name of the feature
        allow_none : If False, None values for `name` will raise an exception
        allow_empty: If False, empty / zero-length `name` will raise an exception

    Returns:
        The original object `name` if no exception was raised
    """
    return ret  # type: ignore[return-value]


def features(
    c_inputs: Optional[Sequence[str]],
    allow_none: bool = False,
    allow_empty: bool = False,
) -> Optional[Sequence[str]]:
    """
    Helper to run routine checks on lists of model features

    Parameters:
        c_inputs   : The list of feature names / column names
        allow_none : If False, None values for `c_inputs` will raise an exception
        allow_empty: If False, empty / zero-length `c_inputs` will raise an exception


    Returns:
        The original object `name` if no exception was raised
    """
    return iterable(
        c_inputs,
        allow_none=allow_none,
        allow_empty=allow_empty,
        allow_duplicates=False,
        allow_none_values=False,
        allow_empty_values=False,
        check_value_types=True,
        allowed_value_types=[str, numpy.str_],
    )


@dataclasses.dataclass(frozen=True)
class _CalibrationMetric:
    """
    A dataclass to represent a calibration metric.

    Attributes
    ----------
    name : str
        the name of the metric
    target_lb : float
        the lower bound for feasible values of the metric
    target_ub : float
        the upper bound for feasible values of the metric
    """

    name: str
    target_lb: float
    target_ub: float


def binary_classification_metric(
    metric: str,
    metric_target: Optional[float],
    allow_none_target: bool = False,
    allow_extremes: bool = True,
) -> tuple[str, Optional[float]]:
    """
    Helper to run routine checks on the pair of a _calibration_ metric and a _calibration_
    metric target

    Parameters:
        metric           : The name of the metric
        metric_target    : The calibration target for the metric
        allow_none_target: If False, None values for `metric_target` will raise an exception
        allow_extremes   : If False, the target lower and upper bound are enforced strictly and
                           constraint violations will will raise an exception

    Returns:
        The original pair of `metric` and `metric_target` if no exception was raised
    """
    known_metrics = (
        _CalibrationMetric("alert_rate", 0.0, 1.0),
        _CalibrationMetric("fpr", 0.0, 1.0),
        _CalibrationMetric("recall", 0.0, 1.0),
        _CalibrationMetric("precision", 0.0, 1.0),
        _CalibrationMetric("tpr", 0.0, 1.0),
    )

    if metric_target is None:
        if not allow_none_target:
            raise ValueError(f"Received None metric_target but {allow_none_target=}")
        return metric, None

    for mtr in known_metrics:
        if metric == mtr.name:
            if (
                (metric_target is None)
                or (mtr.target_lb < metric_target < mtr.target_ub)
                or (allow_extremes and metric_target in (mtr.target_lb, mtr.target_ub))
            ):
                return metric, metric_target

            raise ValueError(f"Metric target for `{metric}` lies outside of allowed bounds")

    raise ValueError(f"Unknown metric `{metric}`")


def regression_error_metric(metric: str) -> str:
    """
    Helper to run routine checks on the _regression_ metrics

    Parameters:
        metric           : The name of the metric

    Returns:
        The original object `metric` if the metric is handled by ShapRank and no exception
        was raised
    """
    known_metrics = ("mae", "mse", "rmse")
    if metric not in known_metrics:
        raise ValueError(f"Unknown metric `{metric}`")
    return metric
