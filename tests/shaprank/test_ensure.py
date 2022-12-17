import unittest.mock
from collections.abc import Sequence
from typing import Union

import numpy
import pytest

from shaprank import ensure


@pytest.mark.parametrize("x", [1, 1.1])
def test_numeric(x: Union[int, float]):
    res = ensure.numeric(x)

    assert res == x


def test_numeric_raises_with_defaults_if_value_is_none():
    with pytest.raises(ValueError):
        ensure.numeric(None)


def test_numeric_handles_none():
    res = ensure.numeric(None, allow_none=True)

    assert res is None


def test_numeric_raises_if_value_is_not_numeric():
    with pytest.raises(ValueError):
        ensure.numeric("abc")


def test_numeric_raises_with_defaults_if_value_is_nan():
    with pytest.raises(ValueError):
        ensure.numeric(numpy.nan)


def test_numeric_handles_nan():
    res = ensure.numeric(numpy.nan, allow_nan=True)

    assert numpy.isnan(res)


def test_string():
    res = ensure.string("abc")

    assert res == "abc"


def test_string_raises_with_defaults_if_value_is_none():
    with pytest.raises(ValueError):
        ensure.string(None)


def test_string_handles_none():
    res = ensure.string(None, allow_none=True)

    assert res is None


def test_string_raises_if_type_is_not_string():
    with pytest.raises(ValueError):
        ensure.string(1)


def test_string_raises_with_defaults_if_value_is_empty():
    with pytest.raises(ValueError):
        ensure.string("")


def test_string_handles_empty_strings():
    res = ensure.string("", allow_empty=True)

    assert res == ""


@pytest.mark.parametrize(
    "x",
    [
        [0, 1, 2],
        [
            0,
        ],
    ],
)
def test_iterable(x: Sequence):
    res = ensure.iterable(x)

    assert res == x


def test_iterable_raises_with_defaults_if_value_is_none():
    with pytest.raises(ValueError):
        ensure.iterable(None)


def test_iterable_handles_none_values():
    res = ensure.iterable(None, allow_none=True)

    assert res is None


def test_iterable_raises_if_value_is_not_a_sequence():
    with pytest.raises(ValueError):
        ensure.iterable(1)


def test_iterable_raises_with_defaults_if_value_is_empty():
    with pytest.raises(ValueError):
        ensure.iterable([])


def test_iterable_handles_empty_values():
    res = ensure.iterable([], allow_empty=True)

    assert res == []


def test_iterable_raises_with_defaults_if_value_contains_duplicates():
    with pytest.raises(ValueError):
        ensure.iterable([0, 0, 1])


def test_iterable_handles_duplicates():
    res = ensure.iterable([0, 0, 1], allow_duplicates=True)

    assert res == [0, 0, 1]


def test_iterable_raises_with_defaults_if_value_contains_none():
    with pytest.raises(ValueError):
        ensure.iterable([0, 0, None])


def test_iterable_handles_values_containing_none_items():
    res = ensure.iterable([0, None, 1], allow_none_values=True)

    assert res == [0, None, 1]


def test_iterable_raises_with_defaults_if_value_contains_empty_values():
    with pytest.raises(ValueError):
        ensure.iterable([0, 1, ""])


def test_iterable_handles_values_containing_empty_items():
    res = ensure.iterable([0, 1, ""], allow_empty_values=True)

    assert res == [0, 1, ""]


def test_iterable_checks_items_types():
    res = ensure.iterable([1, 2, 3], check_value_types=True, allowed_value_types=[int])

    assert res == [1, 2, 3]


def test_iterable_raises_if_value_types_are_disallowed():
    with pytest.raises(ValueError):
        ensure.iterable([0, 1, "a"], check_value_types=True, allowed_value_types=[int])


def test_iterable_checks_items_types_v2():
    res = ensure.iterable([0, 1, "a"], check_value_types=True, allowed_value_types=[int, str])

    assert res == [0, 1, "a"]


@pytest.mark.parametrize("allow_empty", [False, True])
@pytest.mark.parametrize("allow_none", [False, True])
def test_features_should_call_ensure_features(allow_none: bool, allow_empty: bool):
    with unittest.mock.patch("shaprank.ensure.iterable") as mock_iterable:
        mock_obj = unittest.mock.Mock()
        res = ensure.features(mock_obj, allow_none=allow_none, allow_empty=allow_empty)

        assert res == mock_iterable.return_value
        mock_iterable.assert_called_once_with(
            mock_obj,
            allow_none=allow_none,
            allow_empty=allow_empty,
            allow_duplicates=False,
            allow_none_values=False,
            allow_empty_values=False,
            check_value_types=True,
            allowed_value_types=[str, numpy.str_],
        )


@pytest.mark.parametrize("metric", ["alert_rate", "fpr", "recall", "precision", "tpr"])
def test_binary_classification_metric(
    metric: str,
):
    mtr_target = 1 / 2
    mtr, _mtr_target = ensure.binary_classification_metric(metric, mtr_target)

    assert mtr == metric
    assert _mtr_target == mtr_target


@pytest.mark.parametrize("metric", ["alert_rate", "fpr", "recall", "precision", "tpr"])
def test_binary_classification_metric_raises_with_default_if_target_is_none(
    metric: str,
):
    with pytest.raises(ValueError):
        ensure.binary_classification_metric(metric, None)


@pytest.mark.parametrize("metric", ["alert_rate", "fpr", "recall", "precision", "tpr"])
def test_binary_classification_metric_handles_none_target(
    metric: str,
):
    mtr, mtr_target = ensure.binary_classification_metric(metric, None, allow_none_target=True)

    assert mtr == metric
    assert mtr_target is None


def test_binary_classification_metric_raises_if_metric_is_unknown():
    with pytest.raises(ValueError):
        ensure.binary_classification_metric("unkwnon-metric", 1 / 2)


@pytest.mark.parametrize("mtr_target", [0.0, 1.0])
@pytest.mark.parametrize("metric", ["alert_rate", "fpr", "recall", "precision", "tpr"])
def test_binary_classification_metric_raises_if_metric_target_is_at_extremes(
    metric: str, mtr_target: float
):
    with pytest.raises(ValueError):
        ensure.binary_classification_metric(metric, mtr_target, allow_extremes=False)


@pytest.mark.parametrize("mtr_target", [0.0, 1.0])
@pytest.mark.parametrize("metric", ["alert_rate", "fpr", "recall", "precision", "tpr"])
def test_binary_classification_metric_with_defaults_handles_extreme_targets(
    metric: str, mtr_target: float
):
    mtr, _mtr_target = ensure.binary_classification_metric(
        metric,
        mtr_target,
    )

    assert mtr == metric
    assert _mtr_target == mtr_target


@pytest.mark.parametrize("metric", ["mae", "mse", "rmse"])
def test_regression_error_metric(metric: str):
    res = ensure.regression_error_metric(metric)

    assert res == metric


def test_regression_error_metric_raises_when_metric_is_unknown():
    with pytest.raises(ValueError):
        ensure.regression_error_metric("unknown-metric")
