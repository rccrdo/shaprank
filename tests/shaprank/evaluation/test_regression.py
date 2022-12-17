import numpy
import pytest
from numpy import typing as npt

from shaprank.evaluation import regression


@pytest.fixture
def x_target():
    return [1.0, 2.0, 3.0, 0.0, 0.0]


@pytest.fixture
def x_pred():
    return [0.0, 0.0, 1.0, 2.0, 3.0]


def test_eval_error_metric_mae(x_target: npt.ArrayLike, x_pred: npt.ArrayLike):
    res = regression.eval_error_metric(x_target, x_pred, "mae")

    assert res == 2.0


def test_eval_error_metric_mse(x_target: npt.ArrayLike, x_pred: npt.ArrayLike):
    res = regression.eval_error_metric(x_target, x_pred, "mse")

    assert res == 4.4


def test_eval_error_metric_rmse(x_target: npt.ArrayLike, x_pred: npt.ArrayLike):
    res = regression.eval_error_metric(x_target, x_pred, "rmse")

    numpy.testing.assert_almost_equal(res, 2.097617696340, decimal=9)


def test_eval_error_metric_raises_if_metric_is_unknown(
    x_target: npt.ArrayLike, x_pred: npt.ArrayLike
):
    with pytest.raises(ValueError):
        regression.eval_error_metric(x_target, x_pred, "unknown-metric")
