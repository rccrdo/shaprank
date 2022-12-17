import unittest.mock
from typing import Any

import numpy
import pandas
import pytest

from shaprank.calibration import strategies


def test_CalibrationResult_instance_has_expected_fields():
    with unittest.mock.patch("shaprank.ensure.numeric") as mock_numeric:

        threshold = unittest.mock.Mock()
        metrics = unittest.mock.Mock()
        ensured_threshold = unittest.mock.Mock()
        mock_numeric.return_value = ensured_threshold

        cr = strategies.CalibrationResult(threshold, metrics)

        mock_numeric.assert_called_once_with(threshold, allow_none=True, allow_nan=False)

        assert cr._threshold == ensured_threshold
        assert cr._metrics == metrics


@pytest.mark.parametrize("metrics", [{}, {"a", 1}])
def test_CalibrationResult_get(metrics: dict[str, Any]):
    with unittest.mock.patch("shaprank.ensure.numeric") as mock_numeric:

        threshold = unittest.mock.Mock()
        mock_numeric.return_value = threshold

        cr = strategies.CalibrationResult(threshold, metrics)

        res = cr.get()

        assert res == metrics


@pytest.mark.parametrize(
    "metrics, exp", [({}, ""), ({"a": 1}, "a=1"), ({"a": 1, "b": 2}, "a=1, b=2")]
)
def test_CalibrationResult_get_summary(metrics: dict[str, Any], exp: str):
    with unittest.mock.patch("shaprank.ensure.numeric") as mock_numeric:

        threshold = unittest.mock.Mock()
        mock_numeric.return_value = threshold

        cr = strategies.CalibrationResult(threshold, metrics)

        res = cr.get_summary()

        assert res == exp


def test_CalibrationStrategy_instance_has_expected_fields():
    with unittest.mock.patch("shaprank.ensure.string") as mock_string:
        name = unittest.mock.Mock()
        ensured_name = unittest.mock.Mock()
        mock_string.return_value = ensured_name

        cs = strategies.CalibrationStrategy(name)

        assert cs._name == ensured_name
        assert cs._metrics_requested == set()


def test_CalibrationStrategy_name():
    with unittest.mock.patch("shaprank.ensure.string") as mock_string:
        name = unittest.mock.Mock()
        ensured_name = unittest.mock.Mock()
        mock_string.return_value = ensured_name

        cs = strategies.CalibrationStrategy(name)

        assert cs.name == ensured_name


@pytest.mark.parametrize(
    "metrics_required",
    [
        {
            "recall",
        },
        {"recall", "alert_rate"},
    ],
)
def test_CalibrationStrategy_request(metrics_required):
    with unittest.mock.patch("shaprank.ensure.string") as mock_string, unittest.mock.patch(
        "shaprank.calibration.strategies.CalibrationStrategy.provides"
    ) as mock_provides:
        name = unittest.mock.Mock()
        ensured_name = unittest.mock.Mock()
        mock_string.return_value = ensured_name
        mock_provides.return_value = {"recall", "alert_rate", "any-metric"}

        cs = strategies.CalibrationStrategy(name)

        cs.request(metrics_required)

        mock_provides.assert_called_once_with()
        assert cs._metrics_requested == metrics_required


@pytest.mark.parametrize(
    "metrics_required",
    [
        {
            "unknown-metric",
        },
        {"recall", "unknown-metric"},
    ],
)
def test_CalibrationStrategy_request_raises_if_metrics_cannot_be_provided(metrics_required):
    with unittest.mock.patch("shaprank.ensure.string") as mock_string, unittest.mock.patch(
        "shaprank.calibration.strategies.CalibrationStrategy.provides"
    ) as mock_provides:
        name = unittest.mock.Mock()
        ensured_name = unittest.mock.Mock()
        mock_string.return_value = ensured_name
        mock_provides.return_value = {
            "recall",
        }

        cs = strategies.CalibrationStrategy(name)

        with pytest.raises(ValueError):
            cs.request(metrics_required)

        mock_provides.assert_called_once_with()


def test_BinaryClassifierCalibrationStrategy_instance_has_expected_fields():
    c_target_class = unittest.mock.Mock()
    metric = unittest.mock.Mock()
    metric_target = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.calibration.strategies.ensure.string"
    ) as mock_string, unittest.mock.patch(
        "shaprank.calibration.strategies.ensure.binary_classification_metric"
    ) as mock_binary_classification_metric:
        mock_string.return_value = c_target_class
        mock_binary_classification_metric.return_value = (metric, metric_target)

        strat = strategies.BinaryClassifierCalibrationStrategy(
            c_target_class, metric, metric_target
        )

        mock_string.assert_has_calls(
            [
                unittest.mock.call(
                    "BinaryClassifierCalibrationStrategy", allow_none=False, allow_empty=False
                ),
                unittest.mock.call(c_target_class, allow_none=False, allow_empty=False),
            ]
        )
        mock_binary_classification_metric.assert_called_once_with(
            metric, metric_target, allow_none_target=False
        )

        assert strat._c_target_class == c_target_class
        assert strat._metric == metric
        assert strat._metric_target == metric_target


@pytest.fixture
def BinaryClassifierCalibrationStrategy_instance():
    c_target_class = unittest.mock.Mock()
    metric = unittest.mock.Mock()
    metric_target = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.calibration.strategies.ensure.string"
    ) as mock_string, unittest.mock.patch(
        "shaprank.calibration.strategies.ensure.binary_classification_metric"
    ) as mock_binary_classification_metric:
        mock_string.return_value = c_target_class
        mock_binary_classification_metric.return_value = (metric, metric_target)

        strat = strategies.BinaryClassifierCalibrationStrategy(
            c_target_class, metric, metric_target
        )

        yield strat


def test_BinaryClassifierCalibrationStrategy_requires(
    BinaryClassifierCalibrationStrategy_instance,
):
    strat = BinaryClassifierCalibrationStrategy_instance

    res = strat.requires()

    assert res == {strat._c_target_class}


def test_BinaryClassifierCalibrationStrategy_provides(
    BinaryClassifierCalibrationStrategy_instance,
):
    strat = BinaryClassifierCalibrationStrategy_instance

    res = strat.provides()

    assert res == {"alert_rate", "fpr", "recall", "precision", "tpr"}


def test_BinaryClassifierCalibrationStrategy_run():
    n = 10
    c_target_class = "label"
    metric = "alert_rate"
    metric_target = 0.2

    c_greedy_prediction = "_shaprank_greedy_prediction_"
    df = pandas.DataFrame({"label": [True] * 10, c_greedy_prediction: numpy.linspace(0, 1, n)})
    strat = strategies.BinaryClassifierCalibrationStrategy(c_target_class, metric, metric_target)
    strat.request({"fpr", "recall"})

    res = strat.run(df, c_greedy_prediction)

    assert isinstance(res, strategies.CalibrationResult)
    numpy.testing.assert_almost_equal(res._threshold, 0.888888, decimal=6)
    assert res._metrics == {"fpr": 0, "recall": 0.2}
