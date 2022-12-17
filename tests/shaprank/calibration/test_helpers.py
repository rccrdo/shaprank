import unittest.mock

import numpy
import pytest

from shaprank.calibration import helpers


def test_calibrate_threshold_at_target_fpr():
    n = 100
    y_true_class = numpy.full(n, False, dtype=bool)
    y_pred_score = numpy.linspace(0.0, 1, n)

    # drop the extremes
    for target in numpy.linspace(0, 1, n)[1:-1]:
        res = helpers.calibrate_threshold_at_target_fpr(y_true_class, y_pred_score, target)
        numpy.testing.assert_almost_equal(res, min(1, 1 - target + 1 / n), decimal=4)


def test_calibrate_threshold_at_target_fpr_raises_if_no_negative_examples():

    y_true_class = [True, True]
    y_pred_score = [0.1, 0.9]

    with pytest.raises(ValueError):
        helpers.calibrate_threshold_at_target_fpr(y_true_class, y_pred_score, 1 / 2)


def test_calibrate_threshold_at_target_fpr_with_non_feasible_target():
    y_true_class = [False, False]
    y_pred_score = [0.1, 0.9]

    res = helpers.calibrate_threshold_at_target_fpr(
        y_true_class, y_pred_score, 1 / (len(y_true_class) + 1)
    )
    assert res == numpy.Inf


def test_calibrate_threshold_at_target_recall():
    n = 100
    y_true_class = numpy.full(n, True, dtype=bool)
    y_pred_score = numpy.linspace(0.0, 1.0, n)

    # Note: drop the extremes
    for target in numpy.linspace(0, 1, n)[1:-1]:
        res = helpers.calibrate_threshold_at_target_recall(y_true_class, y_pred_score, target)
        numpy.testing.assert_almost_equal(res, 1 - target, decimal=4)


def test_calibrate_threshold_at_target_alert_rate():
    n = 100
    y_pred_score = numpy.linspace(0.0, 1.0, n)

    # Note: drop the extremes
    for target in numpy.linspace(0, 1, n)[1:-1]:
        res = helpers.calibrate_threshold_at_target_alert_rate(y_pred_score, target)
        numpy.testing.assert_almost_equal(res, 1 - target + 1 / n, decimal=4)


def test_calibrate_threshold_at_target_alert_rate_handles_unfeasible_target():
    any_score = 1 / 2
    any_target_below_one = 1 / 2
    y_pred_score = [any_score] * 2

    res = helpers.calibrate_threshold_at_target_alert_rate(
        y_pred_score, target=any_target_below_one
    )
    assert res == -numpy.Inf


def test_calibrate_threshold_at_target_precision():
    n = 100
    y_true_class = numpy.full(n, True, dtype=bool)
    y_pred_score = numpy.linspace(0.0, 1, n)

    # drop the extremes
    for target in numpy.linspace(0, 1, n)[1:-1]:
        res = helpers.calibrate_threshold_at_target_precision(y_true_class, y_pred_score, target)
        numpy.testing.assert_almost_equal(res, 1 - target, decimal=4)


def test_calibrate_threshold_at_target_precision_raises_if_no_positive_examples():

    y_true_class = [False, False]
    y_pred_score = [0.1, 0.9]

    with pytest.raises(ValueError):
        helpers.calibrate_threshold_at_target_precision(y_true_class, y_pred_score, 1 / 2)


def test_calibrate_threshold_with_metric_alert_rate():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    metric_target = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.calibration.helpers.calibrate_threshold_at_target_alert_rate"
    ) as mock_impl:
        res = helpers.calibrate_threshold(y_true_class, y_pred_score, "alert_rate", metric_target)

        mock_impl.assert_called_once_with(y_pred_score, metric_target)
        assert res == mock_impl.return_value


def test_calibrate_threshold_with_metric_fpr():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    metric_target = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.calibration.helpers.calibrate_threshold_at_target_fpr"
    ) as mock_impl:
        res = helpers.calibrate_threshold(y_true_class, y_pred_score, "fpr", metric_target)

        mock_impl.assert_called_once_with(y_true_class, y_pred_score, metric_target)
        assert res == mock_impl.return_value


@pytest.mark.parametrize("metric", ["recall", "tpr"])
def test_calibrate_threshold_with_metric_recall_tpr(metric):
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    metric_target = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.calibration.helpers.calibrate_threshold_at_target_recall"
    ) as mock_impl:
        res = helpers.calibrate_threshold(y_true_class, y_pred_score, metric, metric_target)

        mock_impl.assert_called_once_with(y_true_class, y_pred_score, metric_target)
        assert res == mock_impl.return_value


def test_calibrate_threshold_with_metric_precision():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    metric_target = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.calibration.helpers.calibrate_threshold_at_target_precision"
    ) as mock_impl:
        res = helpers.calibrate_threshold(y_true_class, y_pred_score, "precision", metric_target)

        mock_impl.assert_called_once_with(y_true_class, y_pred_score, metric_target)
        assert res == mock_impl.return_value


def test_calibrate_threshold_raises_if_unhandled_metric():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    metric_target = unittest.mock.Mock()

    with pytest.raises(ValueError):
        helpers.calibrate_threshold(y_true_class, y_pred_score, "unknown-metric", metric_target)
