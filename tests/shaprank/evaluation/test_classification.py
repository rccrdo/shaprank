import unittest.mock

import numpy
import pytest

from shaprank.evaluation import classification


@pytest.mark.parametrize(
    "y_pred_class,exp",
    [
        ([0, 0, 0], 0),
        ([-1, -1, -1], 0),
        ([-1, -1, 1], 1 / 3),
        ([-1, 0, 1], 1 / 3),
        ([-1, 1, 1], 2 / 3),
    ],
)
def test_evaluate_alert_rate(y_pred_class, exp):
    res = classification.evaluate_alert_rate(y_pred_class)

    assert res == exp


@pytest.mark.parametrize(
    "y_true_class,y_pred_class,exp",
    [
        ([True, 1, 1], [0, 0, 0], 0),
        ([True, 1, 1], [1, 1, 1], 0),
        ([False, 0, -1], [1, 1, 1], 1),
        ([False, 0, -1], [1, 1, -1], 2 / 3),
    ],
)
def test_evaluate_fpr(y_true_class, y_pred_class, exp):
    res = classification.evaluate_fpr(y_true_class, y_pred_class)

    assert res == exp


@pytest.mark.parametrize(
    "y_true_class,y_pred_class,exp",
    [
        ([True, 1, 1], [0, 0, 0], 0),
        ([True, 1, 1], [1, 1, 1], 0),
        ([False, 0, -1], [1, 1, 1], 1),
        ([False, 0, -1], [1, 1, -1], 2 / 3),
    ],
)
def test_evaluate_fpr(y_true_class, y_pred_class, exp):
    res = classification.evaluate_fpr(y_true_class, y_pred_class)

    assert res == exp


@pytest.mark.parametrize(
    "y_true_class,y_pred_class,exp",
    [
        ([True, 1, 1], [0, 0, 0], 0),
        ([True, 1, 1], [1, 1, 1], 1),
        ([False, 0, -1], [1, 1, 1], 0),
        ([1, -1, True], [1, 1, -1], 1 / 2),
    ],
)
def test_evaluate_recall(y_true_class, y_pred_class, exp):
    res = classification.evaluate_recall(y_true_class, y_pred_class)

    assert res == exp


@pytest.mark.parametrize(
    "y_true_class,y_pred_class,exp",
    [
        ([True, 1, 1], [0, 0, 0], 1),
        ([True, 1, 1], [1, 1, 1], 1),
        ([False, 0, -1], [1, 1, 1], 0),
        ([1, -1, False], [1, 1, 1], 1 / 3),
    ],
)
def test_evaluate_precision(y_true_class, y_pred_class, exp):
    res = classification.evaluate_precision(y_true_class, y_pred_class)

    assert res == exp


def test_evaluate_metric_with_metric_alert_rate():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    y_pred_class = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.evaluation.classification.evaluate_alert_rate"
    ) as mock_impl:
        res = classification.evaluate_metric(
            y_true_class, y_pred_score, y_pred_class, "alert_rate"
        )

        mock_impl.assert_called_once_with(y_pred_class)
        assert res == mock_impl.return_value


def test_evaluate_metric_with_metric_fpr():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    y_pred_class = unittest.mock.Mock()

    with unittest.mock.patch("shaprank.evaluation.classification.evaluate_fpr") as mock_impl:
        res = classification.evaluate_metric(y_true_class, y_pred_score, y_pred_class, "fpr")

        mock_impl.assert_called_once_with(y_true_class, y_pred_class)
        assert res == mock_impl.return_value


@pytest.mark.parametrize("metric", ["recall", "tpr"])
def test_evaluate_metric_with_metric_recall_tpr(metric):
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    y_pred_class = unittest.mock.Mock()

    with unittest.mock.patch("shaprank.evaluation.classification.evaluate_recall") as mock_impl:
        res = classification.evaluate_metric(y_true_class, y_pred_score, y_pred_class, metric)

        mock_impl.assert_called_once_with(y_true_class, y_pred_class)
        assert res == mock_impl.return_value


def test_evaluate_metric_with_metric_recall_precision():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    y_pred_class = unittest.mock.Mock()

    with unittest.mock.patch("shaprank.evaluation.classification.evaluate_precision") as mock_impl:
        res = classification.evaluate_metric(y_true_class, y_pred_score, y_pred_class, "precision")

        mock_impl.assert_called_once_with(y_true_class, y_pred_class)
        assert res == mock_impl.return_value


def test_evaluate_metric_raises_if_unhandled_metric():
    y_true_class = unittest.mock.Mock()
    y_pred_score = unittest.mock.Mock()
    y_pred_class = unittest.mock.Mock()

    with pytest.raises(ValueError):
        classification.evaluate_metric(y_true_class, y_pred_score, y_pred_class, "unknown-metric")


def test_evaluate_calibrated_metrics_invokes_evaluation_for_requested_metrics():
    y_true_class = [0, 1]
    y_pred_score = numpy.array([0.1, 0.9])
    threshold = 1 / 2
    mtr1 = unittest.mock.Mock()
    mtr2 = unittest.mock.Mock()
    mtr3 = unittest.mock.Mock()
    metrics_requested = [mtr1, mtr2, mtr3]

    mock_any = unittest.mock.ANY
    with unittest.mock.patch(
        "shaprank.evaluation.classification.evaluate_metric"
    ) as mock_evaluate_metric:

        res = classification.evaluate_calibrated_metrics(
            y_true_class, y_pred_score, threshold, metrics_requested
        )

        assert list(res.keys()) == metrics_requested
        assert all(v == mock_evaluate_metric.return_value for v in res.values())
        mock_evaluate_metric.assert_has_calls(
            [
                unittest.mock.call(mock_any, mock_any, mock_any, mtr1),
                unittest.mock.call(mock_any, mock_any, mock_any, mtr2),
                unittest.mock.call(mock_any, mock_any, mock_any, mtr3),
            ]
        )
