import unittest.mock

import pytest

from shaprank import quick_start


@pytest.mark.parametrize("verbose", [False, True])
def test_rank_classifier_features_executes_expected_calls(verbose):

    df_shap = unittest.mock.Mock()
    c_inputs = unittest.mock.Mock()
    c_output = unittest.mock.Mock()
    calib_metric = unittest.mock.Mock()
    calib_metric_target = unittest.mock.Mock()
    eval_metric = unittest.mock.Mock()
    calib_metric_penalty = unittest.mock.Mock()

    show_progress = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.quick_start.criteria"
    ) as mock_criteria, unittest.mock.patch("shaprank.quick_start.ShapRank") as mock_ShapRank:
        res = quick_start.rank_classifier_features(
            df_shap,
            c_inputs,
            c_output,
            calib_metric,
            calib_metric_target,
            eval_metric,
            calib_metric_penalty,
            verbose,
            show_progress,
        )

        mock_criteria.classification.SoftConstrainedClassificationMetric.assert_called_once_with(
            c_output, calib_metric, calib_metric_target, eval_metric, calib_metric_penalty
        )
        criterion = mock_criteria.classification.SoftConstrainedClassificationMetric.return_value
        mock_ShapRank.assert_called_once_with(c_inputs, c_output, criterion)
        gre = mock_ShapRank.return_value
        gre.rank.assert_called_once_with(df_shap, verbose=verbose, show_progress=show_progress)

        result = gre.rank.return_value
        if verbose:
            result.log_summary.assert_called_once_with()
        else:
            result.log_summary.assert_not_called()

        assert res == result


@pytest.mark.parametrize("verbose", [False, True])
def test_rank_regressor_features(verbose):

    df_shap = unittest.mock.Mock()
    c_inputs = unittest.mock.Mock()
    c_output = unittest.mock.Mock()
    eval_metric = unittest.mock.Mock()
    show_progress = unittest.mock.Mock()

    with unittest.mock.patch(
        "shaprank.quick_start.criteria"
    ) as mock_criteria, unittest.mock.patch("shaprank.quick_start.ShapRank") as mock_ShapRank:
        res = quick_start.rank_regressor_features(
            df_shap, c_inputs, c_output, eval_metric, verbose, show_progress
        )

        mock_criteria.regression.RegressionError.assert_called_once_with(eval_metric)
        criterion = mock_criteria.regression.RegressionError.return_value
        mock_ShapRank.assert_called_once_with(c_inputs, c_output, criterion)
        gre = mock_ShapRank.return_value
        gre.rank.assert_called_once_with(df_shap, verbose=verbose, show_progress=show_progress)

        result = gre.rank.return_value
        if verbose:
            result.log_summary.assert_called_once_with()
        else:
            result.log_summary.assert_not_called()

        assert res == result
