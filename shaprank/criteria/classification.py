"""Classification-specific criteria for ShapRank's greedy-search iterations."""

import pandas

from shaprank import objectives
from shaprank.calibration.strategies import (
    BinaryClassifierCalibrationStrategy,
    CalibrationStrategy,
)
from shaprank.constants import RankingProblemType
from shaprank.criteria.definitions import RankingCriterionResult, ShapRankCriterion
from shaprank.objectives import ObjectiveFunction

__all__ = [
    "ClassificationMetric",
    "SoftConstrainedClassificationMetric",
]


class ClassifierRankingCriterion(ShapRankCriterion):
    def __init__(
        self,
        calibration_strategy: CalibrationStrategy,
        objective_function: ObjectiveFunction,
    ):
        super().__init__(RankingProblemType.CLASSIFICATION, objective_function)
        self._calibration_strategy = calibration_strategy

        mtr_required = objective_function.requires()
        calibration_strategy.request(mtr_required)
        self._calibration_strategy = calibration_strategy

    def evaluate(
        self, df: pandas.DataFrame, c_output: str, c_greedy_prediction: str
    ) -> RankingCriterionResult:
        calib_result = self._calibration_strategy.run(df, c_greedy_prediction)
        objective_result = self._objective_function.evaluate(
            df, c_output, c_greedy_prediction, calib_result.get()
        )
        return RankingCriterionResult(objective_result, calib_result)


class SoftConstrainedClassificationMetric(ClassifierRankingCriterion):
    def __init__(
        self,
        c_target_class: str,
        calib_metric: str,
        calib_metric_target: float,
        eval_metric: str,
        calib_metric_penalty: float,
    ):
        calib_strategy = BinaryClassifierCalibrationStrategy(
            c_target_class, calib_metric, calib_metric_target
        )
        eval_strategy = objectives.SoftConstrainedClassifier(
            eval_metric, calib_metric, calib_metric_target, calib_metric_penalty
        )

        super().__init__(calib_strategy, eval_strategy)


class ClassificationMetric(SoftConstrainedClassificationMetric):
    def __init__(
        self,
        c_target_class: str,
        calib_metric: str,
        calib_metric_target: float,
        eval_metric: str,
    ):
        super().__init__(
            c_target_class, calib_metric, calib_metric_target, eval_metric, calib_metric_penalty=0
        )
