"""Regression-specific criteria for ShapRank's greedy-search iterations."""

import pandas

from shaprank import objectives
from shaprank.constants import RankingProblemType
from shaprank.criteria import RankingCriterionResult, ShapRankCriterion
from shaprank.objectives import ObjectiveFunction

__all__ = [
    "RegressionError",
]


class RegressorRankingCriterion(ShapRankCriterion):
    def __init__(
        self,
        objective_function: ObjectiveFunction,
    ):
        super().__init__(RankingProblemType.REGRESSION, objective_function)


class RegressionError(RegressorRankingCriterion):
    def __init__(self, error_metric: str):
        eval_strategy = objectives.RegressionErrorMetric(error_metric)

        super().__init__(eval_strategy)

    def evaluate(
        self, df: pandas.DataFrame, c_output: str, c_greedy_prediction: str
    ) -> RankingCriterionResult:

        objective_result = self._objective_function.evaluate(df, c_output, c_greedy_prediction, {})
        return RankingCriterionResult(objective_result)
