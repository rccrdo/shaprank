"""ShapRank's base Criterion abstractions."""

import abc
from typing import Callable, Optional

import pandas

from shaprank.calibration.strategies import CalibrationResult
from shaprank.constants import RankingProblemType
from shaprank.objectives import ObjectiveFunction, ObjectiveFunctionResult

__all__ = ["ShapRankCriterion", "RankingCriterionResult"]


class RankingCriterionResult:
    def __init__(
        self,
        evaluation_result: ObjectiveFunctionResult,
        calibration_result: Optional[CalibrationResult] = None,
    ):
        self._evaluation_result = evaluation_result
        self._calibration_result = calibration_result

    def get_objective_result(self) -> ObjectiveFunctionResult:
        return self._evaluation_result

    def get_calibration_result(self) -> CalibrationResult:
        if self._calibration_result is None:
            raise ValueError("No calibration result is available")
        return self._calibration_result


class ShapRankCriterion:
    def __init__(
        self,
        problem_type: RankingProblemType,
        objective_function: ObjectiveFunction,
    ):
        self._problem_type = problem_type
        self._objective_function = objective_function

    def get_problem_type(self) -> RankingProblemType:
        return self._problem_type

    def get_optimization_primitive(self) -> Callable:
        return self._objective_function.get_optimization_primitive()

    @abc.abstractmethod
    def evaluate(
        self, df: pandas.DataFrame, c_output: str, c_greedy_prediction: str
    ) -> RankingCriterionResult:
        raise NotImplementedError()
