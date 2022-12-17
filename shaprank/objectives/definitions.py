"""
ShapRank's abstractions for the optimization objective pursude during greedy-search
iterations.
"""

import abc
import copy
import logging
from typing import Any, Callable, Optional

import numpy
import pandas

from shaprank import ensure, evaluation, formatting

__all__ = [
    "ClassificationMetric",
    "ObjectiveFunctionResult",
    "ObjectiveFunction",
    "RegressionErrorMetric",
    "SoftConstrainedClassifier",
]


class ObjectiveFunctionResult:
    def __init__(self, value: float, metrics: dict[str, float]):
        self._value = value
        self._metrics = metrics

    def get(self) -> dict[str, float]:
        return copy.deepcopy(self._metrics)

    @property
    def value(self) -> float:
        if self._value is None:
            raise ValueError("Accessing to non-evaluated ObjectiveFunction")
        return self._value

    def get_notes(self) -> Optional[str]:
        return None


class ConstrainedObjectiveFunctionResult(ObjectiveFunctionResult):
    def __init__(self, value: float, metrics: dict[str, float]):
        super().__init__(value, metrics)

    def get_notes(self) -> Optional[str]:
        metrics = self.get()
        v_raw_objective = metrics["raw_objective"]
        v_constraint_penalty = metrics["constraint_penalty"]
        v_constraint_violation = metrics["constraint_violation"]
        if not v_constraint_violation:
            return None
        return (
            f"Raw objective {formatting.number(v_raw_objective)},"
            f" penalty {formatting.number(v_constraint_penalty)}"
            f" (violation {formatting.number(v_constraint_violation)})"
        )


class ObjectiveFunction:
    def __init__(self, name: str, optimization_primitive: Callable):
        name = ensure.string(name, allow_none=False, allow_empty=False)  # type: ignore[assignment]
        self._name: str = name

        if optimization_primitive not in (min, max):
            logging.warning(
                "Optimization primitive is neither min or max instead: `%s`",
                optimization_primitive,
            )
        self._optimization_primitive = optimization_primitive

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def requires(self) -> set[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(
        self,
        df: pandas.DataFrame,
        c_output: str,
        c_greedy_prediction: str,
        context: dict[str, Any],
    ) -> ObjectiveFunctionResult:
        raise NotImplementedError()

    def get_optimization_primitive(self) -> Callable:
        return self._optimization_primitive


class RegressionErrorMetric(ObjectiveFunction):
    def __init__(
        self,
        regression_error_metric: str,
    ):
        name = RegressionErrorMetric.__name__
        super().__init__(name, optimization_primitive=min)

        self._regression_error_metric = ensure.regression_error_metric(regression_error_metric)

    def requires(self) -> set[str]:
        return set()

    def evaluate(
        self,
        df: pandas.DataFrame,
        c_output: str,
        c_greedy_prediction: str,
        context: dict[str, Any],
    ) -> ObjectiveFunctionResult:

        v_target = df[c_output] - df[c_output].mean()
        v_shap_prediction = df[c_greedy_prediction] - df[c_greedy_prediction].mean()
        v_objective = evaluation.regression.eval_error_metric(
            v_target, v_shap_prediction, self._regression_error_metric
        )

        return ObjectiveFunctionResult(v_objective, {})


class ClassificationMetric(ObjectiveFunction):
    def __init__(self, metric: str):
        name = ClassificationMetric.__name__

        optimization_primitive_map = {
            "recall": max,
            "fpr": min,
            "tpr": max,
        }
        super().__init__(name, optimization_primitive_map[metric])

        self._metric, _ = ensure.binary_classification_metric(metric, None, allow_none_target=True)

    def requires(self) -> set[str]:
        return set((self._metric,))

    def evaluate(
        self,
        df: pandas.DataFrame,
        c_output: str,
        c_greedy_prediction: str,
        context: dict[str, Any],
    ) -> ObjectiveFunctionResult:
        value = context[self._metric]
        return ObjectiveFunctionResult(value=value, metrics=context)


class SoftConstrainedClassifier(ObjectiveFunction):
    def __init__(
        self,
        objective_metric: str,
        constraint_metric: str,
        constraint_metric_target: float,
        penalty_weight: float,
    ):
        name = SoftConstrainedClassifier.__name__

        optimization_primitive_map = {
            "recall": max,
            "fpr": min,
            "tpr": max,
        }
        super().__init__(name, optimization_primitive_map[objective_metric])

        self._objective_metric = objective_metric
        self._constraint_metric = constraint_metric
        self._constraint_metric_target = constraint_metric_target
        _penalty_weight = ensure.numeric(penalty_weight, allow_none=False, allow_nan=False)
        self._penalty_weight = float(_penalty_weight)  # type: ignore[arg-type]

    def requires(self) -> set[str]:
        return set([self._objective_metric, self._constraint_metric])

    def evaluate(
        self,
        df: pandas.DataFrame,
        c_output: str,
        c_greedy_prediction: str,
        context: dict[str, Any],
    ) -> ObjectiveFunctionResult:
        v_objective_metric = context[self._objective_metric]
        v_constraint_metric = context[self._constraint_metric]

        v_constraint_delta = numpy.abs(v_constraint_metric - self._constraint_metric_target)
        if self._constraint_metric in ("recall", "fpr", "alert_rate", "precision"):
            v_constraint_delta = -max(0, v_constraint_delta)
        else:
            raise ValueError(f"Unhandled constraint metric `{self._constraint_metric}`")

        v_constraint_penalty = v_constraint_delta * self._penalty_weight
        v_objective = v_objective_metric + v_constraint_penalty

        metrics = {
            "raw_objective": v_objective_metric,
            "constraint_violation": v_constraint_delta,
            "constraint_penalty": v_constraint_penalty,
        }
        return ConstrainedObjectiveFunctionResult(v_objective, metrics)
