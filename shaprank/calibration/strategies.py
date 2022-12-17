"""Calibration strategies available to ShapRank's greedy search algorithm."""

import abc
import copy
from typing import Optional

import pandas

from shaprank import ensure, formatting
from shaprank.calibration.helpers import calibrate_threshold
from shaprank.evaluation.classification import evaluate_calibrated_metrics

__all__ = [
    "BinaryClassifierCalibrationStrategy",
    "CalibrationResult",
    "CalibrationStrategy",
]


class CalibrationResult:
    def __init__(self, threshold: Optional[float], metrics: dict[str, float]):
        self._threshold = ensure.numeric(threshold, allow_none=True, allow_nan=False)
        self._metrics = metrics

    def get(self) -> dict[str, float]:
        return copy.deepcopy(self._metrics)

    def get_summary(self) -> str:
        args = [f"{k}={formatting.number(v)}" for k, v in self._metrics.items()]
        return ", ".join(args)


class CalibrationStrategy:
    def __init__(self, name: str):
        name = ensure.string(name, allow_none=False, allow_empty=False)  # type: ignore[assignment]
        self._name: str = name
        self._metrics_requested: set[str] = set()

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def requires(self) -> set[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def provides(self) -> set[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, df: pandas.DataFrame, c_greedy_prediction: str) -> CalibrationResult:
        raise NotImplementedError()

    def request(self, metrics_required: set[str]) -> None:
        mtr_provided = self.provides()
        mtr_missing = metrics_required.difference(mtr_provided)

        if mtr_missing:
            n_missing = len(mtr_missing)
            n_show_examples = 5

            examples = sorted(mtr_missing, key=str.casefold)
            examples_str = ", ".join(examples[:n_show_examples])
            if n_show_examples < n_missing:
                examples_str += " ..."
            raise ValueError(
                f"Calibration strategy {self.name} does not provide all the requested"
                f" metrics: {n_missing} missing | {examples_str}."
            )
        self._metrics_requested = copy.deepcopy(metrics_required)


class BinaryClassifierCalibrationStrategy(CalibrationStrategy):
    def __init__(self, c_target_class: str, metric: str, metric_target: float):
        name = BinaryClassifierCalibrationStrategy.__name__
        super().__init__(name)

        c_target_class = ensure.string(
            c_target_class, allow_none=False, allow_empty=False
        )  # type: ignore[assignment]

        metric, metric_target = ensure.binary_classification_metric(  # type: ignore[assignment]
            metric, metric_target, allow_none_target=False
        )
        self._c_target_class: str = c_target_class

        self._metric: str = metric
        self._metric_target: float = metric_target

    def requires(self) -> set[str]:
        return set([self._c_target_class])

    def provides(self) -> set[str]:
        return set(["alert_rate", "fpr", "recall", "precision", "tpr"])

    def run(self, df: pandas.DataFrame, c_greedy_prediction: str) -> CalibrationResult:
        target_class = df[self._c_target_class].to_numpy()
        target_prediction = df[c_greedy_prediction].to_numpy()

        threshold = calibrate_threshold(
            target_class, target_prediction, self._metric, self._metric_target
        )

        metrics = evaluate_calibrated_metrics(
            target_class, target_prediction, threshold, self._metrics_requested
        )
        return CalibrationResult(threshold, metrics)
