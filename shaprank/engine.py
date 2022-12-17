"""ShapRank's greedy-search ranking engines for classification and regression problems."""

import abc
import logging
import textwrap
from collections.abc import Sequence
from typing import Optional

import numpy
import pandas
import tabulate
import tqdm  # type: ignore[import]

from shaprank import ensure, formatting
from shaprank.constants import RankingProblemType
from shaprank.criteria import RankingCriterionResult, ShapRankCriterion

__all__ = [
    "ShapRankResult",
    "ShapRank",
]


class ShapRankIterationResult:
    def __init__(self, input_name: str, objective_value: float, notes: Optional[str]):
        self._input_name: str = ensure.feature(input_name)
        self._objective_value = ensure.numeric(objective_value, allow_none=False, allow_nan=False)
        self._notes = ensure.string(notes, allow_none=True, allow_empty=False)

    @property
    def input_name(self):
        return self._input_name

    @property
    def objective_value(self):
        return self._objective_value

    @property
    def notes(self) -> Optional[str]:
        return self._notes


class ShapRankResult(abc.ABC):
    def __init__(self, ranking: list[ShapRankIterationResult]):
        self._ranking = ranking

    def get_summary_description(self) -> str:
        return (
            "In the summary below, the top ranking features are deemed as most predictive."
            " Note, however, that the estimated performance will be less and less occurate"
            " as one moves from the bottom of this list up to the top (where correctness"
            " pends on the increasingly stronger assumptions on the joint distribution of"
            " the larger set of features)."
        )

    def compile_summary(self) -> str:
        lines = ["SHAP-Rank RankingResult"]

        text_desc = self.get_summary_description()
        for _line in textwrap.wrap(text_desc, width=80):
            lines.append(textwrap.indent(_line, prefix="  > "))

        c_name = "Importance"
        table_rows = []
        has_notes = any(iter_result.notes for iter_result in self._ranking)
        for idx, iter_result in enumerate(reversed(self._ranking)):
            row = {
                "Ranking": idx + 1,
                "Feature": iter_result.input_name,
                c_name: formatting.number(iter_result.objective_value),
            }

            if has_notes:
                row["Notes"] = iter_result.notes
            table_rows.append(row)

        table_headers = list(table_rows[0].keys())
        table_values = [[row[k] for k in table_headers] for row in table_rows]
        table_rankings = tabulate.tabulate(table_values, headers=table_headers)
        lines.append(textwrap.indent(table_rankings, " " * 2))
        return "\n".join(lines)

    def log_summary(self):
        logging.info(self.compile_summary())


class RankingEngine:
    _c_greedy_prediction: str = "_shaprank_greedy_target_prediction_"

    def __init__(
        self,
        c_inputs: list[str],
        c_output: str,
        criterion: ShapRankCriterion,
    ):
        self._c_inputs = c_inputs
        self._c_output = c_output
        self._criterion = criterion

    def _make_greedy_evaluation_frame(
        self, df: pandas.DataFrame, c_selection: list[str]
    ) -> pandas.DataFrame:
        v_greedy_prediction = df[c_selection].sum(axis=1)
        return pandas.DataFrame(
            {
                self._c_output: df[self._c_output].to_numpy(),
                self._c_greedy_prediction: v_greedy_prediction.to_numpy(),
            }
        )

    @abc.abstractmethod
    def rank(
        self, df: pandas.DataFrame, show_progress: bool = False, verbose: bool = False
    ) -> ShapRankResult:
        raise NotImplementedError()


class RegressorRankingEngine(RankingEngine):
    def _eval_criterion(self, df_greedy_eval: pandas.DataFrame) -> RankingCriterionResult:
        return self._criterion.evaluate(df_greedy_eval, self._c_output, self._c_greedy_prediction)

    def rank(
        self, df: pandas.DataFrame, show_progress: bool = False, verbose: bool = False
    ) -> ShapRankResult:
        n_inputs = len(self._c_inputs)

        loop_gen = range(n_inputs)
        if show_progress:
            loop_gen = tqdm.tqdm(loop_gen)

        # evaluate the baseline
        df_greedy_eval = self._make_greedy_evaluation_frame(df, self._c_inputs)
        result = self._eval_criterion(df_greedy_eval)
        baseline_objective = result.get_objective_result()
        del df_greedy_eval
        if verbose:
            logging.info(
                "Baseline objective value: %s", formatting.number(baseline_objective.value)
            )

        pending: set[str] = set(self._c_inputs)
        selection = []
        optimization_primitive = self._criterion.get_optimization_primitive()
        if optimization_primitive is min:
            v_start = numpy.Inf
        else:
            v_start = -numpy.Inf
        for _ in loop_gen:
            best_objective: float = v_start
            best_column: str
            for idx, c in enumerate(pending):
                c_sub_inputs = [k for k in pending if k != c]
                df_greedy_eval = self._make_greedy_evaluation_frame(df, c_sub_inputs)

                result = self._eval_criterion(df_greedy_eval)
                objective_result = result.get_objective_result()

                if (not idx) or best_objective != optimization_primitive(
                    best_objective, objective_result.value
                ):
                    best_objective = objective_result.value
                    best_column = c

            pending.discard(best_column)
            selection.append(ShapRankIterationResult(best_column, best_objective, notes=None))
        return ShapRankResult(selection)


class ClassifierRankingEngine(RankingEngine):
    def _eval_criterion(self, df_greedy_eval: pandas.DataFrame) -> RankingCriterionResult:
        return self._criterion.evaluate(df_greedy_eval, self._c_output, self._c_greedy_prediction)

    def rank(
        self, df: pandas.DataFrame, show_progress: bool = False, verbose: bool = False
    ) -> ShapRankResult:
        n_inputs = len(self._c_inputs)

        loop_gen = range(n_inputs)
        if show_progress:
            loop_gen = tqdm.tqdm(loop_gen)

        # evaluate the baseline
        df_greedy_eval = self._make_greedy_evaluation_frame(df, self._c_inputs)
        result = self._eval_criterion(df_greedy_eval)
        baseline_objective_result = result.get_objective_result()
        baseline_calib_result = result.get_calibration_result()
        del df_greedy_eval
        if verbose:
            logging.info(
                "Baseline objective value: %s | %s",
                formatting.number(baseline_objective_result.value),
                baseline_calib_result.get_summary(),
            )

        pending: set[str] = set(self._c_inputs)
        selection = []
        optimization_primitive = self._criterion.get_optimization_primitive()
        if optimization_primitive is min:
            v_start = numpy.Inf
        else:
            v_start = -numpy.Inf
        for _ in loop_gen:
            best_objective: float = v_start
            best_column: str
            for idx, c in enumerate(pending):
                c_sub_inputs = [k for k in pending if k != c]
                df_greedy_eval = self._make_greedy_evaluation_frame(df, c_sub_inputs)

                result = self._eval_criterion(df_greedy_eval)
                objective_result = result.get_objective_result()
                result.get_calibration_result()

                if (not idx) or best_objective != optimization_primitive(
                    best_objective, objective_result.value
                ):
                    best_objective = objective_result.value
                    best_column = c
                    best_objective_notes = objective_result.get_notes()

            pending.discard(best_column)
            selection.append(
                ShapRankIterationResult(best_column, best_objective, best_objective_notes)
            )
        return ShapRankResult(selection)


def _instantiate_engine(
    c_inputs: list[str], c_output: str, criterion: ShapRankCriterion
) -> RankingEngine:

    problem_type = criterion.get_problem_type()
    if problem_type == RankingProblemType.CLASSIFICATION:
        return ClassifierRankingEngine(c_inputs, c_output, criterion)
    if problem_type == RankingProblemType.REGRESSION:
        return RegressorRankingEngine(c_inputs, c_output, criterion)
    raise ValueError("Unhandled RankingProblemType: `{problem_type}`")


class ShapRank(abc.ABC):
    def __init__(self, c_inputs: Sequence[str], c_output: str, criterion: ShapRankCriterion):
        c_inputs = ensure.features(
            c_inputs, allow_none=False, allow_empty=False
        )  # type: ignore[assignment]

        self._c_inputs: list[str] = list(c_inputs)
        self._c_output: str = ensure.string(
            c_output, allow_none=False, allow_empty=False
        )  # type: ignore[assignment]

        self._engine = _instantiate_engine(self._c_inputs, self._c_output, criterion)

    def _get_required_raw_columns(self, df: pandas.DataFrame) -> pandas.DataFrame:
        # columns with the SHAP values and the target
        c_required = set(self._c_inputs)
        c_required.add(self._c_output)

        c_missing = c_required.difference(df.columns)
        if c_missing:
            n_missing = len(c_missing)
            n_show_examples = 5
            examples = sorted(c_missing, key=str.casefold)
            examples_str = ", ".join(examples[:n_show_examples])
            if n_show_examples < n_missing:
                examples_str += " ..."
            raise ValueError(
                f'Received a frame with missing column{"s" if n_missing > 1 else ""}:'
                f" {n_missing} | {examples_str}"
            )

        return df[list(c_required)]

    def rank(self, df: pandas.DataFrame, **kwargs) -> ShapRankResult:
        if df.empty:
            raise ValueError("Received an empty frame")

        df = self._get_required_raw_columns(df)
        return self._engine.rank(df, **kwargs)
