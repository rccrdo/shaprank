"""Optional TreeSHAP-related utilities dedicated to CatBoost models."""

from collections.abc import Sequence
from typing import Optional, Union

import catboost  # type: ignore[import]
import pandas

from shaprank import ensure


def eval_shap_values(
    cb_model: Union[catboost.CatBoostClassifier, catboost.CatBoostRegressor],
    df: pandas.DataFrame,
    c_keep: Optional[Sequence[str]] = None,
    prefix: Optional[str] = None,
) -> pandas.DataFrame:
    """
    Evaluates the TreeSHAP values given a CatBoost model for each example in the given frame.
    Optionally preserves / copies columns in the source frame to the returned one.

    Parameters:
        cb_model: A trained CatBoost model
        df      : The frame of examples for which SHAP values shoud be evaluated
        c_keep  : The optional list of columns to copy to the output frame, optionally prefixed by
                  the value of `prefix`. Useful to copy, e.g., the column of targets.
        prefx   : The optional prefix value for the colums copied via the `c_keep` mechanism

    Returns:
        df_shap : The frame of TreeSHAP values
    """
    if not cb_model.feature_names_:
        raise ValueError("CatBoost model as empty features list")
    c_inputs: list[str] = list(cb_model.feature_names_)
    c_categoricals = [c_inputs[idx] for idx in cb_model.get_cat_feature_indices()]
    c_text = [c_inputs[idx] for idx in cb_model.get_text_feature_indices()]
    c_embeddings = [c_inputs[idx] for idx in cb_model.get_embedding_feature_indices()]

    c_keep = ensure.features(
        c_keep,
        allow_none=True,
        allow_empty=True,
    )
    if c_keep is None:
        c_keep = []
    prefix = ensure.string(prefix, allow_none=True, allow_empty=True)
    if not prefix:
        prefix = ""

    c_missing = set(c_inputs).difference(df.columns)
    if c_missing:
        n_missing = len(c_missing)
        n_show_examples = 5
        examples = sorted(c_missing, key=str.casefold)
        examples_str = ", ".join(examples[:n_show_examples])
        if n_show_examples < n_missing:
            examples_str += " ..."
        raise ValueError(f"Missing model inputs: {n_missing} | {examples_str}.")

    c_missing = set(c_keep).difference(df.columns)
    if c_missing:
        n_missing = len(c_missing)
        n_show_examples = 5
        examples = sorted(c_missing, key=str.casefold)
        examples_str = ", ".join(examples[:n_show_examples])
        if n_show_examples < n_missing:
            examples_str += " ..."
        raise ValueError(f"Missing `keep` columns: {n_missing} | {examples_str}.")

    c_all: list[str] = list(set(c_inputs).union(c_keep))
    df = df[c_all].reset_index()

    pool = catboost.Pool(
        df[c_inputs],
        label=None,
        cat_features=c_categoricals,
        text_features=c_text,
        embedding_features=c_embeddings,
    )
    shap = cb_model.get_feature_importance(pool, "ShapValues")

    df_shap = pandas.DataFrame(shap, columns=list(c_inputs) + ["shap-0"])
    for c in c_keep:
        df_shap[f"{prefix}{c}"] = df[c]

    return df_shap
