"""Utility routines for dataset loading and model fitting in use across the examples."""

import logging
import sys
from typing import Any, Optional

import catboost
import pandas
import sklearn.datasets


def load_dataset_breast_cancer() -> tuple[pandas.DataFrame, list[str, str]]:
    """
    Load the sklearn dataset on breast cancer, returning the data in a pandas frame
    together with the list of columns specifying the model inputs and the classification
    target / the labels.
    """
    dataset = sklearn.datasets.load_breast_cancer(as_frame=True)
    logging.info(dataset["DESCR"])

    df = dataset["data"]

    c_inputs = list(dataset["feature_names"])
    c_output = "target"
    target_classes = dataset["target_names"]

    df["target"] = [target_classes[idx] for idx in dataset["target"]]
    return df, c_inputs, c_output


def load_dataset_diabetes() -> tuple[pandas.DataFrame, list[str, str]]:
    """
    Load the sklearn dataset on diabetes, returning the data in a pandas frame
    together with the list of columns specifying the model inputs and the regression
    target.
    """
    dataset = sklearn.datasets.load_diabetes(as_frame=True)
    logging.info(dataset["DESCR"])

    df = dataset["data"]
    c_inputs = dataset["feature_names"]
    c_output = "target"
    df[c_output] = dataset["target"]
    return df, c_inputs, c_output


def _get_default_catboost_model_kwargs() -> dict[str, Any]:
    """
    Constructs and reducts the dictionary of default tree-building hyper-parameters
    used across the CatBoost examples.
    """
    return dict(od_type="Iter", od_wait=100, allow_writing_files=False, verbose=100)


def _fit_catboost_model(
    cb_model: catboost.core.CatBoost,
    df: pandas.DataFrame,
    c_inputs: list[str],
    c_output: str,
    c_categoricals: Optional[list[str]] = None,
) -> catboost.core.CatBoost:
    """
    Fits a CatBoost model on the given data.
    """
    pool = catboost.Pool(df[c_inputs], df[c_output], cat_features=c_categoricals)
    cb_model.fit(pool)
    sys.stdout.flush()
    return cb_model


def fit_catboost_classifier(
    df: pandas.DataFrame,
    c_inputs: list[str],
    c_output: str,
    c_categoricals: Optional[list[str]] = None,
) -> catboost.CatBoostClassifier():
    """
    Fits a CatBoost Classifier on the given data.
    """
    logging.info("Fitting the model ...")
    cb_kwargs = _get_default_catboost_model_kwargs()
    cb_model = catboost.CatBoostClassifier(**cb_kwargs)
    return _fit_catboost_model(cb_model, df, c_inputs, c_output, c_categoricals)


def fit_catboost_regressor(
    df: pandas.DataFrame,
    c_inputs: list[str],
    c_output: str,
    c_categoricals: Optional[list[str]] = None,
) -> catboost.CatBoostRegressor():
    """
    Fits a CatBoost Regressor on the given data.
    """
    cb_kwargs = _get_default_catboost_model_kwargs()
    cb_model = catboost.CatBoostRegressor(**cb_kwargs)
    return _fit_catboost_model(cb_model, df, c_inputs, c_output, c_categoricals)
