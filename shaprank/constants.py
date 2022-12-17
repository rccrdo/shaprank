"""Common constants and enumerations."""

import enum


class RankingProblemType(enum.Enum):
    """
    Enum for the problem types handled by ShapRank, namely classifiration and regression
    """

    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
