from shaprank.constants import RankingProblemType


def test_RankingProblemType_has_expected_fields():
    assert RankingProblemType.CLASSIFICATION.value == "Classification"
    assert RankingProblemType.REGRESSION.value == "Regression"

    fields = set(f for f in RankingProblemType)
    assert fields == {RankingProblemType.CLASSIFICATION, RankingProblemType.REGRESSION}
