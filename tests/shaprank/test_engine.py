from shaprank.engine import RankingEngine


def class_RankingEngine_has_expected_defaults():
    RankingEngine._c_greedy_prediction
    assert isinstance(c_greedy_prediction, str)
    assert c_greedy_prediction
    assert c_greedy_prediction == "_shaprank_greedy_target_prediction_"
