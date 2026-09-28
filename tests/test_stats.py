from reteval.metrics import per_query_ndcg
from reteval.stats import bootstrap_ci, paired_randomization_test


def test_identical_systems_are_not_significant():
    a = [0.1, 0.5, 0.9, 0.3] * 25
    assert paired_randomization_test(a, a) == 1.0


def test_consistent_improvement_is_significant():
    a = [0.6 + 0.01 * (i % 5) for i in range(100)]
    b = [x - 0.1 for x in a]
    assert paired_randomization_test(a, b) < 0.001
    low, high = bootstrap_ci(a, b)
    assert 0.099 < low <= high < 0.101


def test_noisy_small_difference_is_not_significant():
    a = [1.0, 0.0] * 10
    b = [0.0, 1.0] * 9 + [0.0, 0.0]
    assert paired_randomization_test(a, b) > 0.05


def test_per_query_ndcg():
    scores = per_query_ndcg({"q1": ["d1"], "q2": ["x"]}, {"q1": {"d1": 1}, "q2": {"d2": 1}})
    assert scores == {"q1": 1.0, "q2": 0.0}
