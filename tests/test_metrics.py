import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from reteval.metrics import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank

RANKED = ["d3", "d1", "d7", "d2", "d9"]
RELEVANT = {"d1": 1, "d2": 1}


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0
    assert reciprocal_rank(RANKED, {"d9": 1}, k=4) == 0.0  # hit at rank 5 is outside k


def test_ndcg_by_hand():
    # hits at ranks 2 and 4; ideal puts them at ranks 1 and 2
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 10) == pytest.approx(expected)


def test_perfect_and_empty_rankings():
    assert ndcg_at_k(["d1", "d2"], RELEVANT) == 1.0
    assert ndcg_at_k([], RELEVANT) == 0.0


def test_ndcg_matches_scikit_learn_with_graded_relevance():
    docs = ["a", "b", "c", "d", "e", "f"]
    relevant = {"b": 2, "d": 1, "f": 3}
    ranked = ["a", "b", "c", "d", "e", "f"]
    scores = np.array([[6, 5, 4, 3, 2, 1]])  # the ranking above as scores
    true = np.array([[relevant.get(d, 0) for d in docs]])
    assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(ndcg_score(true, scores, k=5))


def test_evaluate_averages_over_all_queries():
    qrels = {"q1": {"d1": 1}, "q2": {"d5": 1}}
    run = {"q1": ["d1", "d2"]}  # q2 missing -> scores 0
    result = evaluate(run, qrels)
    assert result["recall@1"] == 0.5
    assert result["mrr@10"] == 0.5
    assert result["ndcg@10"] == 0.5
