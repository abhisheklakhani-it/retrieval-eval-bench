import pytest

from reteval.fusion import min_max, reciprocal_rank_fusion, weighted_fusion


def test_rrf_rewards_documents_ranked_well_by_both():
    bm25 = [("a", 12.0), ("b", 9.0), ("c", 1.0)]
    dense = [("b", 0.9), ("c", 0.8), ("a", 0.1)]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    assert [d for d, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_includes_documents_found_by_only_one_retriever():
    fused = dict(reciprocal_rank_fusion([[("a", 1.0)], [("z", 1.0)]]))
    assert fused.keys() == {"a", "z"}


def test_min_max_scales_to_unit_range():
    assert min_max([("a", 10.0), ("b", 5.0), ("c", 0.0)]) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert min_max([("a", 3.0)]) == {"a": 0.0}
    assert min_max([]) == {}


def test_weighted_fusion_alpha_extremes_follow_one_retriever():
    lexical = [("a", 9.0), ("b", 1.0)]
    dense = [("b", 0.9), ("a", 0.2)]
    assert weighted_fusion(lexical, dense, alpha=0.0)[0][0] == "a"
    assert weighted_fusion(lexical, dense, alpha=1.0)[0][0] == "b"
