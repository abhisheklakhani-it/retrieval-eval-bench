"""Ranking metrics. `ranked` is a list of doc ids, best first; `relevant` maps doc id -> relevance grade (> 0)."""

import math


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant.keys()) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """1 / rank of the first relevant document in the top k, or 0 if there is none."""
    for rank, doc_id in enumerate(ranked[:k], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Normalized discounted cumulative gain: gains discounted by log2(rank + 1), divided by the best possible.

    Uses linear gain (the relevance grade), as trec_eval and the BEIR benchmark do.
    """
    dcg = sum(relevant.get(doc_id, 0) / math.log2(rank + 1) for rank, doc_id in enumerate(ranked[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Average each metric over all queries in qrels (a query missing from the run scores 0)."""
    totals: dict[str, float] = {f"recall@{k}": 0.0 for k in ks} | {"mrr@10": 0.0, "ndcg@10": 0.0}
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
        totals["mrr@10"] += reciprocal_rank(ranked, relevant, 10)
        totals["ndcg@10"] += ndcg_at_k(ranked, relevant, 10)
    return {name: value / len(qrels) for name, value in totals.items()}


def per_query_ndcg(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10) -> dict[str, float]:
    """nDCG@k for each query separately (needed for significance tests)."""
    return {qid: ndcg_at_k(run.get(qid, []), relevant, k) for qid, relevant in qrels.items()}
