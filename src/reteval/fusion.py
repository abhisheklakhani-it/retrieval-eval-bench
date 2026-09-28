"""Ways to merge the rankings of several retrievers into one."""

Hits = list[tuple[str, float]]


def reciprocal_rank_fusion(rankings: list[Hits], k: int = 60) -> Hits:
    """RRF (Cormack et al., 2009): score(d) = sum over rankings of 1 / (k + rank of d).

    Uses only ranks, not scores, so BM25 scores (0 to ~50) and cosine scores (-1 to 1) need
    no rescaling. k = 60 is the value from the paper; it damps the advantage of the very top ranks.
    """
    fused: dict[str, float] = {}
    for hits in rankings:
        for rank, (doc_id, _) in enumerate(hits, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def min_max(hits: Hits) -> dict[str, float]:
    """Rescale one ranking's scores to 0..1 so different retrievers become comparable."""
    if not hits:
        return {}
    scores = [s for _, s in hits]
    low, high = min(scores), max(scores)
    span = (high - low) or 1.0
    return {doc_id: (s - low) / span for doc_id, s in hits}


def weighted_fusion(lexical: Hits, dense: Hits, alpha: float) -> Hits:
    """alpha * dense + (1 - alpha) * lexical on min-max scaled scores; a doc missing from a list scores 0 there."""
    a, b = min_max(lexical), min_max(dense)
    fused = {d: alpha * b.get(d, 0.0) + (1 - alpha) * a.get(d, 0.0) for d in a.keys() | b.keys()}
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)
