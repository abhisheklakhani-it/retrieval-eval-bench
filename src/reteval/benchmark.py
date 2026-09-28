"""Run retrievers over a dataset's queries and score them."""

import time
from dataclasses import dataclass

from .data import Dataset
from .fusion import reciprocal_rank_fusion, weighted_fusion
from .metrics import evaluate
from .retrievers import Hits, Retriever

Run = dict[str, Hits]  # query id -> ranked hits


@dataclass
class Result:
    name: str
    metrics: dict[str, float]
    ms_per_query: float | None = None

    def row(self) -> dict:
        row = {"retriever": self.name, **{k: round(v, 4) for k, v in self.metrics.items()}}
        row["ms_per_query"] = round(self.ms_per_query, 2) if self.ms_per_query is not None else None
        return row


def run_queries(retriever: Retriever, dataset: Dataset, depth: int = 100) -> tuple[Run, float]:
    """Search every query (top `depth` hits, deep enough for fusion); returns the run and ms per query."""
    qids = list(dataset.queries)
    start = time.perf_counter()
    hits = retriever.search_batch([dataset.queries[q] for q in qids], depth)
    ms = (time.perf_counter() - start) * 1000 / len(qids)
    return dict(zip(qids, hits)), ms


def score(name: str, run: Run, dataset: Dataset, ms: float | None = None) -> Result:
    ranked = {qid: [doc_id for doc_id, _ in hits] for qid, hits in run.items()}
    return Result(name, evaluate(ranked, dataset.qrels), ms)


def rrf_run(runs: list[Run], k: int = 60) -> Run:
    return {qid: reciprocal_rank_fusion([r[qid] for r in runs], k) for qid in runs[0]}


def weighted_run(lexical: Run, dense: Run, alpha: float) -> Run:
    return {qid: weighted_fusion(lexical[qid], dense[qid], alpha) for qid in lexical}
