import numpy as np
import pytest

from reteval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, TfidfRetriever, top_k
from reteval.text import tokenize

DOC_IDS = ["cats", "dogs", "cars", "stocks"]
TEXTS = [
    "Cats are small domestic animals that purr and chase mice.",
    "Dogs are loyal animals; puppies grow into dogs that bark.",
    "Electric cars use batteries instead of petrol engines.",
    "Stock markets rise and fall with interest rates.",
]
VOCAB = sorted({w for t in TEXTS for w in tokenize(t)})


class BagOfWordsDense(DenseRetriever):
    """A dense retriever with a fake encoder, so tests need no model download."""

    def __init__(self, **kwargs):
        super().__init__(name="fake-dense", **kwargs)

    def encode(self, texts):
        vectors = np.array([[tokenize(t).count(w) for w in VOCAB] for t in texts], dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


@pytest.mark.parametrize("retriever", [TfidfRetriever(), BM25Retriever(), BagOfWordsDense()], ids=lambda r: r.name)
def test_finds_the_matching_document_first(retriever):
    retriever.index(DOC_IDS, TEXTS)
    assert retriever.search("Why do my puppies bark?", k=2)[0][0] == "dogs"
    assert retriever.search("battery powered car", k=1)[0][0] == "cars"


@pytest.mark.parametrize("retriever", [TfidfRetriever(), BM25Retriever(), BagOfWordsDense()], ids=lambda r: r.name)
def test_results_are_sorted_and_limited_to_k(retriever):
    retriever.index(DOC_IDS, TEXTS)
    hits = retriever.search("animals", k=3)
    scores = [s for _, s in hits]
    assert len(hits) == 3 and scores == sorted(scores, reverse=True)


def test_batch_search_equals_single_search():
    r = TfidfRetriever()
    r.index(DOC_IDS, TEXTS)
    queries = ["cats and mice", "interest rates"]
    assert r.search_batch(queries, 2) == [r.search(q, 2) for q in queries]


def test_hybrid_merges_retrievers():
    hybrid = HybridRetriever([BM25Retriever(), BagOfWordsDense()], depth=4)
    hybrid.index(DOC_IDS, TEXTS)
    assert hybrid.search("purring cats", k=1)[0][0] == "cats"


def test_dense_vectors_are_cached(tmp_path):
    first = BagOfWordsDense(cache_dir=tmp_path)
    first.index(DOC_IDS, TEXTS)
    assert len(list(tmp_path.glob("*.npy"))) == 1

    class Exploding(BagOfWordsDense):
        def encode(self, texts):
            if len(texts) > 1:
                raise AssertionError("documents should come from the cache")
            return super().encode(texts)

    cached = Exploding(cache_dir=tmp_path)
    cached.index(DOC_IDS, TEXTS)
    assert cached.search("stock market", 1)[0][0] == "stocks"


def test_top_k_orders_best_first():
    assert top_k(np.array([0.1, 0.9, 0.5]), ["a", "b", "c"], 2) == [("b", 0.9), ("c", 0.5)]


def test_tokenize_stems_and_drops_stop_words():
    assert tokenize("The cats are running") == ["cat", "run"]
    assert tokenize("The cats are running", stem=False) == ["cats", "running"]
