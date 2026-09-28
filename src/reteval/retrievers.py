"""Retrievers behind one interface: index(doc_ids, texts), then search(query, k) -> [(doc_id, score), ...]."""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .fusion import reciprocal_rank_fusion
from .text import tokenize

Hits = list[tuple[str, float]]


def top_k(scores: np.ndarray, doc_ids: list[str], k: int) -> Hits:
    """The k highest scores, best first (argpartition avoids sorting every document)."""
    k = min(k, len(scores))
    best = np.argpartition(-scores, k - 1)[:k]
    best = best[np.argsort(-scores[best], kind="stable")]
    return [(doc_ids[i], float(scores[i])) for i in best]


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def index(self, doc_ids: list[str], texts: list[str]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 10) -> Hits: ...

    def search_batch(self, queries: list[str], k: int = 10) -> list[Hits]:
        return [self.search(q, k) for q in queries]


class TfidfRetriever(Retriever):
    """TF-IDF vectors (unigrams + bigrams) ranked by cosine similarity."""

    def __init__(self, stem: bool = True, name: str = "TF-IDF") -> None:
        self.name = name
        self.vectorizer = TfidfVectorizer(tokenizer=lambda t: tokenize(t, stem), lowercase=False,
                                          token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)

    def index(self, doc_ids: list[str], texts: list[str]) -> None:
        self.doc_ids = list(doc_ids)
        self.matrix = self.vectorizer.fit_transform(texts)  # rows are L2-normalized

    def search(self, query: str, k: int = 10) -> Hits:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[Hits]:
        scores = (self.vectorizer.transform(queries) @ self.matrix.T).toarray()  # cosine similarity
        return [top_k(row, self.doc_ids, k) for row in scores]


class BM25Retriever(Retriever):
    """Okapi BM25 (rank_bm25). k1 controls term-frequency saturation, b the document-length penalty."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, stem: bool = True, name: str = "BM25") -> None:
        self.k1, self.b, self.stem, self.name = k1, b, stem, name

    def index(self, doc_ids: list[str], texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.doc_ids = list(doc_ids)
        self.bm25 = BM25Okapi([tokenize(t, self.stem) for t in texts], k1=self.k1, b=self.b)

    def search(self, query: str, k: int = 10) -> Hits:
        return top_k(self.bm25.get_scores(tokenize(query, self.stem)), self.doc_ids, k)


class DenseRetriever(Retriever):
    """Sentence-embedding model + FAISS exact inner-product index (cosine, since vectors are normalized)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", query_prefix: str = "",
                 name: str | None = None, cache_dir: Path | None = None, batch_size: int = 64) -> None:
        self.model_name, self.query_prefix = model_name, query_prefix
        self.name = name or model_name.split("/")[-1]
        self.cache_dir, self.batch_size = cache_dir, batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype("float32")

    def _cached_doc_vectors(self, doc_ids: list[str], texts: list[str]) -> np.ndarray:
        """Encoding thousands of documents takes minutes, so vectors are cached on disk per model and corpus."""
        if self.cache_dir is None:
            return self.encode(texts)
        key = hashlib.sha1("\n".join(doc_ids).encode()).hexdigest()[:12]
        path = Path(self.cache_dir) / f"{self.name}-{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.encode(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def index(self, doc_ids: list[str], texts: list[str]) -> None:
        import faiss

        self.doc_ids = list(doc_ids)
        vectors = self._cached_doc_vectors(self.doc_ids, texts)
        self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss_index.add(vectors)

    def search(self, query: str, k: int = 10) -> Hits:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[Hits]:
        scores, idx = self.faiss_index.search(self.encode([self.query_prefix + q for q in queries]), k)
        return [[(self.doc_ids[i], float(s)) for s, i in zip(row_s, row_i) if i >= 0]
                for row_s, row_i in zip(scores, idx)]


class HybridRetriever(Retriever):
    """Runs several retrievers and merges their rankings with Reciprocal Rank Fusion."""

    def __init__(self, retrievers: list[Retriever], depth: int = 100, rrf_k: int = 60, name: str = "Hybrid (RRF)") -> None:
        self.retrievers, self.depth, self.rrf_k, self.name = retrievers, depth, rrf_k, name

    def index(self, doc_ids: list[str], texts: list[str]) -> None:
        for r in self.retrievers:
            r.index(doc_ids, texts)

    def search(self, query: str, k: int = 10) -> Hits:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[Hits]:
        per_retriever = [r.search_batch(queries, self.depth) for r in self.retrievers]
        return [reciprocal_rank_fusion([hits[i] for hits in per_retriever], self.rrf_k)[:k] for i in range(len(queries))]
