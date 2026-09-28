"""Search SciFact with BM25, dense and hybrid retrieval side by side.

  python scripts/search.py "vitamin D and pregnancy outcomes"
  python scripts/search.py --query-id 1368       # a test query; relevant documents are marked ✓
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reteval.data import load  # noqa: E402
from reteval.fusion import weighted_fusion  # noqa: E402
from reteval.retrievers import BM25Retriever, DenseRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--query-id", help="use a SciFact test query and mark its relevant documents")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    ds = load("scifact", "test")
    if args.query_id:
        query, relevant = ds.queries[args.query_id], set(ds.qrels[args.query_id])
    elif args.query:
        query, relevant = args.query, set()
    else:
        parser.error("give a query or --query-id")

    # Use the settings chosen on the train split by run_benchmark.py.
    config = json.loads((ROOT / "results" / "config.json").read_text())
    bm25 = BM25Retriever(config["bm25"]["k1"], config["bm25"]["b"])
    dense = DenseRetriever("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ",
                           "Dense (BGE-small)", cache_dir=ROOT / "cache")
    for r in (bm25, dense):
        r.index(ds.doc_ids, list(ds.corpus.values()))

    lexical_hits, dense_hits = bm25.search(query, 100), dense.search(query, 100)
    results = {
        "BM25": lexical_hits,
        "Dense (BGE-small)": dense_hits,
        f"Hybrid (α={config['weighted_alpha']})": weighted_fusion(lexical_hits, dense_hits, config["weighted_alpha"]),
    }
    print(f"\nQuery: {query}\n")
    for name, hits in results.items():
        print(name)
        for rank, (doc_id, _) in enumerate(hits[: args.k], 1):
            mark = "✓" if doc_id in relevant else " "
            title = textwrap.shorten(ds.corpus[doc_id], 90, placeholder=" …")
            print(f"  {mark} {rank}. {title}")
        print()


if __name__ == "__main__":
    main()
