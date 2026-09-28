"""Benchmark TF-IDF, BM25, dense and hybrid retrieval on a BEIR dataset.

Parameters are tuned on the train split and reported on the test split, so test scores are not
inflated by tuning. Writes CSV and Markdown tables to results/.

  python scripts/run_benchmark.py            # SciFact (5,183 docs, 809 train / 300 test queries)
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reteval.benchmark import Result, rrf_run, run_queries, score, weighted_run  # noqa: E402
from reteval.data import load  # noqa: E402
from reteval.metrics import per_query_ndcg  # noqa: E402
from reteval.stats import bootstrap_ci, paired_randomization_test  # noqa: E402
from reteval.retrievers import BM25Retriever, DenseRetriever, TfidfRetriever  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
MAIN_METRIC = "ndcg@10"

DENSE_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "", "Dense (MiniLM-L6)"),
    # BGE models expect this instruction in front of retrieval queries (see the model card).
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", "Dense (BGE-small)"),
]
BM25_GRID = {"k1": [0.9, 1.2, 1.5, 2.0], "b": [0.5, 0.75, 0.9]}
ALPHAS = [round(0.1 * i, 1) for i in range(11)]  # 0 = BM25 only, 1 = dense only


def log(msg: str) -> None:
    print(msg, flush=True)


def save(df: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    (RESULTS / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    args = parser.parse_args()

    train, test = load(args.dataset, "train"), load(args.dataset, "test")
    doc_ids, texts = test.doc_ids, list(test.corpus.values())
    log(f"{args.dataset}: {len(doc_ids)} docs, {len(train.queries)} train / {len(test.queries)} test queries")
    tuning = []

    # 1) BM25: grid-search k1 and b on train.
    best_bm25, best_ndcg = None, -1.0
    for k1, b in itertools.product(BM25_GRID["k1"], BM25_GRID["b"]):
        r = BM25Retriever(k1=k1, b=b)
        r.index(doc_ids, texts)
        ndcg = score("", run_queries(r, train)[0], train).metrics[MAIN_METRIC]
        tuning.append({"step": "BM25 k1,b", "setting": f"k1={k1}, b={b}", "train_ndcg@10": round(ndcg, 4)})
        log(f"  BM25 k1={k1} b={b}: train nDCG@10 {ndcg:.4f}")
        if ndcg > best_ndcg:
            best_bm25, best_ndcg = (k1, b), ndcg
    log(f"Best BM25 on train: k1={best_bm25[0]}, b={best_bm25[1]}")

    # 2) Build the final retrievers with the chosen settings.
    retrievers = [TfidfRetriever(), BM25Retriever(*best_bm25, name="BM25")]
    retrievers += [DenseRetriever(m, prefix, name, cache_dir=CACHE) for m, prefix, name in DENSE_MODELS]
    ablations = [TfidfRetriever(stem=False, name="TF-IDF, no stemming"),
                 BM25Retriever(*best_bm25, stem=False, name="BM25, no stemming")]

    runs, results = {}, {}
    for r in retrievers + ablations:
        log(f"Indexing and searching: {r.name}")
        r.index(doc_ids, texts)
        runs[r.name] = {split: run_queries(r, ds) for split, ds in [("train", train), ("test", test)]}
        test_run, ms = runs[r.name]["test"]
        results[r.name] = score(r.name, test_run, test, ms)

    # 3) Hybrids: BM25 + the dense model that is best on TRAIN (never choose on test).
    dense_names = [name for _, _, name in DENSE_MODELS]
    best_dense = max(dense_names, key=lambda n: score(n, runs[n]["train"][0], train).metrics[MAIN_METRIC])
    log(f"Best dense model on train: {best_dense}")
    bm25_ms, dense_ms = runs["BM25"]["test"][1], runs[best_dense]["test"][1]

    rrf_name = "Hybrid RRF (BM25 + BGE)" if "BGE" in best_dense else f"Hybrid RRF (BM25 + {best_dense})"
    test_runs = {name: r["test"][0] for name, r in runs.items()}
    test_runs[rrf_name] = rrf_run([test_runs["BM25"], test_runs[best_dense]])
    results[rrf_name] = score(rrf_name, test_runs[rrf_name], test, bm25_ms + dense_ms)

    best_alpha, best_ndcg = None, -1.0
    for alpha in ALPHAS:
        ndcg = score("", weighted_run(runs["BM25"]["train"][0], runs[best_dense]["train"][0], alpha), train).metrics[MAIN_METRIC]
        tuning.append({"step": "hybrid alpha", "setting": f"alpha={alpha}", "train_ndcg@10": round(ndcg, 4)})
        if ndcg > best_ndcg:
            best_alpha, best_ndcg = alpha, ndcg
    log(f"Best weighted-fusion alpha on train: {best_alpha}")
    w_name = rrf_name.replace("RRF", f"weighted, α={best_alpha}")
    test_runs[w_name] = weighted_run(test_runs["BM25"], test_runs[best_dense], best_alpha)
    results[w_name] = score(w_name, test_runs[w_name], test, bm25_ms + dense_ms)

    # Significance: paired tests on per-query nDCG@10 over the same test queries.
    def ndcg_list(name: str) -> list[float]:
        per_query = per_query_ndcg({q: [d for d, _ in hits] for q, hits in test_runs[name].items()}, test.qrels)
        return [per_query[q] for q in test.qrels]

    comparisons = [("BM25", "TF-IDF"), (best_dense, "BM25"), (best_dense, "Dense (MiniLM-L6)"),
                   (rrf_name, best_dense), (w_name, best_dense), (w_name, rrf_name)]
    significance = []
    for a, b in comparisons:
        xa, xb = ndcg_list(a), ndcg_list(b)
        low, high = bootstrap_ci(xa, xb)
        p = paired_randomization_test(xa, xb)
        significance.append({"A": a, "B": b, "ndcg@10 A - B": round(sum(xa) / len(xa) - sum(xb) / len(xb), 4),
                             "95% CI": f"[{low:+.3f}, {high:+.3f}]", "p-value": round(p, 4),
                             "significant (p < 0.05)": "yes" if p < 0.05 else "no"})

    # 4) Save.
    main_names = [r.name for r in retrievers] + [rrf_name, w_name]
    table = pd.DataFrame([results[n].row() for n in main_names])
    save(table, "results")
    save(pd.DataFrame([results[r.name].row() for r in [retrievers[0], retrievers[1]] + ablations]), "ablation_stemming")
    save(pd.DataFrame(tuning), "tuning")
    save(pd.DataFrame(significance), "significance")
    per_query = pd.DataFrame({"query_id": list(test.qrels)} | {n: ndcg_list(n) for n in main_names})
    per_query.to_csv(RESULTS / "per_query_ndcg.csv", index=False)
    (RESULTS / "config.json").write_text(json.dumps({
        "dataset": args.dataset, "split": "test", "queries": len(test.queries), "docs": len(doc_ids),
        "bm25": {"k1": best_bm25[0], "b": best_bm25[1]}, "hybrid_dense_model": best_dense,
        "rrf_k": 60, "weighted_alpha": best_alpha, "depth_for_fusion": 100,
    }, indent=2) + "\n")
    log("\n" + table.to_markdown(index=False, floatfmt=".3f"))
    log("\n" + pd.DataFrame(significance).to_markdown(index=False))


if __name__ == "__main__":
    main()
