import json

from reteval.data import join_title, load


def write_beir(folder):
    (folder / "qrels").mkdir(parents=True)
    docs = [{"_id": "d1", "title": "Cats", "text": "Cats purr."}, {"_id": "d2", "title": "", "text": "Dogs bark."}]
    (folder / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    queries = [{"_id": "q1", "text": "Do cats purr?"}, {"_id": "q2", "text": "No judgements"}]
    (folder / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n")


def test_load_reads_corpus_queries_and_positive_qrels(tmp_path):
    write_beir(tmp_path / "tiny")
    ds = load("tiny", "test", data_dir=tmp_path)  # corpus.jsonl exists, so nothing is downloaded
    assert ds.corpus == {"d1": "Cats. Cats purr.", "d2": "Dogs bark."}
    assert ds.queries == {"q1": "Do cats purr?"}  # q2 has no relevant documents
    assert ds.qrels == {"q1": {"d1": 1}}           # score 0 means "not relevant"


def test_join_title():
    assert join_title("A title", "Text.") == "A title. Text."
    assert join_title("A question?", "Text.") == "A question? Text."
    assert join_title("", "Text.") == "Text."
