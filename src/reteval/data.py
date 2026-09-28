"""Download and load BEIR-format datasets: corpus, queries and relevance judgements (qrels)."""

import csv
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Dataset:
    name: str
    corpus: dict[str, str]              # doc id -> "title. text"
    queries: dict[str, str]             # query id -> query text
    qrels: dict[str, dict[str, int]]    # query id -> {relevant doc id: relevance grade}

    @property
    def doc_ids(self) -> list[str]:
        return list(self.corpus)


def download(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download and unzip a BEIR dataset once; later calls reuse the local copy."""
    target = data_dir / name
    if not (target / "corpus.jsonl").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} from BEIR...")
        with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=120) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(data_dir)
    return target


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def join_title(title: str, text: str) -> str:
    """Index the title with the text: titles are short and full of key terms."""
    title = title.strip()
    if not title:
        return text
    return f"{title} {text}" if title[-1] in ".?!" else f"{title}. {text}"


def load(name: str = "scifact", split: str = "test", data_dir: Path = DATA_DIR) -> Dataset:
    """Load one split. Only queries that have at least one relevant document are kept."""
    folder = download(name, data_dir)
    corpus = {d["_id"]: join_title(d.get("title", ""), d["text"]) for d in read_jsonl(folder / "corpus.jsonl")}
    qrels = read_qrels(folder / "qrels" / f"{split}.tsv")
    queries = {q["_id"]: q["text"] for q in read_jsonl(folder / "queries.jsonl") if q["_id"] in qrels}
    return Dataset(name, corpus, queries, qrels)
