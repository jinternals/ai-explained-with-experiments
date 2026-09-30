"""Download and read BEIR datasets (https://github.com/beir-cellar/beir)."""

import csv
import io
import json
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATA_DIR = Path("/data")


def load(name, split="test"):
    """Return (documents, queries, qrels) for one BEIR dataset.

    documents: [{"id", "title", "text"}, ...]
    queries:   {query_id: text}                  only queries that have judgments
    qrels:     {query_id: {doc_id: relevance}}   expert relevance labels
    """
    folder = _download(name)

    documents = [
        {"id": row["_id"], "title": row.get("title", ""), "text": row["text"]}
        for row in _read_jsonl(folder / "corpus.jsonl")
    ]

    qrels = {}
    with open(folder / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    queries = {
        row["_id"]: row["text"]
        for row in _read_jsonl(folder / "queries.jsonl")
        if row["_id"] in qrels
    }
    return documents, queries, qrels


def _download(name):
    folder = DATA_DIR / name
    if not folder.exists():
        print(f"Downloading {name}...")
        response = requests.get(f"{BASE_URL}/{name}.zip", timeout=600)
        response.raise_for_status()
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(DATA_DIR)
    return folder


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
