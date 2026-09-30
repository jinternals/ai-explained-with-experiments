"""A small OpenSearch client: build an index, then run keyword, vector and hybrid RRF searches."""

import json
import os
import time

import requests

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://opensearch:9200")
BULK_SIZE = 500

session = requests.Session()


def call(method, path, body=None, **params):
    response = session.request(method, OPENSEARCH_URL + path, json=body, params=params, timeout=300)
    response.raise_for_status()
    return response.json()


def wait_until_ready(timeout_s=300):
    """Block until OpenSearch answers, and return its version."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return call("GET", "/")["version"]["number"]
        except requests.RequestException:
            time.sleep(2)
    raise SystemExit("OpenSearch did not start in time")


class SearchIndex:
    def __init__(self, name):
        self.name = name
        self._pipelines = set()

    # ------------------------------------------------------------ building

    def create(self, dimension):
        """(Re)create the index with a text field for BM25 and a vector field for k-NN."""
        session.delete(f"{OPENSEARCH_URL}/{self.name}")
        call("PUT", f"/{self.name}", {
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {
                "title": {"type": "text"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
                },
            }},
        })

    def add(self, documents, vectors):
        """Index documents ({"id", "title", "text"}) with their embeddings, in bulk."""
        for start in range(0, len(documents), BULK_SIZE):
            batch = zip(documents[start:start + BULK_SIZE], vectors[start:start + BULK_SIZE])
            lines = []
            for doc, vector in batch:
                lines.append(json.dumps({"index": {"_id": doc["id"]}}))
                lines.append(json.dumps({"title": doc["title"], "text": doc["text"], "embedding": vector.tolist()}))

            response = session.post(
                f"{OPENSEARCH_URL}/{self.name}/_bulk",
                data="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
                timeout=300,
            )
            response.raise_for_status()
            if response.json()["errors"]:
                raise RuntimeError(f"Bulk indexing failed for batch starting at {start}")

        call("POST", f"/{self.name}/_refresh")

    # ------------------------------------------------------------ searching

    def keyword(self, text, size):
        """BM25 keyword search. Returns [{"id", "score"}, ...], best first."""
        return self._search(self._keyword_query(text), size)

    def vector(self, vector, size):
        """k-NN search on the embeddings. Returns [{"id", "score"}, ...], best first."""
        return self._search(self._vector_query(vector, size), size)

    def hybrid_rrf(self, text, vector, depth, k=60, size=None):
        """Keyword + vector search, fused inside OpenSearch with its built-in RRF.

        `depth` is how many results each searcher contributes; `size` is how many fused results to return.
        """
        hybrid = {
            "hybrid": {
                "queries": [self._keyword_query(text), self._vector_query(vector, depth)],
                "pagination_depth": depth,
            }
        }
        return self._search(hybrid, size or depth, search_pipeline=self._rrf_pipeline(k))

    # ------------------------------------------------------------ internals

    @staticmethod
    def _keyword_query(text):
        # Title and text as separate fields, like BEIR's "multifield" BM25 baseline.
        # (BEIR used Anserini with k1=0.9, b=0.4; OpenSearch defaults to k1=1.2, b=0.75.)
        return {"multi_match": {"query": text, "fields": ["title", "text"], "tie_breaker": 0.5}}

    @staticmethod
    def _vector_query(vector, k):
        return {"knn": {"embedding": {"vector": vector.tolist(), "k": k}}}

    def _rrf_pipeline(self, k):
        name = f"rrf-{k}"
        if name not in self._pipelines:
            call("PUT", f"/_search/pipeline/{name}", {
                "phase_results_processors": [
                    {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": k}}}
                ]
            })
            self._pipelines.add(name)
        return name

    def _search(self, query, size, **params):
        body = {"size": size, "query": query, "_source": False}
        hits = call("POST", f"/{self.name}/_search", body, **params)["hits"]["hits"]
        return [{"id": hit["_id"], "score": hit["_score"]} for hit in hits]
