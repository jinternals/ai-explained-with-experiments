"""
Does RRF actually help? Measure it on BEIR datasets that come with expert relevance judgments.

For every test query we run keyword search (BM25), meaning search (MiniLM + k-NN) and RRF of the two,
then score each ranking with nDCG@10 (quality of the top 10) and Recall@100 (how much it found).

We also check:
  - how the RRF constant k changes quality
  - how many results each searcher should contribute (the depth)
  - that OpenSearch's built-in RRF gives the same quality as our own rrf()
"""

import json
from statistics import mean

import beir
from embeddings import embed
from engine import SearchIndex, wait_until_ready
from fusion import rrf
from metrics import ndcg_at_k, recall_at_k

DATASETS = ["nfcorpus", "scifact", "fiqa"]
DEPTH = 100                                # results each searcher contributes
K = 60                                     # RRF constant
K_SWEEP = [0, 1, 10, 30, 60, 100, 500]
DEPTH_SWEEP = [5, 10, 20, 50, 100]
EXAMPLES_PER_KIND = 5


def main():
    version = wait_until_ready()
    results = {"opensearch_version": version, "depth": DEPTH, "k": K, "datasets": {}}

    for name in DATASETS:
        results["datasets"][name] = evaluate(name)
        print_summary(name, results["datasets"][name])

    with open("/out/benchmark.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def evaluate(name):
    documents, queries, qrels = beir.load(name)
    query_ids = list(queries)
    print(f"\n[{name}] {len(documents):,} documents, {len(query_ids)} test queries")

    doc_vectors = embed([f"{doc['title']} {doc['text']}".strip() for doc in documents], cache_name=f"{name}-docs")
    query_vectors = embed([queries[q] for q in query_ids], cache_name=f"{name}-queries")

    index = SearchIndex(name)
    index.create(dimension=doc_vectors.shape[1])
    index.add(documents, doc_vectors)

    keyword, meaning, engine_hits = {}, {}, {}
    for query_id, query_vector in zip(query_ids, query_vectors):
        text = queries[query_id]
        keyword[query_id] = ids(index.keyword(text, size=DEPTH))
        meaning[query_id] = ids(index.vector(query_vector, size=DEPTH))
        engine_hits[query_id] = index.hybrid_rrf(text, query_vector, depth=DEPTH, k=K)

    fused_with_scores = {q: rrf([keyword[q], meaning[q]], k=K) for q in query_ids}
    fused = {q: [doc_id for doc_id, _ in fused_with_scores[q]] for q in query_ids}
    engine_rrf = {q: ids(engine_hits[q]) for q in query_ids}

    runs = {"keyword": keyword, "meaning": meaning, "rrf": fused, "rrf_opensearch": engine_rrf}
    per_query = {run: {q: ndcg_at_k(ranking[q], qrels[q]) for q in query_ids} for run, ranking in runs.items()}

    return {
        "documents": len(documents),
        "queries": len(query_ids),
        "ndcg@10": {run: average(per_query[run].values()) for run in runs},
        "recall@100": {run: average(recall_at_k(ranking[q], qrels[q]) for q in query_ids) for run, ranking in runs.items()},
        "k_sweep": {
            k: average(ndcg_at_k(fuse(keyword[q], meaning[q], k=k), qrels[q]) for q in query_ids)
            for k in K_SWEEP
        },
        "depth_sweep": {
            depth: average(ndcg_at_k(fuse(keyword[q], meaning[q], depth=depth), qrels[q]) for q in query_ids)
            for depth in DEPTH_SWEEP
        },
        "rrf_vs_keyword": head_to_head(per_query["rrf"], per_query["keyword"]),
        "rrf_vs_meaning": head_to_head(per_query["rrf"], per_query["meaning"]),
        "opensearch_top10_agreement": {
            "identical_order": average(fused[q][:10] == engine_rrf[q][:10] for q in query_ids),
            "identical_up_to_ties": average(
                same_up_to_ties(fused_with_scores[q][:10], [(h["id"], h["score"]) for h in engine_hits[q][:10]])
                for q in query_ids
            ),
        },
        "examples": pick_examples(documents, queries, qrels, runs, per_query),
    }


def fuse(keyword_ids, meaning_ids, k=K, depth=DEPTH):
    return [doc_id for doc_id, _ in rrf([keyword_ids[:depth], meaning_ids[:depth]], k=k)]


def ids(hits):
    return [hit["id"] for hit in hits]


def average(values):
    return round(mean(values), 4)


def same_up_to_ties(mine, theirs, tolerance=1e-6):
    """True if two (doc_id, score) rankings differ only in the order of docs with equal scores.

    Docs tied at the very last score may differ too: the cut-off picks arbitrarily among them.
    """
    if len(mine) != len(theirs):
        return False
    if any(abs(a - b) > tolerance for (_, a), (_, b) in zip(mine, theirs)):
        return False

    cutoff = mine[-1][1]
    above_mine = {doc_id for doc_id, score in mine if score > cutoff + tolerance}
    above_theirs = {doc_id for doc_id, score in theirs if score > cutoff + tolerance}
    mine_scores, their_scores = dict(mine), dict(theirs)
    return above_mine == above_theirs and all(
        abs(mine_scores[doc_id] - their_scores[doc_id]) <= tolerance for doc_id in above_mine
    )


def head_to_head(scores_a, scores_b, tolerance=1e-9):
    """Count queries where ranking A scores better than, the same as, or worse than ranking B."""
    better = sum(scores_a[q] > scores_b[q] + tolerance for q in scores_a)
    worse = sum(scores_a[q] < scores_b[q] - tolerance for q in scores_a)
    return {"better": better, "same": len(scores_a) - better - worse, "worse": worse}


def pick_examples(documents, queries, qrels, runs, per_query):
    """Real queries where RRF beat both searchers, and where it lost to the better one."""
    snippets = {doc["id"]: (doc["title"] or doc["text"])[:160] for doc in documents}

    def gap(query_id):
        best_single = max(per_query["keyword"][query_id], per_query["meaning"][query_id])
        return per_query["rrf"][query_id] - best_single

    def describe(query_id):
        top10 = lambda run: [
            {"id": d, "snippet": snippets[d], "relevance": qrels[query_id].get(d, 0)} for d in runs[run][query_id][:10]
        ]
        return {
            "query_id": query_id,
            "query": queries[query_id],
            "ndcg@10": {run: round(per_query[run][query_id], 4) for run in ("keyword", "meaning", "rrf")},
            "keyword": top10("keyword"),
            "meaning": top10("meaning"),
            "rrf": top10("rrf"),
        }

    ranked = sorted(queries, key=gap)
    helped = [q for q in reversed(ranked) if gap(q) > 0][:EXAMPLES_PER_KIND]
    hurt = [q for q in ranked if gap(q) < 0][:EXAMPLES_PER_KIND]
    return {"helped": [describe(q) for q in helped], "hurt": [describe(q) for q in hurt]}


def print_summary(name, result):
    print(f"[{name}] nDCG@10    {result['ndcg@10']}")
    print(f"[{name}] Recall@100 {result['recall@100']}")
    print(f"[{name}] k sweep    {result['k_sweep']}")
    print(f"[{name}] depth      {result['depth_sweep']}")
    print(f"[{name}] RRF vs keyword {result['rrf_vs_keyword']} | vs meaning {result['rrf_vs_meaning']}")
    agreement = result["opensearch_top10_agreement"]
    print(f"[{name}] top 10 vs OpenSearch: identical {agreement['identical_order']:.1%}, "
          f"identical up to ties {agreement['identical_up_to_ties']:.1%}")


if __name__ == "__main__":
    main()
