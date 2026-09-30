"""Ranking quality metrics, following the trec_eval conventions that BEIR uses."""

import math


def ndcg_at_k(ranked_ids, relevance, k=10):
    """nDCG@k: how good the top k is, from 0 to 1, rewarding relevant docs near the top.

    `relevance` maps doc id -> graded label (0 or missing = not relevant).
    """
    dcg = sum(
        relevance.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )
    best_labels = sorted((label for label in relevance.values() if label > 0), reverse=True)[:k]
    ideal_dcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(best_labels, start=1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def recall_at_k(ranked_ids, relevance, k=100):
    """Share of all relevant docs that appear anywhere in the top k."""
    relevant = {doc_id for doc_id, label in relevance.items() if label > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)
