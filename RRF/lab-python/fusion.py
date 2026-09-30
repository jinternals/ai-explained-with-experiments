"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009)."""


def rrf(ranked_lists, k=60):
    """Merge ranked lists of doc ids. A doc earns 1 / (k + rank) from every list it appears in.

    Returns (doc_id, score) pairs, best first.
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
