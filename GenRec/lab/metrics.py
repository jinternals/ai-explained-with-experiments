"""Scoring a ranking when there is exactly one right answer: the movie the person rated 4-5 stars next.

Every recommender in this lab gives every movie in the catalogue a score. We sort by score,
find where the right movie landed (its rank, 1 = top) and turn that rank into:

    MRR        1 / rank, averaged. 1st -> 1, 2nd -> 0.5, 10th -> 0.1. The paper's main offline metric.
    nDCG@10    1 / log2(rank + 1) if the movie is in the top 10, else 0. With one right answer
               this is the same formula the search labs use.
    Recall@10  1 if the movie is in the top 10, else 0. How often it shows up at all.

Movies the person already rated are removed before ranking: MovieLens has one rating per
person per movie, so they can never be the answer, and every method is treated the same way.
"""
import numpy as np


def rank_of(scores, target, seen):
    """Rank of `target` (catalogue index) among unseen movies. Ties count as half above, half below."""
    scores = scores.astype(np.float64, copy=True)
    scores[seen] = -np.inf
    t = scores[target]
    above = np.count_nonzero(scores > t)
    ties = np.count_nonzero(scores == t) - 1
    return 1 + above + ties / 2


def summarise(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    in_top10 = ranks <= 10
    return {
        "mrr": float(np.mean(1 / ranks)),
        "ndcg@10": float(np.mean(np.where(in_top10, 1 / np.log2(ranks + 1), 0))),
        "recall@10": float(np.mean(in_top10)),
        "median_rank": float(np.median(ranks)),
        "n": len(ranks),
    }
