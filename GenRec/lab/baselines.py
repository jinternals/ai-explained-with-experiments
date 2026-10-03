"""Step 2: the scores the LLM ranker has to beat.

The paper compares GenRec against Netflix's production ranker, a model tuned for years.
We don't have that, so we use classic recommenders that need no training run:

    random        scores every movie at random. The floor: any real method must beat it.
    popular       recommends what most people liked. Same list for everyone.
    item-kNN      "people who liked X also liked Y". Two movies are similar when the same
                  people rated both 4-5 stars (cosine similarity). A person's score for a movie
                  is its total similarity to the movies they liked.
    item-kNN, last N   the same, but only the person's N most recent likes count.

Popularity and similarity are counted from every 4-5 star rating except the held-out
validation and test answers, so no method sees the answers.

    python lab/baselines.py

Writes out/baselines.json.
"""
import json, time

import numpy as np

from data import OUT, POSITIVE, load
from metrics import rank_of, summarise

d = load()
index = {m: i for i, m in enumerate(d.catalog)}
held_out = {(ex.user, ex.index) for ex in d.valid + d.test}

# people x movies, 1 where the person rated the movie 4-5 stars (held-out answers excluded)
users = sorted(d.history)
row = {u: r for r, u in enumerate(users)}
likes = np.zeros((len(users), len(d.catalog)), dtype=np.float32)
for u, events in d.history.items():
    for i, e in enumerate(events):
        if e.rating >= POSITIVE and (u, i) not in held_out:
            likes[row[u], index[e.movie]] = 1

popularity = likes.sum(0)
norms = np.sqrt(popularity)
similarity = (likes.T @ likes) / np.maximum(np.outer(norms, norms), 1)
np.fill_diagonal(similarity, 0)
print(f"Built popularity and a {similarity.shape[0]:,} x {similarity.shape[1]:,} similarity table")

rng = np.random.default_rng(0)


def recent_likes(past, n=None):
    liked = [index[e.movie] for e in past if e.rating >= POSITIVE]
    return liked[-n:] if n else liked


METHODS = {
    "random": lambda past: rng.random(len(d.catalog)),
    "popular": lambda past: popularity,
    "item-kNN": lambda past: similarity[recent_likes(past)].sum(0),
    "item-kNN, last 20": lambda past: similarity[recent_likes(past, 20)].sum(0),
    "item-kNN, last 5": lambda past: similarity[recent_likes(past, 5)].sum(0),
    "item-kNN, last 1": lambda past: similarity[recent_likes(past, 1)].sum(0),
}

results = {}
# "valid, first 500" is the group train.py checks during training, so its curve can be read against these.
for split, examples in [("valid", d.valid), ("valid, first 500", d.valid[:500]), ("test", d.test)]:
    results[split] = {}
    for name, score in METHODS.items():
        start = time.perf_counter()
        ranks = []
        for ex in examples:
            past = d.past(ex)
            seen = [index[e.movie] for e in past]
            ranks.append(rank_of(score(past), index[d.target(ex)], seen))
        results[split][name] = summarise(ranks) | {"seconds": round(time.perf_counter() - start, 1)}

OUT.mkdir(exist_ok=True)
(OUT / "baselines.json").write_text(json.dumps(results, indent=2) + "\n")

for split in results:
    n = next(iter(results[split].values()))["n"]
    print(f"\n{split} ({n:,} people, {len(d.catalog):,} movies)")
    print(f"  {'method':<20}{'MRR':>8}{'nDCG@10':>9}{'Recall@10':>11}{'median rank':>13}")
    for name, r in results[split].items():
        print(f"  {name:<20}{r['mrr']:>8.4f}{r['ndcg@10']:>9.4f}{r['recall@10']:>11.4f}{r['median_rank']:>13.0f}")
print("\nWrote out/baselines.json")
