"""Step 6: the trained models on the test set, next to the baselines.

Training picked its checkpoints on 500 validation people, which flatters them a little. The test
set is each of the 6,035 people's last 4-5 star rating, never used for training or for choosing
anything. Every method ranks all 3,706 movies, minus the ones the person already rated.

Single scores move by a few thousandths from one group of people to another, so for each model
the script also reports how much better or worse it is than item-kNN on the last 5 likes, with
a 95% interval from a paired bootstrap: resample the 6,035 people 2,000 times, keeping each
person's two ranks together, and take the middle 95% of the differences. If the interval
includes zero, the two can't be told apart on this data.

Results are also split by how long a person's history is, since the prompt writes out only the
last 50 ratings and summarises the rest.

    python lab/evaluate.py part1:last full40k:best full40k:last

Each model takes about 10 minutes. Writes out/evaluate.json.
"""
import argparse, json, time

import mlx.core as mx
import numpy as np

from data import OUT, POSITIVE, load
from metrics import rank_of, summarise
from model import CACHE, batches, build, make_trainable
from verbalize import prompt_tokens

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="run:checkpoint, e.g. full40k:best (best, last or after_20000)")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--bootstrap", type=int, default=2000)
args = ap.parse_args()
mx.set_cache_limit(2 * 1024**3)  # as in train.py: batch shapes vary, so an uncapped cache only grows

d, model, tok = build()
index = {m: i for i, m in enumerate(d.catalog)}
test = d.test
targets = [index[d.target(e)] for e in test]
seen = [[index[x.movie] for x in d.past(e)] for e in test]
history = np.array([e.index for e in test])
GROUPS = {"up to 50 past ratings": history <= 50, "51 to 150": (history > 50) & (history <= 150),
          "over 150": history > 150}

# The baselines, recomputed here so every method has a rank per person (as in baselines.py).
held_out = {(ex.user, ex.index) for ex in d.valid + d.test}
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


def last_likes(e, n):
    return [index[x.movie] for x in d.past(e) if x.rating >= POSITIVE][-n:]


ranks = {
    "popular": np.array([rank_of(popularity, targets[i], seen[i]) for i in range(len(test))]),
    "item-kNN, last 5 likes": np.array([rank_of(similarity[last_likes(e, 5)].sum(0), targets[i], seen[i])
                                         for i, e in enumerate(test)]),
}

lists = [prompt_tokens(tok, d, e) for e in test]
make_trainable(model)
model.eval()
for spec in args.models:
    run, checkpoint = spec.split(":")
    model.load_weights(str(CACHE / "runs" / run / f"{checkpoint}.safetensors"), strict=False)
    start = time.perf_counter()
    r = np.zeros(len(test))
    for idx, tokens, lengths in batches(lists, args.batch, tok.pad_token_id):
        scores = np.array(model(tokens, lengths))
        for k, i in enumerate(idx):
            r[i] = rank_of(scores[k], targets[i], seen[i])
    ranks[f"GenRec {spec}"] = r
    print(f"Scored {spec} on {len(test):,} test people in {(time.perf_counter() - start) / 60:.1f} min", flush=True)

rng = np.random.default_rng(0)
samples = rng.integers(0, len(test), (args.bootstrap, len(test)))
reference = "item-kNN, last 5 likes"


def interval(a, b, metric):
    f = (lambda r: 1 / r) if metric == "mrr" else (lambda r: np.where(r <= 10, 1 / np.log2(r + 1), 0))
    diff = f(a) - f(b)
    means = diff[samples].mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


results = {"test_people": len(test), "bootstrap": args.bootstrap, "methods": {}}
print(f"\nTest set: {len(test):,} people, {len(d.catalog):,} movies")
print(f"  {'method':<26}{'MRR':>8}{'nDCG@10':>9}{'Recall@10':>11}{'median':>8}   MRR vs item-kNN (95%)     nDCG@10 vs item-kNN (95%)")
for name, r in ranks.items():
    s = summarise(r)
    entry = {k: round(v, 4) for k, v in s.items() if k != "n"}
    entry["groups"] = {g: {k: round(v, 4) for k, v in summarise(r[m]).items()} for g, m in GROUPS.items()}
    line = f"  {name:<26}{s['mrr']:>8.4f}{s['ndcg@10']:>9.4f}{s['recall@10']:>11.4f}{s['median_rank']:>8.0f}"
    if name != reference:
        entry["vs_item_knn"] = {"mrr": round(float(np.mean(1 / r) - np.mean(1 / ranks[reference])), 4),
                                "mrr_95": interval(r, ranks[reference], "mrr"),
                                "ndcg@10_95": interval(r, ranks[reference], "ndcg@10")}
        v = entry["vs_item_knn"]
        line += f"   {v['mrr']:+.4f} [{v['mrr_95'][0]:+.4f}, {v['mrr_95'][1]:+.4f}]   [{v['ndcg@10_95'][0]:+.4f}, {v['ndcg@10_95'][1]:+.4f}]"
    results["methods"][name] = entry
    print(line)

print("\nMRR by history length:")
print(f"  {'method':<26}" + "".join(f"{g:>24}" for g in GROUPS))
print(f"  {'(people)':<26}" + "".join(f"{int(m.sum()):>24,}" for m in GROUPS.values()))
for name, entry in results["methods"].items():
    print(f"  {name:<26}" + "".join(f"{entry['groups'][g]['mrr']:>24.4f}" for g in GROUPS))

OUT.mkdir(exist_ok=True)
(OUT / "evaluate.json").write_text(json.dumps(results, indent=2) + "\n")
np.savez(CACHE / "test_ranks.npz", **{k: v for k, v in ranks.items()})
print("\nWrote out/evaluate.json")
