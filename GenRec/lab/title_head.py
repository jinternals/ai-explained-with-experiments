"""Does a scoring head built from the LLM's own title words connect ranking to language training?

The language loss trains h, the hidden state at the end of the prompt, to predict the next
words, through the LLM's output layer (its word embeddings). This script scores every movie
with that same arithmetic, a dot product between h and the title's word embeddings:

    first word   h . embedding(first token of "Title (year)"), the LLM's logit for starting that title
    all words    h . mean of the embeddings of all the title's tokens

each alone and plus the popularity bias the training runs start from. It needs no training:
it compares the original model with LoRA weights saved by a training run, to see whether
language training already moves the ranking.

    python lab/title_head.py --runs check300

Writes out/title_head.json.
"""
import argparse, json

import mlx.core as mx
import numpy as np

from data import OUT, like_counts
from metrics import rank_of, summarise
from model import CACHE, build, make_trainable, person_vectors
from verbalize import prompt_tokens, title

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=[], help="training runs whose best weights to test")
ap.add_argument("--valid", type=int, default=200)
args = ap.parse_args()

d, model, tok = build()
index = {m: i for i, m in enumerate(d.catalog)}
valid = d.valid[:args.valid]
lists = [prompt_tokens(tok, d, e) for e in valid]
popularity = np.log(np.array(like_counts(d), dtype=np.float32) + 1)
popularity -= popularity.mean()

words = np.array(model.llm.model.embed_tokens.weight.astype(mx.float32))
title_ids = [tok.encode(title(d.movies[m])) for m in d.catalog]
first_word = words[[ids[0] for ids in title_ids]]
all_words = np.stack([words[ids].mean(0) for ids in title_ids])
print(f"{len(d.catalog):,} titles, {np.mean([len(t) for t in title_ids]):.1f} tokens on average; "
      f"{len({t[0] for t in title_ids}):,} different first tokens")


def score_all(label):
    h = np.array(person_vectors(model, lists, tok.pad_token_id).astype(mx.float32))
    out = {}
    for head, table in [("first word", first_word), ("all words", all_words)]:
        s = h @ table.T
        for with_pop in (False, True):
            name = f"{head}{' + popularity' if with_pop else ''}"
            scores = s + popularity if with_pop else s
            r = summarise([rank_of(scores[i], index[d.target(e)], [index[x.movie] for x in d.past(e)])
                           for i, e in enumerate(valid)])
            out[name] = {k: round(v, 4) for k, v in r.items() if k != "n"}
            print(f"  {label:<22} {name:<26} MRR {r['mrr']:.4f}  nDCG@10 {r['ndcg@10']:.4f}  "
                  f"Recall@10 {r['recall@10']:.3f}  median rank {r['median_rank']:.0f}", flush=True)
    return out


results = {"valid_people": len(valid), "popularity alone": None}
r = summarise([rank_of(popularity, index[d.target(e)], [index[x.movie] for x in d.past(e)]) for e in valid])
results["popularity alone"] = {k: round(v, 4) for k, v in r.items() if k != "n"}
print(f"  {'':<22} {'popularity alone':<26} MRR {r['mrr']:.4f}  nDCG@10 {r['ndcg@10']:.4f}  "
      f"Recall@10 {r['recall@10']:.3f}  median rank {r['median_rank']:.0f}")
results["original model"] = score_all("original model")

if args.runs:
    make_trainable(model)
    for run in args.runs:
        model.load_weights(str(CACHE / "runs" / run / "best.safetensors"), strict=False)
        results[f"run {run}"] = score_all(f"after run {run}")

OUT.mkdir(exist_ok=True)
(OUT / "title_head.json").write_text(json.dumps(results, indent=2) + "\n")
print("Wrote out/title_head.json")
