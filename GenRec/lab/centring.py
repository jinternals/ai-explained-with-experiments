"""Why the scoring head centres its vectors: an LLM's hidden states all point the same way.

Takes the untrained model's person vector h for 200 validation people and the starting movie
vectors, and compares cosine similarity before and after subtracting the average direction
(average person h from 200 training prompts, average movie vector). Then ranks the 200 people
by popularity plus k times the centred cosine, for several k, to show why the cosine weight
starts small.

    python lab/centring.py

Writes out/centring.json.
"""
import json

import numpy as np

from data import OUT, like_counts
from metrics import rank_of, summarise
from model import CACHE, build, person_vectors
from verbalize import MODEL, prompt_tokens

d, model, tok = build()
index = {m: i for i, m in enumerate(d.catalog)}
popularity = np.log(np.array(like_counts(d), dtype=np.float32) + 1)
popularity -= popularity.mean()

valid = d.valid[:200]
H = np.array(person_vectors(model, [prompt_tokens(tok, d, e) for e in valid], tok.pad_token_id))
E = np.load(CACHE / f"movie_vectors_{MODEL.split('/')[-1]}.npy").astype(np.float32)
person_mean = np.load(CACHE / f"person_mean_{MODEL.split('/')[-1]}.npy")
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)

results = {}
for name, h, e, scales in [("raw", unit(H), unit(E), [20]),
                           ("centred", unit(H - person_mean), unit(E - E.mean(0)), [0, 1, 2, 5, 10, 20])]:
    cos = h @ e.T
    results[name] = {"cosine_mean": round(float(cos.mean()), 3),
                     "spread_across_movies": round(float(cos.std(1).mean()), 3), "popularity_plus": {}}
    print(f"{name:<8} cosine mean {cos.mean():+.3f}, spread across movies {cos.std(1).mean():.3f}")
    for k in scales:
        r = summarise([rank_of(popularity + k * cos[i], index[d.target(ex)], [index[x.movie] for x in d.past(ex)])
                       for i, ex in enumerate(valid)])
        results[name]["popularity_plus"][f"{k} x cosine"] = {key: round(v, 4) for key, v in r.items() if key != "n"}
        print(f"   popularity + {k:>2} x cosine: MRR {r['mrr']:.4f}  median rank {r['median_rank']:.0f}")

OUT.mkdir(exist_ok=True)
(OUT / "centring.json").write_text(json.dumps(results, indent=2) + "\n")
print("Wrote out/centring.json")
