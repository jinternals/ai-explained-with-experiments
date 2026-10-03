"""LP-FT, step 1: train only the scoring head on the frozen LLM's vectors (a "linear probe").

Fine-tuning an LLM while its new head is still untrained sends noisy gradients into the LLM;
the joint runs answered by switching the head off (the cosine weight sank toward zero) and the
ranking loss never moved. The usual remedy is LP-FT (Kumar et al., 2022): train the head alone
on frozen features until it works, then fine-tune everything together starting from it.

This script reads 8,000 training prompts and 500 validation prompts once with the untrained LLM,
saves each h, and then trains many small heads on the saved vectors, which takes seconds each:

    features   centre        h minus the average training h
               standardise   the same, divided by each dimension's spread
    head       cosine        scale x cosine(feature, e_i) + bias_i, scale learned
               dot           feature . e_i + bias_i
    start      fan           e_i = the LLM's h for a made-up person who rated only movie i
               title words   e_i = the average output-word embedding of the title's tokens
               random        small random numbers
    lr         1e-3 or 1e-2

Every head's bias starts at log popularity, as in train.py. Each is trained for 20 passes over the
8,000 examples and scored on the 500 validation people after every pass; the best pass counts.

The grid's heads overfit: one 896-number vector per movie is 3.3 million numbers for 8,000
examples. --stage size then tries smaller heads, centred cosine heads whose movie vectors have
64 or 128 numbers (a learned projection takes h down to that size) or the full 896, each with
weight decay 0, 0.01 or 0.1 on the movie vectors and projection.

    python lab/probe.py                # the 24-head grid
    python lab/probe.py --stage size   # 9 head sizes and weight decays, added to out/probe.json

Saves the vectors in cache/ (about 15 minutes the first time), writes out/probe.json and the best
head to cache/probe_head.safetensors.
"""
import argparse, itertools, json, math, time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from data import OUT, like_counts, training_pool
from metrics import rank_of, summarise
from model import CACHE, build, person_vectors
from verbalize import MODEL, prompt_tokens, title

N_TRAIN, N_VALID, EPOCHS, BATCH = 8_000, 500, 20, 512
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["grid", "size"], default="grid")
args = ap.parse_args()

d, model, tok = build()
index = {m: i for i, m in enumerate(d.catalog)}
train_ex = training_pool(d)[:N_TRAIN]
valid_ex = d.valid[:N_VALID]

path = CACHE / f"probe_h_{MODEL.split('/')[-1]}_{N_TRAIN}.npz"
if not path.exists():
    start = time.perf_counter()
    for name, exs in [("valid", valid_ex), ("train", train_ex)]:
        lists = [prompt_tokens(tok, d, e) for e in exs]
        h = np.array(person_vectors(model, lists, tok.pad_token_id).astype(mx.float32))
        np.save(CACHE / f"_probe_{name}.npy", h)
        print(f"Read {len(exs):,} {name} prompts ({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    np.savez(path, train=np.load(CACHE / "_probe_train.npy"), valid=np.load(CACHE / "_probe_valid.npy"))
    for name in ("train", "valid"):
        (CACHE / f"_probe_{name}.npy").unlink()
saved = np.load(path)
H_train, H_valid = saved["train"], saved["valid"]
print(f"Vectors: {H_train.shape[0]:,} training, {H_valid.shape[0]:,} validation, {H_train.shape[1]} numbers each")

targets = np.array([index[d.target(e)] for e in train_ex])
valid_targets = [index[d.target(e)] for e in valid_ex]
valid_seen = [[index[x.movie] for x in d.past(e)] for e in valid_ex]
popularity = np.log(np.array(like_counts(d), dtype=np.float32) + 1)
popularity -= popularity.mean()

mu, sigma = H_train.mean(0), H_train.std(0) + 1e-6
fan = np.load(CACHE / f"movie_vectors_{MODEL.split('/')[-1]}.npy").astype(np.float32)
words = np.array(model.llm.model.embed_tokens.weight.astype(mx.float32))
title_words = np.stack([words[tok.encode(title(d.movies[m]))].mean(0) for m in d.catalog])
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)


def evaluate(scores):
    return summarise([rank_of(scores[i], valid_targets[i], valid_seen[i]) for i in range(len(valid_ex))])


class Head(nn.Module):
    def __init__(self, kind, vectors, start_scale=5.0, project=None):
        super().__init__()
        self.kind = kind
        self.movies = mx.array(vectors)
        self.bias = mx.array(popularity)
        self.log_scale = mx.array(math.log(start_scale))
        if project is not None:
            self.project = mx.array(project)

    def __call__(self, z):
        if "project" in self:
            z = z @ self.project
        if self.kind == "cosine":
            z = z / mx.linalg.norm(z, axis=-1, keepdims=True)
            e = self.movies / mx.linalg.norm(self.movies, axis=-1, keepdims=True)
            return mx.exp(self.log_scale) * (z @ e.T) + self.bias
        return z @ self.movies.T + self.bias


def start_vectors(start, features):
    if start == "fan":
        v = fan - fan.mean(0)          # same space as h: centre (and standardise) like the features
        v = v / sigma if features == "standardise" else v
    elif start == "title words":
        # logits are h . E, so (h - mu) . E = z . (E * sigma) when z is standardised
        v = title_words * sigma if features == "standardise" else title_words.copy()
    else:
        v = np.random.default_rng(0).normal(0, 0.01, (len(d.catalog), H_train.shape[1])).astype(np.float32)
    return v


def train_head(features, kind, start, lr, size=None, weight_decay=0.0):
    z_train = (H_train - mu) / (sigma if features == "standardise" else 1)
    z_valid = (H_valid - mu) / (sigma if features == "standardise" else 1)
    project = None
    if size:  # a random projection from h down to `size` numbers, and random movie vectors that size
        rng0 = np.random.default_rng(0)
        project = (rng0.normal(0, 1, (H_train.shape[1], size)) / math.sqrt(H_train.shape[1])).astype(np.float32)
        v = rng0.normal(0, 0.01, (len(d.catalog), size)).astype(np.float32)
    else:
        v = start_vectors(start, features)
    if kind == "dot":  # start with dot products of about unit spread, so they don't swamp the bias
        v = v / max(float((z_train[:256] @ v.T).std()), 1e-6)
    head = Head(kind, v, project=project)
    # weight decay shrinks the movie vectors and projection, never the popularity bias or the scale
    decayed = lambda path, _: path in ("movies", "project")
    opt = optim.MultiOptimizer([optim.AdamW(learning_rate=lr, weight_decay=weight_decay),
                                optim.Adam(learning_rate=lr)], [decayed])
    step = nn.value_and_grad(head, lambda m, z, y: nn.losses.cross_entropy(m(z), y, reduction="mean"))
    rng = np.random.default_rng(0)
    zt, zv = mx.array(z_train), mx.array(z_valid)
    best, curve = None, []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(targets))
        losses = []
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            loss, grads = step(head, zt[mx.array(b)], mx.array(targets[b]))
            opt.update(head, grads)
            mx.eval(head.parameters(), opt.state)
            losses.append(loss.item())
        r = evaluate(np.array(head(zv)))
        curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "mrr": round(r["mrr"], 4)})
        if best is None or r["mrr"] > best["valid"]["mrr"]:
            best = {"epoch": epoch, "valid": {k: round(v_, 4) for k, v_ in r.items() if k != "n"},
                    "params": {k: np.array(v_) for k, v_ in head.parameters().items()}}
    return best, curve


r = evaluate(np.tile(popularity, (len(valid_ex), 1)))
pop = {k: round(v, 4) for k, v in r.items() if k != "n"}
print(f"popularity alone: MRR {r['mrr']:.4f}  nDCG@10 {r['ndcg@10']:.4f}  median rank {r['median_rank']:.0f}\n")

if args.stage == "grid":
    results = {"train_examples": N_TRAIN, "valid_people": N_VALID, "epochs": EPOCHS, "popularity": pop, "heads": []}
    runs = [dict(features=f, head=k, start=s, lr=lr) for f, k, s, lr in itertools.product(
        ["centre", "standardise"], ["cosine", "dot"], ["fan", "title words", "random"], [1e-3, 1e-2])]
    key, top = "heads", None
else:
    results = json.loads((OUT / "probe.json").read_text())
    runs = [dict(features="centre", head="cosine", start="random" if size else "title words", lr=1e-3,
                 size=size, weight_decay=wd) for size, wd in itertools.product([64, 128, None], [0.0, 0.01, 0.1])]
    key, top = "size_heads", (results["best"], None)
    results[key] = []

for run in runs:
    t0 = time.perf_counter()
    best, curve = train_head(run["features"], run["head"], run["start"], run["lr"],
                             run.get("size"), run.get("weight_decay", 0.0))
    row = {**run, "best_epoch": best["epoch"], "valid": best["valid"], "curve": curve,
           "seconds": round(time.perf_counter() - t0, 1)}
    results[key].append(row)
    if top is None or best["valid"]["mrr"] > top[0]["valid"]["mrr"]:
        top = (row, best["params"])
    label = (f"{run['features']:<12}{run['head']:<7}{run['start']:<12}lr {run['lr']:<7g}" if args.stage == "grid" else
             f"size {run['size'] or 896:<4} weight decay {run['weight_decay']:<5g}")
    print(f"{label}best epoch {best['epoch']:>2}  MRR {best['valid']['mrr']:.4f}  nDCG@10 {best['valid']['ndcg@10']:.4f}  "
          f"median rank {best['valid']['median_rank']:>4.0f}  train loss {curve[0]['train_loss']:.2f} -> "
          f"{curve[-1]['train_loss']:.2f}  ({row['seconds']} s)", flush=True)

row, params = top
if params is not None:  # a new best head
    results["best"] = {k: v for k, v in row.items() if k != "curve"}
    mx.save_safetensors(str(CACHE / "probe_head.safetensors"),
                        {**{k: mx.array(v) for k, v in params.items()}, "mu": mx.array(mu), "sigma": mx.array(sigma)})
OUT.mkdir(exist_ok=True)
(OUT / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
b = results["best"]
print(f"\nBest so far: {b['features']}, {b['head']}, {b['start']}, lr {b['lr']:g}, size {b.get('size') or 896}, "
      f"weight decay {b.get('weight_decay', 0.0):g}: MRR {b['valid']['mrr']:.4f} (popularity {pop['mrr']:.4f}). "
      f"{'New head saved to' if params is not None else 'Kept the head in'} cache/probe_head.safetensors")
