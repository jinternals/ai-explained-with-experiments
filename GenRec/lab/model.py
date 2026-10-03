"""Step 4: the GenRec model. An LLM reads the chat once and a scoring head scores every movie.

This follows the paper's section 4.5 ("Scoring Model"):

    1. Verbalization   the chat from verbalize.py, as token ids
    2. Pooled vector   the LLM reads the prompt; its hidden state at the last token is a vector h
                       that sums up "this person, right now" (896 numbers for Qwen2.5-0.5B)
    3. Catalog scoring every movie i has its own learned vector e_i, and its score is how
                       closely h points the same way as e_i (cosine), times a learned scale, plus
                       a learned per-movie bias

Centring. Straight out of the LLM, every person's h and every movie's vector point almost the
same way (cosine about 0.94, varying by under 0.01 from movie to movie), so cosine can't tell
movies apart. Subtracting the average direction first fixes that: the average person vector
(learned, starting from 200 training prompts) is taken off h, and the average movie vector is
taken off the movie vectors once at the start.

One forward pass gives a score for all 3,706 movies, so there is no word-by-word generation
("prefill-only"), and the model can only ever recommend movies that are in the catalogue.

The LLM keeps its own output layer too (`words`), used during training for the paper's
language-modelling objective; ranking at inference time doesn't need it.

Starting movie vectors. Each movie starts as the h of a one-line history: a person who has
rated just that movie 5 stars, asked the same question. So before any training, a person is
scored as close to a movie when the LLM "sees" them the way it sees a fan of that movie.
That gives an off-the-shelf score to compare training against.

    python lab/model.py                 # zero-shot score on the test set, plus speed and memory
    python lab/model.py --limit 500     # quicker, on the first 500 test people
"""
import argparse, json, math, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from data import OUT, ROOT, Event, Example, load
from verbalize import DEFAULT, MODEL, Style, messages, prompt_tokens

CACHE = ROOT / "cache"
HEAD = ["movies", "bias", "center"]  # the scoring head's numbers, apart from the scale
START_SCALE = 2.0  # small at first: before training the cosine part is mostly noise


class GenRec(nn.Module):
    def __init__(self, llm, movie_vectors, person_mean=None):
        super().__init__()
        self.llm = llm
        n, dim = movie_vectors.shape
        e = movie_vectors.astype(mx.float32)
        if n > 1:
            e = e - e.mean(axis=0)
        self.movies = e / mx.maximum(mx.linalg.norm(e, axis=-1, keepdims=True), 1e-6)  # e_i, unit length
        self.bias = mx.zeros((n,))
        self.center = mx.zeros((dim,)) if person_mean is None else person_mean.astype(mx.float32)
        self.log_scale = mx.array(math.log(START_SCALE))

    def person(self, tokens, lengths):
        """h: the LLM's last hidden state at each prompt's final token. Prompts are right-padded,
        and attention is causal, so padding after a prompt never changes its h."""
        hidden = self.llm.model(tokens)
        return hidden[mx.arange(tokens.shape[0]), lengths - 1].astype(mx.float32)

    def score(self, h):
        """Scores for every movie from pooled vectors h, one row per person."""
        h = h.astype(mx.float32) - self.center
        h = h / mx.linalg.norm(h, axis=-1, keepdims=True)
        e = self.movies / mx.linalg.norm(self.movies, axis=-1, keepdims=True)
        return mx.exp(self.log_scale) * (h @ e.T) + self.bias

    def words(self, hidden):
        """Next-token scores over the LLM's vocabulary, from its own output layer."""
        if self.llm.args.tie_word_embeddings:
            return self.llm.model.embed_tokens.as_linear(hidden)
        return self.llm.lm_head(hidden)

    def __call__(self, tokens, lengths):
        return self.score(self.person(tokens, lengths))


def batches(token_lists, size, pad):
    """Groups prompts of similar length (less padding), right-pads them and remembers the order."""
    order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
    for start in range(0, len(order), size):
        idx = order[start:start + size]
        longest = max(len(token_lists[i]) for i in idx)
        tokens = np.full((len(idx), longest), pad, dtype=np.int32)
        for row, i in enumerate(idx):
            tokens[row, :len(token_lists[i])] = token_lists[i]
        yield idx, mx.array(tokens), mx.array([len(token_lists[i]) for i in idx])


def person_vectors(model, token_lists, pad, size=8):
    out = [None] * len(token_lists)
    for idx, tokens, lengths in batches(token_lists, size, pad):
        h = model.person(tokens, lengths)
        mx.eval(h)
        for row, i in enumerate(idx):
            out[i] = h[row]
    return mx.stack(out)


def movie_vectors(d, llm, tok):
    """Each movie's starting vector: h for a person whose only rating is that movie, 5 stars."""
    path = CACHE / f"movie_vectors_{MODEL.split('/')[-1]}.npy"
    if path.exists():
        return mx.array(np.load(path))
    one_movie = Style(recent=1, summary=False, genres="liked")
    lists = []
    for m in d.catalog:
        fake = Dataset1(d, m)
        lists.append(tok.apply_chat_template(messages(fake, fake.ex, one_movie, answer=False)[:2],
                                             add_generation_prompt=True))
    probe = GenRec(llm, mx.zeros((1, llm.args.hidden_size)))
    vectors = person_vectors(probe, lists, tok.pad_token_id, size=64)
    CACHE.mkdir(exist_ok=True)
    np.save(path, np.array(vectors))
    return vectors


class Dataset1:
    """A stand-in dataset holding one made-up person who rated one movie 5 stars."""
    def __init__(self, d, movie):
        self.movies = d.movies
        self.users = {0: {"gender": "person", "age": "any age", "job": "any job"}}
        ts = d.history[next(iter(d.history))][0].ts
        self.history = {0: [Event(movie, 5, ts), Event(movie, 5, ts)]}
        self.ex = Example(0, 1)

    def past(self, ex):
        return self.history[ex.user][:ex.index]

    def target(self, ex):
        return self.history[ex.user][ex.index].movie


def make_trainable(model, rank=16):
    """Freeze the LLM, add LoRA to every layer, and train only LoRA plus the scoring head."""
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model.freeze()
    linear_to_lora_layers(model.llm, len(model.llm.layers), {"rank": rank, "scale": 20.0, "dropout": 0.0})
    model.unfreeze(keys=HEAD + ["log_scale"], recurse=False)
    return model


def evaluate(model, tok, d, examples, batch=8):
    """Score every movie for each example in one pass, and summarise where the answer landed."""
    from metrics import rank_of, summarise
    index = {m: i for i, m in enumerate(d.catalog)}
    lists = [prompt_tokens(tok, d, e) for e in examples]
    ranks = []
    for idx, tokens, lengths in batches(lists, batch, tok.pad_token_id):
        scores = np.array(model(tokens, lengths))
        for row, i in enumerate(idx):
            e = examples[i]
            ranks.append(rank_of(scores[row], index[d.target(e)], [index[x.movie] for x in d.past(e)]))
    return summarise(ranks), sum(map(len, lists))


def person_mean(d, llm, tok):
    """The average h over 200 training prompts, the direction every person vector shares."""
    path = CACHE / f"person_mean_{MODEL.split('/')[-1]}.npy"
    if path.exists():
        return mx.array(np.load(path))
    probe = GenRec(llm, mx.zeros((1, llm.args.hidden_size)))
    h = person_vectors(probe, [prompt_tokens(tok, d, e) for e in d.train[1::2800][:200]], tok.pad_token_id)
    mean = h.astype(mx.float32).mean(axis=0)
    CACHE.mkdir(exist_ok=True)
    np.save(path, np.array(mean))
    return mean


def build(d=None):
    from mlx_lm import load as load_llm
    d = d or load()
    llm, tok = load_llm(MODEL)
    return d, GenRec(llm, movie_vectors(d, llm, tok), person_mean(d, llm, tok)), tok


if __name__ == "__main__":
    from metrics import rank_of, summarise

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N test people")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    start = time.perf_counter()
    d, model, tok = build()
    model.eval()
    index = {m: i for i, m in enumerate(d.catalog)}
    print(f"Loaded {MODEL} and {len(d.catalog):,} movie vectors in {time.perf_counter() - start:.0f} s")

    # 1. One person, one pass, every movie scored.
    ex = d.test[0]
    t = prompt_tokens(tok, d, ex)
    scores = model(mx.array([t]), mx.array([len(t)]))[0]
    print(f"\nPerson {ex.user}: {len(t)} prompt tokens in, {scores.shape[0]:,} movie scores out (one forward pass)")
    seen = [index[e.movie] for e in d.past(ex)]
    s = np.array(scores)
    s[seen] = -np.inf
    print("Top 10 before any training:")
    for i in np.argsort(-s)[:10]:
        m = d.movies[d.catalog[i]]
        print(f"  {m['title']} ({m['year']})  {', '.join(m['genres'])}")
    print(f"The answer, {d.movies[d.target(ex)]['title']}, is ranked {rank_of(s, index[d.target(ex)], seen):.0f}")

    # 2. Off-the-shelf score on the test set, and how fast prefill-only inference is.
    examples = d.test[:args.limit] if args.limit else d.test
    mx.reset_peak_memory()
    start = time.perf_counter()
    zero_shot, n_tokens = evaluate(model, tok, d, examples, args.batch)
    seconds = time.perf_counter() - start
    speed = {"people": len(examples), "tokens": n_tokens, "seconds": round(seconds, 1),
             "ms_per_person": round(1000 * seconds / len(examples), 1),
             "tokens_per_second": round(n_tokens / seconds), "batch": args.batch,
             "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}
    print(f"\nZero-shot on {len(examples):,} test people: MRR {zero_shot['mrr']:.4f}, nDCG@10 {zero_shot['ndcg@10']:.4f}, "
          f"Recall@10 {zero_shot['recall@10']:.4f}, median rank {zero_shot['median_rank']:.0f}")
    print(f"Inference: {speed['tokens_per_second']:,} tokens/s, {speed['ms_per_person']} ms per person, "
          f"peak memory {speed['peak_memory_gb']} GB")

    # 3. How long one training step takes: LoRA on every layer, ranking loss over the full catalogue.
    import mlx.optimizers as optim

    make_trainable(model)
    model.train()
    trainable = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))

    def loss_fn(model, tokens, lengths, targets):
        return nn.losses.cross_entropy(model(tokens, lengths), targets, reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)
    train_lists = [prompt_tokens(tok, d, e) for e in d.train[:64]]
    train_targets = [index[d.target(e)] for e in d.train[:64]]
    times = []
    mx.reset_peak_memory()
    for idx, tokens, lengths in batches(train_lists, args.batch, tok.pad_token_id):
        start = time.perf_counter()
        loss, grads = step(model, tokens, lengths, mx.array([train_targets[i] for i in idx]))
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        times.append((time.perf_counter() - start, int(tokens.size)))
    warm = times[1:]
    train_speed = {"trainable_parameters": trainable, "total_parameters": total, "batch": args.batch,
                   "seconds_per_step": round(sum(t for t, _ in warm) / len(warm), 2),
                   "tokens_per_second": round(sum(n for _, n in warm) / sum(t for t, _ in warm)),
                   "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}
    print(f"\nTraining step (LoRA rank 16 on all {len(model.llm.layers)} layers + scoring head): "
          f"{trainable / 1e6:.1f}M of {total / 1e6:.0f}M parameters trainable")
    print(f"  {train_speed['seconds_per_step']} s per batch of {args.batch}, {train_speed['tokens_per_second']:,} tokens/s, "
          f"peak memory {train_speed['peak_memory_gb']} GB")
    print(f"  -> about {train_speed['seconds_per_step'] / args.batch:.2f} s per training example")

    OUT.mkdir(exist_ok=True)
    (OUT / "model.json").write_text(json.dumps({"model": MODEL, "style": DEFAULT.__dict__, "zero_shot": zero_shot,
                                                "inference": speed, "training": train_speed}, indent=2) + "\n")
    print("\nWrote out/model.json")
