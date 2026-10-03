"""Step 5: Phase-2 post-training. Teach the LLM + scoring head to rank movies.

The paper trains with two objectives at once (section 4.4):

    ranking    a cross-entropy loss over the whole catalogue, from the scoring head, so the movie
               the person actually liked next gets the highest score
    language   the LLM's usual next-token loss over the verbalized inputs and outputs: the history
               lines and the answer's title. It keeps the model's language skills and gives it many
               more "which movie comes next" examples than the one answer per prompt.

    loss = alpha * ranking + beta * language,   alpha + beta = 1

One forward pass over the full chat (prompt + answer) gives both: the scoring head reads h at the
prompt's last token, which causal attention keeps blind to the answer after it. --lm answer
limits the language loss to the answer's tokens; --lm none trains on ranking alone.

Scoring all 152,000 vocabulary words at every position of 8 prompts of ~1,000 tokens needs about
5 GB per copy, and the backward pass keeps several: the first test peaked at 52 GB. So each step
scores --lm-positions randomly chosen positions per prompt (128 by default; 256 still peaked at 33.6 GB). On average that gives
the same loss as using every position, and over many steps every part of the text is trained.
Only LoRA layers (small add-ons inside each LLM layer) and the scoring head are trained.

Each movie's bias starts at the log of how many people liked it, so the model starts out knowing
what plain popularity knows and spends its training on what popularity misses. (A first run with
the bias at zero was still worse than popularity after 2,000 examples.) The scoring head learns
10 times faster than LoRA, and the cosine scale, a single number, 100 times faster.

Training data is a sample, because one example takes about a quarter of a second on a laptop:
a fixed, shuffled pool of 40,000 examples, made of each person's 3 most recent training examples
(closest to what the test asks) topped up with random older ones. A run takes a slice of the pool,
so training can happen in parts: part 2 starts from part 1's weights and sees 20,000 new examples.

--reward turns on the paper's reward-weighted ranking loss (section 4.6). Netflix weights each
example with separate reward models for long-term satisfaction. Our stand-in for "how valuable
was this engagement" is the star rating: a 5-star example counts twice as much as a 4-star one.

    python lab/train.py --name part1                                   # pool examples 0-19,999
    python lab/train.py --name part2 --skip 20000 --init part1         # 20,000-39,999, continuing part 1
    python lab/train.py --name reward --reward

LP-FT (attempt 4): --head-from probe starts the scoring head from the one lab/probe.py trained on
the frozen LLM, so the head already works before LoRA starts changing h. The head then learns
slowly (--head-lr 1e-4) while LoRA, the ranking loss and the language loss reshape h.

    python lab/train.py --name part1 --head-from probe --head-lr 1e-4 --scale-lr 1e-3

Batches are built by token budget (--tokens, padded tokens per batch, at most 16 prompts), so long
prompts go in smaller batches and memory stays level: 8 prompts of up to 1,480 tokens peaked at
34.5 GB.

A run saves the optimizer's state with its last weights, and --init restores both, so a later part
keeps Adam's running averages and continues at the learning rate where the earlier part ended. (A first part 2 that restored only the weights
began with empty Adam averages and a re-warmed learning rate; it fell below popularity within 250
steps: out/train_part2_rewarmed.json.) --save-at N also keeps the weights after N examples.

Every --every steps it scores 500 validation people and keeps the best weights in
cache/runs/<name>/. The learning curve goes to out/train_<name>.json.
"""
import argparse, json, math, random, time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from data import OUT, like_counts, training_pool
from model import CACHE, HEAD, batches, build, evaluate, make_trainable
from verbalize import messages, prompt_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--examples", type=int, default=20_000)
ap.add_argument("--skip", type=int, default=0, help="start this far into the pool")
ap.add_argument("--pool", type=int, default=40_000)
ap.add_argument("--init", help="start from the last weights of this earlier run")
ap.add_argument("--head-from", choices=["probe"], help="start the scoring head from lab/probe.py's best head")
ap.add_argument("--save-at", type=int, nargs="*", default=[], help="also keep the weights after this many examples")
ap.add_argument("--tokens", type=int, default=7_600, help="padded tokens per batch (8 prompts of an average 950)")
ap.add_argument("--recent", type=int, default=3, help="most recent training examples per person")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4, help="peak learning rate for LoRA")
ap.add_argument("--head-lr", type=float, default=1e-3, help="peak learning rate for movie vectors, bias and centre")
ap.add_argument("--scale-lr", type=float, default=1e-2, help="peak learning rate for the cosine scale")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--reward", action="store_true")
ap.add_argument("--lm", choices=["all", "answer", "none"], default="all", help="tokens the language loss covers")
ap.add_argument("--alpha", type=float, default=0.5, help="weight of the ranking loss; the language loss gets 1 - alpha")
ap.add_argument("--lm-positions", type=int, default=128, help="positions per prompt the language loss samples each step")
ap.add_argument("--every", type=int, default=250)
ap.add_argument("--valid", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--memory-gb", type=float, default=30, help="ceiling for MLX's memory use")
args = ap.parse_args()

# Batches have different lengths, so MLX's cache of freed GPU buffers fills with sizes it never
# reuses. Uncapped, part 1 grew to 36 GB, pushed macOS into swap and ran 3-4 times slower.
mx.set_cache_limit(2 * 1024**3)
mx.set_memory_limit(int(args.memory_gb * 1024**3))


def training_batches(lists, budget, seed, most=16):
    """Random batches of similar-length prompts, each at most `budget` padded tokens: shuffle,
    sort within chunks of 512, fill batches greedily, then shuffle the batches."""
    rng = random.Random(seed)
    order = list(range(len(lists)))
    rng.shuffle(order)
    out = []
    for start in range(0, len(order), 512):
        batch = []
        for i in sorted(order[start:start + 512], key=lambda i: len(lists[i])):
            if batch and (len(batch) == most or (len(batch) + 1) * len(lists[i]) > budget):
                out.append(batch)
                batch = []
            batch.append(i)
        out.append(batch)
    rng.shuffle(out)
    return out


d, model, tok = build()
index = {m: i for i, m in enumerate(d.catalog)}
make_trainable(model, args.rank)
log_likes = mx.log(mx.array(like_counts(d), dtype=mx.float32) + 1)
model.bias = log_likes - log_likes.mean()
pad = tok.pad_token_id

if args.head_from == "probe":
    head = mx.load(str(CACHE / "probe_head.safetensors"))
    assert "project" not in head, "train.py expects a full-size head without a projection"
    model.movies, model.bias, model.log_scale = head["movies"], head["bias"], head["log_scale"]
    model.center = head["mu"]
    print(f"Scoring head from lab/probe.py: cosine weight {math.exp(head['log_scale'].item()):.2f}", flush=True)
if args.init:
    model.load_weights(str(CACHE / "runs" / args.init / "last.safetensors"), strict=False)
    print(f"Starting from the weights at the end of run '{args.init}'", flush=True)
examples = training_pool(d, args.pool, args.recent, args.seed)[args.skip:args.skip + args.examples]
lists = [tok.apply_chat_template(messages(d, ex), tokenize=True) for ex in examples]  # prompt + answer
prompt_lengths = [len(prompt_tokens(tok, d, ex)) for ex in examples]
assert all(lists[i][:n] == prompt_tokens(tok, d, examples[i]) for i, n in enumerate(prompt_lengths[:20]))
targets = [index[d.target(ex)] for ex in examples]
stars = [d.history[ex.user][ex.index].rating for ex in examples]
weights = [2.0 if s == 5 else 1.0 for s in stars] if args.reward else [1.0] * len(examples)
plan = training_batches(lists, args.tokens, args.seed)
seen_after = np.cumsum([len(b) for b in plan]).tolist()
valid = d.valid[:args.valid]

steps = len(plan)
warmup = max(1, min(100, steps // 10))


def schedule(peak):
    return optim.join_schedules([optim.linear_schedule(peak / 100, peak, warmup),
                                 optim.cosine_decay(peak, steps - warmup, peak / 10)], [warmup])


is_head = lambda path, _: path.split(".")[0] in HEAD
is_scale = lambda path, _: path == "log_scale"
opt = optim.MultiOptimizer([optim.AdamW(schedule(args.head_lr), weight_decay=0.0),
                            optim.AdamW(schedule(args.scale_lr), weight_decay=0.0),
                            optim.AdamW(schedule(args.lr), weight_decay=0.0)], [is_head, is_scale])


alpha = 1.0 if args.lm == "none" else args.alpha


lm_rng = np.random.default_rng(args.seed)


def language_positions(idx):
    """For each prompt in the batch, the positions whose next token the language loss predicts.
    Position p predicts token p + 1; padding is never predicted. Returns positions and a mask."""
    chosen = []
    for i in idx:
        first = prompt_lengths[i] - 1 if args.lm == "answer" else 0
        valid = np.arange(first, len(lists[i]) - 1)
        if len(valid) > args.lm_positions:
            valid = np.sort(lm_rng.choice(valid, args.lm_positions, replace=False))
        chosen.append(valid)
    width = max(len(c) for c in chosen)
    positions = np.zeros((len(idx), width), dtype=np.int32)
    mask = np.zeros((len(idx), width), dtype=np.float32)
    for row, c in enumerate(chosen):
        positions[row, :len(c)] = c
        mask[row, :len(c)] = 1
    return mx.array(positions), mx.array(mask)


def loss_fn(model, tokens, prompt_lengths, targets, weights, positions, mask):
    hidden = model.llm.model(tokens)
    rows = mx.arange(tokens.shape[0])[:, None]
    scores = model.score(hidden[rows[:, 0], prompt_lengths - 1])
    ranking = (nn.losses.cross_entropy(scores, targets, reduction="none") * weights).sum() / weights.sum()
    if args.lm == "none":
        return ranking, (ranking, mx.array(0.0))
    logits = model.words(hidden[rows, positions]).astype(mx.float32)
    per_token = nn.losses.cross_entropy(logits, tokens[rows, positions + 1], reduction="none")
    language = (per_token * mask).sum() / mask.sum()
    return alpha * ranking + (1 - alpha) * language, (ranking, language)


step_fn = nn.value_and_grad(model, loss_fn)
if args.init and (CACHE / "runs" / args.init / "optimizer.safetensors").exists():
    opt.init(model.trainable_parameters())
    opt.state = tree_unflatten(list(mx.load(str(CACHE / "runs" / args.init / "optimizer.safetensors")).items()))
    print(f"Restored the optimizer state of run '{args.init}'", flush=True)
run = CACHE / "runs" / args.name
run.mkdir(parents=True, exist_ok=True)
trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"Run '{args.name}': {len(examples):,} examples ({sum(s == 5 for s in stars):,} are 5★), {steps:,} steps of up to {args.tokens:,} tokens (about {len(examples) / steps:.1f} prompts), "
      f"{trainable / 1e6:.1f}M trainable parameters, reward weighting {'on' if args.reward else 'off'}, "
      f"language loss on {args.lm} tokens, alpha {alpha}", flush=True)

curve, best, losses, rank_losses, lang_losses, start = [], None, [], [], [], time.perf_counter()


def check(step):
    global best
    model.eval()
    r, _ = evaluate(model, tok, d, valid, args.batch)
    model.train()
    point = {"step": step, "examples_seen": args.skip + (seen_after[step - 1] if step else 0), "minutes": round((time.perf_counter() - start) / 60, 1),
             "train_loss": round(float(np.mean(losses)), 4) if losses else None,
             "ranking_loss": round(float(np.mean(rank_losses)), 4) if rank_losses else None,
             "language_loss": round(float(np.mean(lang_losses)), 4) if lang_losses else None,
             "valid": {k: round(v, 4) for k, v in r.items() if k != "n"}}
    curve.append(point)
    improved = best is None or r["mrr"] > best["valid"]["mrr"]
    if improved:
        best = point
        mx.save_safetensors(str(run / "best.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    point["scale"] = round(math.exp(model.log_scale.item()), 2)
    print(f"step {step:>5}/{steps}  {point['minutes']:>5} min  ranking {point['ranking_loss'] or float('nan'):.3f}  "
          f"language {point['language_loss'] or float('nan'):.3f}  scale {point['scale']:<5} "
          f"valid MRR {r['mrr']:.4f}  nDCG@10 {r['ndcg@10']:.4f}  Recall@10 {r['recall@10']:.3f}  "
          f"median rank {r['median_rank']:.0f}{'  *best*' if improved else ''}", flush=True)
    losses.clear(); rank_losses.clear(); lang_losses.clear()
    (OUT / f"train_{args.name}.json").write_text(json.dumps(
        {"args": vars(args), "examples": len(examples), "steps": steps, "trainable_parameters": trainable,
         "valid_people": len(valid), "best": best, "curve": curve}, indent=2) + "\n")


check(0)
for step, idx in enumerate(plan, 1):
    longest = max(len(lists[i]) for i in idx)
    tokens = np.full((len(idx), longest), pad, dtype=np.int32)
    for row, i in enumerate(idx):
        tokens[row, :len(lists[i])] = lists[i]
    positions, mask = language_positions(idx)
    (loss, (ranking, language)), grads = step_fn(
        model, mx.array(tokens), mx.array([prompt_lengths[i] for i in idx]),
        mx.array([targets[i] for i in idx]), mx.array([weights[i] for i in idx]), positions, mask)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state, loss, ranking, language)
    losses.append(loss.item()); rank_losses.append(ranking.item()); lang_losses.append(language.item())
    if step % 50 == 0 and step % args.every:
        print(f"step {step:>5}/{steps}  {(time.perf_counter() - start) / 60:>5.1f} min  ranking {np.mean(rank_losses[-50:]):.3f}  "
              f"language {np.mean(lang_losses[-50:]):.3f}  memory {mx.get_active_memory() / 1e9:.1f} GB "
              f"(peak {mx.get_peak_memory() / 1e9:.1f}) + cache {mx.get_cache_memory() / 1e9:.1f} GB", flush=True)
    for n in args.save_at:
        if seen_after[step - 1] >= n > (seen_after[step - 2] if step > 1 else 0):
            mx.save_safetensors(str(run / f"after_{n}.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            print(f"step {step:>5}/{steps}  saved the weights after {n:,} examples", flush=True)
    if step % args.every == 0 or step == steps:
        check(step)

mx.save_safetensors(str(run / "last.safetensors"), dict(tree_flatten(model.trainable_parameters())))
mx.save_safetensors(str(run / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
print(f"\nBest valid MRR {best['valid']['mrr']:.4f} at step {best['step']}. Weights in {run}/best.safetensors", flush=True)
