"""Experiment 5: does a real LLM's greedy (temperature 0) answer change with batch size?

We ask the same question at temperature 0. The only thing that changes between runs is
how many *other* requests share the GPU batch with it -- exactly what happens on a busy
server. Then we apply the fix (always run one fixed batch size) and check again.

    python lab/model_experiment.py
    python lab/model_experiment.py --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 --tokens 400
"""
import argparse, json, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
ap.add_argument("--prompt", default="Tell me about Richard Feynman.")
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--fixed", type=int, default=32, help="fixed batch size used by the fix")
args = ap.parse_args()

model, tok = load(args.model)
prompt_ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                     add_generation_prompt=True)


def greedy(batch_size, pad_to=None):
    """Greedy-decode our prompt as row 0 of a batch.

    Rows 1..batch_size-1 are other users (here: the same prompt, so no padding masks are
    needed and every row is a real, independent request). If pad_to is set, the batch is
    topped up with filler rows so the GPU always sees exactly pad_to rows.
    """
    rows = pad_to or batch_size
    cache = make_prompt_cache(model)
    x = mx.array([prompt_ids] * rows)
    logits = model(x, cache=cache)[:, -1]
    out = []
    t0 = time.perf_counter()
    for _ in range(args.tokens):
        nxt = mx.argmax(logits, axis=-1)          # temperature 0, every row
        mx.eval(nxt)
        t = int(nxt[0].item())
        out.append(t)
        if t == tok.eos_token_id:
            break
        logits = model(nxt[:, None], cache=cache)[:, -1]
    return out, time.perf_counter() - t0


def first_diff(a, b):
    for i, (p, q) in enumerate(zip(a, b)):
        if p != q:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def around(ids, i, before=12, after=10):
    return tok.decode(ids[max(0, i - before): i]), tok.decode(ids[i: i + after])


res = {"model": args.model, "prompt": args.prompt, "max_tokens": args.tokens,
       "chip": None, "mlx": mx.__version__}
try:
    import subprocess
    res["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
except Exception:
    pass

# --- A. Plain batching, like a normal server --------------------------------------
sizes = [1, 1, 2, 3, 4, 8, 16, 32, 64, 128, 129, 160]
base, _ = greedy(1)
runs = []
for bs in sizes:
    ids, secs = greedy(bs)
    d = first_diff(base, ids)
    row = {"batch_size": bs, "first_different_token": d, "tokens": len(ids), "seconds": round(secs, 2)}
    if d is not None:
        row["shared_text_before"], row["alone_continues"] = around(base, d)
        row["in_batch_continues"] = around(ids, d)[1]
    runs.append(row)
    print(json.dumps(row, ensure_ascii=False))
res["A_plain"] = runs
res["A_distinct_completions"] = len({tuple(greedy(bs)[0]) for bs in sorted(set(sizes))})
res["A_baseline_text"] = tok.decode(base)

# --- B. The fix: always run exactly `fixed` rows ------------------------------------
fixed_runs, ref = [], None
for bs in [1, 2, 8, args.fixed]:
    ids, secs = greedy(bs, pad_to=args.fixed)
    ref = ref or ids
    fixed_runs.append({"real_requests": bs, "gpu_batch": args.fixed,
                       "identical_to_first": ids == ref, "seconds": round(secs, 2)})
    print(json.dumps(fixed_runs[-1]))
res["B_fixed_batch"] = fixed_runs

print(json.dumps({k: v for k, v in res.items() if k.startswith("A_distinct")}, indent=2))
OUT = Path(__file__).resolve().parent.parent / "out"  # results live in <project>/out/
OUT.mkdir(exist_ok=True)
name = args.model.split("/")[-1]
Path(OUT / f"model-{name}-{(res['chip'] or 'unknown').replace(' ', '-')}.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False))
