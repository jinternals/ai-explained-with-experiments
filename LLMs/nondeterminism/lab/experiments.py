"""Experiments 1-4: floating point order, run-to-run determinism, batch invariance.

Reproduces the small examples from "Defeating Nondeterminism in LLM Inference"
(Horace He, Thinking Machines Lab, Sep 2025) with MLX on Apple Silicon.

    python lab/experiments.py        # prints results and writes out/kernels-<chip>.json
"""
import json, platform, random, subprocess, sys
from pathlib import Path

import numpy as np
import mlx.core as mx


def chip():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        return platform.processor() or platform.machine()


def f32(x):
    return np.array(x.astype(mx.float32))


out = {"env": {"chip": chip(), "mlx": mx.__version__, "python": sys.version.split()[0],
               "device": str(mx.default_device())}}

# 1. Order of addition changes the answer ---------------------------------------
a, big = mx.array(0.1), mx.array(1e20)
out["1_order"] = {
    "(0.1 + 1e20) - 1e20": ((a + big) - big).item(),
    "0.1 + (1e20 - 1e20)": (a + (big - big)).item(),
}

# The article's shuffle example, verbatim (plain Python floats).
vals = [1e-10, 1e-5, 1e-2, 1]
vals = vals + [-v for v in vals]  # true sum is exactly 0
loop, builtin = [], []
random.seed(42)
for _ in range(10000):
    random.shuffle(vals)
    t = 0.0
    for v in vals:  # plain left-to-right: the same on every Python version
        t += v
    loop.append(t)
    builtin.append(sum(vals))  # the article's code; sum() changed in 3.12 and 3.14
out["1_shuffle"] = {"shuffles": 10000, "true_sum": 0,
                    "plain_loop": {"distinct_results": len(set(loop)), "times_exactly_zero": loop.count(0.0)},
                    "builtin_sum": {"python": sys.version.split()[0], "distinct_results": len(set(builtin)),
                                    "times_exactly_zero": builtin.count(0.0)}}

# 2a. The smallest float16 case: one pass vs. two halves (what split-K does).
#     Above 2048, float16 can only store even numbers, so +1 rounds away.
v = [np.float16(n) for n in (2048, 1, 1, 1, 1)]
one_pass = np.float16(0)
for n in v:
    one_pass = np.float16(one_pass + n)
half1 = np.float16(v[0] + v[1])
half2 = np.float16(np.float16(v[2] + v[3]) + v[4])
out["2a_fp16_halves"] = {"numbers": [2048, 1, 1, 1, 1], "true_sum": 2052,
                         "one_pass_left_to_right": float(one_pass),
                         "two_halves": [float(half1), float(half2)],
                         "halves_added": float(np.float16(half1 + half2))}

# 2b. One reduction, three ways of chunking it -----------------------------------
x = mx.random.normal((1_048_576,), key=mx.random.key(0))  # float32
exact = float(np.sum(np.array(x, dtype=np.float64)))
ways = {"mx.sum in one call": mx.sum(x).item()}
for c in [256, 4096, 65536]:
    ways[f"sum of {c}-element chunks"] = mx.sum(mx.sum(x.reshape(-1, c), axis=1)).item()
ways["numpy sequential (float32)"] = float(np.cumsum(np.array(x))[-1])
out["2_chunking"] = {"n": x.size, "exact_float64": exact, "float32_results": ways,
                     "distinct": len(set(ways.values()))}

# 3. Same matmul, same inputs, 1000 times -> run-to-run deterministic? -----------
mx.random.seed(0)
A = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
B = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
ref = f32(A @ B)
worst = 0.0
for _ in range(1000):
    worst = max(worst, float(np.abs(f32(A @ B) - ref).max()))
out["3_run_to_run"] = {"op": "2048x2048 @ 2048x2048 bf16", "runs": 1000, "max_diff_vs_first": worst}

# 4. Same row, different batch size -> batch invariant? --------------------------
# 4a. The article's linspace example, verbatim shapes.
Bsz, D = 2048, 4096
la = mx.linspace(-1000, 1000, Bsz * D).reshape(Bsz, D)
lb = mx.linspace(-1000, 1000, D * D).reshape(D, D)
out1 = la[:1] @ lb
out2 = (la @ lb)[:1]
mx.eval(out1, out2)
out["4a_linspace"] = {"dtype": "float32", "max_abs_diff": mx.abs(out1 - out2).max().item(),
                      "typical_magnitude": mx.abs(out2).mean().item()}

# 4b. One row of random activations, multiplied alone vs. inside batches.
def batch_sweep(dtype, sizes):
    mx.random.seed(1)
    X = mx.random.normal((max(sizes), 4096)).astype(dtype)
    W = mx.random.normal((4096, 4096)).astype(dtype) * 0.02
    alone = f32(X[:1] @ W)[0]
    rows = {}
    for bs in sizes:
        r = f32(X[:bs] @ W)[0]
        d = np.abs(r - alone)
        rows[bs] = {"max_diff": float(d.max()), "outputs_changed": int((d > 0).sum()), "of": int(d.size)}
    return rows

sizes = [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024]
out["4b_matmul_bf16"] = batch_sweep(mx.bfloat16, sizes)
out["4b_matmul_fp16"] = batch_sweep(mx.float16, sizes)
out["4b_matmul_fp32"] = batch_sweep(mx.float32, sizes)

# 4e. Where exactly does row 0 change? Sweep every batch size 1..1024.
#     Then the fix: always pad the batch to 1024 rows, so one kernel is always used.
def switch_points(dtype):
    mx.random.seed(1)
    X = mx.random.normal((1024, 4096)).astype(dtype)
    W = (mx.random.normal((4096, 4096)) * 0.02).astype(dtype)
    prev, changes = None, []
    for bs in range(1, 1025):
        r = f32(X[:bs] @ W)[0]
        if prev is None or not np.array_equal(r, prev):
            changes.append(bs)
        prev = r
    padded = []
    for bs in [1, 2, 7, 64, 129, 300, 1024]:
        Xp = mx.concatenate([X[:bs], mx.zeros((1024 - bs, 4096), dtype=dtype)])
        padded.append(f32(Xp @ W)[0])
    return {"row0_changes_at_batch_sizes": changes,
            "padded_to_1024_identical": all(np.array_equal(padded[0], p) for p in padded)}

out["4e_switch_points"] = {str(dt): switch_points(dt) for dt in [mx.bfloat16, mx.float16, mx.float32]}

# 4c. RMSNorm: each row is reduced on its own, so batch size should not matter.
mx.random.seed(2)
X = mx.random.normal((1024, 4096)).astype(mx.bfloat16)
w = mx.random.normal((4096,)).astype(mx.bfloat16)
alone = f32(mx.fast.rms_norm(X[:1], w, 1e-6))[0]
out["4c_rmsnorm_bf16"] = {bs: float(np.abs(f32(mx.fast.rms_norm(X[:bs], w, 1e-6))[0] - alone).max())
                          for bs in [1, 2, 8, 64, 1024]}

# 4d. Attention: one query token, same KV cache, alone vs. with other queries.
def attn(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)

mx.random.seed(3)
H, Dh = 8, 128
q1 = mx.random.normal((1, H, 1, Dh)).astype(mx.bfloat16)
extra = mx.random.normal((1, H, 15, Dh)).astype(mx.bfloat16)
att = {}
for L in [128, 1000, 4096, 16384]:
    k = mx.random.normal((1, H, L, Dh)).astype(mx.bfloat16)
    v = mx.random.normal((1, H, L, Dh)).astype(mx.bfloat16)
    alone = f32(attn(q1, k, v))
    row = {}
    for nq in [2, 8, 16]:
        batched = f32(attn(mx.concatenate([q1, extra[:, :, : nq - 1]], axis=2), k, v))[:, :, :1]
        row[f"with {nq} queries"] = float(np.abs(alone - batched).max())
    # also: same single query batched with 3 other *sequences* (batch dim)
    kb = mx.concatenate([k] * 4, axis=0); vb = mx.concatenate([v] * 4, axis=0)
    qb = mx.concatenate([q1] * 4, axis=0)
    row["4 sequences in batch"] = float(np.abs(alone - f32(attn(qb, kb, vb))[:1]).max())
    att[L] = row
out["4d_attention_bf16"] = att

print(json.dumps(out, indent=2))
OUT = Path(__file__).resolve().parent.parent / "out"  # results live in <project>/out/
OUT.mkdir(exist_ok=True)
tag = out["env"]["chip"].replace(" ", "-")
Path(OUT / f"kernels-{tag}.json").write_text(json.dumps(out, indent=2))
