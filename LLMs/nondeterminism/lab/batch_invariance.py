"""Section 06 of the post: batch size changes the order of additions.

Needs MLX (Apple Silicon):  python lab/batch_invariance.py

1. Batch invariance, tested: one row alone vs. the same row inside bigger batches.
2. Why the plan changes: how many 16x16 output tiles there are to hand out.
3. Why MLX does it: time the small-batch plan against forcing the big-batch plan.
"""
import time
import numpy as np
import mlx.core as mx

K = N = 4096  # one row of 4,096 numbers times a 4,096 x 4,096 weight matrix
mx.random.seed(1)
X = mx.random.normal((1024, K)).astype(mx.bfloat16)          # up to 1,024 users' rows
W = (mx.random.normal((K, N)) * 0.02).astype(mx.bfloat16)    # the weights


DTYPES = [mx.bfloat16, mx.float16, mx.float32]


def row0(batch, weights=W):
    """Your row's output when it runs as row 0 of `batch`."""
    return np.array((batch @ weights).astype(mx.float32))[0]


print(f"GPU: {mx.device_info()['device_name']} ({mx.device_info()['architecture']}), MLX {mx.__version__}\n")

# 1. Your row alone, then with 1, 7, 127, 128 ... other users --------------------
print("1. your row alone vs. in a batch: how many of its 4,096 outputs changed")
print(f"   {'batch':>6} {'bf16':>12} {'float16':>12} {'float32':>12}")
alone = {dt: row0(X[:1].astype(dt), W.astype(dt)) for dt in DTYPES}
for bs in [1, 2, 8, 128, 129, 512]:
    cells = []
    for dt in DTYPES:
        d = np.abs(row0(X[:bs].astype(dt), W.astype(dt)) - alone[dt])
        cells.append(f"{int((d > 0).sum()):>5} ({float(d.max()):.0e})" if d.max() else f"{0:>5}        ")
    print(f"   {bs:>6} " + " ".join(f"{c:>12}" for c in cells))

# 2. How many 16x16 tiles of output are there to hand out? -----------------------
print("\n2. output tiles of 16 x 16 numbers (MLX uses split-K while tiles <= 2,048 on Pro/Max)")
for bs in [1, 2, 16, 128, 129, 1024]:
    tiles = -(-bs // 16) * (N // 16)
    if bs == 1:
        print(f"   batch {bs:>5}:      - (no tiles)  -> gemv, the single-row kernel")
    else:
        plan = "split-K" if tiles <= 2048 else "tiled GEMM"
        print(f"   batch {bs:>5}: {tiles:>6} tiles     -> {plan}")


# 3. Why bother splitting? Time it. ------------------------------------------------
def ms(fn, reps=200):
    for _ in range(10):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return (time.perf_counter() - t) / reps * 1000

pad = mx.zeros((127, K), dtype=mx.bfloat16)
print("\n3. time for 2 users' rows")
t_split = ms(lambda: X[:2] @ W)
t_big = ms(lambda: mx.concatenate([X[:2], pad]) @ W)
print(f"   2 rows, split-K plan:                    {t_split:.3f} ms")
print(f"   2 rows padded to 129, big-batch plan:    {t_big:.3f} ms  ({t_big / t_split:.1f}x slower)")
same = np.array_equal(row0(mx.concatenate([X[:1], mx.zeros((128, K), dtype=mx.bfloat16)])),
                      row0(mx.concatenate([X[:2], pad])))
print(f"   padded runs give your row identical bits whatever the real batch: {same}")
