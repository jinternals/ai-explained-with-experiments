"""At the token where the answers split, show the model's top-2 candidates at each batch size.

Every batch size is fed the *same* 203-token prefix (the batch-1 answer) through the normal
decode path (KV cache, one token per step), so the only difference is the batch size.

    python lab/logit_gap.py [model] [step]
"""
import json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 203
model, tok = load(MODEL)
ids = tok.apply_chat_template([{"role": "user", "content": "Tell me about Richard Feynman."}],
                              add_generation_prompt=True)


def decode(bs, forced=None):
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids] * bs), cache=cache)[:, -1]
    toks = []
    for i in range(STEP):
        t = forced[i] if forced else int(mx.argmax(logits, axis=-1)[0].item())
        toks.append(t)
        logits = model(mx.array([[t]] * bs), cache=cache)[:, -1]
    return toks, logits[0].astype(mx.float32)


prefix, _ = decode(1)
out = {"model": MODEL, "step": STEP, "text_before": tok.decode(prefix[-12:]), "by_batch": {}}
for bs in [1, 2, 3, 4, 8, 16, 32, 64, 128]:
    _, lg = decode(bs, forced=prefix)
    top = mx.argsort(-lg)[:3].tolist()
    out["by_batch"][bs] = [[tok.decode([i]), lg[i].item()] for i in top]
    print(bs, out["by_batch"][bs])
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "out"  # results live in <project>/out/
OUT.mkdir(exist_ok=True)
(OUT / "logit-gap.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
