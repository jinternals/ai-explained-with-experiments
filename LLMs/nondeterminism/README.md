# Nondeterminism in LLM inference

Why a language model at temperature 0 can still give different answers to the same question, reproduced on a MacBook with MLX.

**Post:** [`blog/index.html`](blog/index.html) · **Based on:** [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Horace He, Thinking Machines Lab, September 2025)

## The short version

A server runs your request in a batch with other people's requests, and the size of that batch depends on traffic. Running the same calculation twice gives identical bits, so random GPU timing is not the cause. The cause is that the GPU picks a different way of adding numbers for different batch sizes, and a different order of additions rounds differently. When two candidate words score almost the same, that rounding decides which one you get.

## Results

Apple M4 Pro, MLX 0.32.2. Section numbers refer to the post.

| Question | Result | Script |
|---|---|---|
| Does the order of addition change the answer? (03) | Yes. In float16, 2048 + 1 + 1 + 1 + 1 is 2048 added left to right and 2052 added in two halves. | `lab/rounding.py` |
| Does summing in chunks change it? (04) | Yes. One million float32 numbers, 5 ways of summing, 3 different answers. | `lab/experiments.py` |
| Does the same matmul repeat exactly? (05) | Yes. 1,000 repeats, largest difference `0.0`. | `lab/experiments.py` |
| Does your row change with batch size? (06) | Yes, at batch 2 (split-K starts) and 129 (more than 2,048 tiles). float16: 15, then 12 of 4,096 outputs change. | `lab/batch_invariance.py` |
| What does avoiding that cost? (06) | Padding 2 rows to 129 gives identical bits and takes 2.5–3.1 times longer. | `lab/batch_invariance.py` |
| RMSNorm and attention (07) | RMSNorm never changes. Attention changes by 0.000244 with 4,096+ cached tokens and 8+ new tokens. | `lab/experiments.py` |
| Does a real model's answer change? (01, 02) | Qwen2.5-0.5B at temperature 0: 8 different answers across 11 batch sizes. | `lab/model_experiment.py` |
| Why did the words change? (08) | " particle" and " quantum" both score 21.0 alone. In a batch of 2, " quantum" scores 21.125 and wins. | `lab/logit_gap.py` |
| Does a fixed batch size fix it? (09) | Yes, 4 of 4 identical, at 2.4 times the time for a single request. | `lab/model_experiment.py` |

## Requirements

- A Mac with Apple Silicon (M1 or newer). MLX does not run on Intel Macs, Linux or Windows.
- Python 3.10 or newer, and about 2 GB of free disk space for the model.
- `lab/rounding.py` is the exception: it uses only the standard library and runs on any computer.

## Run it

From this folder:

```bash
python3 lab/rounding.py              # section 03, no install needed

python3 -m venv .venv && source .venv/bin/activate
pip install -r lab/requirements.txt

python lab/batch_invariance.py       # section 06, about 1 second
python lab/experiments.py            # sections 03–07, about 15 seconds
python lab/model_experiment.py       # sections 01, 02 and 09, about 4 minutes, 1 GB download
python lab/logit_gap.py              # section 08, about 30 seconds
python blog/build_page.py            # rebuild blog/index.html
```

Each script prints its results and saves JSON to `out/`, named after your chip. With [uv](https://docs.astral.sh/uv/): `uv venv .venv && uv pip install -r lab/requirements.txt`.

More options:

```bash
python lab/model_experiment.py --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 --tokens 400
python lab/model_experiment.py --prompt "Explain how a bicycle stays upright." --fixed 64
python lab/logit_gap.py mlx-community/Qwen2.5-0.5B-Instruct-bf16 203
```

`logit_gap.py` needs the step where *your* run split. Take it from `first_different_token` in the `model_experiment.py` output.

## Layout

```
nondeterminism/
├── blog/      the post
├── lab/       the experiments
└── out/       results, one JSON file per script and chip
```

| Path | What it is |
|---|---|
| `blog/index.html` | The post. Built by `build_page.py`, so don't edit it by hand |
| `blog/page/body.html` | The post's text and figures |
| `blog/page/base.css`, `blog/page/runtime.js` | Shared stylesheet and script |
| `blog/build_page.py` | Builds `index.html`. The batch-size demo reads its data from `out/model-*.json` |
| `lab/rounding.py` | Rounding and the order of addition, standard library only |
| `lab/batch_invariance.py` | One row alone vs. in a batch, tile counts, and the cost of padding |
| `lab/experiments.py` | Every kernel-level check, including the full 1-to-1,024 batch sweep |
| `lab/model_experiment.py` | A real model at many batch sizes, then with a fixed batch |
| `lab/logit_gap.py` | The top candidate words at the step where the answers split |
| `lab/requirements.txt` | Pinned MLX and mlx-lm versions |
| `out/` | JSON results from each run |

## Notes

- **Why the switch is at 2 and 129.** In MLX's `mlx/backend/metal/matmul.cpp` (v0.32.2), 1 row uses `gemv`. 2 to 128 rows use split-K, which cuts each 4,096-long sum into two halves and adds them at the end. Split-K is used while the output has at most 2,048 tiles of 16 × 16 on Pro and Max chips, and 1,024 on others. From 129 rows it uses the regular tiled matmul. Check your GPU with `python -c "import mlx.core as mx; print(mx.device_info()['architecture'])"`. A name ending in `s` or `d` means Pro or Max.
- **What will differ on another Mac.** The switch points depend on the chip and MLX version (a base M-series chip should switch at 65, not 129). The token where the answers split will differ too. Three things should hold everywhere: repeated calls give identical bits, RMSNorm doesn't change with batch size, and a fixed batch size gives identical answers. If your results break one of those, keep the JSON.
- **Python's `sum()` changed.** The article's "102 different results" only reproduces on Python 3.12 and 3.13. A plain loop gives 258 on every version.
- **The model.** A 0.5B model gets some facts about Feynman wrong. That doesn't matter here, because we only compare its answers with each other. Every row in a batch is the same prompt, so no padding masks are needed and only the batch size changes.
- **The fix.** A fixed batch size is the simplest fix, and it only holds while real requests fit inside it. The article's fix is better: kernels that add in the same order for any batch size.

## References

- Horace He and Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), Connectionism, September 2025. DOI 10.64434/tml.20250910
- [thinking-machines-lab/batch-invariant-ops](https://github.com/thinking-machines-lab/batch-invariant-ops), the article's reference kernels
- [MLX matmul dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/matmul.cpp), v0.32.2
- [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), and the model [mlx-community/Qwen2.5-0.5B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-bf16)
