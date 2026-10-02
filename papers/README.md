# Papers

Papers saved for reading. Each one is kept as its PDF, named `<ShortName>-<arXiv id>.pdf`.

| Paper | Authors | Year | File | Link |
|---|---|---|---|---|
| **GenRec: An LLM-Backed Recommendation Ranker at Netflix** | Li, Sehgal, Rao, Houthooft, Zhu, Rastogi | 2026 | [GenRec-2608.10257v2.pdf](GenRec-2608.10257v2.pdf) | [arXiv 2608.10257](https://arxiv.org/abs/2608.10257) |

## GenRec

Netflix replaces a feature-engineered ranker with one built on its own foundation LLM. Phase 1 adapts an open-source LLM to Netflix data; this paper covers Phase 2, which post-trains that model for ranking. It covers turning user histories into text, building training data, adding rewards, the model architecture, and a prefill-only serving design that keeps costs down. The paper reports an A/B test against the production ranker.
