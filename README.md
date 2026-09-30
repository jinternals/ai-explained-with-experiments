# AI, explained with experiments

Plain-language write-ups of ideas from AI research. Each project is written for readers who don't want heavy maths, and each comes with the code that produced every number in it.

The rule for every post here: **if a number is on the page, a script in the same folder printed it.** Numbers quoted from a paper are marked as coming from the paper.

## Projects

| Project | Question it answers | Headline result | Stack |
|---|---|---|---|
| [**BM25**](BM25/) | How does keyword search score a document, and which of its settings change results? | Rebuilds one real search score, 28.64, word by word. k1 and b moved nDCG@10 by at most 0.005 near the defaults; stemming added up to 0.034, and reproduces the BEIR paper's BM25 on NFCorpus and FiQA exactly. | Java 21, OpenSearch 2.19.1, Docker |
| [**Reciprocal Rank Fusion**](RRF/) | How does hybrid search merge a keyword list and a meaning list into one? | On 1,271 BEIR questions, RRF beat both searchers on NFCorpus (+9%) and SciFact (+4%), and lost on FiQA (−5%), where one searcher is much weaker. | Java 21, OpenSearch 2.19.1, Docker |
| [**Cross-encoder re-ranking**](CrossEncoder/) | Does a slower model that reads the question and each document together improve search results, and what does it cost? | Re-ranking improved all 9 combinations of dataset and first stage (keyword search on FiQA +40%). For RRF, re-ranking the top 20 beat the top 100 at a sixth of the time: 0.67 s vs 4.0 s per question on CPU. | Java 21, OpenSearch 2.19.1, ONNX, Docker |
| [**Nondeterminism in LLM inference**](LLMs/nondeterminism/) | Why can a model at temperature 0 give different answers to the same question? | The same calculation repeats bit for bit. The batch size changes the order of additions. A 0.5B model gave 8 different answers across 11 batch sizes, and 4 of 4 identical ones with the batch size fixed. | Python, MLX, Apple Silicon |

## BM25

`BM25/` · part 1: how keyword search works

BM25 adds up IDF × TF for each matching word. IDF makes rare words count more; TF lets repeats count a little more each time (k1) and gives long documents a small penalty (b). The post shows the full formula and fills it in with real numbers. The post takes one real search score apart using OpenSearch's explain API, then tests 42 combinations of k1 and b and two text analyzers on three BEIR datasets.

```bash
cd BM25
docker compose run --rm --build lab example     # the 28.64 score, word by word
docker compose run --rm --build lab benchmark   # about 6 minutes
docker compose down
```

Needs Docker. Details are in [BM25/README.md](BM25/README.md).

## Reciprocal Rank Fusion

`RRF/` · [Read it on Medium](https://medium.com/jinternals/reciprocal-rank-fusion-without-the-scary-math-68421476c505)

Keyword search and meaning search score results on different scales, so their scores can't be added. RRF ignores the scores and adds up points for each result's position in each list. The post works through one real NFCorpus question by hand, then tests the method on three BEIR datasets and sweeps the constant k.

```bash
cd RRF
docker compose run --rm --build lab step3       # the worked example from the post
docker compose run --rm --build lab benchmark   # 1,271 questions, 3 datasets
docker compose down
```

Needs Docker. The first benchmark run downloads about 23 MB of data and embeds about 66,000 documents on the CPU. Details are in [RRF/README.md](RRF/README.md).

## Cross-encoder re-ranking

`CrossEncoder/` · follows the RRF post

Meaning search reads the question and each document separately, so it is fast but never sees them together. A cross-encoder reads the question and one document as a single input and outputs one relevance score. It is used to re-sort the top results of a fast search. The post re-ranks keyword, meaning and RRF lists on the same three BEIR datasets as the RRF post, and measures the gain, the cost, and how far the result is from a perfect order.

```bash
cd CrossEncoder
docker compose run --rm --build lab example     # the RRF post's worked example, re-ranked
docker compose run --rm --build lab benchmark   # 1,271 questions, ~2 hours of CPU the first time
docker compose down
```

Needs Docker. Details are in [CrossEncoder/README.md](CrossEncoder/README.md).

## Nondeterminism in LLM inference

`LLMs/nondeterminism/` · based on [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Thinking Machines Lab, 2025)

A server runs your request in a batch with other people's. The GPU picks a different way of adding numbers for different batch sizes, and a different order rounds differently. When two next words score almost the same, that rounding decides which one you get. The post reproduces the article's claims on a MacBook with MLX, finds where the batch size changes the result (row counts 2 and 129 on an M4 Pro), and traces those switch points to MLX's source.

```bash
cd LLMs/nondeterminism
python3 lab/rounding.py                  # rounding examples, any computer, no install
python3 -m venv .venv && source .venv/bin/activate
pip install -r lab/requirements.txt
python lab/batch_invariance.py           # the batch-size effect in one second
python lab/model_experiment.py           # a real model, ~4 minutes, 1 GB download
```

Needs an Apple Silicon Mac for everything except `rounding.py`. Details are in [LLMs/nondeterminism/README.md](LLMs/nondeterminism/README.md).

## Tools

Apps built along the way, not posts, so they don't use the `blog/ lab/ out/` layout.

| Tool | What it does |
|---|---|
| [**apple-foundation-model**](apple-foundation-model/) | A macOS menu-bar app that serves Apple's on-device Foundation Model through an OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`, streaming), so Open WebUI and other chat apps can use it. Swift, no dependencies. |

## Layout

Every project uses the same folders and the same README sections, so you know where to look in any of them.

| Folder | What goes in it |
|---|---|
| `blog/` | The post as a web page |
| `lab/` | The code that produces every number in the post |
| `out/` | Results as JSON, written by the lab and read by the post's figures |
| `medium/` | The Medium version, when there is one |
| `README.md` | The short version, results, requirements, how to run, layout, notes, references |

```
AI/
├── RRF/
│   ├── blog/          rrf-blog.html
│   ├── medium/        rrf-medium.md, rrf-medium.html, images/, build-images.mjs
│   ├── lab/           Java 21: OpenSearch, BM25, all-MiniLM-L6-v2, BEIR
│   ├── lab-python/    the first version of the lab
│   ├── out/           benchmark and example results
│   └── data/ cache/   downloaded datasets and embeddings, created on first run
├── CrossEncoder/
│   ├── blog/          index.html, page/ (source), build_page.py
│   ├── lab/           Java 21: OpenSearch, all-MiniLM-L6-v2, ms-marco-MiniLM-L6-v2 cross-encoder
│   ├── out/           example and benchmark results
│   └── data/ cache/   datasets, embeddings, the ONNX model, cross-encoder scores
├── LLMs/
│   └── nondeterminism/
│       ├── blog/      index.html, page/ (source), build_page.py
│       ├── lab/       rounding, batch_invariance, experiments, model_experiment, logit_gap
│       └── out/       JSON results, named after the chip
└── BM25/
    ├── blog/          index.html, page/ (source), build_page.py
    ├── lab/           Java 21: OpenSearch BM25, explain and analyze APIs
    └── out/           example and benchmark results
```

`RRF/data/` and `RRF/cache/` take about 350 MB together, and `CrossEncoder/data/` and `CrossEncoder/cache/` about 270 MB. Delete them to free space; the next run rebuilds them.

## How the posts are made

- **Real data only.** Examples use public datasets (BEIR) or real model output, never made-up numbers.
- **Every figure is rebuilt from results.** The RRF images are rendered from `out/*.json` by `medium/build-images.mjs`. The nondeterminism, cross-encoder and BM25 pages fill their numbers from `out/*.json` when they are built.
- **Sources are marked.** The nondeterminism post tags every number "ran on Mac" or "paper". The RRF post names the source under each figure, such as "Raw output from OpenSearch" or "Table 1 of Cormack et al. (2009)".
- **Differences are reported.** When a result disagrees with the paper, the post says so and explains why. For example, Python's `sum()` gives the article's 102 only on Python 3.12 and 3.13.
