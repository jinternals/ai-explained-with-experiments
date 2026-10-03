# AI, explained with experiments

Plain-language write-ups of ideas from AI research. Each project is written for readers who don't want heavy maths, and each comes with the code that produced every number in it.

The rule for every post here: **if a number is on the page, a script in the same folder printed it.** Numbers quoted from a paper are marked as coming from the paper.

## Reading order

The four search posts build on each other; the LLM post stands on its own.

1. **[BM25](BM25/)**: how keyword search scores a document.
2. **[Semantic search](SemanticSearch/)**: how search by meaning works, and when it beats keywords.
3. **[Reciprocal Rank Fusion](RRF/)**: merging keyword search and meaning search into one list.
4. **[Cross-encoder re-ranking](CrossEncoder/)**: re-sorting the top results with a slower, more careful model.
5. **[Nondeterminism in LLM inference](LLMs/nondeterminism/)**: why temperature 0 can still give different answers.

Also in the repository: **[Apple Foundation Model server](apple-foundation-model/)**, a macOS app that lets Open WebUI and other chat apps use the language model built into macOS.

**[Papers](papers/)**: research papers saved for reading, with an index of what each one is about.

## Getting started

```bash
git clone https://github.com/jinternals/ai-explained-with-experiments.git
cd ai-explained-with-experiments
```

Each project runs on its own, so you only need what the project you want uses:

| To run | You need |
|---|---|
| BM25, SemanticSearch, RRF, CrossEncoder | Docker. The labs bring OpenSearch and download the BEIR datasets and models on the first run. |
| LLMs/nondeterminism | An Apple Silicon Mac and Python 3.10+ (`lab/rounding.py` runs anywhere). |
| apple-foundation-model | An Apple Silicon Mac with macOS 27 and Apple Intelligence turned on, and Xcode 27 to build. |

The posts themselves are plain HTML files (`blog/index.html` or `blog/rrf-blog.html`) that open in any browser. Datasets, cached embeddings, models and build output are not in the repository; the labs recreate them (see [What isn't in the repository](#what-isnt-in-the-repository)).

## Projects

| Project | Question it answers | Headline result | Stack |
|---|---|---|---|
| [**BM25**](BM25/) | How does keyword search score a document, and which of its settings change results? | Rebuilds one real search score, 28.64, word by word. k1 and b moved nDCG@10 by at most 0.005 near the defaults; stemming added up to 0.034, and reproduces the BEIR paper's BM25 on NFCorpus and FiQA exactly. | Java 21, OpenSearch 2.19.1, Docker |
| [**Semantic search**](SemanticSearch/) | How does search by meaning work, and when does it beat keyword search? | On 1,271 BEIR questions, meaning search won FiQA (0.239 → 0.366 nDCG@10) and lost SciFact (0.656 → 0.624). Shared words decide it: with under a third of the question's words in the answers, it won 229 questions and lost 85. HNSW matched exact search's top 10 for 94% of questions. | Java 21, OpenSearch 2.19.1, Docker |
| [**Reciprocal Rank Fusion**](RRF/) | How does hybrid search merge a keyword list and a meaning list into one? | On 1,271 BEIR questions, RRF beat both searchers on NFCorpus (+9%) and SciFact (+4%), and lost on FiQA (−5%), where one searcher is much weaker. | Java 21, OpenSearch 2.19.1, Docker |
| [**Cross-encoder re-ranking**](CrossEncoder/) | Does a slower model that reads the question and each document together improve search results, and what does it cost? | Re-ranking improved all 9 combinations of dataset and first stage (keyword search on FiQA +40%). For RRF, re-ranking the top 20 beat the top 100 at a sixth of the time: 0.67 s vs 4.0 s per question on CPU. | Java 21, OpenSearch 2.19.1, ONNX, Docker |
| [**Nondeterminism in LLM inference**](LLMs/nondeterminism/) | Why can a model at temperature 0 give different answers to the same question? | The same calculation repeats bit for bit. The batch size changes the order of additions. A 0.5B model gave 8 different answers across 11 batch sizes, and 4 of 4 identical ones with the batch size fixed. | Python, MLX, Apple Silicon |
| [**Apple Foundation Model server**](apple-foundation-model/) | Can chat apps like Open WebUI use the language model built into macOS? | Yes: a menu-bar app serves it through an OpenAI-compatible API. Tested with Open WebUI v0.11.4 and the OpenAI Python SDK: streaming, conversation history and OpenAI-style errors all work. The larger Private Cloud Compute model needs an entitlement from Apple. | Swift, Foundation Models, macOS 27 |

## BM25

`BM25/` · part 1: how keyword search works

BM25 adds up IDF × TF for each matching word. IDF makes rare words count more; TF lets repeats count a little more each time (k1) and gives long documents a small penalty (b). The post shows the full formula, fills it in with the numbers behind one real search score (OpenSearch's explain API), then tests 42 combinations of k1 and b and two text analyzers on three BEIR datasets.

```bash
cd BM25
docker compose run --rm --build lab example     # the 28.64 score, word by word
docker compose run --rm --build lab benchmark   # about 6 minutes
docker compose down
```

Needs Docker. Details are in [BM25/README.md](BM25/README.md).

## Semantic search

`SemanticSearch/` · part 2

An embedding model turns a word, a sentence or a paragraph into 384 numbers, an arrow, through one recipe: split into word pieces, adjust each by its neighbours, average. Cosine similarity compares arrows by direction. The post builds that up with interactive examples, then compares meaning search with BM25 on the same three BEIR datasets, tests HNSW against exact search, and shows that LangChain4j reads only the first 128 word pieces of each document.

```bash
cd SemanticSearch
docker compose run --rm --build lab example     # the worked examples
docker compose run --rm --build lab benchmark   # 1,271 questions, 3 datasets
docker compose run --rm --build lab length      # how much of each document the model reads
docker compose down
```

Needs Docker. Details are in [SemanticSearch/README.md](SemanticSearch/README.md).

## Reciprocal Rank Fusion

`RRF/` · part 3 · [Read it on Medium](https://medium.com/jinternals/reciprocal-rank-fusion-without-the-scary-math-68421476c505)

Keyword search and meaning search score results on different scales, so their scores can't be added. RRF ignores the scores and adds up points for each result's position in each list. The post works through one real NFCorpus question by hand, then tests the method on three BEIR datasets and sweeps the constant k.

```bash
cd RRF
docker compose run --rm --build lab step3       # the worked example from the post
docker compose run --rm --build lab benchmark   # 1,271 questions, 3 datasets
docker compose down
```

Needs Docker. The first benchmark run downloads about 23 MB of data and embeds about 66,000 documents on the CPU. Details are in [RRF/README.md](RRF/README.md).

## Cross-encoder re-ranking

`CrossEncoder/` · part 4

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

## Apple Foundation Model server

`apple-foundation-model/` · a tool, not a post

macOS 27 includes a language model that runs entirely on your Mac, but Apple only exposes it to Swift code. This menu-bar app wraps it in the OpenAI API (`GET /v1/models`, `POST /v1/chat/completions`, with streaming), so Open WebUI and other chat apps can use it like any other model. It has a log window, an optional API key, and no third-party dependencies.

```bash
cd apple-foundation-model
scripts/build-app.sh --install                  # build, sign and copy to ~/Applications
open ~/Applications/"Foundation Model Server.app"
docker compose up -d                            # Open WebUI at http://localhost:3000, already connected
```

Base URL: `http://127.0.0.1:11535/v1` (from Docker: `http://host.docker.internal:11535/v1`). Needs an Apple Silicon Mac with macOS 27, Apple Intelligence turned on, and Xcode 27. Details, including how to use Apple's Private Cloud Compute model, are in [apple-foundation-model/README.md](apple-foundation-model/README.md).

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
├── BM25/
│   ├── blog/          index.html, page/ (source), build_page.py
│   ├── lab/           Java 21: OpenSearch BM25, explain and analyze APIs
│   └── out/           example and benchmark results
├── SemanticSearch/
│   ├── blog/          index.html, page/ (source), build_page.py
│   ├── lab/           Java 21: OpenSearch k-NN, all-MiniLM-L6-v2, BEIR
│   └── out/           example, benchmark and length results
├── RRF/
│   ├── blog/          rrf-blog.html
│   ├── medium/        rrf-medium.md, rrf-medium.html, images/, build-images.mjs
│   ├── lab/           Java 21: OpenSearch, BM25, all-MiniLM-L6-v2, BEIR
│   └── lab-python/    the first version of the lab
├── CrossEncoder/
│   ├── blog/          index.html, page/ (source), build_page.py
│   ├── lab/           Java 21: OpenSearch, all-MiniLM-L6-v2, ms-marco-MiniLM-L6-v2 cross-encoder
│   └── out/           example and benchmark results
├── LLMs/
│   └── nondeterminism/
│       ├── blog/      index.html, page/ (source), build_page.py
│       ├── lab/       rounding, batch_invariance, experiments, model_experiment, logit_gap
│       └── out/       JSON results, named after the chip
├── apple-foundation-model/
│   ├── Sources/       the server (FMServerCore) and the menu-bar app
│   ├── Tests/         unit tests
│   ├── Resources/     the app icon
│   └── scripts/       build-app.sh, make-icon.swift
└── papers/            PDFs of papers to read, indexed in papers/README.md
```

## What isn't in the repository

Everything the labs download or rebuild is ignored by `.gitignore`, about 1.2 GB on the machine these were made on:

| Not committed | Size | Recreated by |
|---|---|---|
| `*/data/`: the BEIR datasets | about 64 MB per project | the first lab run |
| `RRF/cache/`, `SemanticSearch/cache/`, `CrossEncoder/cache/`: embeddings, models, cross-encoder scores | about 285 MB, 400 MB and 200 MB | the first lab runs |
| `RRF/out/`: RRF's results JSON | under 1 MB | `docker compose run --rm lab benchmark` in `RRF/` |
| `LLMs/nondeterminism/.venv/` | about 335 MB | `pip install -r lab/requirements.txt` |
| `apple-foundation-model/.build/`, `build/` | about 225 MB | `scripts/build-app.sh` |

To free space, delete any of these; the next run rebuilds them.

## How the posts are made

- **One score for search quality.** The search posts score results with nDCG@10: a number from 0 to 1 for how good the first 10 results are, where relevant results near the top count for more and 1 is a perfect top 10. Each project's README explains it under Notes.
- **Real data only.** Examples use public datasets (BEIR) or real model output, never made-up numbers.
- **Every figure is rebuilt from results.** The RRF images are rendered from `out/*.json` by `medium/build-images.mjs`. The nondeterminism, BM25, semantic search and cross-encoder pages fill their numbers from `out/*.json` when they are built.
- **Sources are marked.** The nondeterminism post tags every number "ran on Mac" or "paper". The RRF post names the source under each figure, such as "Raw output from OpenSearch" or "Table 1 of Cormack et al. (2009)".
- **Differences are reported.** When a result disagrees with the paper, the post says so and explains why. For example, Python's `sum()` gives the article's 102 only on Python 3.12 and 3.13.
