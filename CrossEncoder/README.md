# Cross-encoder re-ranking

How a cross-encoder re-sorts search results by reading the question and each document together, what it costs, and when it helps. Tested on the same data and search engine as the RRF post.

**Post:** [`blog/index.html`](blog/index.html) · **Follows:** [Reciprocal Rank Fusion, Without the Scary Math](https://medium.com/jinternals/reciprocal-rank-fusion-without-the-scary-math-68421476c505) · **Model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

## The short version

Meaning search (a bi-encoder) turns the question and every document into 384 numbers separately and compares them. That is fast because documents are prepared ahead of time, but the model never sees the question and the document together. A cross-encoder joins the question and one document into one input and outputs one relevance score. It is more careful and must run once per (question, document) pair, so it is used only to re-rank the top results of a fast search.

## Results

OpenSearch 2.19.1, top 100 from each first stage, cross-encoder/ms-marco-MiniLM-L6-v2 on 12 CPU cores in Docker. Scores are nDCG@10 (higher is better); "top N" means the cross-encoder re-sorted the first N results.

| Dataset | First stage | Before | Top 20 re-ranked | Top 100 re-ranked |
|---|---|---|---|---|
| NFCorpus | keyword | 0.307 | 0.331 | 0.343 |
| | meaning | 0.315 | 0.338 | 0.338 |
| | RRF | 0.338 | **0.354** | 0.353 |
| SciFact | keyword | 0.656 | 0.675 | 0.683 |
| | meaning | 0.624 | 0.680 | **0.694** |
| | RRF | 0.682 | 0.695 | 0.691 |
| FiQA | keyword | 0.239 | 0.304 | 0.334 |
| | meaning | 0.364 | **0.380** | 0.376 |
| | RRF | 0.343 | 0.379 | 0.370 |

- **Every combination improved.** The weakest list gained most: keyword search on FiQA, +40% with the top 100 re-ranked.
- **For RRF, the top 20 beat the top 100** on all three datasets, at a sixth of the time: 672 ms against 4,001 ms per question (median, NFCorpus). Keyword search and meaning search each took under 10 ms.
- **The model is the main limit.** Putting RRF's top 100 in perfect order would reach 0.657, 0.948 and 0.726. The cross-encoder reached 0.353, 0.691 and 0.370.
- **Averages hide losses.** On SciFact, re-ranking made 59 questions worse and 57 better. The average rose because the wins were bigger.
- **The worked example got worse:** the one labelled paper fell from 1st to 4th (nDCG@10 0.639 to 0.275). See Notes.

## Requirements

- Docker. The first build downloads Maven dependencies, which takes a few minutes.
- About 270 MB of disk space: the BEIR datasets (64 MB), cached embeddings (99 MB), the ONNX model (97 MB) and the cached cross-encoder scores (6 MB).
- To rebuild the page: Python 3 (standard library only).

## Run it

From this folder:

```bash
docker compose run --rm --build lab example     # the worked example, about a minute
docker compose run --rm --build lab benchmark   # three BEIR datasets
docker compose down                             # stop OpenSearch when you're done

python3 blog/build_page.py                      # rebuild blog/index.html from out/*.json
```

The first benchmark run cross-encodes 216,813 (question, document) pairs on the CPU, at about 28 pairs per second on 12 cores: a little over two hours of computing. Keep the Mac awake; our run paused whenever it slept. Scores are saved to `cache/scores/` every 30 seconds, so a stopped run picks up where it left off, and later runs only redo the arithmetic. The timing section never uses the cache.

## Layout

```
CrossEncoder/
├── blog/      the post
├── lab/       the experiments, in Java 21
└── out/       results as JSON
```

| Path | What it is |
|---|---|
| `blog/index.html` | The post. Built by `build_page.py`, so don't edit it by hand |
| `blog/page/body.html` | The post's text. Numbers are placeholders such as `{{b:datasets.nfcorpus.ndcg@10.rrf.100\|3}}` |
| `blog/build_page.py` | Fills every placeholder from `out/benchmark.json` and `out/example.json`. Fails if one is missing, unless run with `--draft` |
| `lab/src/main/java/ce/Reranker.java` | The cross-encoder: downloads the pinned ONNX model, scores pairs in batches of 32, caches scores |
| `lab/src/main/java/ce/Example.java` | The worked example: the NFCorpus question from the RRF post, re-ranked |
| `lab/src/main/java/ce/Benchmark.java` | Re-ranks the top 10/20/50/100 of keyword, meaning and RRF lists; timing; recall and perfect-re-ranking limits |
| `lab/src/main/java/ce/SearchIndex.java` | OpenSearch client: keyword (BM25) and vector (k-NN) search |
| `lab/src/main/java/ce/Embeddings.java` | all-MiniLM-L6-v2 embeddings, cached on disk |
| `lab/src/main/java/ce/Fusion.java`, `Metrics.java`, `Beir.java` | RRF, nDCG@10 and Recall@100, BEIR loading. Shared with the RRF lab |
| `out/example.json` | The worked example's scores |
| `out/benchmark.json` | Every benchmark number in the post |
| `data/`, `cache/` | Created at run time |

## Notes

- **The worked example goes the wrong way.** On "Neurobiology of artificial sweeteners" the cross-encoder moved the one labelled paper from 1st to 4th, in favour of "Possible neurologic effects of aspartame". That paper arguably fits the question better. NFCorpus labels come from which papers NutritionFacts.org articles cite, not from judging each answer.
- **Timings are CPU numbers.** The lab runs ONNX Runtime inside Docker on the CPU. The model card's 1,800 documents per second was measured on an NVIDIA V100 GPU.
- **Long documents are cut.** The cross-encoder reads at most 512 tokens per (question, document) pair.
- **The model is pinned** to revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, so a re-run downloads exactly the same weights.

## References

- [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) and its model card
- N. Thakur et al., [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), NeurIPS 2021 Datasets and Benchmarks
- [MS MARCO](https://microsoft.github.io/msmarco/), the training data
- Sentence Transformers, [Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
- OpenSearch, [rerank processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/rerank-processor/)
