# Reciprocal Rank Fusion

How hybrid search merges a keyword-search list and a meaning-search list into one, explained with fractions and tested on a real search engine.

**Post:** [`blog/rrf-blog.html`](blog/rrf-blog.html) · **Medium version:** [`medium/rrf-medium.html`](medium/rrf-medium.html), [live on Medium](https://medium.com/jinternals/reciprocal-rank-fusion-without-the-scary-math-68421476c505) · **Based on:** [Cormack, Clarke and Büttcher, SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

## The short version

Keyword search and meaning search score results on different scales, so their scores can't be added. RRF ignores the scores and uses only positions. Each result gets 1 ÷ (k + position) points from every list it appears in, and the points are added up. Results that several searchers agree on rise to the top. It works best when the searchers are about equally good.

## Results

OpenSearch 2.19.1, BM25 for keyword search, all-MiniLM-L6-v2 for meaning search, top 100 from each, k = 60. Scores are nDCG@10 (higher is better).

| Dataset | Questions | Keyword | Meaning | RRF | RRF vs. the better searcher |
|---|---|---|---|---|---|
| NFCorpus | 323 | 0.307 | 0.315 | **0.342** | +9% |
| SciFact | 300 | 0.656 | 0.624 | **0.682** | +4% |
| FiQA | 648 | 0.239 | **0.363** | 0.344 | −5% |

- **The worked example:** for "Neurobiology of artificial sweeteners", the one relevant paper was 2nd in the keyword list and 4th in the meaning list. RRF put it 1st, by 0.000008.
- **k:** a smaller k did better on all three datasets. With k = 10, FiQA reaches 0.367 and beats meaning search alone.
- **Checked twice:** our keyword scores are close to the BEIR paper's BM25 results. Our RRF matched OpenSearch's built-in RRF on all 1,271 questions, apart from the order of tied documents.

## Requirements

- Docker. The first build downloads Maven dependencies, which takes a few minutes.
- About 350 MB of disk space for the datasets and cached embeddings.
- To rebuild the Medium figures: Google Chrome and Node 22 or newer.

## Run it

From this folder:

```bash
docker compose run --rm --build lab goa         # the small "cheap flights to Goa" example
docker compose run --rm --build lab step3       # the real question worked through in Step 3
docker compose run --rm --build lab benchmark   # NFCorpus, SciFact and FiQA from BEIR
docker compose down                             # stop OpenSearch when you're done

node medium/build-images.mjs                    # rebuild the Medium figures from out/
```

The first benchmark run downloads the datasets (about 23 MB zipped) and embeds about 66,000 documents on the CPU. FiQA is the slow one. Later runs reuse `data/` and `cache/`. Results go to `out/`.

The Python version still works: `docker compose run --rm --build lab-python python benchmark.py`. Its results go to `out/python/`.

## Layout

```
RRF/
├── blog/         the post
├── medium/       the Medium version and its figures
├── lab/          the experiments, in Java 21
├── lab-python/   the first version of the lab, in Python
└── out/          results as JSON
```

| Path | What it is |
|---|---|
| `blog/rrf-blog.html` | The original post |
| `medium/rrf-medium.md` | The Medium story, in Medium-friendly Markdown |
| `medium/rrf-medium.html` | The same story as a standalone web page |
| `medium/images/` | The figures as PNGs, rendered from `out/*.json` |
| `medium/build-images.mjs` | Renders `medium/images/` with headless Chrome |
| `medium/PUBLISHING.md` | How to put the story on Medium |
| `lab/src/main/java/rrf/Fusion.java` | `rrf()`: the whole algorithm in a few lines |
| `lab/src/main/java/rrf/Metrics.java` | nDCG@10 and Recall@100 |
| `lab/src/main/java/rrf/SearchIndex.java` | OpenSearch: create an index, then keyword, vector and hybrid-RRF search |
| `lab/src/main/java/rrf/Embeddings.java` | Embeds text with all-MiniLM-L6-v2 (LangChain4j, ONNX), cached on disk |
| `lab/src/main/java/rrf/Beir.java` | Downloads and reads BEIR datasets |
| `lab/src/main/java/rrf/GoaDemo.java` | The "cheap flights to Goa" example |
| `lab/src/main/java/rrf/Step3Example.java` | The real NFCorpus question from Step 3 |
| `lab/src/main/java/rrf/Benchmark.java` | Keyword vs. meaning vs. RRF, sweeps k and depth, checks against OpenSearch |
| `lab/src/main/java/rrf/Main.java` | Picks `goa`, `step3` or `benchmark` |
| `lab-python/` | The same lab in Python |
| `data/`, `cache/`, `out/` | Created at run time |

## Notes

- **Two sets of numbers.** `blog/rrf-blog.html` uses the numbers from the first Python run. The Medium version and the table above use the Java run. They differ slightly. On FiQA, RRF's Recall@100 no longer beats meaning search in the Java run.
- **Tuning k.** The best k values were found by looking at the test questions themselves, which flatters them. Tune k on your own data.
- **Where 60 comes from.** The original paper fixed k = 60 in a small pilot test and never changed it. Any k from about 30 to 100 scored almost the same there.
- **The labels.** NFCorpus marks a paper relevant when a NutritionFacts.org article links to it. Some papers that sound on-topic are therefore marked not relevant.

## References

- G. V. Cormack, C. L. A. Clarke, S. Büttcher, [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), SIGIR 2009
- N. Thakur et al., [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), NeurIPS 2021 Datasets and Benchmarks
- OpenSearch, [Score ranker processor](https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/)
