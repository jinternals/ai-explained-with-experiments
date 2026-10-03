# BM25

How keyword search scores a document, one word at a time, and which of its settings actually change results. Part 1 of a series on how search works.

**Post:** [`blog/index.html`](blog/index.html) · **Based on:** [BEIR](https://arxiv.org/abs/2104.08663), Lucene's [BM25Similarity](https://lucene.apache.org/core/9_12_0/core/org/apache/lucene/search/similarities/BM25Similarity.html)

## The short version

BM25 adds up points for each word of the question that a document contains. Rare words earn more (**IDF**). Repeating a word earns a little more each time, with the extra credit running out at a speed set by **k1**, and long documents get a small penalty set by **b** (together, **TF**). How text is split into words (the **analyzer**) decides what counts as the same word in the first place.

## The formula

```
score = Σ  IDF × TF          (Σ: add up over every question word the document contains)

IDF   = ln( 1 + (N − n + 0.5) / (n + 0.5) )

TF    = freq × (k1 + 1) / ( freq + k1 × (1 − b + b × length / avg length) )
```

| Symbol | Meaning |
|---|---|
| N | number of documents (3,633 in NFCorpus) |
| n | documents that contain the word |
| freq | times the word appears in this document's field |
| length, avg length | terms in this document's field, and the average over all documents |
| k1 | how quickly repeats stop adding more (default 1.2) |
| b | how much length counts, 0 to 1 (default 0.75) |

Worked example, "sweeteners" in the text of the top result: IDF = ln(1 + (3,633 − 11 + 0.5) / (11 + 0.5)) = 5.7557, TF = 4 × 2.2 / (4 + 1.2 × (0.25 + 0.75 × 152 / 227.6)) = 1.796, so IDF × TF = 10.33. Six such (field, word) scores make up the final 28.64.

OpenSearch's explain API shows TF in two pieces: `boost` (k1 + 1 = 2.2) and `tf` (the rest, 0.8162 here). Their product is TF as written above.

## Results

OpenSearch 2.19.1, BM25 over title and text (`multi_match`, `tie_breaker` 0.5), nDCG@10 (higher is better).

| Dataset | standard, defaults (1.2, 0.75) | english, defaults | english, BEIR's k1 0.9 b 0.4 | BEIR paper | Best of 42 k1 × b (english) |
|---|---|---|---|---|---|
| NFCorpus | 0.307 | 0.327 | **0.325** | 0.325 | 0.333 |
| SciFact | 0.656 | 0.691 | 0.681 | 0.665 | 0.691 |
| FiQA | 0.239 | 0.254 | **0.236** | 0.236 | 0.254 |

- **The worked example rebuilds a real score of 28.64** (the top result for the NFCorpus question "Neurobiology of Artificial Sweeteners") from six (field, word) pairs returned by OpenSearch's explain API.
- **k1 and b barely matter near the defaults.** The best of 42 combinations beat the defaults by at most 0.005. Extremes hurt: with the standard analyzer, FiQA ranges from 0.132 to 0.241 across the grid, worst when document length is ignored (b = 0).
- **Stemming matters more:** +0.021 NFCorpus, +0.034 SciFact, +0.015 FiQA.
- **The BEIR gap is explained.** The paper used Anserini (Porter stemming, k1 0.9, b 0.4). Matching that reproduces NFCorpus and FiQA to three decimals. SciFact comes out higher than the paper, for reasons not tracked down.

## Requirements

- Docker. The first build downloads Maven dependencies, which takes a few minutes.
- About 64 MB of disk space for the BEIR datasets. No embedding model.
- To rebuild the page: Python 3 (standard library only).

## Run it

From this folder:

```bash
docker compose run --rm --build lab example     # the worked example, under a minute
docker compose run --rm --build lab benchmark   # 3 datasets x 2 analyzers x 42 settings, about 6 minutes
docker compose down                             # stop OpenSearch when you're done

python3 blog/build_page.py                      # rebuild blog/index.html from out/*.json
```

## Layout

```
BM25/
├── blog/      the post
├── lab/       the experiments, in Java 21
└── out/       results as JSON
```

| Path | What it is |
|---|---|
| `blog/index.html` | The post. Built by `build_page.py`, so don't edit it by hand |
| `blog/page/body.html` | The post's text. Numbers are placeholders like `{{e:terms.2.idf\|4}}`, `{{grid:nfcorpus.english}}` draws the k1 × b table |
| `blog/build_page.py` | Fills every placeholder from `out/example.json` and `out/benchmark.json`. Fails if one is missing, unless run with `--draft` |
| `lab/src/main/java/bm25/Index.java` | OpenSearch client: build an index with an analyzer, change k1 and b without re-indexing, search, explain, analyze |
| `lab/src/main/java/bm25/Example.java` | The worked example: OpenSearch's explanation of the 28.64 score, plus curves worked out from the formula |
| `lab/src/main/java/bm25/Benchmark.java` | The k1 × b grid for both analyzers on NFCorpus, SciFact and FiQA |
| `lab/src/main/java/bm25/Metrics.java`, `Beir.java` | nDCG@10, Recall@100 and BEIR loading |
| `out/example.json`, `out/benchmark.json` | Every number in the post |
| `data/` | BEIR datasets, downloaded on first run |

## Notes

- **k1 and b change without re-indexing.** Lucene stores each field's length at index time and applies k1 and b at search time, so the lab closes the index, updates `index.similarity.default` and reopens it.
- **"Default Lucene parameters".** The BEIR paper describes k1 0.9, b 0.4 that way, but those are Anserini's defaults. Lucene's own defaults are 1.2 and 0.75, the same as OpenSearch's.
- **Two tables are worked out from the formula, not measured:** how repeats add up for different k1 (section 05) and how length changes a word's weight for different b (section 06). The post labels them.
- **What nDCG@10 means.** Every score here is nDCG@10, a number from 0 to 1 for how good the first 10 results are. Each relevant result earns its label as points: NFCorpus grades papers 2 (very relevant) or 1 (somewhat relevant), while SciFact and FiQA mark every relevant document 1. Points are divided by log₂(position + 1), so a result counts in full at 1st, about 63% at 2nd, half at 3rd and about 29% at 10th. The total is then divided by the score of a perfect top 10, so 1 means the best possible order. Unlike MRR (mean reciprocal rank), it rewards every relevant result in the top 10, not only the first. The code is `ndcgAtK` in `lab/src/main/java/bm25/Metrics.java`.

## References

- S. Robertson and H. Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf), 2009
- N. Thakur et al., [BEIR](https://arxiv.org/abs/2104.08663), NeurIPS 2021 Datasets and Benchmarks, section 4 and Table 2
- Anserini, [IndexCollection `-stemmer` option](https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/index/IndexCollection.java) (defaults to Porter)
- OpenSearch: [English analyzer](https://docs.opensearch.org/latest/analyzers/language-analyzers/english/), [Explain API](https://docs.opensearch.org/latest/api-reference/search-apis/explain/), [Analyze API](https://docs.opensearch.org/latest/api-reference/analyze-apis/)
