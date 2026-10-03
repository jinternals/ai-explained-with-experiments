# Semantic search

How search by meaning works: text turned into arrows of 384 numbers, compared by the angle between them. Part 2 of a series on how search works, after [BM25](../BM25/).

**Post:** [`blog/index.html`](blog/index.html) · **Model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · **Based on:** [BEIR](https://arxiv.org/abs/2104.08663), [Sentence-BERT](https://arxiv.org/abs/1908.10084), Spring AI's [Understanding Vectors](https://docs.spring.io/spring-ai/reference/api/vectordbs/understand-vectordbs.html)

## The short version

An embedding model turns any text into a list of 384 numbers, which you can picture as an arrow. Words, sentences and paragraphs go through the same recipe: split into word pieces, adjust each piece by its neighbours, average them into one arrow. Texts that mean similar things point in similar directions, and **cosine similarity** measures that: multiply matching numbers and add them up (the dot product), then divide by the two arrows' lengths. Semantic search embeds every document once, embeds the question at search time, and returns the documents whose arrows point most nearly the same way.

## Results

OpenSearch 2.19.1, all-MiniLM-L6-v2 through LangChain4j, nDCG@10 (higher is better).

| Dataset | Keyword (BM25) | Meaning, exact | Meaning, HNSW | Recall@100 keyword → meaning |
|---|---|---|---|---|
| NFCorpus | 0.307 | 0.315 | 0.315 | 0.244 → 0.309 |
| SciFact | **0.656** | 0.624 | 0.624 | 0.884 → 0.923 |
| FiQA | 0.239 | **0.366** | 0.361 | 0.506 → 0.694 |

- **Shared words decide it.** When a question's relevant documents contain under a third of its words, meaning search won 229 questions and lost 85. With two thirds or more, it won 36 and lost 54.
- **Words vs sentences.** "bank" alone is about as close to "river" (0.500) as to "money" (0.469). In sentences, a river bank and a savings bank score 0.262, below a river bank and a stream (0.394), which share no words. Averaging each word's own arrow instead would score the two banks 0.874.
- **Paragraphs blend.** A question about parking permits scores 0.743 against the parking sentence and 0.507 against the three-sentence paragraph that contains it.
- **HNSW is nearly free.** It returned the same top 10 as checking every document for 94% of questions, in about 1 ms. Exact search took 17 ms on FiQA's 57,638 documents.
- **Cosine, dot product and distance agree** on all 1,271 questions, because every vector has length 1.
- **LangChain4j reads only the first 128 word pieces** of each text. Raised to 256, all three datasets match MTEB's published scores for this model (0.317, 0.645, 0.369). Reading whole documents helped SciFact (0.624 → 0.654) but was slightly worse than 256 on NFCorpus and FiQA.

## Requirements

- Docker. The first build downloads Maven dependencies, which takes a few minutes.
- About 500 MB of disk space: the BEIR datasets (64 MB) and cached embeddings (about 100 MB per token limit).
- To rebuild the page: Python 3 (standard library only).

## Run it

From this folder:

```bash
docker compose run --rm --build lab example     # sections 01–06, about a minute
docker compose run --rm --build lab benchmark   # keyword vs meaning, HNSW, metrics, timing
docker compose run --rm --build lab length      # how much of each document the model reads; re-embeds everything twice
docker compose down                             # stop OpenSearch when you're done

python3 blog/build_page.py                      # rebuild blog/index.html from out/*.json
```

The first `benchmark` run embeds about 66,000 documents on the CPU. `length` embeds them twice more, with different token limits; FiQA is most of the time.

## Layout

```
SemanticSearch/
├── blog/      the post
├── lab/       the experiments, in Java 21
└── out/       results as JSON
```

| Path | What it is |
|---|---|
| `blog/index.html` | The post. Built by `build_page.py`, so don't edit it by hand |
| `blog/page/body.html` | The post's text and interactive parts. Numbers are placeholders such as `{{b:datasets.fiqa.exact.ndcg@10\|3}}` |
| `blog/build_page.py` | Fills every placeholder from `out/example.json`, `out/benchmark.json` and `out/length.json`. Fails if one is missing, unless run with `--draft` |
| `lab/src/main/java/semantic/Example.java` | Arrows by hand, word vs sentence vs paragraph embeddings, the pairs, and part 1's NFCorpus question |
| `lab/src/main/java/semantic/Benchmark.java` | Keyword vs meaning on three BEIR datasets, word-overlap groups, HNSW vs exact, cosine vs dot vs distance, timing |
| `lab/src/main/java/semantic/Length.java` | Shows the 128-piece cut, then re-embeds every document with 256 pieces and with no cut |
| `lab/src/main/java/semantic/VectorMath.java` | Dot product, length, cosine and distance, written out in full |
| `lab/src/main/java/semantic/SearchIndex.java` | OpenSearch client: BM25, HNSW (`knn`) and exact (`script_score`) search |
| `lab/src/main/java/semantic/Embeddings.java`, `Tokens.java` | all-MiniLM-L6-v2 through LangChain4j, cached on disk; the model's word pieces |
| `lab/src/main/java/semantic/Words.java`, `Metrics.java`, `Beir.java` | Word overlap, nDCG@10 and Recall@100, BEIR loading |
| `out/*.json` | Every number in the post |
| `data/`, `cache/` | Created at run time |

## Notes

- **The 128-piece cut.** LangChain4j's `AllMiniLmL6V2EmbeddingModel` ships a tokenizer file with `truncation.max_length` 128, so everything after roughly the first 100 words is ignored and its code for splitting texts over 510 pieces never runs. The full-text vector of a long document is identical (cosine 1.0) to the vector of its first 126 pieces. The model card says 256. The RRF and cross-encoder labs use the same model the same way, so their meaning-search numbers were measured with this cut too.
- **Spring AI** (2.0.1, read in the source, not run): `TransformersEmbeddingModel` loads a tokenizer file with the same 128 cut and doesn't normalise its vectors, so dot product and cosine would rank differently there.
- **Word overlap** is measured without stemming and ignores Lucene's 33 English stop words, so "cataract" and "cataracts" count as different words.
- **The HNSW animation in section 10 is an illustration**, not data. The measured HNSW numbers are in the table below it.
- **Timings** are medians over 200 questions per dataset, inside Docker on an M4 Pro MacBook Pro's CPU.
- **What nDCG@10 means.** Every score here is nDCG@10, a number from 0 to 1 for how good the first 10 results are. Each relevant result earns its label as points: NFCorpus grades papers 2 (very relevant) or 1 (somewhat relevant), while SciFact and FiQA mark every relevant document 1. Points are divided by log₂(position + 1), so a result counts in full at 1st, about 63% at 2nd, half at 3rd and about 29% at 10th. The total is then divided by the score of a perfect top 10, so 1 means the best possible order. Unlike MRR (mean reciprocal rank), it rewards every relevant result in the top 10, not only the first. The code is `ndcgAtK` in `lab/src/main/java/semantic/Metrics.java`.

## References

- N. Reimers and I. Gurevych, [Sentence-BERT](https://arxiv.org/abs/1908.10084), EMNLP 2019
- T. Mikolov et al., [word2vec](https://arxiv.org/abs/1301.3781), 2013
- N. Thakur et al., [BEIR](https://arxiv.org/abs/2104.08663), NeurIPS 2021 Datasets and Benchmarks
- [MTEB results for all-MiniLM-L6-v2](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2)
- Y. Malkov and D. Yashunin, [HNSW](https://arxiv.org/abs/1603.09320), IEEE TPAMI 2018
- OpenSearch, [k-NN methods and engines](https://docs.opensearch.org/latest/field-types/supported-field-types/knn-methods-engines/)
- D. Jurafsky and J. H. Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), chapter "Embeddings"
