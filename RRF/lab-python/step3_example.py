"""
The real example used in Step 3 of the blog post.

One NFCorpus question, the top 5 from each searcher with their raw scores and the expert
relevance labels, then RRF worked out by hand on those two short lists.
"""

import json

import beir
from embeddings import embed
from engine import SearchIndex, wait_until_ready
from fusion import rrf

DATASET = "nfcorpus"
QUERY_ID = "PLAIN-2830"  # "Neurobiology of Artificial Sweeteners"
TOP_N = 5
K = 60


def main():
    wait_until_ready()

    documents, queries, qrels = beir.load(DATASET)
    query, labels = queries[QUERY_ID], qrels[QUERY_ID]
    titles = {doc["id"]: doc["title"] for doc in documents}

    doc_vectors = embed([f"{doc['title']} {doc['text']}".strip() for doc in documents], cache_name=f"{DATASET}-docs")
    query_vector = embed([query])[0]

    index = SearchIndex(f"{DATASET}-step3")
    index.create(dimension=doc_vectors.shape[1])
    index.add(documents, doc_vectors)

    keyword = index.keyword(query, size=TOP_N)
    meaning = index.vector(query_vector, size=TOP_N)
    fused = rrf([[hit["id"] for hit in keyword], [hit["id"] for hit in meaning]], k=K)

    relevant_total = sum(label > 0 for label in labels.values())
    print(f"{DATASET} {QUERY_ID}: {query!r} | {relevant_total} docs judged relevant | top {TOP_N} per list | k = {K}\n")

    for name, hits in (("KEYWORD (BM25)", keyword), ("MEANING (embeddings)", meaning)):
        print(name)
        for rank, hit in enumerate(hits, start=1):
            print(f"  {rank}. score {hit['score']:.4f}  relevance {labels.get(hit['id'], 0)}  {titles[hit['id']]}")
        print()

    keyword_rank = {hit["id"]: rank for rank, hit in enumerate(keyword, start=1)}
    meaning_rank = {hit["id"]: rank for rank, hit in enumerate(meaning, start=1)}

    print("RRF by hand")
    rows = []
    for position, (doc_id, total) in enumerate(fused, start=1):
        row = {
            "id": doc_id,
            "title": titles[doc_id],
            "relevance": labels.get(doc_id, 0),
            "keyword_rank": keyword_rank.get(doc_id),
            "meaning_rank": meaning_rank.get(doc_id),
            "keyword_points": 1 / (K + keyword_rank[doc_id]) if doc_id in keyword_rank else 0.0,
            "meaning_points": 1 / (K + meaning_rank[doc_id]) if doc_id in meaning_rank else 0.0,
            "rrf": total,
        }
        rows.append(row)
        print(
            f"  {position}. {total:.6f} = {row['keyword_points']:.6f} (kw {row['keyword_rank']}) "
            f"+ {row['meaning_points']:.6f} (mn {row['meaning_rank']})  relevance {row['relevance']}  {row['title']}"
        )

    with open("/out/step3_example.json", "w") as f:
        json.dump({
            "dataset": DATASET, "query_id": QUERY_ID, "query": query, "top_n": TOP_N, "k": K,
            "relevant_total": relevant_total,
            "keyword": [{**hit, "title": titles[hit["id"]], "relevance": labels.get(hit["id"], 0)} for hit in keyword],
            "meaning": [{**hit, "title": titles[hit["id"]], "relevance": labels.get(hit["id"], 0)} for hit in meaning],
            "rrf": rows,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
