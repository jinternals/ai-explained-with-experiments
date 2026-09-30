"""
The worked example from the blog post.

10 travel-page titles, one query, the top 5 from each searcher,
fused with RRF by hand and by OpenSearch so we can check they agree.
"""

import json

from embeddings import embed
from engine import SearchIndex, wait_until_ready
from fusion import rrf

QUERY = "cheap flights to Goa"
TOP_N = 5
K = 60

TITLES = [
    "Cheap flights to Goa from Mumbai, Delhi and Bengaluru",
    "Budget airfare to Goa: how to find low-cost tickets this winter",
    "Cheap hotels in Goa near Baga and Calangute beach",
    "Flights to Goa delayed as monsoon storms hit Dabolim airport",
    "Low-cost airlines flying Mumbai to Goa for under ₹2,000",
    "Cheap flights to Bali from India: best deals this month",
    "How to book inexpensive plane tickets to India's beach state",
    "Goa train timetable: Konkan Railway from Mumbai",
    "Best time to visit Goa for beaches and festivals",
    "Airline fare sale: domestic tickets from ₹1,499",
]


def main():
    version = wait_until_ready()

    documents = [{"id": f"D{n}", "title": title, "text": ""} for n, title in enumerate(TITLES, start=1)]
    titles = {doc["id"]: doc["title"] for doc in documents}
    query_vector = embed([QUERY])[0]

    index = SearchIndex("goa-demo")
    vectors = embed(TITLES)
    index.create(dimension=vectors.shape[1])
    index.add(documents, vectors)

    keyword = index.keyword(QUERY, size=TOP_N)
    meaning = index.vector(query_vector, size=TOP_N)
    by_hand = rrf([[hit["id"] for hit in keyword], [hit["id"] for hit in meaning]], k=K)
    by_engine = index.hybrid_rrf(QUERY, query_vector, depth=TOP_N, k=K, size=TOP_N * 2)

    print(f"OpenSearch {version} | query: {QUERY!r} | top {TOP_N} per list | k = {K}\n")
    for name, hits in (("KEYWORD (BM25)", keyword), ("MEANING (embeddings)", meaning)):
        print(name)
        for rank, hit in enumerate(hits, start=1):
            print(f"  {rank}. {hit['id']:<3} score {hit['score']:<7.4f} {titles[hit['id']]}")
        print()

    print("RRF: by hand vs OpenSearch")
    for rank, ((doc_id, score), engine_hit) in enumerate(zip(by_hand, by_engine), start=1):
        print(f"  {rank}. {doc_id:<3} {score:.5f}   |   {engine_hit['id']:<3} {engine_hit['score']:.5f}")

    same_order = [doc_id for doc_id, _ in by_hand] == [hit["id"] for hit in by_engine]
    print(f"\nSame order: {same_order}")

    with open("/out/goa_demo.json", "w") as f:
        json.dump({
            "opensearch_version": version, "query": QUERY, "top_n": TOP_N, "k": K, "titles": titles,
            "keyword": keyword, "meaning": meaning,
            "rrf_by_hand": [{"id": doc_id, "score": score} for doc_id, score in by_hand],
            "rrf_opensearch": by_engine, "same_order": same_order,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
