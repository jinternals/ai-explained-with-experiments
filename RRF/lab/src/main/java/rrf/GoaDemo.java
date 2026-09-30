package rrf;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * The "cheap flights to Goa" example: 10 travel-page titles, one query, the top 5 from each searcher,
 * fused with RRF by hand and by OpenSearch so we can check they agree.
 */
final class GoaDemo {

    private static final String QUERY = "cheap flights to Goa";
    private static final int TOP_N = 5;
    private static final int K = 60;

    private static final List<String> TITLES = List.of(
            "Cheap flights to Goa from Mumbai, Delhi and Bengaluru",
            "Budget airfare to Goa: how to find low-cost tickets this winter",
            "Cheap hotels in Goa near Baga and Calangute beach",
            "Flights to Goa delayed as monsoon storms hit Dabolim airport",
            "Low-cost airlines flying Mumbai to Goa for under ₹2,000",
            "Cheap flights to Bali from India: best deals this month",
            "How to book inexpensive plane tickets to India's beach state",
            "Goa train timetable: Konkan Railway from Mumbai",
            "Best time to visit Goa for beaches and festivals",
            "Airline fare sale: domestic tickets from ₹1,499");

    private GoaDemo() {}

    static void run() throws Exception {
        String version = SearchIndex.waitUntilReady();

        List<Document> documents = new ArrayList<>();
        for (int i = 0; i < TITLES.size(); i++) {
            documents.add(new Document("D" + (i + 1), TITLES.get(i), ""));
        }
        Map<String, String> titles = documents.stream().collect(Collectors.toMap(Document::id, Document::title));

        float[][] vectors = Embeddings.embed(TITLES, null);
        float[] queryVector = Embeddings.embed(QUERY);

        SearchIndex index = new SearchIndex("goa-demo");
        index.create(vectors[0].length);
        index.add(documents, vectors);

        List<SearchIndex.Hit> keyword = index.keyword(QUERY, TOP_N);
        List<SearchIndex.Hit> meaning = index.vector(queryVector, TOP_N);
        List<Fusion.Scored> byHand = Fusion.rrf(List.of(SearchIndex.ids(keyword), SearchIndex.ids(meaning)), K);
        List<SearchIndex.Hit> byEngine = index.hybridRrf(QUERY, queryVector, TOP_N, K, TOP_N * 2);

        System.out.printf("OpenSearch %s | query: '%s' | top %d per list | k = %d%n%n", version, QUERY, TOP_N, K);
        printHits("KEYWORD (BM25)", keyword, titles);
        printHits("MEANING (embeddings)", meaning, titles);

        System.out.println("RRF: by hand vs OpenSearch");
        for (int i = 0; i < Math.min(byHand.size(), byEngine.size()); i++) {
            System.out.printf("  %d. %-3s %.5f   |   %-3s %.5f%n",
                    i + 1, byHand.get(i).id(), byHand.get(i).score(), byEngine.get(i).id(), byEngine.get(i).score());
        }
        boolean sameOrder = byHand.stream().map(Fusion.Scored::id).toList().equals(SearchIndex.ids(byEngine));
        System.out.println("\nSame order: " + sameOrder);

        Json.writeFile(Path.of("/out/goa_demo.json"), Json.object(
                "opensearch_version", version, "query", QUERY, "top_n", TOP_N, "k", K, "titles", titles,
                "keyword", keyword, "meaning", meaning,
                "rrf_by_hand", byHand, "rrf_opensearch", byEngine, "same_order", sameOrder));
    }

    private static void printHits(String heading, List<SearchIndex.Hit> hits, Map<String, String> titles) {
        System.out.println(heading);
        for (int i = 0; i < hits.size(); i++) {
            SearchIndex.Hit hit = hits.get(i);
            System.out.printf("  %d. %-3s score %-7.4f %s%n", i + 1, hit.id(), hit.score(), titles.get(hit.id()));
        }
        System.out.println();
    }
}
