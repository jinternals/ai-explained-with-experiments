package rrf;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * The real example used in Step 3 of the blog post.
 *
 * <p>One NFCorpus question, the top 5 from each searcher with their raw scores and the expert
 * relevance labels, then RRF worked out by hand on those two short lists.
 */
final class Step3Example {

    private static final String DATASET = "nfcorpus";
    private static final String QUERY_ID = "PLAIN-2830"; // "Neurobiology of Artificial Sweeteners"
    private static final int TOP_N = 5;
    private static final int K = 60;

    private Step3Example() {}

    static void run() throws Exception {
        SearchIndex.waitUntilReady();

        Beir.Dataset data = Beir.load(DATASET);
        String query = data.queries().get(QUERY_ID);
        Map<String, Integer> labels = data.qrels().get(QUERY_ID);
        Map<String, String> titles = data.documents().stream().collect(Collectors.toMap(Document::id, Document::title));

        float[][] docVectors = Embeddings.embed(data.documents().stream().map(Document::content).toList(), DATASET + "-docs");
        float[] queryVector = Embeddings.embed(query);

        SearchIndex index = new SearchIndex(DATASET + "-step3");
        index.create(docVectors[0].length);
        index.add(data.documents(), docVectors);

        List<SearchIndex.Hit> keyword = index.keyword(query, TOP_N);
        List<SearchIndex.Hit> meaning = index.vector(queryVector, TOP_N);
        List<Fusion.Scored> fused = Fusion.rrf(List.of(SearchIndex.ids(keyword), SearchIndex.ids(meaning)), K);

        long relevantTotal = labels.values().stream().filter(label -> label > 0).count();
        System.out.printf("%s %s: '%s' | %d docs judged relevant | top %d per list | k = %d%n%n",
                DATASET, QUERY_ID, query, relevantTotal, TOP_N, K);
        printHits("KEYWORD (BM25)", keyword, titles, labels);
        printHits("MEANING (embeddings)", meaning, titles, labels);

        Map<String, Integer> keywordRank = ranks(keyword);
        Map<String, Integer> meaningRank = ranks(meaning);

        System.out.println("RRF by hand");
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < fused.size(); i++) {
            Fusion.Scored doc = fused.get(i);
            Integer inKeyword = keywordRank.get(doc.id());
            Integer inMeaning = meaningRank.get(doc.id());
            double keywordPoints = inKeyword == null ? 0 : 1.0 / (K + inKeyword);
            double meaningPoints = inMeaning == null ? 0 : 1.0 / (K + inMeaning);
            int relevance = labels.getOrDefault(doc.id(), 0);

            rows.add(Json.object(
                    "id", doc.id(), "title", titles.get(doc.id()), "relevance", relevance,
                    "keyword_rank", inKeyword, "meaning_rank", inMeaning,
                    "keyword_points", keywordPoints, "meaning_points", meaningPoints,
                    "rrf", doc.score()));
            System.out.printf("  %d. %.6f = %.6f (kw %s) + %.6f (mn %s)  relevance %d  %s%n",
                    i + 1, doc.score(), keywordPoints, inKeyword == null ? "-" : inKeyword,
                    meaningPoints, inMeaning == null ? "-" : inMeaning, relevance, titles.get(doc.id()));
        }

        Json.writeFile(Path.of("/out/step3_example.json"), Json.object(
                "dataset", DATASET, "query_id", QUERY_ID, "query", query, "top_n", TOP_N, "k", K,
                "relevant_total", relevantTotal,
                "keyword", describe(keyword, titles, labels),
                "meaning", describe(meaning, titles, labels),
                "rrf", rows));
    }

    private static Map<String, Integer> ranks(List<SearchIndex.Hit> hits) {
        Map<String, Integer> ranks = new LinkedHashMap<>();
        for (int i = 0; i < hits.size(); i++) {
            ranks.put(hits.get(i).id(), i + 1);
        }
        return ranks;
    }

    private static List<Map<String, Object>> describe(
            List<SearchIndex.Hit> hits, Map<String, String> titles, Map<String, Integer> labels) {
        return hits.stream()
                .map(hit -> Json.object(
                        "id", hit.id(), "score", hit.score(),
                        "title", titles.get(hit.id()), "relevance", labels.getOrDefault(hit.id(), 0)))
                .toList();
    }

    private static void printHits(
            String heading, List<SearchIndex.Hit> hits, Map<String, String> titles, Map<String, Integer> labels) {
        System.out.println(heading);
        for (int i = 0; i < hits.size(); i++) {
            SearchIndex.Hit hit = hits.get(i);
            System.out.printf("  %d. score %.4f  relevance %d  %s%n",
                    i + 1, hit.score(), labels.getOrDefault(hit.id(), 0), titles.get(hit.id()));
        }
        System.out.println();
    }
}
