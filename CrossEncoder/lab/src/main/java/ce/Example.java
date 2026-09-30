package ce;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * The worked example: the NFCorpus question from Step 3 of the RRF post, "Neurobiology of artificial
 * sweeteners". Take the same top-5 keyword and meaning lists, merge them with RRF, then re-rank the
 * merged list with the cross-encoder, and compare with the bi-encoder's cosine similarity.
 */
final class Example {

    private static final String DATASET = "nfcorpus";
    private static final String QUERY_ID = "PLAIN-2830";
    private static final int TOP_N = 5;
    private static final int K = 60;

    private Example() {}

    static void run() throws Exception {
        SearchIndex.waitUntilReady();
        Beir.Dataset data = Beir.load(DATASET);
        Map<String, Document> byId = data.documents().stream()
                .collect(Collectors.toMap(Document::id, Function.identity()));
        String query = data.queries().get(QUERY_ID);
        Map<String, Integer> labels = data.qrels().get(QUERY_ID);

        float[][] docVectors = Embeddings.embed(data.documents().stream().map(Document::content).toList(), DATASET + "-docs");
        SearchIndex index = new SearchIndex(DATASET);
        index.create(docVectors[0].length);
        index.add(data.documents(), docVectors);

        float[] queryVector = Embeddings.embed(query);
        List<String> keyword = SearchIndex.ids(index.keyword(query, TOP_N));
        List<String> meaning = SearchIndex.ids(index.vector(queryVector, TOP_N));
        List<Fusion.Scored> fused = Fusion.rrf(List.of(keyword, meaning), K);
        List<String> fusedIds = fused.stream().map(Fusion.Scored::id).toList();

        // Everything the searchers found, plus the relevant papers they missed.
        List<String> relevant = labels.entrySet().stream().filter(e -> e.getValue() > 0).map(Map.Entry::getKey).toList();
        List<String> scored = new ArrayList<>(fusedIds);
        relevant.stream().filter(id -> !scored.contains(id)).forEach(scored::add);

        Map<String, float[]> vectorById = new LinkedHashMap<>();
        for (int i = 0; i < data.documents().size(); i++) {
            vectorById.put(data.documents().get(i).id(), docVectors[i]);
        }

        double[] crossScores;
        long nanos;
        try (Reranker reranker = new Reranker()) {
            reranker.score(query, List.of(byId.get(scored.get(0)))); // warm-up
            long start = System.nanoTime();
            crossScores = reranker.score(query, scored.stream().map(byId::get).toList());
            nanos = System.nanoTime() - start;
        }
        Map<String, Double> crossById = new LinkedHashMap<>();
        for (int i = 0; i < scored.size(); i++) {
            crossById.put(scored.get(i), crossScores[i]);
        }
        List<String> reranked = Reranker.sortByScore(fusedIds, crossById);

        List<Map<String, Object>> rows = new ArrayList<>();
        for (String id : scored) {
            Document doc = byId.get(id);
            rows.add(Json.object(
                    "id", id,
                    "title", doc.title(),
                    "relevance", labels.getOrDefault(id, 0),
                    "keyword_rank", rankOf(keyword, id),
                    "meaning_rank", rankOf(meaning, id),
                    "rrf_rank", rankOf(fusedIds, id),
                    "rrf_score", fused.stream().filter(s -> s.id().equals(id)).mapToDouble(Fusion.Scored::score).findFirst().orElse(0),
                    "cosine", round(cosine(queryVector, vectorById.get(id)), 4),
                    "cross_encoder_score", round(crossById.get(id), 3),
                    "cross_encoder_rank", rankOf(reranked, id)));
        }

        Map<String, Object> result = Json.object(
                "dataset", DATASET,
                "query_id", QUERY_ID,
                "query", query,
                "model", Reranker.MODEL,
                "model_revision", Reranker.REVISION,
                "top_n", TOP_N,
                "k", K,
                "pairs_scored", scored.size(),
                "milliseconds", round(nanos / 1e6, 1),
                "rrf_order", fusedIds,
                "cross_encoder_order", reranked,
                "ndcg@10", Json.object(
                        "rrf", round(Metrics.ndcgAtK(fusedIds, labels, 10), 4),
                        "rrf_then_cross_encoder", round(Metrics.ndcgAtK(reranked, labels, 10), 4)),
                "documents", rows);
        Json.writeFile(Path.of("/out/example.json"), result);

        System.out.printf("%nQuestion: %s%n%n", query);
        System.out.printf("%-4s %-4s %-9s %-7s %-60s%n", "RRF", "CE", "CE score", "cosine", "document");
        for (Map<String, Object> row : rows) {
            System.out.printf("%-4s %-4s %9.3f %7.4f %-60s%s%n",
                    orDash(row.get("rrf_rank")), orDash(row.get("cross_encoder_rank")),
                    (double) row.get("cross_encoder_score"), (double) row.get("cosine"),
                    truncate((String) row.get("title"), 60), (int) row.get("relevance") > 0 ? "  <- relevant" : "");
        }
        System.out.printf("%nScored %d pairs in %.1f ms. Written to out/example.json%n", scored.size(), nanos / 1e6);
    }

    private static Object rankOf(List<String> ranked, String id) {
        int i = ranked.indexOf(id);
        return i < 0 ? null : i + 1;
    }

    private static String orDash(Object value) {
        return value == null ? "-" : value.toString();
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }

    static double round(double value, int digits) {
        double scale = Math.pow(10, digits);
        return Math.round(value * scale) / scale;
    }

    private static String truncate(String text, int max) {
        return text.length() <= max ? text : text.substring(0, max - 1) + "…";
    }
}
