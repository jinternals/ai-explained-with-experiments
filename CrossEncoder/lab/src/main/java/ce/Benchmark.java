package ce;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Does a cross-encoder improve the top 10, and what does it cost?
 *
 * <p>For every BEIR test query we get the top 100 from keyword search (BM25), meaning search
 * (MiniLM + k-NN) and RRF of the two. Then the cross-encoder re-sorts the first N of each list
 * (N = 10, 20, 50, 100) and we score the new top 10 with nDCG@10. Cross-encoder scores are cached in
 * cache/scores/, so a second run only re-does the arithmetic. The timing section never uses the cache.
 */
final class Benchmark {

    private static final List<String> DATASETS = List.of("nfcorpus", "scifact", "fiqa");
    private static final int DEPTH = 100;
    private static final int K = 60;
    private static final List<Integer> RERANK_DEPTHS = List.of(10, 20, 50, 100);
    private static final List<String> STAGES = List.of("keyword", "meaning", "rrf");
    private static final int TIMING_QUERIES = 50;
    private static final int EXAMPLES = 3;

    private Benchmark() {}

    static void run() throws Exception {
        String version = SearchIndex.waitUntilReady();
        Map<String, Object> datasets = new LinkedHashMap<>();
        Map<String, Object> timing = null;
        try (Reranker reranker = new Reranker()) {
            for (String name : DATASETS) {
                Loaded loaded = load(name);
                if (timing == null) {
                    timing = time(loaded, reranker); // on the first dataset, before anything is cached in memory
                }
                Map<String, Object> result = evaluate(loaded, reranker);
                printSummary(name, result);
                datasets.put(name, result);
            }
        }
        Json.writeFile(Path.of("/out/benchmark.json"), Json.object(
                "opensearch_version", version,
                "model", Reranker.MODEL,
                "model_revision", Reranker.REVISION,
                "cpus", Runtime.getRuntime().availableProcessors(),
                "depth", DEPTH,
                "k", K,
                "timing", timing,
                "datasets", datasets));
        System.out.println("\nWritten to out/benchmark.json");
    }

    // ------------------------------------------------------------ first stage

    private record Loaded(
            String name,
            Beir.Dataset data,
            List<String> queryIds,
            Map<String, Document> byId,
            float[][] queryVectors,
            SearchIndex index,
            Map<String, Map<String, List<String>>> stages) {}

    private static Loaded load(String name) throws Exception {
        Beir.Dataset data = Beir.load(name);
        List<String> queryIds = List.copyOf(data.queries().keySet());
        System.out.printf("%n[%s] %,d documents, %d test queries%n", name, data.documents().size(), queryIds.size());

        float[][] docVectors = Embeddings.embed(data.documents().stream().map(Document::content).toList(), name + "-docs");
        float[][] queryVectors = Embeddings.embed(queryIds.stream().map(data.queries()::get).toList(), name + "-queries");
        SearchIndex index = new SearchIndex(name);
        index.create(docVectors[0].length);
        index.add(data.documents(), docVectors);

        Map<String, List<String>> keyword = new HashMap<>();
        Map<String, List<String>> meaning = new HashMap<>();
        Map<String, List<String>> rrf = new HashMap<>();
        for (int i = 0; i < queryIds.size(); i++) {
            String q = queryIds.get(i);
            keyword.put(q, SearchIndex.ids(index.keyword(data.queries().get(q), DEPTH)));
            meaning.put(q, SearchIndex.ids(index.vector(queryVectors[i], DEPTH)));
            rrf.put(q, Fusion.rrf(List.of(keyword.get(q), meaning.get(q)), K).stream()
                    .map(Fusion.Scored::id).limit(DEPTH).toList());
        }
        Map<String, Map<String, List<String>>> stages = new LinkedHashMap<>();
        stages.put("keyword", keyword);
        stages.put("meaning", meaning);
        stages.put("rrf", rrf);

        Map<String, Document> byId = data.documents().stream()
                .collect(Collectors.toMap(Document::id, Function.identity()));
        return new Loaded(name, data, queryIds, byId, queryVectors, index, stages);
    }

    // ------------------------------------------------------------ quality

    private static Map<String, Object> evaluate(Loaded loaded, Reranker reranker) throws Exception {
        List<String> queryIds = loaded.queryIds();
        Map<String, Map<String, Integer>> qrels = loaded.data().qrels();

        // Score each query against every document any first stage returned, once.
        Map<String, List<String>> candidates = new LinkedHashMap<>();
        for (String q : queryIds) {
            LinkedHashSet<String> union = new LinkedHashSet<>();
            STAGES.forEach(stage -> union.addAll(loaded.stages().get(stage).get(q)));
            candidates.put(q, List.copyOf(union));
        }
        long pairs = candidates.values().stream().mapToLong(List::size).sum();
        System.out.printf("  cross-encoding %,d (query, document) pairs%n", pairs);
        Map<String, Map<String, Double>> scores =
                reranker.scoreAll(loaded.name(), loaded.data().queries(), candidates, loaded.byId());

        Map<String, Object> ndcg = new LinkedHashMap<>();
        Map<String, Object> recall = new LinkedHashMap<>();
        Map<String, Object> ceiling = new LinkedHashMap<>();
        Map<String, Map<String, Double>> perQuery = new LinkedHashMap<>();
        for (String stage : STAGES) {
            Map<String, List<String>> ranking = loaded.stages().get(stage);
            Map<String, Object> byDepth = new LinkedHashMap<>();
            Map<String, Double> before = byQuery(queryIds, q -> Metrics.ndcgAtK(ranking.get(q), qrels.get(q), 10));
            byDepth.put("0", average(before.values()));
            perQuery.put(stage, before);
            for (int n : RERANK_DEPTHS) {
                Map<String, Double> after = byQuery(queryIds,
                        q -> Metrics.ndcgAtK(rerankTop(ranking.get(q), n, scores.get(q)), qrels.get(q), 10));
                byDepth.put(String.valueOf(n), average(after.values()));
                perQuery.put(stage + "+ce" + n, after);
            }
            ndcg.put(stage, byDepth);
            recall.put(stage, average(queryIds.stream().map(q -> Metrics.recallAtK(ranking.get(q), qrels.get(q), DEPTH)).toList()));
            // Best possible nDCG@10 if the top 100 were put in perfect order: the most a re-ranker could reach.
            ceiling.put(stage, average(queryIds.stream()
                    .map(q -> Metrics.ndcgAtK(perfectOrder(ranking.get(q), qrels.get(q)), qrels.get(q), 10)).toList()));
        }

        return Json.object(
                "documents", loaded.data().documents().size(),
                "queries", queryIds.size(),
                "pairs_scored", pairs,
                "ndcg@10", ndcg,
                "recall@100", recall,
                "perfect_rerank_ndcg@10", ceiling,
                "rrf_ce100_vs_rrf", headToHead(perQuery.get("rrf+ce100"), perQuery.get("rrf")),
                "meaning_ce100_vs_meaning", headToHead(perQuery.get("meaning+ce100"), perQuery.get("meaning")),
                "examples", examples(loaded, scores, perQuery.get("rrf"), perQuery.get("rrf+ce100")));
    }

    /** Re-sort the first n results by cross-encoder score; the rest keep their places. */
    static List<String> rerankTop(List<String> ranked, int n, Map<String, Double> scores) {
        int cut = Math.min(n, ranked.size());
        List<String> result = new ArrayList<>(Reranker.sortByScore(ranked.subList(0, cut), scores));
        result.addAll(ranked.subList(cut, ranked.size()));
        return result;
    }

    private static List<String> perfectOrder(List<String> ranked, Map<String, Integer> labels) {
        List<String> sorted = new ArrayList<>(ranked);
        sorted.sort(Comparator.comparingInt((String id) -> labels.getOrDefault(id, 0)).reversed());
        return sorted;
    }

    // ------------------------------------------------------------ cost

    /** Time the cross-encoder without any cache, next to the cost of the first stage. */
    private static Map<String, Object> time(Loaded loaded, Reranker reranker) throws Exception {
        List<String> queryIds = loaded.queryIds().subList(0, Math.min(TIMING_QUERIES, loaded.queryIds().size()));
        Map<String, List<String>> rrf = loaded.stages().get("rrf");
        String firstQuery = loaded.data().queries().get(queryIds.get(0));
        reranker.score(firstQuery, rrf.get(queryIds.get(0)).stream().limit(32).map(loaded.byId()::get).toList()); // warm-up
        Embeddings.embed(firstQuery);

        Map<String, Object> rerank = new LinkedHashMap<>();
        for (int n : RERANK_DEPTHS) {
            List<Double> millis = new ArrayList<>();
            for (String q : queryIds) {
                List<Document> docs = rrf.get(q).stream().limit(n).map(loaded.byId()::get).toList();
                long start = System.nanoTime();
                reranker.score(loaded.data().queries().get(q), docs);
                millis.add((System.nanoTime() - start) / 1e6);
            }
            rerank.put(String.valueOf(n), Json.object(
                    "median_ms", Example.round(median(millis), 1),
                    "mean_ms", Example.round(average(millis), 1),
                    "pairs_per_second", Math.round(n / (average(millis) / 1000))));
        }

        List<Double> embedMs = new ArrayList<>();
        List<Double> keywordMs = new ArrayList<>();
        List<Double> vectorMs = new ArrayList<>();
        for (int i = 0; i < queryIds.size(); i++) {
            String text = loaded.data().queries().get(queryIds.get(i));
            long t0 = System.nanoTime();
            float[] v = Embeddings.embed(text);
            long t1 = System.nanoTime();
            loaded.index().keyword(text, DEPTH);
            long t2 = System.nanoTime();
            loaded.index().vector(v, DEPTH);
            long t3 = System.nanoTime();
            embedMs.add((t1 - t0) / 1e6);
            keywordMs.add((t2 - t1) / 1e6);
            vectorMs.add((t3 - t2) / 1e6);
        }

        @SuppressWarnings("unchecked")
        long pairsPerSecond = (long) ((Map<String, Object>) rerank.get("100")).get("pairs_per_second");
        int corpus = loaded.data().documents().size();
        return Json.object(
                "dataset", loaded.name(),
                "queries_timed", queryIds.size(),
                "cross_encoder_rerank", rerank,
                "embed_query_median_ms", Example.round(median(embedMs), 1),
                "keyword_search_top100_median_ms", Example.round(median(keywordMs), 1),
                "vector_search_top100_median_ms", Example.round(median(vectorMs), 1),
                "whole_corpus_documents", corpus,
                "whole_corpus_estimated_seconds_per_query", Example.round((double) corpus / pairsPerSecond, 1));
    }

    // ------------------------------------------------------------ helpers

    private static List<Map<String, Object>> examples(
            Loaded loaded, Map<String, Map<String, Double>> scores,
            Map<String, Double> before, Map<String, Double> after) {
        List<String> byGain = loaded.queryIds().stream()
                .sorted(Comparator.comparingDouble((String q) -> after.get(q) - before.get(q)).reversed())
                .toList();
        List<Map<String, Object>> rows = new ArrayList<>();
        for (String q : byGain.subList(0, EXAMPLES)) {
            rows.add(example(loaded, scores, q, "helped", before.get(q), after.get(q)));
        }
        for (String q : byGain.subList(byGain.size() - EXAMPLES, byGain.size())) {
            rows.add(example(loaded, scores, q, "hurt", before.get(q), after.get(q)));
        }
        return rows;
    }

    private static Map<String, Object> example(
            Loaded loaded, Map<String, Map<String, Double>> scores, String q, String kind, double before, double after) {
        Map<String, Integer> labels = loaded.data().qrels().get(q);
        List<String> rrf = loaded.stages().get("rrf").get(q);
        List<String> reranked = rerankTop(rrf, 100, scores.get(q));
        Function<List<String>, List<Map<String, Object>>> top3 = ids -> ids.stream().limit(3)
                .map(id -> Json.object(
                        "id", id,
                        "title", snippet(loaded.byId().get(id)),
                        "relevance", labels.getOrDefault(id, 0),
                        "cross_encoder_score", Example.round(scores.get(q).get(id), 3)))
                .toList();
        return Json.object(
                "kind", kind,
                "query_id", q,
                "query", loaded.data().queries().get(q),
                "ndcg@10", Json.object("rrf", Example.round(before, 4), "rrf_then_cross_encoder", Example.round(after, 4)),
                "rrf_top3", top3.apply(rrf),
                "cross_encoder_top3", top3.apply(reranked));
    }

    private static String snippet(Document doc) {
        String text = doc.title().isBlank() ? doc.text() : doc.title();
        return text.length() <= 120 ? text : text.substring(0, 119) + "…";
    }

    private static Map<String, Object> headToHead(Map<String, Double> a, Map<String, Double> b) {
        int better = 0, same = 0, worse = 0;
        for (String q : a.keySet()) {
            double diff = a.get(q) - b.get(q);
            if (Math.abs(diff) < 1e-9) same++;
            else if (diff > 0) better++;
            else worse++;
        }
        return Json.object("better", better, "same", same, "worse", worse);
    }

    private static <T> Map<String, T> byQuery(List<String> queryIds, Function<String, T> f) {
        Map<String, T> result = new LinkedHashMap<>();
        queryIds.forEach(q -> result.put(q, f.apply(q)));
        return result;
    }

    private static double average(Collection<Double> values) {
        return Example.round(values.stream().mapToDouble(Double::doubleValue).average().orElse(0), 4);
    }

    private static double median(List<Double> values) {
        List<Double> sorted = values.stream().sorted().toList();
        int n = sorted.size();
        return n % 2 == 1 ? sorted.get(n / 2) : (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2;
    }

    @SuppressWarnings("unchecked")
    private static void printSummary(String name, Map<String, Object> result) {
        System.out.printf("  nDCG@10 (first stage -> after re-ranking the top 10 / 20 / 50 / 100)%n");
        Map<String, Object> ndcg = (Map<String, Object>) result.get("ndcg@10");
        ndcg.forEach((stage, value) -> {
            Map<String, Object> byDepth = (Map<String, Object>) value;
            System.out.printf("  %-8s %.3f -> %.3f / %.3f / %.3f / %.3f%n", stage,
                    byDepth.get("0"), byDepth.get("10"), byDepth.get("20"), byDepth.get("50"), byDepth.get("100"));
        });
    }
}
