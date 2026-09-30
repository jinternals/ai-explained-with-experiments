package bm25;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * How much do BM25's settings matter? For each BEIR dataset we index title and text twice: with the
 * "standard" analyzer (OpenSearch's default) and the "english" analyzer (stop words removed, words
 * stemmed). Then, without re-indexing, we try a grid of k1 and b values and score every test query
 * with nDCG@10 and Recall@100.
 */
final class Benchmark {

    private static final List<String> DATASETS = List.of("nfcorpus", "scifact", "fiqa");
    private static final List<String> ANALYZERS = List.of("standard", "english");
    private static final double[] K1 = {0.0, 0.5, 0.9, 1.2, 1.5, 2.0, 3.0};
    private static final double[] B = {0.0, 0.25, 0.4, 0.6, 0.75, 1.0};
    private static final int DEPTH = 100;
    private static final int EXAMPLES = 3;

    private Benchmark() {}

    static void run() throws Exception {
        String version = Index.waitUntilReady();
        Map<String, Object> datasets = new LinkedHashMap<>();
        for (String name : DATASETS) {
            datasets.put(name, evaluate(name));
        }
        Json.writeFile(Path.of("/out/benchmark.json"), Json.object(
                "opensearch_version", version,
                "k1_values", K1,
                "b_values", B,
                "defaults", Json.object("k1", Example.DEFAULT_K1, "b", Example.DEFAULT_B),
                "datasets", datasets));
        System.out.println("\nWritten to out/benchmark.json");
    }

    private static Map<String, Object> evaluate(String name) throws Exception {
        Beir.Dataset data = Beir.load(name);
        List<String> queryIds = List.copyOf(data.queries().keySet());
        Map<String, Map<String, Integer>> qrels = data.qrels();
        System.out.printf("%n[%s] %,d documents, %d test queries%n", name, data.documents().size(), queryIds.size());

        Map<String, Object> byAnalyzer = new LinkedHashMap<>();
        Map<String, Map<String, Double>> defaultPerQuery = new LinkedHashMap<>();
        for (String analyzer : ANALYZERS) {
            long start = System.nanoTime();
            Index index = new Index(name + "-" + analyzer);
            index.create(analyzer, Example.DEFAULT_K1, Example.DEFAULT_B);
            index.add(data.documents());

            Map<String, Object> grid = new LinkedHashMap<>();
            String bestKey = null;
            double best = -1;
            for (double k1 : K1) {
                for (double b : B) {
                    index.setBm25(k1, b);
                    Map<String, List<String>> runs = new HashMap<>();
                    for (String q : queryIds) {
                        runs.put(q, Index.ids(index.search(data.queries().get(q), DEPTH)));
                    }
                    Map<String, Double> perQuery = new LinkedHashMap<>();
                    queryIds.forEach(q -> perQuery.put(q, Metrics.ndcgAtK(runs.get(q), qrels.get(q), 10)));
                    double ndcg = average(perQuery.values());
                    double recall = average(queryIds.stream().map(q -> Metrics.recallAtK(runs.get(q), qrels.get(q), DEPTH)).toList());
                    String key = key(k1, b);
                    grid.put(key, Json.object("ndcg@10", ndcg, "recall@100", recall));
                    if (ndcg > best) {
                        best = ndcg;
                        bestKey = key;
                    }
                    if (k1 == Example.DEFAULT_K1 && b == Example.DEFAULT_B) {
                        defaultPerQuery.put(analyzer, perQuery);
                    }
                }
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> defaults = (Map<String, Object>) grid.get(key(Example.DEFAULT_K1, Example.DEFAULT_B));
            @SuppressWarnings("unchecked")
            Map<String, Object> beirLike = (Map<String, Object>) grid.get(key(0.9, 0.4));
            @SuppressWarnings("unchecked")
            Map<String, Object> bestRow = (Map<String, Object>) grid.get(bestKey);
            byAnalyzer.put(analyzer, Json.object(
                    "default", defaults,
                    "k1_0.9_b_0.4", beirLike,
                    "best", Json.object("setting", bestKey, "ndcg@10", bestRow.get("ndcg@10"), "recall@100", bestRow.get("recall@100")),
                    "seconds", Math.round((System.nanoTime() - start) / 1e9),
                    "grid", grid));
            System.out.printf("  %-8s default %.3f | k1=0.9,b=0.4 %.3f | best %s %.3f%n",
                    analyzer, defaults.get("ndcg@10"), beirLike.get("ndcg@10"), bestKey, bestRow.get("ndcg@10"));
        }

        return Json.object(
                "documents", data.documents().size(),
                "queries", queryIds.size(),
                "document_words", lengthStats(data.documents()),
                "analyzers", byAnalyzer,
                "english_vs_standard_at_default", headToHead(defaultPerQuery.get("english"), defaultPerQuery.get("standard")),
                "stemming_examples", examples(data, queryIds, defaultPerQuery.get("standard"), defaultPerQuery.get("english")));
    }

    private static String key(double k1, double b) {
        return "k1=" + k1 + ",b=" + b;
    }

    /** Words per document (title + text, split on spaces): BM25's b decides how much length matters. */
    private static Map<String, Object> lengthStats(List<Document> documents) {
        List<Integer> lengths = documents.stream().map(d -> d.content().split("\\s+").length).sorted().toList();
        int n = lengths.size();
        return Json.object(
                "mean", Math.round(lengths.stream().mapToInt(Integer::intValue).average().orElse(0)),
                "median", lengths.get(n / 2),
                "p10", lengths.get(n / 10),
                "p90", lengths.get(n * 9 / 10),
                "max", lengths.get(n - 1));
    }

    private static List<Map<String, Object>> examples(
            Beir.Dataset data, List<String> queryIds, Map<String, Double> standard, Map<String, Double> english) {
        List<String> byGain = queryIds.stream()
                .sorted(Comparator.comparingDouble((String q) -> english.get(q) - standard.get(q)).reversed())
                .toList();
        List<Map<String, Object>> rows = new ArrayList<>();
        for (String q : byGain.subList(0, EXAMPLES)) {
            rows.add(Json.object("kind", "stemming helped", "query_id", q, "query", data.queries().get(q),
                    "standard", Example.round(standard.get(q), 4), "english", Example.round(english.get(q), 4)));
        }
        for (String q : byGain.subList(byGain.size() - EXAMPLES, byGain.size())) {
            rows.add(Json.object("kind", "stemming hurt", "query_id", q, "query", data.queries().get(q),
                    "standard", Example.round(standard.get(q), 4), "english", Example.round(english.get(q), 4)));
        }
        return rows;
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

    private static double average(Collection<Double> values) {
        return Example.round(values.stream().mapToDouble(Double::doubleValue).average().orElse(0), 4);
    }
}
