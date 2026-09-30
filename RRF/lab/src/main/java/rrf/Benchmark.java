package rrf;

import java.nio.file.Path;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Does RRF actually help? Measure it on BEIR datasets that come with expert relevance judgments.
 *
 * <p>For every test query we run keyword search (BM25), meaning search (MiniLM + k-NN) and RRF of the two,
 * then score each ranking with nDCG@10 (quality of the top 10) and Recall@100 (how much it found).
 *
 * <p>We also check how the RRF constant k and the depth (results per searcher) change quality,
 * and that OpenSearch's built-in RRF agrees with {@link Fusion#rrf}.
 */
final class Benchmark {

    private static final List<String> DATASETS = List.of("nfcorpus", "scifact", "fiqa");
    private static final int DEPTH = 100; // results each searcher contributes
    private static final int K = 60;      // RRF constant
    private static final List<Integer> K_SWEEP = List.of(0, 1, 10, 30, 60, 100, 500);
    private static final List<Integer> DEPTH_SWEEP = List.of(5, 10, 20, 50, 100);
    private static final int EXAMPLES_PER_KIND = 5;

    private Benchmark() {}

    static void run() throws Exception {
        String version = SearchIndex.waitUntilReady();

        Map<String, Object> datasets = new LinkedHashMap<>();
        for (String name : DATASETS) {
            Map<String, Object> result = evaluate(name);
            printSummary(name, result);
            datasets.put(name, result);
        }

        Json.writeFile(Path.of("/out/benchmark.json"),
                Json.object("opensearch_version", version, "depth", DEPTH, "k", K, "datasets", datasets));
    }

    private static Map<String, Object> evaluate(String name) throws Exception {
        Beir.Dataset data = Beir.load(name);
        List<String> queryIds = List.copyOf(data.queries().keySet());
        Map<String, Map<String, Integer>> qrels = data.qrels();
        System.out.printf("%n[%s] %,d documents, %d test queries%n", name, data.documents().size(), queryIds.size());

        float[][] docVectors = Embeddings.embed(data.documents().stream().map(Document::content).toList(), name + "-docs");
        float[][] queryVectors = Embeddings.embed(queryIds.stream().map(data.queries()::get).toList(), name + "-queries");

        SearchIndex index = new SearchIndex(name);
        index.create(docVectors[0].length);
        index.add(data.documents(), docVectors);

        // Run every query through both searchers, and through OpenSearch's own RRF.
        Map<String, List<String>> keyword = new HashMap<>();
        Map<String, List<String>> meaning = new HashMap<>();
        Map<String, List<SearchIndex.Hit>> engineHits = new HashMap<>();
        for (int i = 0; i < queryIds.size(); i++) {
            String queryId = queryIds.get(i);
            String text = data.queries().get(queryId);
            keyword.put(queryId, SearchIndex.ids(index.keyword(text, DEPTH)));
            meaning.put(queryId, SearchIndex.ids(index.vector(queryVectors[i], DEPTH)));
            engineHits.put(queryId, index.hybridRrf(text, queryVectors[i], DEPTH, K, DEPTH));
        }

        Map<String, List<Fusion.Scored>> fusedWithScores =
                byQuery(queryIds, q -> Fusion.rrf(List.of(keyword.get(q), meaning.get(q)), K));

        Map<String, Map<String, List<String>>> runs = new LinkedHashMap<>();
        runs.put("keyword", keyword);
        runs.put("meaning", meaning);
        runs.put("rrf", byQuery(queryIds, q -> fusedWithScores.get(q).stream().map(Fusion.Scored::id).toList()));
        runs.put("rrf_opensearch", byQuery(queryIds, q -> SearchIndex.ids(engineHits.get(q))));

        // Score every ranking.
        Map<String, Map<String, Double>> ndcgPerQuery = new LinkedHashMap<>();
        Map<String, Object> ndcg = new LinkedHashMap<>();
        Map<String, Object> recall = new LinkedHashMap<>();
        runs.forEach((run, ranking) -> {
            ndcgPerQuery.put(run, byQuery(queryIds, q -> Metrics.ndcgAtK(ranking.get(q), qrels.get(q), 10)));
            ndcg.put(run, average(ndcgPerQuery.get(run).values()));
            recall.put(run, average(queryIds.stream().map(q -> Metrics.recallAtK(ranking.get(q), qrels.get(q), 100)).toList()));
        });

        Map<String, Object> kSweep = new LinkedHashMap<>();
        for (int k : K_SWEEP) {
            kSweep.put(String.valueOf(k), average(queryIds.stream()
                    .map(q -> Metrics.ndcgAtK(fuse(keyword.get(q), meaning.get(q), k, DEPTH), qrels.get(q), 10))
                    .toList()));
        }

        Map<String, Object> depthSweep = new LinkedHashMap<>();
        for (int depth : DEPTH_SWEEP) {
            depthSweep.put(String.valueOf(depth), average(queryIds.stream()
                    .map(q -> Metrics.ndcgAtK(fuse(keyword.get(q), meaning.get(q), K, depth), qrels.get(q), 10))
                    .toList()));
        }

        Map<String, List<String>> fused = runs.get("rrf");
        Map<String, List<String>> engine = runs.get("rrf_opensearch");
        Map<String, Object> agreement = Json.object(
                "identical_order", share(queryIds, q -> top(fused.get(q), 10).equals(top(engine.get(q), 10))),
                "identical_up_to_ties", share(queryIds, q -> sameUpToTies(top(fusedWithScores.get(q), 10), top(engineHits.get(q), 10))));

        return Json.object(
                "documents", data.documents().size(),
                "queries", queryIds.size(),
                "ndcg@10", ndcg,
                "recall@100", recall,
                "k_sweep", kSweep,
                "depth_sweep", depthSweep,
                "rrf_vs_keyword", headToHead(ndcgPerQuery.get("rrf"), ndcgPerQuery.get("keyword")),
                "rrf_vs_meaning", headToHead(ndcgPerQuery.get("rrf"), ndcgPerQuery.get("meaning")),
                "opensearch_top10_agreement", agreement,
                "examples", pickExamples(data, runs, ndcgPerQuery));
    }

    private static List<String> fuse(List<String> keywordIds, List<String> meaningIds, int k, int depth) {
        return Fusion.rrf(List.of(top(keywordIds, depth), top(meaningIds, depth)), k).stream()
                .map(Fusion.Scored::id)
                .toList();
    }

    /**
     * True if two top-k rankings differ only in the order of docs with equal scores.
     * Docs tied at the very last score may differ too: the cut-off picks arbitrarily among them.
     */
    private static boolean sameUpToTies(List<Fusion.Scored> mine, List<SearchIndex.Hit> theirs) {
        double tolerance = 1e-6;
        if (mine.size() != theirs.size()) {
            return false;
        }
        for (int i = 0; i < mine.size(); i++) {
            if (Math.abs(mine.get(i).score() - theirs.get(i).score()) > tolerance) {
                return false;
            }
        }
        if (mine.isEmpty()) {
            return true;
        }

        double cutoff = mine.getLast().score();
        Map<String, Double> mineAbove = mine.stream()
                .filter(doc -> doc.score() > cutoff + tolerance)
                .collect(Collectors.toMap(Fusion.Scored::id, Fusion.Scored::score));
        Map<String, Double> theirsAbove = theirs.stream()
                .filter(hit -> hit.score() > cutoff + tolerance)
                .collect(Collectors.toMap(SearchIndex.Hit::id, SearchIndex.Hit::score));
        return mineAbove.keySet().equals(theirsAbove.keySet())
                && mineAbove.keySet().stream().allMatch(id -> Math.abs(mineAbove.get(id) - theirsAbove.get(id)) <= tolerance);
    }

    /** Count queries where ranking A scores better than, the same as, or worse than ranking B. */
    private static Map<String, Object> headToHead(Map<String, Double> scoresA, Map<String, Double> scoresB) {
        double tolerance = 1e-9;
        int better = 0;
        int worse = 0;
        for (String queryId : scoresA.keySet()) {
            double a = scoresA.get(queryId);
            double b = scoresB.get(queryId);
            if (a > b + tolerance) {
                better++;
            } else if (a < b - tolerance) {
                worse++;
            }
        }
        return Json.object("better", better, "same", scoresA.size() - better - worse, "worse", worse);
    }

    /** Real queries where RRF beat both searchers, and where it lost to the better one. */
    private static Map<String, Object> pickExamples(
            Beir.Dataset data, Map<String, Map<String, List<String>>> runs, Map<String, Map<String, Double>> ndcg) {
        Map<String, String> snippets = data.documents().stream().collect(Collectors.toMap(Document::id, doc -> {
            String snippet = doc.title().isBlank() ? doc.text() : doc.title();
            return snippet.substring(0, Math.min(160, snippet.length()));
        }));

        Function<String, Double> gap = q ->
                ndcg.get("rrf").get(q) - Math.max(ndcg.get("keyword").get(q), ndcg.get("meaning").get(q));
        List<String> byGap = data.queries().keySet().stream().sorted(Comparator.comparing(gap)).toList();

        Function<String, Map<String, Object>> describe = q -> {
            Function<String, List<Map<String, Object>>> top10 = run -> top(runs.get(run).get(q), 10).stream()
                    .map(id -> Json.object("id", id, "snippet", snippets.get(id), "relevance", data.qrels().get(q).getOrDefault(id, 0)))
                    .toList();
            return Json.object(
                    "query_id", q,
                    "query", data.queries().get(q),
                    "ndcg@10", Json.object(
                            "keyword", round(ndcg.get("keyword").get(q)),
                            "meaning", round(ndcg.get("meaning").get(q)),
                            "rrf", round(ndcg.get("rrf").get(q))),
                    "keyword", top10.apply("keyword"),
                    "meaning", top10.apply("meaning"),
                    "rrf", top10.apply("rrf"));
        };

        return Json.object(
                "helped", byGap.reversed().stream().filter(q -> gap.apply(q) > 0).limit(EXAMPLES_PER_KIND).map(describe).toList(),
                "hurt", byGap.stream().filter(q -> gap.apply(q) < 0).limit(EXAMPLES_PER_KIND).map(describe).toList());
    }

    // ------------------------------------------------------------ small helpers

    private static <T> Map<String, T> byQuery(List<String> queryIds, Function<String, T> compute) {
        Map<String, T> result = new LinkedHashMap<>();
        for (String queryId : queryIds) {
            result.put(queryId, compute.apply(queryId));
        }
        return result;
    }

    private static <T> List<T> top(List<T> list, int n) {
        return list.subList(0, Math.min(n, list.size()));
    }

    private static double share(List<String> queryIds, Function<String, Boolean> test) {
        return average(queryIds.stream().map(q -> test.apply(q) ? 1.0 : 0.0).toList());
    }

    private static double average(Collection<Double> values) {
        return round(values.stream().mapToDouble(Double::doubleValue).average().orElse(0));
    }

    private static double round(double value) {
        return Math.round(value * 10_000) / 10_000.0;
    }

    @SuppressWarnings("unchecked")
    private static void printSummary(String name, Map<String, Object> result) {
        System.out.printf("[%s] nDCG@10    %s%n", name, result.get("ndcg@10"));
        System.out.printf("[%s] Recall@100 %s%n", name, result.get("recall@100"));
        System.out.printf("[%s] k sweep    %s%n", name, result.get("k_sweep"));
        System.out.printf("[%s] depth      %s%n", name, result.get("depth_sweep"));
        System.out.printf("[%s] RRF vs keyword %s | vs meaning %s%n", name, result.get("rrf_vs_keyword"), result.get("rrf_vs_meaning"));
        Map<String, Object> agreement = (Map<String, Object>) result.get("opensearch_top10_agreement");
        System.out.printf("[%s] top 10 vs OpenSearch: identical %.1f%%, identical up to ties %.1f%%%n",
                name, 100 * (double) agreement.get("identical_order"), 100 * (double) agreement.get("identical_up_to_ties"));
    }
}
