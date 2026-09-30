package semantic;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Keyword search (BM25) against meaning search (all-MiniLM-L6-v2 vectors) on three BEIR datasets:
 * <ul>
 *   <li>quality: nDCG@10 and Recall@100, and question-by-question wins and losses;</li>
 *   <li>when meaning search helps: questions grouped by how many of their words the answers share;</li>
 *   <li>the vector database: HNSW (approximate) against scoring every document (exact);</li>
 *   <li>cosine, dot product and distance: do they ever rank differently on these vectors?</li>
 *   <li>speed: median time per question for each kind of search.</li>
 * </ul>
 */
final class Benchmark {

    private static final List<String> DATASETS = List.of("nfcorpus", "scifact", "fiqa");
    private static final int DEPTH = 100;
    private static final int TIMED_QUERIES = 200;
    private static final int EXAMPLES = 5;

    /** Word-overlap groups: share of the question's words found in its relevant documents. */
    private static final double[] GROUP_EDGES = {1.0 / 3, 2.0 / 3};
    private static final List<String> GROUP_NAMES = List.of("low", "middle", "high");

    private Benchmark() {}

    static void run() throws Exception {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("opensearch_version", SearchIndex.waitUntilReady());
        result.put("model", "sentence-transformers/all-MiniLM-L6-v2 (LangChain4j 1.20.0-beta30, ONNX)");
        result.put("group_edges", GROUP_EDGES);

        Map<String, Object> datasets = new LinkedHashMap<>();
        for (String name : DATASETS) {
            datasets.put(name, runDataset(name, result));
        }
        result.put("datasets", datasets);

        Json.writeFile(Path.of("/out", "benchmark.json"), result);
        System.out.println("\nWrote /out/benchmark.json");
    }

    private static Map<String, Object> runDataset(String name, Map<String, Object> result) throws Exception {
        System.out.println("\n=== " + name);
        Beir.Dataset data = Beir.load(name);
        List<Document> documents = data.documents();
        Map<String, Document> byId = documents.stream().collect(Collectors.toMap(Document::id, Function.identity()));
        List<String> queryIds = new ArrayList<>(data.queries().keySet());

        float[][] docVectors = Embeddings.embed(documents.stream().map(Document::content).toList(), name + "-docs");
        float[][] queryVectors = Embeddings.embed(queryIds.stream().map(data.queries()::get).toList(), name + "-queries");

        System.out.println("Indexing " + documents.size() + " documents...");
        SearchIndex index = new SearchIndex(name);
        index.create(docVectors[0].length);
        index.add(documents, docVectors);
        result.putIfAbsent("hnsw_mapping", Json.read(Json.write(index.hnswSettings())));

        // ---- Search every question three ways
        Map<String, double[]> ndcg = Map.of("keyword", new double[queryIds.size()], "exact", new double[queryIds.size()], "hnsw", new double[queryIds.size()]);
        Map<String, double[]> recall = Map.of("keyword", new double[queryIds.size()], "exact", new double[queryIds.size()], "hnsw", new double[queryIds.size()]);
        double[] hnswOverlap = new double[queryIds.size()];
        int hnswSameTop10 = 0;
        List<List<String>> keywordLists = new ArrayList<>();
        List<List<String>> exactLists = new ArrayList<>();

        for (int i = 0; i < queryIds.size(); i++) {
            String queryId = queryIds.get(i);
            Map<String, Integer> labels = data.qrels().get(queryId);
            List<String> keyword = SearchIndex.ids(index.keyword(data.queries().get(queryId), DEPTH));
            List<String> exact = SearchIndex.ids(index.exact(queryVectors[i], DEPTH));
            List<String> hnsw = SearchIndex.ids(index.hnsw(queryVectors[i], DEPTH));
            keywordLists.add(keyword);
            exactLists.add(exact);

            for (var entry : Map.of("keyword", keyword, "exact", exact, "hnsw", hnsw).entrySet()) {
                ndcg.get(entry.getKey())[i] = Metrics.ndcgAtK(entry.getValue(), labels, 10);
                recall.get(entry.getKey())[i] = Metrics.recallAtK(entry.getValue(), labels, DEPTH);
            }
            Set<String> exactTop10 = new HashSet<>(top(exact, 10));
            hnswOverlap[i] = top(hnsw, 10).stream().filter(exactTop10::contains).count() / 10.0;
            if (top(hnsw, 10).equals(top(exact, 10))) {
                hnswSameTop10++;
            }
        }

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("documents", documents.size());
        out.put("queries", queryIds.size());
        for (String method : List.of("keyword", "exact", "hnsw")) {
            out.put(method, Json.object(
                    "ndcg@10", mean(ndcg.get(method)),
                    "recall@100", mean(recall.get(method))));
        }
        out.put("hnsw_vs_exact", Json.object(
                "top10_overlap", mean(hnswOverlap),
                "same_top10", hnswSameTop10,
                "same_top10_share", (double) hnswSameTop10 / queryIds.size()));
        System.out.printf("nDCG@10  keyword %.3f  meaning exact %.3f  meaning HNSW %.3f%n",
                mean(ndcg.get("keyword")), mean(ndcg.get("exact")), mean(ndcg.get("hnsw")));

        // ---- Question by question: meaning (exact) against keyword
        int better = 0, same = 0, worse = 0;
        double[] overlap = new double[queryIds.size()];
        for (int i = 0; i < queryIds.size(); i++) {
            double diff = ndcg.get("exact")[i] - ndcg.get("keyword")[i];
            if (Math.abs(diff) < 1e-9) {
                same++;
            } else if (diff > 0) {
                better++;
            } else {
                worse++;
            }
            overlap[i] = wordOverlap(data.queries().get(queryIds.get(i)), data.qrels().get(queryIds.get(i)), byId);
        }
        out.put("meaning_vs_keyword", Json.object("better", better, "same", same, "worse", worse));
        out.put("mean_word_overlap", mean(overlap));

        // ---- Grouped by word overlap
        Map<String, Object> groups = new LinkedHashMap<>();
        for (int g = 0; g < GROUP_NAMES.size(); g++) {
            List<Integer> members = new ArrayList<>();
            for (int i = 0; i < queryIds.size(); i++) {
                if (group(overlap[i]) == g) {
                    members.add(i);
                }
            }
            int wins = (int) members.stream().filter(i -> ndcg.get("exact")[i] > ndcg.get("keyword")[i] + 1e-9).count();
            int losses = (int) members.stream().filter(i -> ndcg.get("exact")[i] < ndcg.get("keyword")[i] - 1e-9).count();
            groups.put(GROUP_NAMES.get(g), Json.object(
                    "queries", members.size(),
                    "keyword", meanOf(ndcg.get("keyword"), members),
                    "meaning", meanOf(ndcg.get("exact"), members),
                    "meaning_better", wins,
                    "meaning_worse", losses));
        }
        out.put("by_word_overlap", groups);

        // ---- The biggest wins and losses, with what each search put first
        Integer[] order = new Integer[queryIds.size()];
        Arrays.setAll(order, i -> i);
        Comparator<Integer> byGain = Comparator.comparingDouble(i -> ndcg.get("exact")[i] - ndcg.get("keyword")[i]);
        List<Integer> sorted = Arrays.stream(order).sorted(byGain).toList();
        out.put("biggest_wins", examples(sorted.reversed().subList(0, EXAMPLES), queryIds, data, byId, ndcg, overlap, keywordLists, exactLists));
        out.put("biggest_losses", examples(sorted.subList(0, EXAMPLES), queryIds, data, byId, ndcg, overlap, keywordLists, exactLists));

        // ---- Cosine, dot product and distance, computed here over every document
        out.put("metrics_agree", compareMetrics(docVectors, queryVectors));

        // ---- Speed
        out.put("timing_ms", timing(index, queryIds.subList(0, Math.min(TIMED_QUERIES, queryIds.size())), data));
        System.out.println("Timing (median ms): " + out.get("timing_ms"));
        return out;
    }

    /** Average, over a question's relevant documents, of the share of its words each one contains. */
    private static double wordOverlap(String query, Map<String, Integer> labels, Map<String, Document> byId) {
        double[] shares = labels.entrySet().stream()
                .filter(e -> e.getValue() > 0 && byId.containsKey(e.getKey()))
                .mapToDouble(e -> Words.overlap(query, byId.get(e.getKey()).content()))
                .toArray();
        return shares.length == 0 ? 0 : Arrays.stream(shares).average().orElse(0);
    }

    private static int group(double overlap) {
        int g = 0;
        while (g < GROUP_EDGES.length && overlap >= GROUP_EDGES[g]) {
            g++;
        }
        return g;
    }

    private static List<Map<String, Object>> examples(List<Integer> picks, List<String> queryIds, Beir.Dataset data,
            Map<String, Document> byId, Map<String, double[]> ndcg, double[] overlap,
            List<List<String>> keywordLists, List<List<String>> exactLists) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i : picks) {
            String queryId = queryIds.get(i);
            String query = data.queries().get(queryId);
            Map<String, Integer> labels = data.qrels().get(queryId);
            List<Map<String, Object>> relevant = new ArrayList<>();
            labels.entrySet().stream()
                    .filter(e -> e.getValue() > 0 && byId.containsKey(e.getKey()))
                    .sorted(Comparator.comparingInt(e -> rank(exactLists.get(i), e.getKey())))
                    .limit(3)
                    .forEach(e -> {
                        Document doc = byId.get(e.getKey());
                        relevant.add(Json.object(
                                "id", doc.id(),
                                "title", doc.title(),
                                "start", start(doc.text(), 160),
                                "keyword_rank", rankOrNull(keywordLists.get(i), doc.id()),
                                "meaning_rank", rankOrNull(exactLists.get(i), doc.id()),
                                "shared_words", Words.shared(query, doc.content())));
                    });
            Document keywordFirst = byId.get(keywordLists.get(i).get(0));
            Document meaningFirst = byId.get(exactLists.get(i).get(0));
            rows.add(Json.object(
                    "query_id", queryId,
                    "query", query,
                    "keyword", ndcg.get("keyword")[i],
                    "meaning", ndcg.get("exact")[i],
                    "word_overlap", overlap[i],
                    "relevant", relevant,
                    "keyword_first", Json.object("title", keywordFirst.title(), "start", start(keywordFirst.text(), 160), "relevance", labels.getOrDefault(keywordFirst.id(), 0)),
                    "meaning_first", Json.object("title", meaningFirst.title(), "start", start(meaningFirst.text(), 160), "relevance", labels.getOrDefault(meaningFirst.id(), 0))));
        }
        return rows;
    }

    /**
     * Rank every document by cosine, by dot product and by distance (smallest first), and count the
     * questions whose top 10 comes out the same all three ways. Also report how long the vectors are.
     */
    private static Map<String, Object> compareMetrics(float[][] docs, float[][] queries) {
        double minLength = Double.MAX_VALUE, maxLength = 0;
        for (float[] doc : docs) {
            double length = VectorMath.length(doc);
            minLength = Math.min(minLength, length);
            maxLength = Math.max(maxLength, length);
        }
        int sameTop10 = 0;
        for (float[] q : queries) {
            double[] cosine = new double[docs.length], dot = new double[docs.length], distance = new double[docs.length];
            for (int d = 0; d < docs.length; d++) {
                cosine[d] = VectorMath.cosine(q, docs[d]);
                dot[d] = VectorMath.dot(q, docs[d]);
                distance[d] = -VectorMath.euclidean(q, docs[d]); // smaller distance = better
            }
            List<Integer> a = topIndexes(cosine, 10), b = topIndexes(dot, 10), c = topIndexes(distance, 10);
            if (a.equals(b) && b.equals(c)) {
                sameTop10++;
            }
        }
        return Json.object(
                "queries", queries.length,
                "same_top10", sameTop10,
                "vector_length_min", minLength,
                "vector_length_max", maxLength);
    }

    private static Map<String, Object> timing(SearchIndex index, List<String> queryIds, Beir.Dataset data) throws Exception {
        List<Double> embed = new ArrayList<>(), keyword = new ArrayList<>(), hnsw = new ArrayList<>(), exact = new ArrayList<>();
        Embeddings.embed("warm-up");
        for (int round = 0; round < 2; round++) { // the first round warms caches and is thrown away
            embed.clear();
            keyword.clear();
            hnsw.clear();
            exact.clear();
            for (String queryId : queryIds) {
                String text = data.queries().get(queryId);
                long t0 = System.nanoTime();
                float[] vector = Embeddings.embed(text);
                long t1 = System.nanoTime();
                index.keyword(text, 10);
                long t2 = System.nanoTime();
                index.hnsw(vector, 10);
                long t3 = System.nanoTime();
                index.exact(vector, 10);
                long t4 = System.nanoTime();
                embed.add((t1 - t0) / 1e6);
                keyword.add((t2 - t1) / 1e6);
                hnsw.add((t3 - t2) / 1e6);
                exact.add((t4 - t3) / 1e6);
            }
        }
        return Json.object(
                "queries", queryIds.size(),
                "embed_question", median(embed),
                "keyword", median(keyword),
                "hnsw", median(hnsw),
                "exact", median(exact));
    }

    // ------------------------------------------------------------ small helpers

    private static List<Integer> topIndexes(double[] scores, int k) {
        Integer[] idx = new Integer[scores.length];
        Arrays.setAll(idx, i -> i);
        return Arrays.stream(idx).sorted((x, y) -> Double.compare(scores[y], scores[x])).limit(k).toList();
    }

    private static List<String> top(List<String> ids, int k) {
        return ids.subList(0, Math.min(k, ids.size()));
    }

    private static int rank(List<String> ids, String id) {
        int i = ids.indexOf(id);
        return i < 0 ? Integer.MAX_VALUE : i;
    }

    private static Integer rankOrNull(List<String> ids, String id) {
        int i = ids.indexOf(id);
        return i < 0 ? null : i + 1;
    }

    private static String start(String text, int chars) {
        if (text.length() <= chars) {
            return text;
        }
        int cut = text.lastIndexOf(' ', chars);
        return text.substring(0, cut > 0 ? cut : chars) + " …";
    }

    private static double mean(double[] values) {
        return Arrays.stream(values).average().orElse(0);
    }

    private static double meanOf(double[] values, List<Integer> members) {
        return members.stream().mapToDouble(i -> values[i]).average().orElse(0);
    }

    private static double median(List<Double> values) {
        double[] sorted = values.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        int n = sorted.length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    }
}
