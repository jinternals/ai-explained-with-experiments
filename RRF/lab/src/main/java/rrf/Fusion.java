package rrf;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Reciprocal Rank Fusion (Cormack, Clarke &amp; Büttcher, SIGIR 2009). */
public final class Fusion {

    public record Scored(String id, double score) {}

    private Fusion() {}

    /**
     * Merge ranked lists of doc ids. A doc earns 1 / (k + rank) from every list it appears in.
     *
     * @return docs with their RRF scores, best first (ties keep the order they were first seen in)
     */
    public static List<Scored> rrf(List<List<String>> rankedLists, int k) {
        Map<String, Double> scores = new LinkedHashMap<>();
        for (List<String> ranked : rankedLists) {
            for (int i = 0; i < ranked.size(); i++) {
                int rank = i + 1;
                scores.merge(ranked.get(i), 1.0 / (k + rank), Double::sum);
            }
        }
        return scores.entrySet().stream()
                .map(entry -> new Scored(entry.getKey(), entry.getValue()))
                .sorted(Comparator.comparingDouble(Scored::score).reversed())
                .toList();
    }
}
