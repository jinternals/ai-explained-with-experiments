package semantic;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** Ranking quality metrics, following the trec_eval conventions that BEIR uses. */
public final class Metrics {

    private Metrics() {}

    /**
     * nDCG@k: how good the top k is, from 0 to 1, rewarding relevant docs near the top.
     *
     * @param relevance doc id to graded label (0 or missing = not relevant)
     */
    public static double ndcgAtK(List<String> ranked, Map<String, Integer> relevance, int k) {
        double dcg = 0;
        for (int i = 0; i < Math.min(k, ranked.size()); i++) {
            dcg += relevance.getOrDefault(ranked.get(i), 0) / log2(i + 2);
        }

        List<Integer> bestLabels = relevance.values().stream()
                .filter(label -> label > 0)
                .sorted(Comparator.reverseOrder())
                .limit(k)
                .toList();
        double idealDcg = 0;
        for (int i = 0; i < bestLabels.size(); i++) {
            idealDcg += bestLabels.get(i) / log2(i + 2);
        }
        return idealDcg == 0 ? 0 : dcg / idealDcg;
    }

    /** Share of all relevant docs that appear anywhere in the top k. */
    public static double recallAtK(List<String> ranked, Map<String, Integer> relevance, int k) {
        Set<String> relevant = relevance.entrySet().stream()
                .filter(entry -> entry.getValue() > 0)
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
        if (relevant.isEmpty()) {
            return 0;
        }
        long found = ranked.stream().limit(k).filter(relevant::contains).count();
        return (double) found / relevant.size();
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2);
    }
}
