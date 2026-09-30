package bm25;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The worked example: why did keyword search give "The potential toxicity of artificial sweeteners"
 * a score of 28.64 for the NFCorpus question "Neurobiology of Artificial Sweeteners"?
 *
 * <p>OpenSearch's _explain API returns the exact arithmetic behind a score. We flatten it into one row
 * per (field, word), and add a few numbers worked out from the same formula to show what k1 and b do.
 */
final class Example {

    private static final String DATASET = "nfcorpus";
    private static final String QUERY_ID = "PLAIN-2830";
    static final double DEFAULT_K1 = 1.2; // OpenSearch's defaults
    static final double DEFAULT_B = 0.75;

    private Example() {}

    static void run() throws Exception {
        Index.waitUntilReady();
        Beir.Dataset data = Beir.load(DATASET);
        String query = data.queries().get(QUERY_ID);

        Index index = new Index("bm25-example");
        index.create("standard", DEFAULT_K1, DEFAULT_B);
        index.add(data.documents());

        List<Index.Hit> hits = index.search(query, 5);
        Index.Hit top = hits.get(0);
        JsonNode explanation = index.explain(query, top.id());
        Document doc = data.documents().stream().filter(d -> d.id().equals(top.id())).findFirst().orElseThrow();

        List<Map<String, Object>> terms = new ArrayList<>();
        collectTerms(explanation.path("explanation"), terms);
        Map<String, Object> fieldTotals = new LinkedHashMap<>();
        for (JsonNode field : explanation.path("explanation").path("details")) {
            // multi_match gives one child per field; each child's terms all mention that field.
            String name = field.toString().contains("title:") ? "title" : "text";
            fieldTotals.put(name, round(field.path("value").asDouble(), 4));
        }

        Map<String, Object> result = Json.object(
                "dataset", DATASET,
                "query_id", QUERY_ID,
                "query", query,
                "k1", DEFAULT_K1,
                "b", DEFAULT_B,
                "analyzed_query", Json.object(
                        "standard", index.analyze("standard", query),
                        "english", index.analyze("english", query)),
                "analyzed_title", Json.object(
                        "standard", index.analyze("standard", doc.title()),
                        "english", index.analyze("english", doc.title())),
                "top_hits", hits.stream().map(h -> Json.object(
                        "id", h.id(),
                        "score", round(h.score(), 4),
                        "title", title(data, h.id()))).toList(),
                "document", Json.object("id", doc.id(), "title", doc.title(), "text_words", doc.text().split("\\s+").length),
                "score", round(explanation.path("explanation").path("value").asDouble(), 4),
                "combination", explanation.path("explanation").path("description").asText(),
                "field_totals", fieldTotals,
                "terms", terms,
                "saturation", saturation(),
                "length", lengthEffect(),
                "raw_explanation", explanation.path("explanation"));
        Json.writeFile(Path.of("/out/example.json"), result);

        System.out.printf("%nQuery: %s%nTop result: %s (score %.2f)%n%n", query, doc.title(), top.score());
        System.out.printf("%-6s %-12s %5s %6s %8s %6s %6s %7s %7s%n", "field", "word", "freq", "len", "avg len", "n", "N", "idf", "score");
        for (Map<String, Object> t : terms) {
            System.out.printf("%-6s %-12s %5s %6s %8s %6s %6s %7s %7s%n",
                    t.get("field"), t.get("term"), t.get("freq"), t.get("dl"), t.get("avgdl"), t.get("n"), t.get("N"), t.get("idf"), t.get("score"));
        }
        System.out.printf("%nField totals: %s. Final score: %.4f%n", fieldTotals, top.score());
        System.out.println("Written to out/example.json");
    }

    /** Walk the explanation tree and pull out one row per weight(field:term) node. */
    private static void collectTerms(JsonNode node, List<Map<String, Object>> out) {
        String description = node.path("description").asText();
        if (description.startsWith("weight(")) {
            String inside = description.substring("weight(".length(), description.indexOf(' '));
            String[] fieldTerm = inside.split(":", 2);
            Map<String, Object> row = Json.object("field", fieldTerm[0], "term", fieldTerm[1], "score", round(node.path("value").asDouble(), 4));
            readValues(node, row);
            out.add(row);
            return;
        }
        for (JsonNode child : node.path("details")) {
            collectTerms(child, out);
        }
    }

    private static void readValues(JsonNode node, Map<String, Object> row) {
        String d = node.path("description").asText();
        double v = node.path("value").asDouble();
        if (d.startsWith("boost")) row.put("boost", round(v, 4));
        else if (d.startsWith("idf")) row.put("idf", round(v, 4));
        else if (d.startsWith("tf")) row.put("tf", round(v, 4));
        else if (d.startsWith("freq")) row.put("freq", round(v, 1));
        else if (d.startsWith("n,")) row.put("n", (long) v);
        else if (d.startsWith("N,")) row.put("N", (long) v);
        else if (d.startsWith("dl,")) row.put("dl", round(v, 1));
        else if (d.startsWith("avgdl")) row.put("avgdl", round(v, 1));
        else if (d.startsWith("k1")) row.put("k1", v);
        else if (d.startsWith("b,")) row.put("b", v);
        for (JsonNode child : node.path("details")) {
            readValues(child, row);
        }
    }

    /**
     * Worked out from the formula (not measured): how much the word-count part of BM25 grows as a word
     * appears 1, 2, 3 ... times, for a document of average length. Shown relative to one appearance.
     */
    private static Map<String, Object> saturation() {
        Map<String, Object> byK1 = new LinkedHashMap<>();
        for (double k1 : new double[] {0.5, 1.2, 2.0, 5.0}) {
            List<Double> values = new ArrayList<>();
            for (int freq = 1; freq <= 10; freq++) {
                values.add(round(tf(freq, k1, DEFAULT_B, 1.0) / tf(1, k1, DEFAULT_B, 1.0), 3));
            }
            byK1.put(String.valueOf(k1), values);
        }
        return byK1;
    }

    /** Worked out from the formula: one appearance of a word, in documents of different lengths. */
    private static Map<String, Object> lengthEffect() {
        Map<String, Object> byB = new LinkedHashMap<>();
        for (double b : new double[] {0.0, 0.4, 0.75, 1.0}) {
            Map<String, Object> byLength = new LinkedHashMap<>();
            for (double ratio : new double[] {0.5, 1.0, 2.0, 4.0}) {
                byLength.put(String.valueOf(ratio), round(tf(1, DEFAULT_K1, b, ratio) / tf(1, DEFAULT_K1, b, 1.0), 3));
            }
            byB.put(String.valueOf(b), byLength);
        }
        return byB;
    }

    /** Lucene's BM25 word-count part: freq / (freq + k1 * (1 - b + b * length / average length)). */
    static double tf(double freq, double k1, double b, double lengthRatio) {
        return freq / (freq + k1 * (1 - b + b * lengthRatio));
    }

    private static String title(Beir.Dataset data, String id) {
        return data.documents().stream().filter(d -> d.id().equals(id)).map(Document::title).findFirst().orElse("");
    }

    static double round(double value, int digits) {
        double scale = Math.pow(10, digits);
        return Math.round(value * scale) / scale;
    }
}
