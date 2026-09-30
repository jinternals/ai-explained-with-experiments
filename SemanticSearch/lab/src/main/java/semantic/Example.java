package semantic;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * The worked example, in four parts:
 * <ol>
 *   <li>cosine similarity on arrows with two numbers, small enough to check by hand;</li>
 *   <li>a real sentence turned into 384 numbers, and five sentences compared with a question;</li>
 *   <li>pairs of words and sentences that show what the numbers do and don't capture;</li>
 *   <li>the NFCorpus question from part 1, searched by keyword and by meaning.</li>
 * </ol>
 */
final class Example {

    private static final String DATASET = "nfcorpus";
    private static final String QUERY_ID = "PLAIN-2830"; // "Neurobiology of Artificial Sweeteners", as in part 1
    private static final int TOP_N = 5;
    private static final int DEPTH = 100;

    private static final String QUESTION = "How can I lower my blood pressure?";
    private static final List<String> SENTENCES = List.of(
            "Eating less salt helps reduce hypertension.",
            "Ways to bring down high blood pressure without medication.",
            "Blood pressure is measured in millimetres of mercury.",
            "Lower the tyre pressure before driving on sand.",
            "The stock market fell sharply today.");

    private static final List<List<String>> PAIRS = List.of(
            List.of("car", "automobile"),
            List.of("car", "truck"),
            List.of("car", "banana"),
            List.of("hot", "cold"),
            List.of("The dog bit the man.", "The man bit the dog."),
            List.of("I love this film.", "I do not love this film."),
            List.of("Order 4471 has shipped.", "Order 9012 has shipped."));

    private Example() {}

    static void run() throws Exception {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("arrows", arrows());
        result.put("longer_arrow", longerArrow());

        // ---- 2. One real sentence as numbers, then five sentences against a question
        float[] q = Embeddings.embed(QUESTION);
        List<Double> firstValues = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            firstValues.add(round(q[i], 4));
        }
        result.put("question", QUESTION);
        result.put("vector", Json.object(
                "dimensions", q.length,
                "first_values", firstValues,
                "length", round(VectorMath.length(q), 4)));

        List<Map<String, Object>> sentences = new ArrayList<>();
        for (String sentence : SENTENCES) {
            float[] s = Embeddings.embed(sentence);
            sentences.add(Json.object(
                    "text", sentence,
                    "shared_words", Words.shared(QUESTION, sentence),
                    "cosine", round(VectorMath.cosine(q, s), 3),
                    "dot", round(VectorMath.dot(q, s), 3),
                    "euclidean", round(VectorMath.euclidean(q, s), 3),
                    "degrees", round(VectorMath.degrees(VectorMath.cosine(q, s)), 0)));
        }
        result.put("sentences", sentences);

        // The same recipe with 384 numbers: the question against the first sentence, position by position.
        float[] first = Embeddings.embed(SENTENCES.get(0));
        List<Map<String, Object>> products = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            products.add(Json.object("question", round(q[i], 4), "sentence", round(first[i], 4), "product", round((double) q[i] * first[i], 5)));
        }
        result.put("recipe_384", Json.object(
                "sentence", SENTENCES.get(0),
                "first_products", products,
                "dot", round(VectorMath.dot(q, first), 4),
                "length_question", round(VectorMath.length(q), 4),
                "length_sentence", round(VectorMath.length(first), 4),
                "cosine", round(VectorMath.cosine(q, first), 4)));
        print("Sentences compared with \"" + QUESTION + "\"", sentences);

        // ---- Words, sentences and paragraphs
        result.put("levels", levels());

        // ---- 3. Pairs
        List<Map<String, Object>> pairs = new ArrayList<>();
        for (List<String> pair : PAIRS) {
            double cosine = VectorMath.cosine(Embeddings.embed(pair.get(0)), Embeddings.embed(pair.get(1)));
            pairs.add(Json.object("a", pair.get(0), "b", pair.get(1), "cosine", round(cosine, 3)));
        }
        result.put("pairs", pairs);
        print("Pairs", pairs);

        // ---- 4. Part 1's question, both ways
        result.put("nfcorpus", nfcorpus());

        Json.writeFile(Path.of("/out", "example.json"), result);
        System.out.println("\nWrote /out/example.json");
    }

    /** Arrows with two numbers each: a against three others, worked out step by step. */
    private static List<Map<String, Object>> arrows() {
        float[] a = {4, 3};
        Map<String, float[]> others = new LinkedHashMap<>();
        others.put("b", new float[] {3, 4});
        others.put("c", new float[] {-3, 4});
        others.put("d", new float[] {-4, -3});

        List<Map<String, Object>> rows = new ArrayList<>();
        for (var entry : others.entrySet()) {
            float[] other = entry.getValue();
            double cosine = VectorMath.cosine(a, other);
            rows.add(Json.object(
                    "name", entry.getKey(),
                    "vector", List.of(other[0], other[1]),
                    "dot", VectorMath.dot(a, other),
                    "length_a", VectorMath.length(a),
                    "length_other", VectorMath.length(other),
                    "cosine", round(cosine, 2),
                    "degrees", round(VectorMath.degrees(cosine), 0)));
        }
        print("Arrows compared with a = (4, 3)", rows);
        return rows;
    }

    private static final String PARAGRAPH_QUESTION = "How do I renew my parking permit?";
    private static final List<String> PARAGRAPH = List.of(
            "Our office moved to a new building on Park Street in March.",
            "The canteen on the ground floor now serves vegetarian meals every day.",
            "Parking permits must be renewed online before the end of June.");

    /**
     * The same model at three sizes of text: single words (one fixed arrow per word, whatever the
     * sentence), whole sentences (context decides which "bank"), and a paragraph (one arrow for
     * several topics, compared with an arrow per sentence).
     */
    private static Map<String, Object> levels() {
        List<List<String>> wordPairs = List.of(
                List.of("bank", "river"), List.of("bank", "money"), List.of("river", "money"));
        List<Map<String, Object>> words = new ArrayList<>();
        for (List<String> pair : wordPairs) {
            words.add(Json.object("a", pair.get(0), "b", pair.get(1),
                    "cosine", round(VectorMath.cosine(Embeddings.embed(pair.get(0)), Embeddings.embed(pair.get(1))), 3)));
        }

        String riverBank = "She sat on the bank of the river and watched the water.";
        String moneyBank = "He opened a savings account at the bank.";
        List<List<String>> sentencePairs = List.of(
                List.of(riverBank, moneyBank),
                List.of(riverBank, "They had a picnic beside the stream."),
                List.of(moneyBank, "She paid her salary into her current account."));
        List<Map<String, Object>> sentences = new ArrayList<>();
        for (List<String> pair : sentencePairs) {
            sentences.add(Json.object("a", pair.get(0), "b", pair.get(1),
                    "shared_words", Words.shared(pair.get(0), pair.get(1)),
                    "cosine", round(VectorMath.cosine(Embeddings.embed(pair.get(0)), Embeddings.embed(pair.get(1))), 3),
                    "cosine_of_word_averages", round(VectorMath.cosine(wordAverage(pair.get(0)), wordAverage(pair.get(1))), 3)));
        }

        float[] question = Embeddings.embed(PARAGRAPH_QUESTION);
        List<Map<String, Object>> parts = new ArrayList<>();
        for (String sentence : PARAGRAPH) {
            parts.add(Json.object("text", sentence, "cosine", round(VectorMath.cosine(question, Embeddings.embed(sentence)), 3)));
        }
        String paragraph = String.join(" ", PARAGRAPH);
        float[] paragraphVector = Embeddings.embed(paragraph);

        // Is a paragraph's arrow a blend of its sentences' arrows? Compare it with each sentence, and
        // with the plain average of the three sentence arrows.
        float[] average = new float[paragraphVector.length];
        List<Double> paragraphVsSentence = new ArrayList<>();
        for (String sentence : PARAGRAPH) {
            float[] v = Embeddings.embed(sentence);
            paragraphVsSentence.add(round(VectorMath.cosine(paragraphVector, v), 3));
            for (int i = 0; i < v.length; i++) {
                average[i] += v[i] / PARAGRAPH.size();
            }
        }

        // The word pieces the model actually reads, for a word, a long word and a sentence.
        List<Map<String, Object>> pieces = new ArrayList<>();
        for (String text : List.of("bank", "hypertension", riverBank)) {
            pieces.add(Json.object("text", text, "pieces", Tokens.pieces(text)));
        }

        Map<String, Object> result = Json.object(
                "pieces", pieces,
                "words", words,
                "sentences", sentences,
                "paragraph", Json.object(
                        "question", PARAGRAPH_QUESTION,
                        "whole", round(VectorMath.cosine(question, paragraphVector), 3),
                        "paragraph_vs_each_sentence", paragraphVsSentence,
                        "paragraph_vs_average_of_sentences", round(VectorMath.cosine(paragraphVector, average), 3),
                        "sentences", parts));
        System.out.println("\nWords, sentences, paragraph\n  " + result);
        return result;
    }

    /**
     * What a sentence's arrow would be if it were just its words' arrows averaged: embed every word on
     * its own, with no neighbours, and average. The model does not work this way; this is the comparison.
     */
    private static float[] wordAverage(String sentence) {
        String[] words = sentence.toLowerCase().replaceAll("[^a-z ]", "").trim().split("\\s+");
        float[] sum = null;
        for (String word : words) {
            float[] v = Embeddings.embed(word);
            if (sum == null) {
                sum = new float[v.length];
            }
            for (int i = 0; i < v.length; i++) {
                sum[i] += v[i] / words.length;
            }
        }
        return sum;
    }

    /**
     * e = (8, 6) points exactly the same way as a = (4, 3) but is twice as long. Against b = (3, 4),
     * its dot product doubles while its cosine stays the same: why cosine divides by the lengths.
     */
    private static Map<String, Object> longerArrow() {
        float[] a = {4, 3}, e = {8, 6}, b = {3, 4};
        return Json.object(
                "vector", List.of(e[0], e[1]),
                "dot_with_b", VectorMath.dot(e, b),
                "length", VectorMath.length(e),
                "cosine_with_b", round(VectorMath.cosine(e, b), 2),
                "cosine_with_a", round(VectorMath.cosine(e, a), 2),
                "dot_a_with_b", VectorMath.dot(a, b));
    }

    private static Map<String, Object> nfcorpus() throws Exception {
        String version = SearchIndex.waitUntilReady();
        Beir.Dataset data = Beir.load(DATASET);
        Map<String, Document> byId = data.documents().stream()
                .collect(Collectors.toMap(Document::id, Function.identity()));
        String query = data.queries().get(QUERY_ID);
        Map<String, Integer> labels = data.qrels().get(QUERY_ID);

        float[][] docVectors = Embeddings.embed(data.documents().stream().map(Document::content).toList(), DATASET + "-docs");
        Map<String, float[]> vectorById = new LinkedHashMap<>();
        for (int i = 0; i < data.documents().size(); i++) {
            vectorById.put(data.documents().get(i).id(), docVectors[i]);
        }
        SearchIndex index = new SearchIndex(DATASET);
        index.create(docVectors[0].length);
        index.add(data.documents(), docVectors);

        float[] queryVector = Embeddings.embed(query);
        List<SearchIndex.Hit> keywordHits = index.keyword(query, DEPTH);
        List<SearchIndex.Hit> meaningHits = index.exact(queryVector, DEPTH);
        List<String> keyword = SearchIndex.ids(keywordHits);
        List<String> meaning = SearchIndex.ids(meaningHits);

        Function<String, Map<String, Object>> row = id -> {
            Document doc = byId.get(id);
            return Json.object(
                    "id", id,
                    "title", doc.title(),
                    "relevance", labels.getOrDefault(id, 0),
                    "keyword_rank", rankOf(keyword, id),
                    "meaning_rank", rankOf(meaning, id),
                    "cosine", round(VectorMath.cosine(queryVector, vectorById.get(id)), 3),
                    "shared_words", Words.shared(query, doc.content()));
        };

        List<String> relevant = labels.entrySet().stream().filter(e -> e.getValue() > 0).map(Map.Entry::getKey).toList();
        Map<String, Object> out = Json.object(
                "opensearch_version", version,
                "query_id", QUERY_ID,
                "query", query,
                "documents", data.documents().size(),
                "keyword_top", keyword.subList(0, TOP_N).stream().map(row).toList(),
                "meaning_top", meaning.subList(0, TOP_N).stream().map(row).toList(),
                "relevant", relevant.stream().map(row).toList(),
                "ndcg@10", Json.object(
                        "keyword", round(Metrics.ndcgAtK(keyword, labels, 10), 3),
                        "meaning", round(Metrics.ndcgAtK(meaning, labels, 10), 3)));
        print("Keyword top " + TOP_N + " for \"" + query + "\"", (List<Map<String, Object>>) out.get("keyword_top"));
        print("Meaning top " + TOP_N, (List<Map<String, Object>>) out.get("meaning_top"));
        print("Relevant papers", (List<Map<String, Object>>) out.get("relevant"));
        return out;
    }

    /** 1-based position in the list, or null if it isn't there. */
    private static Integer rankOf(List<String> ids, String id) {
        int i = ids.indexOf(id);
        return i < 0 ? null : i + 1;
    }

    private static double round(double value, int digits) {
        double scale = Math.pow(10, digits);
        return Math.round(value * scale) / scale;
    }

    private static void print(String title, List<Map<String, Object>> rows) {
        System.out.println("\n" + title);
        rows.forEach(r -> System.out.println("  " + r));
    }
}
