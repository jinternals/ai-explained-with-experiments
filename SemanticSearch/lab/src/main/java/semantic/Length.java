package semantic;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.onnx.PoolingMode;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * How much of each document the model reads.
 *
 * <p>The model counts text in word pieces (tokens). LangChain4j's all-MiniLM-L6-v2 ships a tokenizer
 * file that cuts every text at 128 of them, including the two special tokens [CLS] and [SEP]. So its
 * code for splitting texts longer than 510 tokens into parts never runs. This experiment:
 * <ol>
 *   <li>shows the cut: a long document's vector is the same as the vector of its first 126 tokens;</li>
 *   <li>measures how long the documents are;</li>
 *   <li>embeds every document again with the same model and a 256-token cut (what sentence-transformers
 *       and the MTEB leaderboard use), and with no cut (LangChain4j then splits and averages), and
 *       compares search quality.</li>
 * </ol>
 */
final class Length {

    private static final List<String> DATASETS = List.of("nfcorpus", "scifact", "fiqa");
    private static final Path MODEL_DIR = Path.of("/cache", "models");

    private Length() {}

    static void run() throws Exception {
        Path onnx = extract("/all-minilm-l6-v2.onnx", "all-minilm-l6-v2.onnx");
        Path tokenizerFile = extract("/all-minilm-l6-v2-tokenizer.json", "all-minilm-l6-v2-tokenizer.json");
        JsonNode original = Json.read(Files.readString(tokenizerFile));

        // The same model with three token limits: 128 (as shipped, to check we match it), 256, and none.
        Map<String, EmbeddingModel> models = new LinkedHashMap<>();
        for (String limit : List.of("128", "256", "none")) {
            ObjectNode config = original.deepCopy();
            // "none": a limit no document reaches. (Removing the truncation setting from the file still
            // left a 128 cut, so we raise it instead.) LangChain4j then splits texts over 510 pieces into
            // parts, embeds each, and averages them, weighted by length.
            int maxLength = limit.equals("none") ? 1_000_000 : Integer.parseInt(limit);
            ((ObjectNode) config.get("truncation")).put("max_length", maxLength);
            Path file = MODEL_DIR.resolve("tokenizer-" + limit + ".json");
            Files.writeString(file, Json.write(config));
            models.put(limit, new OnnxEmbeddingModel(onnx, file, PoolingMode.MEAN));
        }

        // For counting, with no cut at all.
        HuggingFaceTokenizer counter = HuggingFaceTokenizer.newInstance(tokenizerFile, Map.of("padding", "false", "truncation", "false"));

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("shipped_tokenizer_truncation", original.get("truncation"));
        Map<String, Object> datasets = new LinkedHashMap<>();
        for (String name : DATASETS) {
            System.out.println("\n=== " + name);
            Beir.Dataset data = Beir.load(name);
            List<String> texts = data.documents().stream().map(Document::content).toList();
            List<String> ids = data.documents().stream().map(Document::id).toList();
            List<String> queryIds = new ArrayList<>(data.queries().keySet());

            int[] lengths = texts.stream().mapToInt(t -> tokens(counter, t).length + 2).toArray(); // + [CLS], [SEP]
            int[] sorted = Arrays.stream(lengths).sorted().toArray();
            int longestQuery = queryIds.stream().mapToInt(id -> tokens(counter, data.queries().get(id)).length + 2).max().orElse(0);

            // 1. The cut, shown on the longest document.
            int longest = 0;
            for (int i = 1; i < lengths.length; i++) {
                if (lengths[i] > lengths[longest]) {
                    longest = i;
                }
            }
            String text = texts.get(longest);
            float[] shipped = Embeddings.embed(text);
            Map<String, Object> probe = Json.object(
                    "tokens", lengths[longest],
                    "shipped_vs_its_first_126_tokens", VectorMath.cosine(shipped, Embeddings.embed(firstTokens(counter, text, 126))),
                    "shipped_vs_limit_128_model", VectorMath.cosine(shipped, models.get("128").embed(text).content().vector()),
                    "shipped_vs_limit_256_model", VectorMath.cosine(shipped, models.get("256").embed(text).content().vector()),
                    "shipped_vs_no_limit_model", VectorMath.cosine(shipped, models.get("none").embed(text).content().vector()));
            System.out.println("Longest document: " + probe);

            // 3. Search quality with each limit. Questions are all shorter than 128 tokens, so their vectors don't change.
            float[][] queries = Embeddings.embed(queryIds.stream().map(data.queries()::get).toList(), name + "-queries");
            Map<String, Object> ndcg = new LinkedHashMap<>();
            ndcg.put("128", ndcg(Embeddings.embed(texts, name + "-docs"), queries, queryIds, ids, data));
            ndcg.put("256", ndcg(Embeddings.embed(models.get("256"), texts, name + "-docs-limit256"), queries, queryIds, ids, data));
            ndcg.put("none", ndcg(Embeddings.embed(models.get("none"), texts, name + "-docs-nolimit"), queries, queryIds, ids, data));
            System.out.println("nDCG@10 by token limit: " + ndcg);

            Map<String, Object> share = new LinkedHashMap<>();
            for (int limit : List.of(128, 256, 512)) {
                share.put(String.valueOf(limit), Arrays.stream(lengths).filter(n -> n > limit).count() / (double) lengths.length);
            }
            datasets.put(name, Json.object(
                    "documents", lengths.length,
                    "median_tokens", sorted[sorted.length / 2],
                    "p90_tokens", sorted[(int) (sorted.length * 0.9)],
                    "share_longer_than", share,
                    "longest_query_tokens", longestQuery,
                    "longest_document_probe", probe,
                    "ndcg@10", ndcg));
        }
        result.put("datasets", datasets);
        Json.writeFile(Path.of("/out", "length.json"), result);
        System.out.println("\nWrote /out/length.json");
    }

    /** Copy a file bundled in the LangChain4j jar to disk, so a tokenizer can be loaded next to it. */
    private static Path extract(String resource, String fileName) throws IOException {
        Files.createDirectories(MODEL_DIR);
        Path target = MODEL_DIR.resolve(fileName);
        if (!Files.exists(target)) {
            try (InputStream in = Length.class.getResourceAsStream(resource)) {
                Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        return target;
    }

    private static String[] tokens(HuggingFaceTokenizer tokenizer, String text) {
        return tokenizer.encode(text, false, false).getTokens();
    }

    /** The text's first n word pieces, turned back into text. */
    private static String firstTokens(HuggingFaceTokenizer tokenizer, String text, int n) {
        String[] tokens = tokens(tokenizer, text);
        return tokens.length <= n ? text : tokenizer.buildSentence(Arrays.asList(tokens).subList(0, n));
    }

    /** Exact search in memory (the vectors have length 1, so the dot product is the cosine), then nDCG@10. */
    private static double ndcg(float[][] docs, float[][] queries, List<String> queryIds, List<String> ids, Beir.Dataset data) {
        double sum = 0;
        for (int q = 0; q < queries.length; q++) {
            double[] scores = new double[docs.length];
            for (int d = 0; d < docs.length; d++) {
                scores[d] = VectorMath.dot(queries[q], docs[d]);
            }
            Integer[] order = new Integer[docs.length];
            Arrays.setAll(order, i -> i);
            List<String> top = Arrays.stream(order)
                    .sorted(Comparator.comparingDouble((Integer i) -> scores[i]).reversed())
                    .limit(10)
                    .map(ids::get)
                    .toList();
            sum += Metrics.ndcgAtK(top, data.qrels().get(queryIds.get(q)), 10);
        }
        return sum / queries.length;
    }
}
