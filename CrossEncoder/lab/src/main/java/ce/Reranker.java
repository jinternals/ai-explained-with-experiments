package ce;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.scoring.onnx.OnnxScoringModel;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A cross-encoder: reads the query and one document together and outputs one relevance score.
 *
 * <p>Model: cross-encoder/ms-marco-MiniLM-L6-v2 (trained on MS MARCO passage ranking), run in-process
 * with ONNX Runtime through LangChain4j. Scores are raw logits: higher means more relevant, and only
 * the order matters.
 */
public final class Reranker implements AutoCloseable {

    public static final String MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2";
    public static final String REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"; // pinned, so results can be reproduced
    private static final Path MODEL_DIR = Path.of("/cache", "models", "ms-marco-MiniLM-L6-v2");
    private static final Path SCORE_DIR = Path.of("/cache", "scores");
    private static final int BATCH_SIZE = 32; // pairs per ONNX call
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final OnnxScoringModel model;

    public Reranker() throws IOException, InterruptedException {
        Path onnx = download("onnx/model.onnx");
        Path tokenizer = download("tokenizer.json");
        this.model = new OnnxScoringModel(onnx.toString(), tokenizer.toString());
    }

    /** Score every document against the query, in the documents' order. Nothing is cached here. */
    public double[] score(String query, List<Document> documents) {
        double[] scores = new double[documents.size()];
        for (int start = 0; start < documents.size(); start += BATCH_SIZE) {
            int end = Math.min(start + BATCH_SIZE, documents.size());
            List<TextSegment> batch = documents.subList(start, end).stream()
                    .map(doc -> TextSegment.from(doc.content()))
                    .toList();
            List<Double> batchScores = model.scoreAll(batch, query).content();
            for (int i = 0; i < batchScores.size(); i++) {
                scores[start + i] = batchScores.get(i);
            }
        }
        return scores;
    }

    /**
     * Score many (query, document) pairs, reusing scores saved by an earlier run.
     *
     * @param candidates query id to the document ids to score for it
     * @return query id to (document id to score)
     */
    public Map<String, Map<String, Double>> scoreAll(
            String cacheName,
            Map<String, String> queries,
            Map<String, List<String>> candidates,
            Map<String, Document> documents) throws IOException {
        Path cacheFile = SCORE_DIR.resolve(cacheName + ".json");
        Map<String, Map<String, Double>> scores = Files.exists(cacheFile)
                ? MAPPER.readValue(cacheFile.toFile(), new TypeReference<>() {})
                : new HashMap<>();

        long total = candidates.values().stream().mapToLong(List::size).sum();
        long done = 0;
        long lastReport = System.nanoTime();
        for (var entry : candidates.entrySet()) {
            String queryId = entry.getKey();
            Map<String, Double> known = scores.computeIfAbsent(queryId, id -> new HashMap<>());
            List<String> missing = entry.getValue().stream().filter(id -> !known.containsKey(id)).toList();
            if (!missing.isEmpty()) {
                double[] fresh = score(queries.get(queryId), missing.stream().map(documents::get).toList());
                for (int i = 0; i < missing.size(); i++) {
                    known.put(missing.get(i), fresh[i]);
                }
            }
            done += entry.getValue().size();
            if (System.nanoTime() - lastReport > 30_000_000_000L) {
                System.out.printf("  cross-encoded %,d / %,d pairs%n", done, total);
                lastReport = System.nanoTime();
                save(cacheFile, scores);
            }
        }
        save(cacheFile, scores);
        return scores;
    }

    private static void save(Path file, Map<String, Map<String, Double>> scores) throws IOException {
        Files.createDirectories(file.getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        MAPPER.writeValue(tmp.toFile(), scores);
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
    }

    private static Path download(String file) throws IOException, InterruptedException {
        Path target = MODEL_DIR.resolve(file);
        if (Files.exists(target)) {
            return target;
        }
        System.out.println("Downloading " + MODEL + " " + file + "...");
        Files.createDirectories(target.getParent());
        String url = "https://huggingface.co/" + MODEL + "/resolve/" + REVISION + "/" + file;
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpResponse<InputStream> response = client.send(
                HttpRequest.newBuilder(URI.create(url)).build(), HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            throw new IOException("Downloading " + url + " failed: HTTP " + response.statusCode());
        }
        Path tmp = target.resolveSibling(target.getFileName() + ".part");
        try (InputStream body = response.body()) {
            Files.copy(body, tmp, StandardCopyOption.REPLACE_EXISTING);
        }
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
        return target;
    }

    /** Rank ids by score, best first. Ties keep their original order. */
    public static List<String> sortByScore(List<String> ids, Map<String, Double> scores) {
        List<String> sorted = new ArrayList<>(ids);
        sorted.sort((a, b) -> Double.compare(scores.get(b), scores.get(a)));
        return sorted;
    }

    @Override
    public void close() {
        model.close();
    }
}
