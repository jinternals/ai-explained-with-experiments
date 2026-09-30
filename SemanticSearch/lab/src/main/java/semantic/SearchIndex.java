package semantic;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A small OpenSearch client: build an index with a text field and a vector field, then search it
 * three ways: by keyword (BM25), by vector with the HNSW graph (approximate), and by vector
 * comparing the question with every document (exact).
 */
public final class SearchIndex {

    public record Hit(String id, double score) {}

    private static final String OPENSEARCH_URL = System.getenv().getOrDefault("OPENSEARCH_URL", "http://opensearch:9200");
    private static final HttpClient HTTP = HttpClient.newHttpClient();
    private static final int BULK_SIZE = 500;

    private final String name;

    public SearchIndex(String name) {
        this.name = name;
    }

    /** Block until OpenSearch answers, and return its version. */
    public static String waitUntilReady() throws InterruptedException {
        long deadline = System.nanoTime() + Duration.ofMinutes(5).toNanos();
        while (System.nanoTime() < deadline) {
            try {
                return call("GET", "/", null).path("version").path("number").asText();
            } catch (IOException notReadyYet) {
                Thread.sleep(2_000);
            }
        }
        throw new IllegalStateException("OpenSearch did not start in time");
    }

    public static List<String> ids(List<Hit> hits) {
        return hits.stream().map(Hit::id).toList();
    }

    // ------------------------------------------------------------ building

    /**
     * (Re)create the index: title and text for BM25, and a 384-number vector per document for k-NN.
     * The HNSW settings are left at OpenSearch's defaults; {@link #hnswSettings()} reads them back.
     */
    public void create(int dimension) throws IOException, InterruptedException {
        request("DELETE", "/" + name, null, "application/json"); // fine if it didn't exist
        call("PUT", "/" + name, Json.object(
                "settings", Json.object("index", Json.object("knn", true, "number_of_shards", 1, "number_of_replicas", 0)),
                "mappings", Json.object("properties", Json.object(
                        "title", Json.object("type", "text"),
                        "text", Json.object("type", "text"),
                        "embedding", Json.object(
                                "type", "knn_vector",
                                "dimension", dimension,
                                "method", Json.object("name", "hnsw", "space_type", "cosinesimil", "engine", "lucene"))))));
    }

    /** Index documents with their embeddings (same order), in bulk, then merge into one segment. */
    public void add(List<Document> documents, float[][] vectors) throws IOException, InterruptedException {
        for (int start = 0; start < documents.size(); start += BULK_SIZE) {
            StringBuilder ndjson = new StringBuilder();
            for (int i = start; i < Math.min(start + BULK_SIZE, documents.size()); i++) {
                Document doc = documents.get(i);
                ndjson.append(Json.write(Json.object("index", Json.object("_id", doc.id())))).append('\n');
                ndjson.append(Json.write(Json.object("title", doc.title(), "text", doc.text(), "embedding", vectors[i]))).append('\n');
            }
            JsonNode response = call("POST", "/" + name + "/_bulk", ndjson.toString(), "application/x-ndjson");
            if (response.path("errors").asBoolean()) {
                throw new IOException("Bulk indexing failed for batch starting at " + start);
            }
        }
        call("POST", "/" + name + "/_refresh", null);
        // One segment means one HNSW graph, so every run searches the same structure.
        call("POST", "/" + name + "/_forcemerge?max_num_segments=1", null);
        call("POST", "/" + name + "/_refresh", null);
    }

    /** The vector field's settings as OpenSearch reports them, including defaults it filled in. */
    public JsonNode hnswSettings() throws IOException, InterruptedException {
        return call("GET", "/" + name + "/_mapping", null)
                .path(name).path("mappings").path("properties").path("embedding");
    }

    // ------------------------------------------------------------ searching

    /** BM25 keyword search over title and text, best first. */
    public List<Hit> keyword(String text, int size) throws IOException, InterruptedException {
        // Title and text as separate fields, like BEIR's "multifield" BM25 baseline.
        return search(Json.object("multi_match", Json.object(
                "query", text, "fields", List.of("title", "text"), "tie_breaker", 0.5)), size);
    }

    /** Approximate nearest neighbours: walk the HNSW graph. */
    public List<Hit> hnsw(float[] vector, int size) throws IOException, InterruptedException {
        return search(Json.object("knn", Json.object("embedding", Json.object("vector", vector, "k", size))), size);
    }

    /** Exact nearest neighbours: score every document against the question (a brute-force scan). */
    public List<Hit> exact(float[] vector, int size) throws IOException, InterruptedException {
        return search(Json.object("script_score", Json.object(
                "query", Json.object("match_all", Json.object()),
                "script", Json.object(
                        "source", "knn_score",
                        "lang", "knn",
                        "params", Json.object("field", "embedding", "query_value", vector, "space_type", "cosinesimil")))), size);
    }

    // ------------------------------------------------------------ internals

    private List<Hit> search(Map<String, Object> query, int size) throws IOException, InterruptedException {
        JsonNode response = call("POST", "/" + name + "/_search", Json.object("size", size, "query", query, "_source", false));

        List<Hit> hits = new ArrayList<>();
        for (JsonNode hit : response.path("hits").path("hits")) {
            hits.add(new Hit(hit.get("_id").asText(), hit.get("_score").asDouble()));
        }
        return hits;
    }

    private static JsonNode call(String method, String path, Object body) throws IOException, InterruptedException {
        return call(method, path, body == null ? null : Json.write(body), "application/json");
    }

    private static JsonNode call(String method, String path, String body, String contentType)
            throws IOException, InterruptedException {
        HttpResponse<String> response = request(method, path, body, contentType);
        if (response.statusCode() >= 300) {
            throw new IOException(method + " " + path + " failed: HTTP " + response.statusCode() + " " + response.body());
        }
        return Json.read(response.body());
    }

    private static HttpResponse<String> request(String method, String path, String body, String contentType)
            throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder(URI.create(OPENSEARCH_URL + path))
                .timeout(Duration.ofMinutes(10))
                .header("Content-Type", contentType)
                .method(method, body == null ? BodyPublishers.noBody() : BodyPublishers.ofString(body))
                .build();
        return HTTP.send(request, BodyHandlers.ofString());
    }
}
