package ce;

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

/** A small OpenSearch client: build an index, then run keyword and vector searches. */
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

    /** (Re)create the index with a text field for BM25 and a vector field for k-NN. */
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

    /** Index documents with their embeddings (same order), in bulk. */
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
    }

    // ------------------------------------------------------------ searching

    /** BM25 keyword search, best first. */
    public List<Hit> keyword(String text, int size) throws IOException, InterruptedException {
        return search(keywordQuery(text), size);
    }

    /** k-NN search on the embeddings, best first. */
    public List<Hit> vector(float[] vector, int size) throws IOException, InterruptedException {
        return search(vectorQuery(vector, size), size);
    }

    // ------------------------------------------------------------ internals

    private static Map<String, Object> keywordQuery(String text) {
        // Title and text as separate fields, like BEIR's "multifield" BM25 baseline.
        // (BEIR used Anserini with k1=0.9, b=0.4; OpenSearch defaults to k1=1.2, b=0.75.)
        return Json.object("multi_match", Json.object("query", text, "fields", List.of("title", "text"), "tie_breaker", 0.5));
    }

    private static Map<String, Object> vectorQuery(float[] vector, int k) {
        return Json.object("knn", Json.object("embedding", Json.object("vector", vector, "k", k)));
    }

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
                .timeout(Duration.ofMinutes(5))
                .header("Content-Type", contentType)
                .method(method, body == null ? BodyPublishers.noBody() : BodyPublishers.ofString(body))
                .build();
        return HTTP.send(request, BodyHandlers.ofString());
    }
}
