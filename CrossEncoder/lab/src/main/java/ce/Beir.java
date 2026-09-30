package ce;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** Download and read BEIR datasets (https://github.com/beir-cellar/beir). */
public final class Beir {

    /**
     * One BEIR dataset.
     *
     * @param queries only the test queries that have judgments, by id
     * @param qrels   expert relevance labels: query id to (doc id to label)
     */
    public record Dataset(
            List<Document> documents,
            Map<String, String> queries,
            Map<String, Map<String, Integer>> qrels) {}

    private static final String BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets";
    private static final Path DATA_DIR = Path.of("/data");

    private Beir() {}

    public static Dataset load(String name) throws IOException, InterruptedException {
        Path folder = download(name);

        List<Document> documents = readJsonl(folder.resolve("corpus.jsonl")).stream()
                .map(row -> new Document(row.get("_id").asText(), row.path("title").asText(""), row.get("text").asText()))
                .toList();

        Map<String, Map<String, Integer>> qrels = new LinkedHashMap<>();
        List<String> lines = Files.readAllLines(folder.resolve("qrels").resolve("test.tsv"));
        for (String line : lines.subList(1, lines.size())) { // first line is the header
            if (line.isBlank()) {
                continue;
            }
            String[] fields = line.split("\t");
            qrels.computeIfAbsent(fields[0], queryId -> new LinkedHashMap<>()).put(fields[1], Integer.parseInt(fields[2]));
        }

        Map<String, String> queries = new LinkedHashMap<>();
        for (JsonNode row : readJsonl(folder.resolve("queries.jsonl"))) {
            String queryId = row.get("_id").asText();
            if (qrels.containsKey(queryId)) {
                queries.put(queryId, row.get("text").asText());
            }
        }
        return new Dataset(documents, queries, qrels);
    }

    private static Path download(String name) throws IOException, InterruptedException {
        Path folder = DATA_DIR.resolve(name);
        if (Files.exists(folder)) {
            return folder;
        }

        System.out.println("Downloading " + name + "...");
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpRequest request = HttpRequest.newBuilder(URI.create(BASE_URL + "/" + name + ".zip")).build();
        HttpResponse<InputStream> response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            throw new IOException("Downloading " + name + " failed: HTTP " + response.statusCode());
        }

        try (ZipInputStream zip = new ZipInputStream(response.body())) {
            for (ZipEntry entry; (entry = zip.getNextEntry()) != null; ) {
                Path target = DATA_DIR.resolve(entry.getName()).normalize();
                if (!target.startsWith(DATA_DIR)) {
                    throw new IOException("Unsafe path in zip: " + entry.getName());
                }
                if (entry.isDirectory()) {
                    Files.createDirectories(target);
                } else {
                    Files.createDirectories(target.getParent());
                    Files.copy(zip, target, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
        return folder;
    }

    private static List<JsonNode> readJsonl(Path path) throws IOException {
        try (Stream<String> lines = Files.lines(path)) {
            return lines.filter(line -> !line.isBlank()).map(Json::read).toList();
        }
    }
}
