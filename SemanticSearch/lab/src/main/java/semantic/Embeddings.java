package semantic;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Text embeddings with all-MiniLM-L6-v2 (in-process ONNX via LangChain4j), cached on disk. Each vector
 * has length 1. LangChain4j's tokenizer file cuts every text at 128 word pieces; see {@link Length}.
 */
public final class Embeddings {

    private static final Path CACHE_DIR = Path.of("/cache", "vectors-java");
    private static final int BATCH_SIZE = 1_000;

    private static EmbeddingModel model;

    private Embeddings() {}

    /**
     * Embed texts into 384-dimensional vectors.
     *
     * @param cacheName if not null, vectors are saved under this name and reused on the next run
     */
    public static float[][] embed(List<String> texts, String cacheName) throws IOException {
        return embed(model(), texts, cacheName);
    }

    /** The same, with another model (the length experiment uses one with a different token limit). */
    public static float[][] embed(EmbeddingModel model, List<String> texts, String cacheName) throws IOException {
        Path cacheFile = cacheName == null ? null : CACHE_DIR.resolve(cacheName + ".bin");
        if (cacheFile != null && Files.exists(cacheFile)) {
            return readVectors(cacheFile);
        }

        float[][] vectors = new float[texts.size()][];
        for (int start = 0; start < texts.size(); start += BATCH_SIZE) {
            int end = Math.min(start + BATCH_SIZE, texts.size());
            List<TextSegment> batch = texts.subList(start, end).stream().map(TextSegment::from).toList();
            List<Embedding> embeddings = model.embedAll(batch).content();
            for (int i = 0; i < embeddings.size(); i++) {
                vectors[start + i] = embeddings.get(i).vector();
            }
            if (texts.size() > BATCH_SIZE) {
                System.out.printf("  embedded %,d / %,d%n", end, texts.size());
            }
        }

        if (cacheFile != null) {
            writeVectors(cacheFile, vectors);
        }
        return vectors;
    }

    public static float[] embed(String text) {
        return model().embed(text).content().vector();
    }

    private static synchronized EmbeddingModel model() {
        if (model == null) {
            model = new AllMiniLmL6V2EmbeddingModel();
        }
        return model;
    }

    // File format: vector count, dimension, then every value as a float.

    private static void writeVectors(Path file, float[][] vectors) throws IOException {
        Files.createDirectories(file.getParent());
        try (var out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(vectors.length);
            out.writeInt(vectors.length == 0 ? 0 : vectors[0].length);
            for (float[] vector : vectors) {
                for (float value : vector) {
                    out.writeFloat(value);
                }
            }
        }
    }

    private static float[][] readVectors(Path file) throws IOException {
        try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            int count = in.readInt();
            int dimension = in.readInt();
            float[][] vectors = new float[count][dimension];
            for (float[] vector : vectors) {
                for (int i = 0; i < dimension; i++) {
                    vector[i] = in.readFloat();
                }
            }
            return vectors;
        }
    }
}
