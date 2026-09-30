package semantic;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/** Small JSON helpers around one shared Jackson mapper. */
public final class Json {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private Json() {}

    /** Build a JSON object that keeps key order: {@code object("size", 10, "query", query)}. */
    public static Map<String, Object> object(Object... keysAndValues) {
        if (keysAndValues.length % 2 != 0) {
            throw new IllegalArgumentException("Expected key/value pairs");
        }
        Map<String, Object> object = new LinkedHashMap<>();
        for (int i = 0; i < keysAndValues.length; i += 2) {
            object.put((String) keysAndValues[i], keysAndValues[i + 1]);
        }
        return object;
    }

    public static String write(Object value) {
        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static JsonNode read(String json) {
        try {
            return MAPPER.readTree(json);
        } catch (JsonProcessingException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static void writeFile(Path path, Object value) throws IOException {
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), value);
    }
}
