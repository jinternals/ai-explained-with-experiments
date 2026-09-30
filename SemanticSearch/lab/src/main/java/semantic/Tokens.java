package semantic;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** The word pieces all-MiniLM-L6-v2 reads, using the tokenizer file bundled with LangChain4j. */
public final class Tokens {

    private static HuggingFaceTokenizer tokenizer;

    private Tokens() {}

    /** The pieces of a text, including the [CLS] and [SEP] markers the model adds at the start and end. */
    public static List<String> pieces(String text) {
        return Arrays.asList(tokenizer().encode(text).getTokens());
    }

    private static synchronized HuggingFaceTokenizer tokenizer() {
        if (tokenizer == null) {
            try (InputStream in = Tokens.class.getResourceAsStream("/all-minilm-l6-v2-tokenizer.json")) {
                tokenizer = HuggingFaceTokenizer.newInstance(in, Map.of("padding", "false"));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        return tokenizer;
    }
}
