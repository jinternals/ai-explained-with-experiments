package semantic;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A plain word splitter for measuring how many of a question's words a document shares: lowercase,
 * split on anything that isn't a letter or digit, and drop Lucene's 33 English stop words (the same
 * list OpenSearch's english analyzer uses). No stemming, so "avocado" and "avocados" differ.
 */
public final class Words {

    private static final Set<String> STOP_WORDS = Set.of(
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
            "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
            "they", "this", "to", "was", "will", "with");

    private Words() {}

    public static Set<String> of(String text) {
        Set<String> words = new LinkedHashSet<>();
        for (String word : text.toLowerCase().split("[^\\p{L}\\p{N}]+")) {
            if (!word.isEmpty() && !STOP_WORDS.contains(word)) {
                words.add(word);
            }
        }
        return words;
    }

    /** The question's words that also appear in the text, in question order. */
    public static List<String> shared(String question, String text) {
        Set<String> inText = of(text);
        return of(question).stream().filter(inText::contains).toList();
    }

    /** Share of the question's words that appear in the text, from 0 to 1. */
    public static double overlap(String question, String text) {
        Set<String> words = of(question);
        return words.isEmpty() ? 0 : (double) shared(question, text).size() / words.size();
    }
}
