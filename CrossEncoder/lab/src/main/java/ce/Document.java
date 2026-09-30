package ce;

/** A searchable document: an id, a title (may be empty) and body text. */
public record Document(String id, String title, String text) {

    /** Title and text together: what the embedding model reads. */
    public String content() {
        String content = (title + " " + text).strip();
        return content.isEmpty() ? id : content;
    }
}
