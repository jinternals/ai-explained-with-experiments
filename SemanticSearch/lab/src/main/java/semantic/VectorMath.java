package semantic;

/** The three measures a vector database can rank by, written out in full. */
public final class VectorMath {

    private VectorMath() {}

    /** Multiply matching positions and add them all up. */
    public static double dot(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += (double) a[i] * b[i];
        }
        return sum;
    }

    /** The length of the arrow: Pythagoras, in as many dimensions as the vector has. */
    public static double length(float[] a) {
        return Math.sqrt(dot(a, a));
    }

    /** The cosine of the angle between two arrows: 1 same direction, 0 unrelated, −1 opposite. */
    public static double cosine(float[] a, float[] b) {
        return dot(a, b) / (length(a) * length(b));
    }

    /** Straight-line distance between the two arrow tips. */
    public static double euclidean(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = (double) a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /** The angle, in degrees, that a cosine stands for. */
    public static double degrees(double cosine) {
        return Math.toDegrees(Math.acos(Math.max(-1, Math.min(1, cosine))));
    }
}
