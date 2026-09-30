package bm25;

/** Entry point: {@code java -jar bm25-lab.jar [example | benchmark]}. */
public final class Main {

    private Main() {}

    public static void main(String[] args) throws Exception {
        String command = args.length == 0 ? "benchmark" : args[0];
        switch (command) {
            case "example" -> Example.run();
            case "benchmark" -> Benchmark.run();
            default -> {
                System.err.println("Usage: java -jar bm25-lab.jar [example | benchmark]");
                System.exit(2);
            }
        }
    }
}
