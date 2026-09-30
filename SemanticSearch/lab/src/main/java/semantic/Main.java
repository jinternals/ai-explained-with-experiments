package semantic;

/** Entry point: {@code java -jar semantic-search-lab.jar [example | benchmark | length]}. */
public final class Main {

    private Main() {}

    public static void main(String[] args) throws Exception {
        String command = args.length == 0 ? "benchmark" : args[0];
        switch (command) {
            case "example" -> Example.run();
            case "benchmark" -> Benchmark.run();
            case "length" -> Length.run();
            default -> {
                System.err.println("Usage: java -jar semantic-search-lab.jar [example | benchmark | length]");
                System.exit(2);
            }
        }
    }
}
