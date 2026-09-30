package rrf;

/** Entry point: {@code java -jar rrf-lab.jar [goa | step3 | benchmark]}. */
public final class Main {

    private Main() {}

    public static void main(String[] args) throws Exception {
        String command = args.length == 0 ? "benchmark" : args[0];
        switch (command) {
            case "goa" -> GoaDemo.run();
            case "step3" -> Step3Example.run();
            case "benchmark" -> Benchmark.run();
            default -> {
                System.err.println("Usage: java -jar rrf-lab.jar [goa | step3 | benchmark]");
                System.exit(2);
            }
        }
    }
}
