// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "FoundationModelServer",
    platforms: [.macOS("27.0")],
    products: [
        .executable(name: "FoundationModelServer", targets: ["FoundationModelServer"]),
    ],
    targets: [
        // HTTP server, OpenAI-compatible API and the bridge to Apple's Foundation Models framework.
        .target(name: "FMServerCore"),
        // The menu-bar app (and --headless mode) around it.
        .executableTarget(name: "FoundationModelServer", dependencies: ["FMServerCore"]),
        .testTarget(name: "FMServerCoreTests", dependencies: ["FMServerCore"]),
    ],
    swiftLanguageModes: [.v5]
)
