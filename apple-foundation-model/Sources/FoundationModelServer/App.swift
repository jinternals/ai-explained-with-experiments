import AppKit
import FMServerCore
import SwiftUI

@main
enum Entry {
    static func main() {
        if CommandLine.arguments.contains("--headless") {
            Headless.run(arguments: CommandLine.arguments)
        } else if let i = CommandLine.arguments.firstIndex(of: "--snapshot"), i + 1 < CommandLine.arguments.count {
            // Render the menu-bar panel to a PNG (for the README), without starting a server.
            MainActor.assumeIsolated { Snapshot.render(to: CommandLine.arguments[i + 1]) }
        } else {
            ServerApp.main()
        }
    }
}

// MARK: - Headless mode

/// `FoundationModelServer --headless [--host 127.0.0.1] [--port 11535] [--api-key KEY] [--log-messages]`
enum Headless {
    static func run(arguments: [String]) -> Never {
        setvbuf(stdout, nil, _IOLBF, 0) // print each line right away, even when output goes to a file
        func value(_ flag: String) -> String? {
            guard let i = arguments.firstIndex(of: flag), i + 1 < arguments.count else { return nil }
            return arguments[i + 1]
        }
        var config = ServerConfig()
        if let host = value("--host") { config.host = host }
        if let port = value("--port").flatMap(UInt16.init) { config.port = port }
        config.apiKey = value("--api-key")

        for model in AppleModel.allCases {
            print("\(model.rawValue): \(model.unavailableReason ?? "available")")
        }
        let logs = LogStore(echo: { print($0) })
        logs.includeMessageText = arguments.contains("--log-messages")
        let server = FoundationModelServer(config: config, logs: logs)
        let baseURL = config.baseURL
        do {
            try server.start { error in
                if let error {
                    print("Could not start the server: \(error)")
                    exit(1)
                }
                print("OpenAI-compatible API at \(baseURL)")
            }
        } catch {
            print("Could not start the server: \(error)")
            exit(1)
        }
        dispatchMain()
    }
}

// MARK: - Snapshot

@MainActor
enum Snapshot {
    static func render(to path: String) {
        let controller = ServerController(autoStart: false)
        let view = PanelView(controller: controller).background(Color(nsColor: .windowBackgroundColor))
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2
        guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
              let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:])
        else { print("Could not render the panel"); exit(1) }
        try? png.write(to: URL(fileURLWithPath: path))
        print("Wrote \(path)")
        exit(0)
    }
}

// MARK: - Menu-bar app

struct ServerApp: App {
    @StateObject private var controller = ServerController()

    var body: some Scene {
        MenuBarExtra {
            PanelView(controller: controller)
        } label: {
            Image(nsImage: MenuBarIcon.image(running: controller.running))
        }
        .menuBarExtraStyle(.window)

        Window("Server Logs", id: "logs") {
            LogsView(controller: controller)
        }
        .defaultSize(width: 860, height: 520)
    }
}

@MainActor
final class ServerController: ObservableObject {
    @Published private(set) var running = false
    @Published private(set) var startError: String?
    @Published private(set) var stats = ServerStats()
    @Published private(set) var models: [(model: AppleModel, reason: String?)] = []
    @Published var port: String
    @Published var allowNetwork: Bool
    @Published var apiKey: String
    @Published private(set) var logEntries: [LogEntry] = []
    @Published var logMessageText: Bool {
        didSet {
            logs.includeMessageText = logMessageText
            defaults.set(logMessageText, forKey: "logMessageText")
        }
    }

    /// One log for the app's lifetime, shared by every server it starts.
    let logs = LogStore()
    private var server: FoundationModelServer?
    private var timer: Timer?
    private let defaults = UserDefaults.standard

    init(autoStart: Bool = true) {
        let savedPort = defaults.integer(forKey: "port")
        port = String(savedPort == 0 ? 11535 : savedPort)
        allowNetwork = defaults.bool(forKey: "allowNetwork")
        apiKey = defaults.string(forKey: "apiKey") ?? ""
        logMessageText = defaults.bool(forKey: "logMessageText")
        logs.includeMessageText = logMessageText
        refreshModels()
        guard autoStart else { return }
        start()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    var config: ServerConfig {
        ServerConfig(host: allowNetwork ? "0.0.0.0" : "127.0.0.1",
                     port: UInt16(port) ?? 11535,
                     apiKey: apiKey.isEmpty ? nil : apiKey)
    }

    func start() {
        stop()
        guard UInt16(port) != nil else {
            startError = "The port must be a number between 1 and 65535."
            return
        }
        defaults.set(Int(port), forKey: "port")
        defaults.set(allowNetwork, forKey: "allowNetwork")
        defaults.set(apiKey, forKey: "apiKey")

        let server = FoundationModelServer(config: config, logs: logs)
        do {
            try server.start { [weak self] error in
                Task { @MainActor in
                    self?.running = error == nil
                    self?.startError = error.map { "Could not listen on port \(self?.port ?? ""): \($0)" }
                }
            }
            self.server = server
        } catch {
            startError = "Could not start the server: \(error.localizedDescription)"
        }
    }

    func stop() {
        server?.stop()
        server = nil
        running = false
    }

    func copyBaseURL() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(config.baseURL, forType: .string)
    }

    private func tick() {
        if let server { stats = server.stats }
        refreshModels()
        if logs.lastID != logEntries.last?.id ?? 0 { logEntries = logs.all() }
    }

    func clearLogs() {
        logs.clear()
        logEntries = []
    }

    func copyLogs(_ entries: [LogEntry]) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(entries.map(\.line).joined(separator: "\n"), forType: .string)
    }

    func revealLogFile() {
        guard let url = logs.fileURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func refreshModels() {
        models = AppleModel.allCases.map { ($0, $0.unavailableReason) }
    }
}

struct PanelView: View {
    @ObservedObject var controller: ServerController
    @Environment(\.openWindow) private var openWindow
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            modelsSection
            Divider()
            statsSection
            Divider()
            settingsSection
            Divider()
            openWebUIHint
            HStack {
                Button(controller.running ? "Stop server" : "Start server") {
                    controller.running ? controller.stop() : controller.start()
                }
                Button("Logs") {
                    openWindow(id: "logs")
                    NSApp.activate()
                }
                Spacer()
                Button("Quit") { NSApplication.shared.terminate(nil) }
            }
        }
        .padding(16)
        .frame(width: 360)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle().fill(controller.running ? Color.green : Color.red).frame(width: 9, height: 9)
                Text(controller.running ? "Running" : "Stopped").font(.headline)
                Spacer()
                Text("OpenAI-compatible").font(.caption).foregroundStyle(.secondary)
            }
            if let error = controller.startError {
                Text(error).font(.caption).foregroundStyle(.red)
            }
            HStack {
                Text(controller.config.baseURL)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                Spacer()
                Button(copied ? "Copied" : "Copy") {
                    controller.copyBaseURL()
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                }
            }
        }
    }

    private var modelsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Models").font(.subheadline.weight(.semibold))
            ForEach(controller.models, id: \.model) { entry in
                HStack(alignment: .top) {
                    Image(systemName: entry.reason == nil ? "checkmark.circle.fill" : "xmark.circle")
                        .foregroundStyle(entry.reason == nil ? .green : .secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(entry.model.rawValue).font(.system(.callout, design: .monospaced))
                        Text(entry.reason ?? (entry.model == .onDevice ? "Runs on this Mac" : "Runs on Apple's Private Cloud Compute servers"))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Requests: \(controller.stats.requests)   Active: \(controller.stats.active)").font(.callout)
            if let last = controller.stats.lastRequest {
                Text("Last: \(last)").font(.caption).foregroundStyle(.secondary)
            }
            if let error = controller.stats.lastError {
                Text("Last error: \(error)").font(.caption).foregroundStyle(.orange).lineLimit(3)
            }
        }
    }

    private var settingsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Settings").font(.subheadline.weight(.semibold))
            HStack {
                Text("Port")
                TextField("11535", text: $controller.port).frame(width: 80)
            }
            Toggle("Allow other devices on my network", isOn: $controller.allowNetwork)
            HStack {
                Text("API key")
                SecureField("optional", text: $controller.apiKey)
            }
            Button("Apply and restart") { controller.start() }
        }
    }

    private var openWebUIHint: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Open WebUI").font(.subheadline.weight(.semibold))
            Text("Admin Settings → Connections → OpenAI API. URL: http://host.docker.internal:\(controller.port)/v1 if Open WebUI runs in Docker, otherwise \(controller.config.baseURL). Key: your API key, or any text if none is set.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Logs window

struct LogsView: View {
    @ObservedObject var controller: ServerController
    @State private var level: LevelFilter = .all
    @State private var search = ""
    @State private var follow = true

    enum LevelFilter: String, CaseIterable, Identifiable {
        case all = "All", problems = "Warnings and errors", errors = "Errors"
        var id: Self { self }

        func includes(_ level: LogLevel) -> Bool {
            switch self {
            case .all: true
            case .problems: level != .info
            case .errors: level == .error
            }
        }
    }

    private var shown: [LogEntry] {
        controller.logEntries.filter { entry in
            level.includes(entry.level) && (search.isEmpty || entry.message.localizedCaseInsensitiveContains(search))
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Picker("Show", selection: $level) {
                    ForEach(LevelFilter.allCases) { Text($0.rawValue).tag($0) }
                }
                .frame(width: 240)
                TextField("Search", text: $search)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 240)
                Toggle("Follow", isOn: $follow)
                Toggle("Include message text", isOn: $controller.logMessageText)
                    .help("Adds short previews of each question and answer. Off by default, because prompts can be private.")
                Spacer()
                Button("Copy") { controller.copyLogs(shown) }
                Button("Clear") { controller.clearLogs() }
                Button("Log file") { controller.revealLogFile() }
                    .help(controller.logs.fileURL?.path ?? "")
            }
            .padding(10)
            Divider()
            if shown.isEmpty {
                ContentUnavailableView(controller.logEntries.isEmpty ? "No log entries yet" : "Nothing matches",
                                       systemImage: "text.alignleft",
                                       description: Text(controller.logEntries.isEmpty
                                            ? "Requests to \(controller.config.baseURL) will appear here."
                                            : "Change the filter or the search."))
            } else {
                ScrollViewReader { proxy in
                    List(shown) { entry in
                        LogRow(entry: entry).id(entry.id)
                    }
                    .listStyle(.plain)
                    .font(.system(.callout, design: .monospaced))
                    .onChange(of: controller.logEntries.last?.id) {
                        if follow, let last = shown.last { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                    .onAppear {
                        if let last = shown.last { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
            Divider()
            HStack {
                Text("\(shown.count) of \(controller.logEntries.count) entries · the app keeps the last 1,000; the log file keeps everything")
                Spacer()
                Circle().fill(controller.running ? Color.green : Color.red).frame(width: 8, height: 8)
                Text(controller.running ? "Running on \(controller.config.baseURL)" : "Stopped")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
        }
        .frame(minWidth: 640, minHeight: 320)
    }
}

struct LogRow: View {
    let entry: LogEntry

    private static let time: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(Self.time.string(from: entry.date)).foregroundStyle(.secondary)
            Text(entry.level.rawValue.uppercased())
                .font(.system(.caption, design: .monospaced).weight(.semibold))
                .foregroundStyle(color)
                .frame(width: 64, alignment: .leading)
            Text(entry.message).textSelection(.enabled)
        }
    }

    private var color: Color {
        switch entry.level {
        case .info: .secondary
        case .warning: .orange
        case .error: .red
        }
    }
}
