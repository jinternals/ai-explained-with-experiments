import Foundation

public enum LogLevel: String, Sendable, CaseIterable {
    case info, warning, error
}

public struct LogEntry: Identifiable, Sendable, Equatable {
    public let id: Int
    public let date: Date
    public let level: LogLevel
    public let message: String

    /// One line for the log file and for copying: "2026-09-29 22:13:05.123  INFO  message".
    public var line: String {
        "\(LogStore.timestamp.string(from: date))  \(level.rawValue.uppercased().padding(toLength: 9, withPad: " ", startingAt: 0))\(message)"
    }
}

/// The server's log: the last entries in memory (for the app's log window) and every entry appended to a file.
public final class LogStore: @unchecked Sendable {
    public static let defaultFile = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "Library/Logs/FoundationModelServer/server.log")

    static let timestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()

    public let fileURL: URL?
    private let capacity: Int
    private let echo: (@Sendable (String) -> Void)?
    private let lock = NSLock()
    private var entries: [LogEntry] = []
    private var nextID = 1
    private var _includeMessageText = false
    private var file: FileHandle?

    /// - Parameters:
    ///   - fileURL: where to append every entry; nil keeps entries in memory only.
    ///   - capacity: how many entries to keep in memory.
    ///   - echo: also pass each line here (headless mode prints them).
    public init(fileURL: URL? = LogStore.defaultFile, capacity: Int = 1_000, echo: (@Sendable (String) -> Void)? = nil) {
        self.fileURL = fileURL
        self.capacity = capacity
        self.echo = echo
        if let fileURL { file = Self.open(fileURL) }
    }

    /// Whether request logs include short previews of the question and the answer. Off by default,
    /// because prompts can be private.
    public var includeMessageText: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _includeMessageText }
        set { lock.lock(); _includeMessageText = newValue; lock.unlock() }
    }

    public func add(_ level: LogLevel, _ message: String) {
        lock.lock()
        let entry = LogEntry(id: nextID, date: Date(), level: level, message: message)
        nextID += 1
        entries.append(entry)
        if entries.count > capacity { entries.removeFirst(entries.count - capacity) }
        let file = self.file
        lock.unlock()

        let line = entry.line
        file?.write(Data((line + "\n").utf8))
        echo?(line)
    }

    public func info(_ message: String) { add(.info, message) }
    public func warning(_ message: String) { add(.warning, message) }
    public func error(_ message: String) { add(.error, message) }

    /// Entries still in memory, oldest first.
    public func all() -> [LogEntry] {
        lock.lock(); defer { lock.unlock() }
        return entries
    }

    /// The id of the newest entry, or 0. Cheap way for the app to see whether anything changed.
    public var lastID: Int {
        lock.lock(); defer { lock.unlock() }
        return entries.last?.id ?? 0
    }

    /// Empty the in-memory list. The log file keeps everything.
    public func clear() {
        lock.lock(); entries.removeAll(); lock.unlock()
    }

    /// Open for appending. A file over 5 MB is started afresh, keeping the old one as server.log.1.
    private static func open(_ url: URL) -> FileHandle? {
        let fm = FileManager.default
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? fm.attributesOfItem(atPath: url.path))?[.size] as? Int, size > 5_000_000 {
            let old = url.appendingPathExtension("1")
            try? fm.removeItem(at: old)
            try? fm.moveItem(at: url, to: old)
        }
        if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
        let handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
        return handle
    }

    /// A short, single-line preview of message text for the log.
    static func preview(_ text: String, limit: Int = 120) -> String {
        let flat = text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
        return flat.count <= limit ? "\"\(flat)\"" : "\"\(flat.prefix(limit))…\""
    }
}
