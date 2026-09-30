import Foundation
import Network

/// One parsed HTTP request. Header names are lowercased.
public struct HTTPRequest: Sendable {
    public let method: String
    public let path: String
    public let headers: [String: String]
    public let body: Data
}

/// Writes one response to a connection, either all at once or as a stream (server-sent events).
/// Every response closes the connection afterwards, which keeps the server simple and is fine for local use.
public final class HTTPResponseWriter: @unchecked Sendable {
    private let connection: NWConnection
    private let lock = NSLock()
    private var closed = false
    private var _status = 0

    init(connection: NWConnection) {
        self.connection = connection
    }

    /// True once the client has gone away or the response has finished.
    public var isClosed: Bool {
        lock.lock(); defer { lock.unlock() }
        return closed
    }

    func markClosed() {
        lock.lock(); closed = true; lock.unlock()
    }

    /// The HTTP status sent, or 0 before anything was sent.
    public var status: Int {
        lock.lock(); defer { lock.unlock() }
        return _status
    }

    private func setStatus(_ status: Int) {
        lock.lock(); _status = status; lock.unlock()
    }

    /// A complete response with a body, then close.
    public func respond(status: Int, headers: [String: String] = [:], body: Data) async {
        setStatus(status)
        var all = headers
        all["Content-Length"] = String(body.count)
        var data = head(status: status, headers: all)
        data.append(body)
        try? await send(data)
        finish()
    }

    public func respondJSON(status: Int = 200, _ value: some Encodable) async {
        let body = (try? OpenAI.encoder.encode(value)) ?? Data("{}".utf8)
        await respond(status: status, headers: ["Content-Type": "application/json"], body: body)
    }

    /// Start a server-sent-events stream. The body ends when the connection closes.
    public func startEventStream() async throws {
        setStatus(200)
        try await send(head(status: 200, headers: [
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        ]))
    }

    /// One `data: ...` event.
    public func sendEvent(_ payload: String) async throws {
        try await send(Data("data: \(payload)\n\n".utf8))
    }

    public func finish() {
        markClosed()
        connection.send(content: nil, contentContext: .finalMessage, isComplete: true, completion: .contentProcessed { [connection] _ in
            connection.cancel()
        })
    }

    private func head(status: Int, headers: [String: String]) -> Data {
        var lines = ["HTTP/1.1 \(status) \(Self.reason(status))"]
        var all = headers
        all["Connection"] = "close"
        all["Access-Control-Allow-Origin"] = "*"
        all["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        all["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        for (name, value) in all.sorted(by: { $0.key < $1.key }) {
            lines.append("\(name): \(value)")
        }
        return Data((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8)
    }

    private func send(_ data: Data) async throws {
        if isClosed { throw CancellationError() }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
            })
        }
    }

    static func reason(_ status: Int) -> String {
        switch status {
        case 200: "OK"
        case 204: "No Content"
        case 400: "Bad Request"
        case 401: "Unauthorized"
        case 404: "Not Found"
        case 405: "Method Not Allowed"
        case 413: "Payload Too Large"
        case 429: "Too Many Requests"
        case 500: "Internal Server Error"
        case 503: "Service Unavailable"
        case 504: "Gateway Timeout"
        default: "Status"
        }
    }
}

/// A small HTTP/1.1 server on Apple's Network framework: no third-party dependencies.
public final class HTTPServer: @unchecked Sendable {
    public typealias Handler = @Sendable (HTTPRequest, HTTPResponseWriter) async -> Void

    private let queue = DispatchQueue(label: "FoundationModelServer.http")
    private var listener: NWListener?
    private static let maxBody = 16 * 1024 * 1024

    public init() {}

    /// Listen on host:port. Use "127.0.0.1" for this Mac only, "0.0.0.0" for the local network.
    public func start(host: String, port: UInt16, handler: @escaping Handler, onState: @escaping @Sendable (String?) -> Void) throws {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { throw POSIXError(.EINVAL) }
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        if host != "0.0.0.0" {
            parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: nwPort)
        }
        let listener = host == "0.0.0.0" ? try NWListener(using: parameters, on: nwPort) : try NWListener(using: parameters)
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: onState(nil)
            case .failed(let error): onState("\(error)")
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection, handler: handler)
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    private func accept(_ connection: NWConnection, handler: @escaping Handler) {
        let writer = HTTPResponseWriter(connection: connection)
        connection.stateUpdateHandler = { state in
            switch state {
            case .failed, .cancelled: writer.markClosed()
            default: break
            }
        }
        connection.start(queue: queue)
        receive(connection, buffer: Data(), writer: writer, handler: handler)
    }

    private func receive(_ connection: NWConnection, buffer: Data, writer: HTTPResponseWriter, handler: @escaping Handler) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 256 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            switch Self.parse(buffer) {
            case .complete(let request):
                Task { await handler(request, writer) }
            case .needMore where !isComplete && error == nil && buffer.count <= Self.maxBody:
                self.receive(connection, buffer: buffer, writer: writer, handler: handler)
            case .needMore:
                connection.cancel()
            case .invalid(let status):
                Task { await writer.respond(status: status, body: Data()) }
            }
        }
    }

    enum ParseResult {
        case complete(HTTPRequest)
        case needMore
        case invalid(Int)
    }

    /// Parse a request once the headers and the whole body (Content-Length) have arrived.
    static func parse(_ buffer: Data) -> ParseResult {
        guard let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            return buffer.count > 64 * 1024 ? .invalid(400) : .needMore
        }
        guard let head = String(data: buffer[..<headerEnd.lowerBound], encoding: .utf8) else { return .invalid(400) }
        let lines = head.components(separatedBy: "\r\n")
        let requestLine = lines[0].split(separator: " ")
        guard requestLine.count >= 2 else { return .invalid(400) }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        if headers["transfer-encoding"]?.lowercased().contains("chunked") == true { return .invalid(400) }
        let length = Int(headers["content-length"] ?? "0") ?? 0
        if length > maxBody { return .invalid(413) }
        let bodyStart = headerEnd.upperBound
        guard buffer.count - bodyStart >= length else { return .needMore }
        let target = String(requestLine[1])
        let path = String(target.split(separator: "?", maxSplits: 1).first ?? "")
        return .complete(HTTPRequest(
            method: String(requestLine[0]).uppercased(),
            path: path,
            headers: headers,
            body: Data(buffer[bodyStart..<(bodyStart + length)])))
    }
}
