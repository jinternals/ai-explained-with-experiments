import Foundation

/// Where and how the server listens.
public struct ServerConfig: Sendable, Equatable {
    public var host: String
    public var port: UInt16
    /// If set, clients must send "Authorization: Bearer <apiKey>".
    public var apiKey: String?

    public init(host: String = "127.0.0.1", port: UInt16 = 11535, apiKey: String? = nil) {
        self.host = host
        self.port = port
        self.apiKey = apiKey
    }

    public var baseURL: String {
        let shown = host == "0.0.0.0" ? "127.0.0.1" : host
        return "http://\(shown):\(port)/v1"
    }
}

/// Counters the app shows in its menu.
public struct ServerStats: Sendable {
    public var requests = 0
    public var active = 0
    public var lastRequest: String?
    public var lastError: String?

    public init() {}
}

/// Runs the OpenAI-compatible API on top of Apple's Foundation Models.
public final class FoundationModelServer: @unchecked Sendable {
    public let config: ServerConfig
    private let http = HTTPServer()
    private let lock = NSLock()
    private var _stats = ServerStats()
    /// Every request, start and stop is written here.
    public let logs: LogStore

    public init(config: ServerConfig, logs: LogStore = LogStore(fileURL: nil)) {
        self.config = config
        self.logs = logs
    }

    public var stats: ServerStats {
        lock.lock(); defer { lock.unlock() }
        return _stats
    }

    private func update(_ change: (inout ServerStats) -> Void) {
        lock.lock(); change(&_stats); lock.unlock()
    }

    /// Start listening. `onState` gets nil when ready, or an error message.
    public func start(onState: @escaping @Sendable (String?) -> Void = { _ in }) throws {
        let config = config, logs = logs
        try http.start(host: config.host, port: config.port, handler: { [weak self] request, writer in
            await self?.handle(request, writer)
        }, onState: { error in
            if let error {
                logs.error("Could not listen on \(config.host):\(config.port): \(error)")
            } else {
                let models = AppleModel.allCases.map { "\($0.rawValue) \($0.isAvailable ? "available" : "unavailable")" }.joined(separator: ", ")
                logs.info("Listening on \(config.baseURL)\(config.apiKey == nil ? "" : " (API key required)") · \(models)")
            }
            onState(error)
        })
    }

    public func stop() {
        http.stop()
        logs.info("Stopped")
    }

    // MARK: Routing

    /// What to add to a request's log line.
    struct Outcome {
        var detail: String?
        var failed = false
    }

    func handle(_ request: HTTPRequest, _ writer: HTTPResponseWriter) async {
        if request.method == "OPTIONS" {
            await writer.respond(status: 204, body: Data()) // browser preflight; not worth a log line
            return
        }
        let started = Date()
        let path = request.path.hasSuffix("/") && request.path.count > 1 ? String(request.path.dropLast()) : request.path

        var outcome = Outcome()
        switch (request.method, path) {
        case ("GET", "/"), ("GET", "/health"):
            await writer.respondJSON(Health(models: AppleModel.allCases.map {
                Health.ModelStatus(id: $0.rawValue, available: $0.isAvailable, reason: $0.unavailableReason)
            }))
        case (_, "/v1/models"), (_, "/models"):
            if authorized(request) {
                let now = Int(Date().timeIntervalSince1970)
                let models = AppleModel.allCases.filter(\.isAvailable).map { OpenAI.Model(id: $0.rawValue, created: now) }
                await writer.respondJSON(OpenAI.ModelList(data: models))
                outcome.detail = models.map(\.id).joined(separator: ", ")
            } else {
                outcome = await unauthorized(writer)
            }
        case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
            outcome = authorized(request) ? await chatCompletions(request, writer) : await unauthorized(writer)
        default:
            outcome = await send(APIError(404, "No route for \(request.method) \(request.path). This server supports GET /v1/models and POST /v1/chat/completions.",
                                          "invalid_request_error", code: "not_found"), writer)
        }

        let status = writer.status
        let ms = Int(Date().timeIntervalSince(started) * 1000)
        let level: LogLevel = status >= 500 || outcome.failed ? .error : status >= 400 ? .warning : .info
        logs.add(level, "\(request.method) \(request.path) \(status) · \(ms) ms" + (outcome.detail.map { " · \($0)" } ?? ""))
    }

    private func authorized(_ request: HTTPRequest) -> Bool {
        guard let key = config.apiKey, !key.isEmpty else { return true }
        return request.headers["authorization"] == "Bearer \(key)"
    }

    private func unauthorized(_ writer: HTTPResponseWriter) async -> Outcome {
        await send(APIError(401, "Missing or wrong API key.", "invalid_request_error", code: "invalid_api_key"), writer)
    }

    /// Send an error response; its message goes into the log line.
    private func send(_ error: APIError, _ writer: HTTPResponseWriter, context: String? = nil) async -> Outcome {
        update { $0.lastError = error.body.error.message }
        await writer.respondJSON(status: error.status, error.body)
        return Outcome(detail: [context, error.body.error.message].compactMap { $0 }.joined(separator: " · "))
    }

    // MARK: Chat completions

    private func chatCompletions(_ request: HTTPRequest, _ writer: HTTPResponseWriter) async -> Outcome {
        let chatRequest: OpenAI.ChatRequest
        do {
            chatRequest = try JSONDecoder().decode(OpenAI.ChatRequest.self, from: request.body)
        } catch {
            return await send(APIError(400, "The request body isn't a valid chat completion request: \(error.localizedDescription)", "invalid_request_error"), writer)
        }

        let modelID = chatRequest.model ?? AppleModel.onDevice.rawValue
        guard let model = AppleModel(rawValue: modelID) else {
            let known = AppleModel.allCases.map(\.rawValue).joined(separator: ", ")
            return await send(APIError(404, "Unknown model '\(modelID)'. Use one of: \(known).", "invalid_request_error", code: "model_not_found"), writer)
        }
        if let reason = model.unavailableReason {
            return await send(APIError(503, "\(model.displayName) model is unavailable: \(reason)", "server_error", code: "model_unavailable"), writer, context: model.rawValue)
        }

        let chat: PreparedChat
        do {
            chat = try PreparedChat.make(from: chatRequest, model: model)
        } catch {
            return await send(APIError(error), writer, context: model.rawValue)
        }

        let streaming = chatRequest.stream ?? false
        update { $0.requests += 1; $0.active += 1; $0.lastRequest = "\(model.rawValue)\(streaming ? ", streamed" : "")" }
        defer { update { $0.active -= 1 } }
        let summary = "\(model.rawValue) · \(streaming ? "streamed" : "not streamed") · \(chatRequest.messages.count) message\(chatRequest.messages.count == 1 ? "" : "s")"
        let question = logs.includeMessageText ? " · Q: \(LogStore.preview(chat.prompt))" : ""
        func done(_ text: String, _ usage: OpenAI.Usage?, _ finish: String) -> Outcome {
            let tokens = usage.map { " · \($0.promptTokens) + \($0.completionTokens) tokens" } ?? ""
            let answer = logs.includeMessageText ? " · A: \(LogStore.preview(text))" : ""
            return Outcome(detail: "\(summary)\(tokens) · \(finish)\(question)\(answer)")
        }

        let id = "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24).lowercased())"
        let created = Int(Date().timeIntervalSince1970)

        if !streaming {
            do {
                let text = try await chat.respond()
                let usage = await chat.usage(completion: text)
                let finish = finishReason(usage, chatRequest)
                await writer.respondJSON(OpenAI.Completion(
                    id: id, created: created, model: model.rawValue,
                    choices: [.init(message: .init(content: text), finishReason: finish)],
                    usage: usage))
                return done(text, usage, finish)
            } catch {
                return await send(APIError(error), writer, context: summary + question)
            }
        }

        do {
            try await writer.startEventStream()
        } catch {
            writer.finish()
            return Outcome(detail: "\(summary) · client disconnected", failed: true)
        }
        func event(_ delta: OpenAI.Chunk.Delta, finish: String? = nil, usage: OpenAI.Usage? = nil, noChoices: Bool = false) async throws {
            let chunk = OpenAI.Chunk(id: id, created: created, model: model.rawValue,
                                     choices: noChoices ? [] : [.init(delta: delta, finishReason: finish)], usage: usage)
            let json = String(data: try OpenAI.encoder.encode(chunk), encoding: .utf8) ?? "{}"
            try await writer.sendEvent(json)
        }

        var sent = ""
        var outcome = Outcome()
        do {
            try await event(.init(role: "assistant", content: ""))
            for try await text in chat.stream() {
                if writer.isClosed { break }
                // Each snapshot is the full text so far; send only what's new.
                let delta: Substring = text.hasPrefix(sent) ? text.dropFirst(sent.count) : text.dropFirst(text.commonPrefix(with: sent).count)
                sent = text
                if !delta.isEmpty { try await event(.init(content: String(delta))) }
            }
            let usage = await chat.usage(completion: sent)
            let finish = writer.isClosed ? "client disconnected" : finishReason(usage, chatRequest)
            try await event(.init(), finish: finish)
            if chatRequest.streamOptions?.includeUsage == true, let usage {
                try await event(.init(), usage: usage, noChoices: true)
            }
            outcome = done(sent, usage, finish)
        } catch {
            let apiError = APIError(error)
            update { $0.lastError = apiError.body.error.message }
            outcome = Outcome(detail: "\(summary)\(question) · \(writer.isClosed ? "client disconnected" : apiError.body.error.message)",
                              failed: !writer.isClosed)
            if let json = try? OpenAI.encoder.encode(apiError.body), let text = String(data: json, encoding: .utf8) {
                try? await writer.sendEvent(text)
            }
        }
        try? await writer.sendEvent("[DONE]")
        writer.finish()
        return outcome
    }

    /// "length" when the answer used up max_tokens, otherwise "stop".
    private func finishReason(_ usage: OpenAI.Usage?, _ request: OpenAI.ChatRequest) -> String {
        guard let limit = request.maxCompletionTokens ?? request.maxTokens, let usage else { return "stop" }
        return usage.completionTokens >= limit ? "length" : "stop"
    }

    struct Health: Encodable {
        var status = "ok"
        var models: [ModelStatus]

        struct ModelStatus: Encodable {
            var id: String
            var available: Bool
            var reason: String?
        }
    }
}
