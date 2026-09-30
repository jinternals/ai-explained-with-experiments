import Foundation
import FoundationModels
import Security

/// The models this server offers, by the id OpenAI clients use.
public enum AppleModel: String, CaseIterable, Sendable {
    /// Runs entirely on this Mac.
    case onDevice = "apple-on-device"
    /// Apple's larger model on Private Cloud Compute. Requests leave this Mac.
    case privateCloud = "apple-private-cloud"

    public var displayName: String {
        switch self {
        case .onDevice: "On-device"
        case .privateCloud: "Private Cloud Compute"
        }
    }

    /// nil when the model can be used; otherwise the reason it can't.
    public var unavailableReason: String? {
        switch self {
        case .onDevice:
            switch SystemLanguageModel.default.availability {
            case .available: return nil
            case .unavailable(let reason): return Self.describe(reason)
            }
        case .privateCloud:
            // Without this managed entitlement, the framework stops the whole process on the first
            // request instead of throwing an error, so never offer the model unless it's granted.
            guard Self.hasEntitlement(Self.privateCloudEntitlement) else {
                return "Needs Apple's \(Self.privateCloudEntitlement) entitlement. Request it at developer.apple.com/contact/request/private-cloud-compute"
            }
            switch PrivateCloudComputeLanguageModel().availability {
            case .available: return nil
            case .unavailable(let reason): return Self.describe(reason)
            }
        }
    }

    public var isAvailable: Bool { unavailableReason == nil }

    func session(transcript: Transcript) -> LanguageModelSession {
        switch self {
        case .onDevice: LanguageModelSession(model: SystemLanguageModel.default, transcript: transcript)
        case .privateCloud: LanguageModelSession(model: PrivateCloudComputeLanguageModel(), transcript: transcript)
        }
    }

    static let privateCloudEntitlement = "com.apple.developer.private-cloud-compute"

    /// Whether this running app's code signature grants the entitlement.
    static func hasEntitlement(_ name: String) -> Bool {
        guard let task = SecTaskCreateFromSelf(kCFAllocatorDefault),
              let value = SecTaskCopyValueForEntitlement(task, name as CFString, nil)
        else { return false }
        return (value as? Bool) ?? true
    }

    private static func describe(_ reason: some Any) -> String {
        switch String(describing: reason) {
        case "deviceNotEligible": "This Mac can't run Apple Intelligence."
        case "appleIntelligenceNotEnabled": "Turn on Apple Intelligence in System Settings."
        case "modelNotReady": "The model is still downloading. Try again later."
        case let other: other
        }
    }
}

/// One chat, turned into what the Foundation Models framework needs:
/// a transcript of the earlier turns, and the newest user message as the prompt.
public struct PreparedChat: Sendable {
    public let model: AppleModel
    public let transcript: Transcript
    let entries: [Transcript.Entry]
    public let prompt: String
    public let options: GenerationOptions

    /// OpenAI roles map like this: system/developer become instructions, user becomes a prompt,
    /// assistant becomes a response. Tool messages are included as plain text.
    public static func make(from request: OpenAI.ChatRequest, model: AppleModel) throws -> PreparedChat {
        var instructions: [String] = []
        var turns: [(role: String, text: String)] = []
        for message in request.messages {
            let (text, hasNonText) = message.content?.text ?? ("", false)
            if hasNonText {
                throw ChatError.badRequest("Only text is supported. Remove images and other attachments from the message.")
            }
            switch message.role {
            case "system", "developer": instructions.append(text)
            case "user", "assistant": turns.append((message.role, text))
            case "tool": turns.append(("user", "Tool result:\n\(text)"))
            default: throw ChatError.badRequest("Unknown message role '\(message.role)'.")
            }
        }
        guard let last = turns.last, last.role == "user" else {
            throw ChatError.badRequest("The last message must come from the user.")
        }

        var entries: [Transcript.Entry] = []
        let joinedInstructions = instructions.filter { !$0.isEmpty }.joined(separator: "\n\n")
        if !joinedInstructions.isEmpty {
            entries.append(.instructions(Transcript.Instructions(segments: [.text(.init(content: joinedInstructions))], toolDefinitions: [])))
        }
        for turn in turns.dropLast() {
            let segment = Transcript.Segment.text(.init(content: turn.text))
            if turn.role == "user" {
                entries.append(.prompt(Transcript.Prompt(segments: [segment])))
            } else {
                entries.append(.response(Transcript.Response(assetIDs: [], segments: [segment])))
            }
        }

        let temperature = request.temperature.map { min(max($0, 0), 2) }
        let maxTokens = request.maxCompletionTokens ?? request.maxTokens
        return PreparedChat(
            model: model,
            transcript: Transcript(entries: entries),
            entries: entries,
            prompt: last.text,
            options: GenerationOptions(temperature: temperature, maximumResponseTokens: maxTokens))
    }

    /// The whole answer at once.
    public func respond() async throws -> String {
        try await model.session(transcript: transcript).respond(to: prompt, options: options).content
    }

    /// The answer as it grows. Each value is the full text so far.
    public func stream() -> AsyncThrowingStream<String, Error> {
        let session = model.session(transcript: transcript)
        let prompt = prompt, options = options
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await snapshot in session.streamResponse(to: prompt, options: options) {
                        continuation.yield(snapshot.content)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Token counts for the usage field, using the on-device model's tokenizer
    /// (the only one the framework can count with). Nil if counting fails.
    public func usage(completion: String) async -> OpenAI.Usage? {
        let counter = SystemLanguageModel.default
        guard let history = try? await counter.tokenCount(for: entries),
              let promptTokens = try? await counter.tokenCount(for: prompt),
              let completionTokens = try? await counter.tokenCount(for: completion)
        else { return nil }
        let input = history + promptTokens
        return OpenAI.Usage(promptTokens: input, completionTokens: completionTokens, totalTokens: input + completionTokens)
    }
}

public enum ChatError: Error {
    case badRequest(String)
}

/// An OpenAI-style error for anything the model or the request can throw.
struct APIError {
    let status: Int
    let body: OpenAI.ErrorBody

    init(_ error: Error) {
        switch error {
        case ChatError.badRequest(let message):
            self.init(400, message, "invalid_request_error")
        case let error as LanguageModelError:
            switch error {
            case .contextSizeExceeded:
                self.init(400, "The conversation is too long for this model. Start a new chat or shorten it. (\(error.localizedDescription))",
                          "invalid_request_error", code: "context_length_exceeded")
            case .guardrailViolation, .refusal:
                self.init(400, "The model declined this request. (\(error.localizedDescription))", "invalid_request_error", code: "content_filter")
            case .rateLimited:
                self.init(429, error.localizedDescription, "rate_limit_error", code: "rate_limit_exceeded")
            case .unsupportedLanguageOrLocale:
                self.init(400, error.localizedDescription, "invalid_request_error", code: "unsupported_language")
            case .timeout:
                self.init(504, error.localizedDescription, "server_error", code: "timeout")
            default:
                self.init(500, error.localizedDescription, "server_error")
            }
        case let error as LanguageModelSession.GenerationError:
            switch error {
            case .exceededContextWindowSize:
                self.init(400, "The conversation is too long for this model. Start a new chat or shorten it.",
                          "invalid_request_error", code: "context_length_exceeded")
            case .guardrailViolation, .refusal:
                self.init(400, "The model declined this request. (\(error.localizedDescription))", "invalid_request_error", code: "content_filter")
            case .rateLimited, .concurrentRequests:
                self.init(429, error.localizedDescription, "rate_limit_error", code: "rate_limit_exceeded")
            case .unsupportedLanguageOrLocale:
                self.init(400, error.localizedDescription, "invalid_request_error", code: "unsupported_language")
            case .assetsUnavailable:
                self.init(503, "The model isn't ready yet. (\(error.localizedDescription))", "server_error", code: "model_not_ready")
            default:
                self.init(500, error.localizedDescription, "server_error")
            }
        default:
            self.init(500, error.localizedDescription, "server_error")
        }
    }

    init(_ status: Int, _ message: String, _ type: String, code: String? = nil) {
        self.status = status
        self.body = OpenAI.error(message, type: type, code: code)
    }
}
