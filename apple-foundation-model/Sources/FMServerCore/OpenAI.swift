import Foundation

/// The parts of the OpenAI Chat Completions API that chat apps such as Open WebUI use.
public enum OpenAI {
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return encoder
    }()

    // MARK: Requests

    public struct ChatRequest: Decodable, Sendable {
        public var model: String?
        public var messages: [Message]
        public var temperature: Double?
        public var maxTokens: Int?
        public var maxCompletionTokens: Int?
        public var stream: Bool?
        public var streamOptions: StreamOptions?

        enum CodingKeys: String, CodingKey {
            case model, messages, temperature, stream
            case maxTokens = "max_tokens"
            case maxCompletionTokens = "max_completion_tokens"
            case streamOptions = "stream_options"
        }
    }

    public struct StreamOptions: Decodable, Sendable {
        public var includeUsage: Bool?
        enum CodingKeys: String, CodingKey { case includeUsage = "include_usage" }
    }

    public struct Message: Decodable, Sendable {
        public var role: String
        public var content: Content?

        public init(role: String, content: Content?) {
            self.role = role
            self.content = content
        }
    }

    /// Message content is either a string or a list of parts (text, images, ...).
    public enum Content: Decodable, Sendable {
        case text(String)
        case parts([Part])

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
            } else {
                self = .parts(try container.decode([Part].self))
            }
        }

        /// All text parts joined; `hasNonText` is true if the message also had images or other parts.
        public var text: (text: String, hasNonText: Bool) {
            switch self {
            case .text(let text): return (text, false)
            case .parts(let parts):
                let texts = parts.compactMap { $0.type == "text" ? $0.text : nil }
                return (texts.joined(separator: "\n"), parts.contains { $0.type != "text" })
            }
        }
    }

    public struct Part: Decodable, Sendable {
        public var type: String
        public var text: String?
    }

    // MARK: Responses

    public struct ModelList: Encodable {
        var object = "list"
        var data: [Model]
    }

    public struct Model: Encodable {
        var id: String
        var object = "model"
        var created: Int
        var ownedBy = "apple"
    }

    public struct Usage: Encodable, Sendable {
        var promptTokens: Int
        var completionTokens: Int
        var totalTokens: Int
    }

    public struct Completion: Encodable {
        var id: String
        var object = "chat.completion"
        var created: Int
        var model: String
        var choices: [Choice]
        var usage: Usage?

        struct Choice: Encodable {
            var index = 0
            var message: ResponseMessage
            var finishReason: String
        }
    }

    public struct ResponseMessage: Encodable {
        var role = "assistant"
        var content: String
    }

    public struct Chunk: Encodable {
        var id: String
        var object = "chat.completion.chunk"
        var created: Int
        var model: String
        var choices: [Choice]
        var usage: Usage?

        struct Choice: Encodable {
            var index = 0
            var delta: Delta
            var finishReason: String?
        }

        struct Delta: Encodable {
            var role: String?
            var content: String?
        }
    }

    public struct ErrorBody: Encodable {
        var error: Detail

        struct Detail: Encodable {
            var message: String
            var type: String
            var code: String?
        }
    }

    static func error(_ message: String, type: String, code: String? = nil) -> ErrorBody {
        ErrorBody(error: .init(message: message, type: type, code: code))
    }
}
