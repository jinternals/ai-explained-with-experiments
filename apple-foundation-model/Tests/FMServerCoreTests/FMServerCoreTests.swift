import Foundation
import Testing
@testable import FMServerCore

struct HTTPParsingTests {
    @Test func parsesCompleteRequest() throws {
        let raw = "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: a\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
        guard case .complete(let request) = HTTPServer.parse(Data(raw.utf8)) else {
            Issue.record("expected a complete request"); return
        }
        #expect(request.method == "POST")
        #expect(request.path == "/v1/chat/completions")
        #expect(request.headers["content-type"] == "application/json")
        #expect(request.body == Data("{}".utf8))
    }

    @Test func waitsForTheWholeBody() {
        let raw = "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 10\r\n\r\n{\"a\""
        guard case .needMore = HTTPServer.parse(Data(raw.utf8)) else {
            Issue.record("expected needMore"); return
        }
    }

    @Test func rejectsChunkedBodies() {
        let raw = "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        guard case .invalid(400) = HTTPServer.parse(Data(raw.utf8)) else {
            Issue.record("expected 400"); return
        }
    }
}

struct ChatMappingTests {
    private func request(_ json: String) throws -> OpenAI.ChatRequest {
        try JSONDecoder().decode(OpenAI.ChatRequest.self, from: Data(json.utf8))
    }

    @Test func splitsHistoryAndPrompt() throws {
        let chat = try PreparedChat.make(from: request("""
            {"model":"apple-on-device","max_tokens":50,"temperature":5,"messages":[
              {"role":"system","content":"Be brief."},
              {"role":"user","content":"Hi"},
              {"role":"assistant","content":"Hello!"},
              {"role":"user","content":[{"type":"text","text":"What did I say?"}]}]}
            """), model: .onDevice)
        #expect(chat.prompt == "What did I say?")
        #expect(chat.entries.count == 3) // instructions, earlier prompt, earlier response
        #expect(chat.options.maximumResponseTokens == 50)
        #expect(chat.options.temperature == 2) // clamped to OpenAI's range
    }

    @Test func rejectsImages() throws {
        let body = try request("""
            {"messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:,"}}]}]}
            """)
        #expect(throws: ChatError.self) { try PreparedChat.make(from: body, model: .onDevice) }
    }

    @Test func requiresUserMessageLast() throws {
        let body = try request("""
            {"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}
            """)
        #expect(throws: ChatError.self) { try PreparedChat.make(from: body, model: .onDevice) }
    }
}
