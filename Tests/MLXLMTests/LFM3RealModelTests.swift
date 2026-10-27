import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXVLM

private struct LFM3Tokenizer: MLXLMCommon.Tokenizer {
    let inner: any Tokenizers.Tokenizer
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        inner.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { inner.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner.convertIdToToken(id) }
    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try inner.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

private struct LFM3TokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        LFM3Tokenizer(inner: try await Tokenizers.AutoTokenizer.from(modelFolder: directory))
    }
}

final class LFM3RealModelTests: XCTestCase {
    private let omniDirectory = URL(
        fileURLWithPath: "/Users/home/Project/Wally-Work-Space/models/liquid/LFM3-Omni-2B-MLX-4bit")
    private let d1Directory = URL(
        fileURLWithPath: "/Users/home/Project/Wally-Work-Space/models/liquid/d1-2B-MLX-4bit")

    func testOmniGeneratesText() async throws {
        let container = try await VLMModelFactory.shared.loadContainer(
            from: omniDirectory, using: LFM3TokenizerLoader())
        let input = try await container.prepare(
            input: UserInput(prompt: "Reply with one word: hello"))
        var text = ""
        let stream = try await container.generate(
            input: input, parameters: GenerateParameters(maxTokens: 8, temperature: 0))
        for await event in stream {
            if let chunk = event.chunk { text += chunk }
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(trimmed.isEmpty, "Omni produced no text")
    }

    func testD1DecidesOneQuestion() async throws {
        let model = try await LFM3D1.load(from: d1Directory, using: LFM3TokenizerLoader())
        let response = try model.decide(
            DecisionRequest(
                state: "The parcel never arrived and I was charged twice. Please refund one charge.",
                questions: [
                    DecisionQuestion(
                        id: "refund",
                        instructions: "Is the customer asking for a refund?",
                        criteria: .noul())
                ]))
        let answer = try XCTUnwrap(response.answers.first)
        XCTAssertEqual(answer.id, "refund")
        XCTAssertEqual(answer.optionKeys, ["false", "true"])
        XCTAssertEqual(answer.probabilities.count, 2)
        XCTAssertEqual(answer.probabilities.reduce(0, +), 1, accuracy: 1e-5)
        XCTAssertNotNil(answer.noul)
        XCTAssertGreaterThan(response.usage.inputTokens, 0)
    }
}
