// Copyright © 2026 RunAnywhere AI.
//
// Weights tests for the decider. Local only: they load the 27B checkpoint
// from the shared model cache and never run in CI (guarded by `modelURL`).

import Foundation
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXLLM

/// `MLXLMCommon.Tokenizer` over an upstream-initialized tokenizer.
private struct UpstreamTokenizer: MLXLMCommon.Tokenizer {
    let inner: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        inner.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? {
        inner.convertTokenToId(token)
    }
    func convertIdToToken(_ id: Int) -> String? {
        inner.convertIdToToken(id)
    }
    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try inner.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: additionalContext)
    }
}

private struct UpstreamTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        UpstreamTokenizer(inner: try await Tokenizers.AutoTokenizer.from(modelFolder: directory))
    }
}

final class Qwen35DeciderRealModelTests: XCTestCase {
    /// Local checkout of `mlx-community/pplx-decider-v1.1-27b-4bit`, or nil off-machine.
    private var modelURL: URL? {
        let url = URL(
            fileURLWithPath:
                "\(NSHomeDirectory())/.local/share/runanywhere/Models/MLX/mlx-pplx-decider-27b-4bit")
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { return nil }
        return url
    }

    private func loadModel() async throws -> Qwen35DeciderModel {
        guard let directory = modelURL else {
            throw XCTSkip("decider weights not on this machine")
        }
        return try await Qwen35DeciderModel.load(
            from: directory, using: UpstreamTokenizerLoader())
    }

    func testLoadRealWeights() async throws {
        let model = try await loadModel()
        XCTAssertEqual(model.decider.codes.count, 255)
        XCTAssertEqual(model.decider.temperature, 1.0087417, accuracy: 1e-6)
        let headWeight = try XCTUnwrap(model.backbone.languageModel.lmHead?.weight)
        XCTAssertEqual(headWeight.shape, [255, 5120])
    }

    /// Golden from `decider_mlx.py` on the same local checkout: 89 prompt
    /// tokens, `false` 0.00743, `true` 0.99257.
    func testDecideMatchesReference() async throws {
        let model = try await loadModel()
        let request = ClefDecisionRequest(
            state: "I was charged twice.",
            questions: [
                ClefDecisionQuestion(
                    id: "answer", kind: .noul, instructions: "Wants a refund?")
            ])
        let answers = try model.decide(request).answers
        XCTAssertEqual(answers.count, 1)
        let probabilities = try XCTUnwrap(answers.first?.probabilities)
        XCTAssertEqual(probabilities["false"] ?? -1, 0.00743, accuracy: 1e-3)
        XCTAssertEqual(probabilities["true"] ?? -1, 0.99257, accuracy: 1e-3)
        XCTAssertEqual(answers.first?.probabilityOfTrue ?? -1, 0.99257, accuracy: 1e-3)
    }
}
