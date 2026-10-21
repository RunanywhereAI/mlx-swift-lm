// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Loads the real `RecursiveSI/flint-int4` checkpoint and proves it generates sensible code.
///
/// Opt-in: `MLX_FLINT_MODEL=~/Downloads/flint-int4 swift test --filter FlintRealModelTests`.
/// The prompt is pre-tokenized with the checkpoint's own `tokenizer.json` because this
/// target carries no real tokenizer; output is decoded from the same file's byte-level vocab.
final class FlintRealModelTests: XCTestCase {
    /// `<|begin_of_text|>def add(a, b):\n    return`
    private let prompt: [Int32] = [150000, 1182, 836, 3444, 11, 286, 1080, 279, 597]
    private let newTokens = 24

    private func modelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MLX_FLINT_MODEL"], !path.isEmpty
        else {
            throw XCTSkip("set MLX_FLINT_MODEL to a local RecursiveSI/flint-int4 directory")
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    private func loadFlint(from directory: URL) async throws -> any LanguageModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        XCTAssertEqual(base.modelType, "flint")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    func testFlintInt4GeneratesCode() async throws {
        let directory = try modelDirectory()
        let model = try await loadFlint(from: directory)
        let decoder = try ByteLevelDecoder(
            tokenizer: directory.appendingPathComponent("tokenizer.json"))

        let cache = try model.newCache(parameters: nil)
        var output = model(
            LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: cache, state: nil)
        let promptLogits = output.logits[0..., -1, 0...]
        XCTAssertFalse(
            any(isNaN(promptLogits)).item(Bool.self), "prompt logits contain NaN")

        var generated: [Int] = []
        for _ in 0 ..< newTokens {
            let token = argMax(output.logits[0..., -1, 0...], axis: -1).item(Int.self)
            generated.append(token)
            output = model(
                LMInput.Text(tokens: MLXArray([Int32(token)])[.newAxis]), cache: cache,
                state: output.state)
        }

        let text = decoder.decode(generated)
        print("Flint INT4 continuation: \(text.debugDescription)")
        let firstLine = text.split(separator: "\n", omittingEmptySubsequences: false).first ?? ""
        let answer = firstLine.replacingOccurrences(of: " ", with: "")
        XCTAssertTrue(
            answer == "a+b" || answer == "b+a",
            "expected the model to complete `return` with `a + b`, got \(text.debugDescription)")

        // Cached decode must agree with a single uncached pass over the same tokens.
        let all = prompt + generated.prefix(4).map(Int32.init)
        let uncached = model(
            LMInput.Text(tokens: MLXArray(Array(all.dropLast()))[.newAxis]), cache: nil,
            state: nil)
        let next = argMax(uncached.logits[0..., -1, 0...], axis: -1).item(Int.self)
        XCTAssertEqual(next, generated[3], "uncached forward disagrees with cached decode")
    }
}

/// Decodes ids through a byte-level BPE `tokenizer.json` (GPT-2 byte-to-unicode mapping).
private struct ByteLevelDecoder {
    private let tokens: [Int: String]
    private let unicodeToByte: [Character: UInt8]

    init(tokenizer url: URL) throws {
        struct File: Decodable {
            struct Model: Decodable { let vocab: [String: Int] }
            struct Added: Decodable {
                let id: Int
                let content: String
            }
            let model: Model
            let addedTokens: [Added]
            enum CodingKeys: String, CodingKey {
                case model
                case addedTokens = "added_tokens"
            }
        }
        let file = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        var tokens = Dictionary(uniqueKeysWithValues: file.model.vocab.map { ($1, $0) })
        for added in file.addedTokens { tokens[added.id] = added.content }
        self.tokens = tokens

        var printable = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var codepoints = printable
        var extra = 0
        for byte in 0 ... 255 where !printable.contains(byte) {
            printable.append(byte)
            codepoints.append(256 + extra)
            extra += 1
        }
        var map = [Character: UInt8]()
        for (byte, codepoint) in zip(printable, codepoints) {
            map[Character(Unicode.Scalar(UInt32(codepoint))!)] = UInt8(byte)
        }
        self.unicodeToByte = map
    }

    func decode(_ ids: [Int]) -> String {
        let bytes = ids.flatMap { id -> [UInt8] in
            guard let token = tokens[id] else { return Array("<\(id)>".utf8) }
            return token.compactMap { unicodeToByte[$0] }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
