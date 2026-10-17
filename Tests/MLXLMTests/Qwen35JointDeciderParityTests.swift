// Copyright © 2026 RunAnywhere AI.
//
// Parity of the joint inline-readout decider against the reference PyTorch
// package. Local only: set `MLX_JOINT_DECIDER_CHECKPOINT` to a checkpoint
// directory and `MLX_JOINT_DECIDER_REFERENCE` to the JSON that
// `tools/decider/make_joint_reference.py` wrote for it. Works for the
// published `pplx-decider-flash-v1.2` and for a tiny random checkpoint of the
// same layout.

import CoreImage
import Foundation
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXVLM

/// `MLXLMCommon.Tokenizer` over an upstream-initialized tokenizer.
private struct UpstreamTokenizer: MLXLMCommon.Tokenizer {
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

/// The reference file's shape (see `make_joint_reference.py`).
private struct Reference: Decodable {
    struct Sequence: Decodable {
        let keys: [String]
        let inputIds: [Int]
        let readPositions: [Int]
        let logits: [[Double]]

        enum CodingKeys: String, CodingKey {
            case keys, logits
            case inputIds = "input_ids"
            case readPositions = "read_positions"
        }
    }

    struct Answer: Decodable {
        let type: String
        let choice: String?
        let noul: Double?
        let score: Double?
        let probabilities: [String: Double]?
    }

    struct Question: Decodable {
        let id: String
        let type: String
        let instructions: String?
        let options: [[String?]]?
        let levels: [String]?
        let whenTrue: String?
        let whenFalse: String?

        enum CodingKeys: String, CodingKey {
            case id, type, instructions, options, levels
            case whenTrue = "when_true"
            case whenFalse = "when_false"
        }
    }

    struct Request: Decodable {
        let state: String
        let images: [String]?
    }

    struct Case: Decodable {
        let name: String
        let request: Request
        let orderedQuestions: [Question]
        let sequences: [Sequence]
        let response: Response

        enum CodingKeys: String, CodingKey {
            case name, request, sequences, response
            case orderedQuestions = "ordered_questions"
        }
    }

    struct Response: Decodable {
        let answers: [String: Answer]
    }

    let cases: [Case]
}

final class Qwen35JointDeciderParityTests: XCTestCase {
    /// Bound on raw readout-logit drift against the float32 reference. The
    /// published BF16 4B drifts up to about 0.11; a causal full-attention bug
    /// moves the same logits by about 0.34, so 0.15 still separates the two.
    private var logitTolerance: Double {
        Double(ProcessInfo.processInfo.environment["MLX_JOINT_DECIDER_LOGIT_TOLERANCE"] ?? "")
            ?? 0.15
    }

    private func fixtures() throws -> (checkpoint: URL, reference: URL) {
        let environment = ProcessInfo.processInfo.environment
        guard let checkpoint = environment["MLX_JOINT_DECIDER_CHECKPOINT"],
            let reference = environment["MLX_JOINT_DECIDER_REFERENCE"]
        else {
            throw XCTSkip("set MLX_JOINT_DECIDER_CHECKPOINT and MLX_JOINT_DECIDER_REFERENCE")
        }
        return (URL(fileURLWithPath: checkpoint), URL(fileURLWithPath: reference))
    }

    func testMatchesReference() async throws {
        let (checkpointURL, referenceURL) = try fixtures()
        let reference = try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: referenceURL))
        let decider = try await Qwen35JointDecider.load(
            from: checkpointURL, using: UpstreamTokenizerLoader())

        for testCase in reference.cases {
            let request = try makeRequest(
                testCase, imageRoot: referenceURL.deletingLastPathComponent())
            let indices = Array(request.questions.indices)
            let processing = UserInput.Processing(maxPixels: decider.checkpoint.maxPixels)
            let chunks = try decider.encoder.plan(
                request, groups: [indices], maxLength: nil, maxQuestions: nil,
                processing: processing)
            XCTAssertEqual(chunks.count, testCase.sequences.count, testCase.name)

            for (chunk, expected) in zip(chunks, testCase.sequences) {
                let ids = chunk.input.text.tokens.asArray(Int32.self).map(Int.init)
                XCTAssertEqual(ids, expected.inputIds, "\(testCase.name): token ids")
                XCTAssertEqual(chunk.readPositions, expected.readPositions, testCase.name)

                let logits = try decider.readoutLogits(chunk)
                for (row, key) in expected.keys.enumerated() {
                    let count = expected.logits[row].count
                    let drift =
                        zip(logits[row].prefix(count), expected.logits[row])
                        .map { abs($0 - $1) }.max() ?? .infinity
                    XCTAssertLessThan(drift, logitTolerance, "\(testCase.name)/\(key): logit drift")
                }
            }

            // The reference `decide` runs every request jointly.
            let response = try decider.decide(request, options: .init(grouping: .joint))
            for answer in response.answers {
                let expected = try XCTUnwrap(testCase.response.answers[answer.id])
                XCTAssertEqual(answer.kind.rawValue, expected.type)
                if let choice = expected.choice {
                    XCTAssertEqual(answer.choice, choice, "\(testCase.name)/\(answer.id)")
                }
                for (key, probability) in expected.probabilities ?? [:] {
                    let index = try XCTUnwrap(answer.optionKeys.firstIndex(of: key))
                    XCTAssertEqual(
                        answer.probabilities[index], probability, accuracy: 0.02,
                        "\(testCase.name)/\(answer.id)/\(key)")
                }
                if let noul = expected.noul {
                    XCTAssertEqual(
                        answer.noul!, noul, accuracy: 0.02, "\(testCase.name)/\(answer.id)")
                }
            }
        }
    }

    private func makeRequest(_ testCase: Reference.Case, imageRoot: URL) throws -> DecisionRequest {
        let questions = try testCase.orderedQuestions.map { question -> DecisionQuestion in
            let criteria: DecisionCriteria
            switch question.type {
            case "choice":
                let options = try XCTUnwrap(question.options).map { pair in
                    DecisionOption(pair[0]!, description: pair[1])
                }
                criteria = .choice(options)
            case "noul":
                criteria = .noul(whenTrue: question.whenTrue, whenFalse: question.whenFalse)
            case "score":
                criteria = .score(levels: try XCTUnwrap(question.levels))
            default:
                throw DecisionError.invalidRequest("unsupported question type \(question.type)")
            }
            return DecisionQuestion(
                id: question.id, instructions: question.instructions, criteria: criteria)
        }
        let images = (testCase.request.images ?? []).map {
            UserInput.Image.url(imageRoot.appending(component: $0))
        }
        return DecisionRequest(state: testCase.request.state, questions: questions, images: images)
    }
}
