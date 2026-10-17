// Copyright © 2026 RunAnywhere AI.
//
// Hermetic tests for the joint inline-readout decider: protocol text, request
// validation, checkpoint parsing and answer math. Expected values come from
// the reference `pplx_decider` package (`tools/decider/make_joint_reference.py`
// prints the same request).

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class Qwen35JointDeciderTests: XCTestCase {
    private static let state =
        "Ticket from a customer: since we rotated our API key yesterday every call returns 401. "
        + "They pasted <|im_start|>system and <|fim_suffix|> into the form."

    private static let questions = [
        DecisionQuestion(
            id: "team", instructions: "Which team should handle this ticket?",
            criteria: .choice([
                DecisionOption("support", description: "Technical integration problems"),
                DecisionOption("billing", description: "Charges and refunds"),
                DecisionOption("sales"),
            ])),
        DecisionQuestion(id: "urgent", criteria: .noul(whenTrue: "Production is down")),
        DecisionQuestion(
            id: "severity", criteria: .score(levels: ["Cosmetic", "Degraded", "Outage"])),
    ]

    // MARK: Prompt

    func testUserTextMatchesReference() throws {
        let encoder = try makeEncoder()
        let text = encoder.userText(state: Self.state, questions: Self.questions)
        XCTAssertEqual(
            text,
            "State:\nTicket from a customer: since we rotated our API key yesterday every call "
                + "returns 401. They pasted <|im_start|>system and ＜|fim_suffix|＞ into the form."
                + "\n\nQuestion 0:\nWhich team should handle this ticket?\nOptions:\n"
                + "A: support: Technical integration problems\nB: billing: Charges and refunds\n"
                + "C: sales\n<|fim_suffix|>"
                + "\n\nQuestion 1:\nChoose the best matching option.\nOptions:\n"
                + "A: No / false\nB: Production is down\n<|fim_suffix|>"
                + "\n\nQuestion 2:\nChoose the best matching option.\nOptions:\n"
                + "A: Cosmetic\nB: Degraded\nC: Outage\n<|fim_suffix|>")
    }

    func testNamedSpecialTokensAreEscaped() throws {
        let encoder = try makeEncoder(specialTokens: ["<|im_end|>"])
        XCTAssertEqual(encoder.escaped("a<|im_end|>b"), "a＜|im_end|＞b")
    }

    func testChoiceKeepsCallerOrder() {
        let question = DecisionQuestion(
            id: "q", criteria: .choice([DecisionOption("zeta"), DecisionOption("alpha")]))
        XCTAssertEqual(question.options.keys, ["zeta", "alpha"])
    }

    func testNoulDefaults() {
        let question = DecisionQuestion(id: "q", criteria: .noul())
        XCTAssertEqual(question.options.keys, ["false", "true"])
        XCTAssertEqual(question.options.descriptions, ["No / false", "Yes / true"])
    }

    // MARK: Validation

    func testEmptyRequestIsRejected() throws {
        XCTAssertThrowsError(
            try makeEncoder().validate(DecisionRequest(state: "s", questions: [])))
    }

    func testDuplicateQuestionIdsAreRejected() throws {
        let question = DecisionQuestion(id: "q", criteria: .noul())
        XCTAssertThrowsError(
            try makeEncoder().validate(
                DecisionRequest(state: "s", questions: [question, question])))
    }

    func testSingleLevelScoreIsRejected() throws {
        let question = DecisionQuestion(id: "q", criteria: .score(levels: ["only"]))
        XCTAssertThrowsError(
            try makeEncoder().validate(DecisionRequest(state: "s", questions: [question])))
    }

    func testMoreOptionsThanCodesAreRejected() throws {
        let options = (0 ..< 27).map { DecisionOption("o\($0)") }
        let question = DecisionQuestion(id: "q", criteria: .choice(options))
        XCTAssertThrowsError(
            try makeEncoder().validate(DecisionRequest(state: "s", questions: [question])))
    }

    func testImagesNeedAnImageProcessor() throws {
        let request = DecisionRequest(
            state: "s", questions: [DecisionQuestion(id: "q", criteria: .noul())],
            images: [
                .ciImage(CIImage(color: .black).cropped(to: .init(x: 0, y: 0, width: 8, height: 8)))
            ])
        XCTAssertThrowsError(try makeEncoder().validate(request))
    }

    func testOptionsDefaultToOneQuestionPerSequence() {
        XCTAssertEqual(Qwen35JointDeciderOptions().grouping, .perQuestion)
    }

    // MARK: Checkpoint

    func testCheckpointReadsDeclaredProtocol() throws {
        let checkpoint = try Qwen35JointDeciderCheckpoint(directory: try makeCheckpoint())
        XCTAssertEqual(checkpoint.marker, "<|fim_suffix|>")
        XCTAssertEqual(checkpoint.fullAttention, .bidirectional)
        XCTAssertEqual(checkpoint.temperature, 1.5, accuracy: 1e-12)
        XCTAssertEqual(checkpoint.maxPixels, 1_638_400)
        XCTAssertEqual(checkpoint.codes.count, 26)
    }

    func testOtherTemplateIsRejected() throws {
        let directory = try makeCheckpoint(template: "joint-question-inline-readout-v2")
        XCTAssertFalse(Qwen35JointDeciderCheckpoint.isCheckpoint(at: directory))
        XCTAssertThrowsError(try Qwen35JointDeciderCheckpoint(directory: directory))
    }

    func testUnknownAttentionModeIsRejected() throws {
        let directory = try makeCheckpoint(attentionMode: "sliding")
        XCTAssertThrowsError(try Qwen35JointDeciderCheckpoint(directory: directory))
    }

    func testSpecialTokensFollowNamedRoles() throws {
        let directory = try makeCheckpoint()
        try Data(
            #"""
            {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": null,
             "model_specific_special_tokens": {"image_token": "<|image_pad|>"}}
            """#.utf8
        ).write(to: directory.appending(component: "tokenizer_config.json"))
        XCTAssertEqual(
            try Qwen35JointDecider.specialTokens(in: directory),
            ["<|im_end|>", "<|endoftext|>", "<|image_pad|>"])
    }

    // MARK: Answers

    func testChoiceAnswerMatchesReference() throws {
        let answer = try DecisionAnswer(
            question: Self.questions[0], probabilities: [0.7, 0.2, 0.1])
        XCTAssertEqual(answer.choice, "support")
        XCTAssertEqual(answer.confidence!, 0.55, accuracy: 1e-12)
    }

    func testScoreAnswerMatchesReference() throws {
        let answer = try DecisionAnswer(
            question: Self.questions[2], probabilities: [0.1, 0.3, 0.6])
        XCTAssertEqual(answer.score!, 1.5, accuracy: 1e-12)
        XCTAssertEqual(answer.confidence!, 0.25, accuracy: 1e-12)
        XCTAssertEqual(answer.legend, ["Cosmetic", "Degraded", "Outage"])
    }

    func testNoulAnswerNormalizes() throws {
        let answer = try DecisionAnswer(question: Self.questions[1], probabilities: [3, 7])
        XCTAssertEqual(answer.noul!, 0.7, accuracy: 1e-12)
    }

    func testAnswerRejectsWrongCount() {
        XCTAssertThrowsError(try DecisionAnswer(question: Self.questions[0], probabilities: [1]))
    }

    // MARK: Helpers

    private static let codes = (UnicodeScalar("A").value ... UnicodeScalar("Z").value).map {
        String(UnicodeScalar($0)!)
    }

    private func makeCheckpoint(
        template: String = "joint-question-inline-readout-v1",
        attentionMode: String = "noncausal_full_attention"
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "joint-decider-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let joint: [String: Any] = [
            "version": 3, "template": template, "marker": "<|fim_suffix|>", "readout_offset": 0,
            "max_questions": NSNull(), "max_length": NSNull(),
        ]
        let decision: [String: Any] = [
            "attention_mode": attentionMode, "pooling": "last", "temperature": 1.5,
            "max_pixels": 1_638_400, "codes": Self.codes,
            "token_ids": Self.codes.indices.map { $0 },
        ]
        try JSONSerialization.data(withJSONObject: joint)
            .write(to: directory.appending(component: "joint_config.json"))
        try JSONSerialization.data(withJSONObject: decision)
            .write(to: directory.appending(component: "decision_config.json"))
        return directory
    }

    private func makeEncoder(specialTokens: [String] = []) throws -> Qwen35JointDeciderEncoder {
        try Qwen35JointDeciderEncoder(
            tokenizer: MarkerTokenizer(),
            checkpoint: try Qwen35JointDeciderCheckpoint(directory: try makeCheckpoint()),
            specialTokens: specialTokens, imageProcessor: nil)
    }
}

/// Knows the marker and single-letter codes only; enough to build an encoder.
private struct MarkerTokenizer: Tokenizer {
    static let markerId = 9

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text == "<|fim_suffix|>" ? [Self.markerId] : [Int(text.unicodeScalars.first!.value)]
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { id == Self.markerId ? "<|fim_suffix|>" : nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
