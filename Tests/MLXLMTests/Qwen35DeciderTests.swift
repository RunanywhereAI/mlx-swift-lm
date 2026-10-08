// Copyright © 2026 RunAnywhere AI.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class Qwen35DeciderTests: XCTestCase {
    func testNoulDefaults() throws {
        let question = ClefDecisionQuestion(id: "spam", kind: .noul)
        let (keys, descriptions) = try Qwen35DeciderEncoder.optionLists(of: question)
        XCTAssertEqual(keys, ["false", "true"])
        XCTAssertEqual(descriptions, ["No / false", "Yes / true"])
    }

    func testNoulOverrides() throws {
        let question = ClefDecisionQuestion(
            id: "refund", kind: .noul,
            options: [
                ClefDecisionOption(key: "true", description: "Wants a refund?"),
                ClefDecisionOption(key: "false", description: "Something else"),
            ])
        let (keys, descriptions) = try Qwen35DeciderEncoder.optionLists(of: question)
        XCTAssertEqual(keys, ["false", "true"])
        XCTAssertEqual(descriptions, ["Something else", "Wants a refund?"])
    }

    func testChoiceSortsByKey() throws {
        let question = ClefDecisionQuestion(
            id: "route", kind: .choice,
            options: [
                ClefDecisionOption(key: "billing", description: "Money"),
                ClefDecisionOption(key: "abuse", description: "Harm"),
            ])
        let (keys, descriptions) = try Qwen35DeciderEncoder.optionLists(of: question)
        XCTAssertEqual(keys, ["abuse", "billing"])
        XCTAssertEqual(descriptions, ["abuse: Harm", "billing: Money"])
    }

    func testChoiceRendersBareKey() throws {
        let question = ClefDecisionQuestion(
            id: "route", kind: .choice,
            options: [ClefDecisionOption(key: "billing")])
        let (keys, descriptions) = try Qwen35DeciderEncoder.optionLists(of: question)
        XCTAssertEqual(keys, ["billing"])
        XCTAssertEqual(descriptions, ["billing"])
    }

    func testScoreUsesLevelIndices() throws {
        let question = ClefDecisionQuestion(
            id: "urgency", kind: .score,
            options: [
                ClefDecisionOption(key: "low", description: "Whenever"),
                ClefDecisionOption(key: "high", description: "Now"),
            ])
        let (keys, descriptions) = try Qwen35DeciderEncoder.optionLists(of: question)
        XCTAssertEqual(keys, ["0", "1"])
        XCTAssertEqual(descriptions, ["Whenever", "Now"])
    }

    func testScoreNeedsTwoLevels() {
        let question = ClefDecisionQuestion(
            id: "urgency", kind: .score,
            options: [ClefDecisionOption(key: "only")])
        XCTAssertThrowsError(try Qwen35DeciderEncoder.optionLists(of: question))
    }

    func testEmptyRequestFails() {
        let request = ClefDecisionRequest(state: "x", questions: [])
        XCTAssertThrowsError(
            try Qwen35DeciderEncoder.encode(
                request, codes: ["A"], tokenizer: FailingTokenizer(), maxLength: 128))
    }
}

/// A tokenizer that traps if tokenization is attempted.
private struct FailingTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        fatalError("must not tokenize")
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
