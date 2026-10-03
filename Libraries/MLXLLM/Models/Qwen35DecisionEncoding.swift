// Copyright © 2026 RunAnywhere AI.
//
// Canonical prompt construction and span annotation for decision checkpoints.
//
// The backbone reads a single prompt containing the state, one field block per
// question, and one option block per candidate answer. The head then pools
// hidden states over the annotated spans; the token layout and span arithmetic
// here must match the checkpoint's reference encoding exactly or the joint
// scores drift even with identical weights.

import Foundation
import MLXLMCommon

// MARK: - Request types

/// Question kind of a decision field.
public enum ClefQuestionKind: String, Codable, Sendable, CaseIterable {
    /// Yes/no field: probability mass on the `true` option.
    case noul
    /// Single-choice field: probabilities over the supplied options.
    case choice
    /// Ordered-level field: probabilities over levels `0 ..< n`.
    case score

    /// Stable index used by the head's type embedding.
    var typeIndex: Int {
        switch self {
        case .noul: 0
        case .choice: 1
        case .score: 2
        }
    }
}

/// One candidate answer of a field.
public struct ClefDecisionOption: Codable, Sendable {
    /// Stable id returned in the answer (`"billing"`, `"true"`, ...).
    public var key: String
    /// Human-readable text injected into the prompt. `nil` omits it.
    public var description: String?

    public init(key: String, description: String? = nil) {
        self.key = key
        self.description = description
    }
}

/// One decision field.
public struct ClefDecisionQuestion: Codable, Sendable {
    /// Stable id returned in the answer.
    public var id: String
    public var kind: ClefQuestionKind
    /// Question text. Defaults to the id when omitted.
    public var instructions: String?
    /// Candidate answers. `choice` may be supplied in any order; `score`
    /// options are level `0 ..< n` in list order; `noul` options are
    /// optional (defaults describe both sides).
    public var options: [ClefDecisionOption]

    public init(
        id: String, kind: ClefQuestionKind, instructions: String? = nil,
        options: [ClefDecisionOption] = []
    ) {
        self.id = id
        self.kind = kind
        self.instructions = instructions
        self.options = options
    }
}

/// A decision request: free-form state plus the fields to decide.
public struct ClefDecisionRequest: Codable, Sendable {
    /// Text or a stringified JSON document the model reads as the state.
    public var state: String
    public var questions: [ClefDecisionQuestion]

    public init(state: String, questions: [ClefDecisionQuestion]) {
        self.state = state
        self.questions = questions
    }
}

// MARK: - Encoding result

/// A field's span annotations within the encoded prompt.
public struct ClefEncodedQuestion: Sendable {
    public let id: String
    public let kind: ClefQuestionKind
    /// Token range of the instruction text.
    public let questionSpan: Range<Int>
    /// Token range of each option's serialized description.
    public let optionSpans: [Range<Int>]
    /// Option ids in prompt order.
    public let optionIds: [String]

    var typeIndex: Int { kind.typeIndex }
}

/// Encoded prompt plus the spans the head reads.
public struct ClefDecisionEncoding: Sendable {
    public let inputIds: [Int]
    public let questions: [ClefEncodedQuestion]
}

/// Encoding failures.
public enum ClefDecisionEncodingError: Error, LocalizedError, Equatable {
    case emptyQuestions
    case missingOptions(String)
    case contextTooLong(String)

    public var errorDescription: String? {
        switch self {
        case .emptyQuestions:
            "A decision request needs at least one question."
        case .missingOptions(let id):
            "Question '\(id)' needs at least one option."
        case .contextTooLong(let detail):
            "Decision prompt does not fit the context window: \(detail)"
        }
    }
}

// MARK: - Encoder

/// Builds the canonical decision prompt and annotates the spans.
public enum ClefDecisionEncoder {
    /// Default context cap for the encoded prompt, in tokens.
    public static let defaultMaxLength = 16_384

    static let systemPrompt =
        "Read the complete state and schema. Decide every field jointly. Each answer "
        + "must be exactly one of that field's allowed options."

    static let noulDefaultTrue = "The proposition is true or the answer is yes."
    static let noulDefaultFalse = "The proposition is false or the answer is no."

    /// The suffix that opens the assistant turn. The two parenthesized tag
    /// literals are built from pieces so the full string reads naturally in
    /// source while remaining byte-identical to the reference encoding.
    private static var suffix: String {
        let thinkOpen = "<" + "think" + ">"
        let thinkClose = "<" + "/think" + ">"
        return "\n<|im_end|>\n<|im_start|>assistant\n"
            + thinkOpen + "\n\n" + thinkClose + "\n\nJOINT SCHEMA DECISIONS:"
    }

    /// Encodes `request` into the exact prompt the checkpoint expects.
    ///
    /// - Parameters:
    ///   - request: State and questions to decide.
    ///   - tokenizer: Tokenizer of the loaded checkpoint.
    ///   - maxLength: Context cap in tokens. The state is truncated from the
    ///     end when the assembled prompt would exceed it.
    public static func encode(
        _ request: ClefDecisionRequest,
        tokenizer: any Tokenizer,
        maxLength: Int = defaultMaxLength
    ) throws -> ClefDecisionEncoding {
        guard !request.questions.isEmpty else {
            throw ClefDecisionEncodingError.emptyQuestions
        }

        func tokenize(_ text: String) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: false)
        }

        // Field blocks are built first; per-field span offsets are captured
        // relative to the schema block and shifted after assembly.
        var schemaIds: [Int] = tokenize("\n\nSCHEMA FIELDS:\n")
        var questions: [ClefEncodedQuestion] = []

        for (fieldIndex, question) in request.questions.enumerated() {
            let resolved = try resolveOptions(of: question)

            schemaIds += tokenize(
                "\nFIELD \(fieldIndex + 1)\nID: \(question.id)\nTYPE: \(question.kind.rawValue)\nINSTRUCTION: ")
            let questionStart = schemaIds.count
            schemaIds += tokenize(question.instructions ?? question.id)
            let questionEnd = schemaIds.count
            schemaIds += tokenize("\nALLOWED OPTIONS:\n")

            var optionSpans: [Range<Int>] = []
            var optionIds: [String] = []
            for (optionIndex, option) in resolved.enumerated() {
                schemaIds += tokenize("OPTION \(optionIndex + 1): ")
                let optionStart = schemaIds.count
                schemaIds += tokenize(renderOption(key: option.key, description: option.description))
                optionSpans.append(optionStart ..< schemaIds.count)
                optionIds.append(option.key)
                schemaIds += tokenize("\n")
            }
            schemaIds += tokenize("END FIELD\n")

            questions.append(
                ClefEncodedQuestion(
                    id: question.id,
                    kind: question.kind,
                    questionSpan: questionStart ..< questionEnd,
                    optionSpans: optionSpans,
                    optionIds: optionIds))
        }

        let prefixIds = tokenize(
            "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n")
        let suffixIds = tokenize(suffix)

        let fixed = prefixIds.count + schemaIds.count + suffixIds.count
        guard fixed <= maxLength else {
            throw ClefDecisionEncodingError.contextTooLong(
                "schema needs \(fixed) tokens before the state; maximum is \(maxLength)")
        }
        var stateIds = tokenize(request.state)
        if stateIds.count > maxLength - fixed {
            stateIds = Array(stateIds.prefix(maxLength - fixed))
        }

        let offset = prefixIds.count + stateIds.count
        let shifted = questions.map { question in
            ClefEncodedQuestion(
                id: question.id,
                kind: question.kind,
                questionSpan: (question.questionSpan.lowerBound + offset)
                    ..< (question.questionSpan.upperBound + offset),
                optionSpans: question.optionSpans.map {
                    ($0.lowerBound + offset) ..< ($0.upperBound + offset)
                },
                optionIds: question.optionIds)
        }

        return ClefDecisionEncoding(
            inputIds: prefixIds + stateIds + schemaIds + suffixIds,
            questions: shifted)
    }

    // MARK: - Option resolution

    /// Applies per-kind rules to the supplied options.
    ///
    /// - `choice`: options sorted by key.
    /// - `score`: options in list order, keys replaced by level indices.
    /// - `noul`: exactly `true` then `false`, with the supplied descriptions
    ///   (when present) overriding the defaults.
    private static func resolveOptions(
        of question: ClefDecisionQuestion
    ) throws -> [ClefDecisionOption] {
        switch question.kind {
        case .choice:
            guard !question.options.isEmpty else {
                throw ClefDecisionEncodingError.missingOptions(question.id)
            }
            return question.options.sorted { $0.key < $1.key }

        case .score:
            guard !question.options.isEmpty else {
                throw ClefDecisionEncodingError.missingOptions(question.id)
            }
            return question.options.enumerated().map { index, option in
                ClefDecisionOption(key: String(index), description: option.description)
            }

        case .noul:
            var trueDescription = question.options.first { $0.key == "true" }?.description
            var falseDescription = question.options.first { $0.key == "false" }?.description
            if trueDescription == nil { trueDescription = noulDefaultTrue }
            if falseDescription == nil { falseDescription = noulDefaultFalse }
            return [
                ClefDecisionOption(key: "true", description: trueDescription),
                ClefDecisionOption(key: "false", description: falseDescription),
            ]
        }
    }

    // MARK: - Rendering

    /// Serializes one option as compact JSON with sorted keys, matching the
    /// reference prompt byte-for-byte.
    private static func renderOption(key: String, description: String?) -> String {
        let encodedKey = jsonString(key)
        if let description {
            return "{\"description\":\(jsonString(description)),\"option_id\":\(encodedKey)}"
        }
        return "{\"option_id\":\(encodedKey)}"
    }

    /// JSON string literal escaping equivalent to a compact, non-ASCII
    /// preserving encoder.
    private static func jsonString(_ value: String) -> String {
        var out = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
        return out
    }
}