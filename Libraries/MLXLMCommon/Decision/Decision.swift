// Copyright © 2026 RunAnywhere AI.
//
// Request and answer types for decision models: one shared state, several
// named questions, a probability per option. The shapes follow the Decisions
// API (`choice`, `noul`, `score`) so a checkpoint-specific runtime only has to
// turn a request into option probabilities.

import Foundation

/// One option of a `choice` question.
public struct DecisionOption: Sendable, Equatable {
    /// Stable identifier returned in the answer.
    public let key: String
    /// Text shown to the model. `nil` shows the key alone.
    public let description: String?

    public init(_ key: String, description: String? = nil) {
        self.key = key
        self.description = description
    }
}

/// What a question asks for, with the criteria the model reads.
public enum DecisionCriteria: Sendable, Equatable {
    /// Pick one option. Keys must be unique.
    case choice([DecisionOption])
    /// Yes/no. Either description may be `nil` for the default wording.
    case noul(whenTrue: String? = nil, whenFalse: String? = nil)
    /// Ordered scale; `levels[i]` describes level `i`.
    case score(levels: [String])
}

public enum DecisionKind: String, Sendable, Codable {
    case choice
    case noul
    case score
}

public struct DecisionQuestion: Sendable, Equatable {
    /// Identifier echoed on the answer. Unique within a request.
    public let id: String
    /// Instructions read by the model. `nil` uses the checkpoint's default.
    public let instructions: String?
    public let criteria: DecisionCriteria

    public init(id: String, instructions: String? = nil, criteria: DecisionCriteria) {
        self.id = id
        self.instructions = instructions
        self.criteria = criteria
    }

    public var kind: DecisionKind {
        switch criteria {
        case .choice: .choice
        case .noul: .noul
        case .score: .score
        }
    }

    /// Option keys as returned in answers, and the descriptions shown to the
    /// model, in prompt order.
    public var options: (keys: [String], descriptions: [String]) {
        switch criteria {
        case .choice(let options):
            return (
                options.map(\.key),
                options.map { option in
                    option.description.map { "\(option.key): \($0)" } ?? option.key
                }
            )
        case .score(let levels):
            return (levels.indices.map(String.init), levels)
        case .noul(let whenTrue, let whenFalse):
            return (["false", "true"], [whenFalse ?? "No / false", whenTrue ?? "Yes / true"])
        }
    }
}

/// One decision context: every question is answered from the same state and images.
public struct DecisionRequest {
    public let state: String
    public let questions: [DecisionQuestion]
    public let images: [UserInput.Image]

    public init(state: String, questions: [DecisionQuestion], images: [UserInput.Image] = []) {
        self.state = state
        self.questions = questions
        self.images = images
    }
}

public struct DecisionAnswer: Sendable {
    public let id: String
    public let kind: DecisionKind
    /// Option keys in prompt order.
    public let optionKeys: [String]
    /// Probability per option, parallel to `optionKeys`. Sums to 1.
    public let probabilities: [Double]
    /// `choice`: the most probable key.
    public let choice: String?
    /// `noul`: probability of `true`.
    public let noul: Double?
    /// `score`: probability-weighted level.
    public let score: Double?
    /// `choice` and `score`: calibrated confidence in [0, 1].
    public let confidence: Double?
    /// `score`: level descriptions, parallel to `optionKeys`.
    public let legend: [String]?

    /// Builds the answer from the question's option probabilities.
    ///
    /// Confidence is the reference definition: for a choice, how far the top
    /// probability sits above uniform; for a score, one minus the expected
    /// distance from the top level relative to a uniform spread.
    public init(question: DecisionQuestion, probabilities raw: [Double]) throws {
        let (keys, descriptions) = question.options
        let total = raw.reduce(0, +)
        guard raw.count == keys.count, !raw.isEmpty,
            raw.allSatisfy({ $0.isFinite && $0 >= 0 }), total > 0
        else {
            throw DecisionError.invalidOutput(
                "question '\(question.id)': expected \(keys.count) finite probabilities")
        }
        let values = raw.map { $0 / total }
        let best = values.indices.max { values[$0] < values[$1] }!
        let count = Double(values.count)

        self.id = question.id
        self.kind = question.kind
        self.optionKeys = keys
        self.probabilities = values

        switch question.kind {
        case .noul:
            choice = nil
            noul = values[1]
            score = nil
            confidence = nil
            legend = nil
        case .choice:
            choice = keys[best]
            noul = nil
            score = nil
            let lift = values.count == 1 ? 1 : (values[best] - 1 / count) / (1 - 1 / count)
            confidence = min(1, max(0, lift))
            legend = nil
        case .score:
            choice = nil
            noul = nil
            score = values.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
            let distance = values.enumerated().reduce(0) {
                $0 + $1.element * Double(abs($1.offset - best))
            }
            let midpoint = (count - 1) / 2
            let baseline = values.indices.reduce(0) { $0 + abs(Double($1) - midpoint) } / count
            confidence = min(1, max(0, 1 - distance / baseline))
            legend = descriptions
        }
    }
}

public struct DecisionUsage: Sendable, Equatable {
    /// Tokens of every forward pass, template and image positions included.
    public let inputTokens: Int
    /// Sequences the questions ran in; more than 1 when split.
    public let sequences: Int

    public init(inputTokens: Int, sequences: Int) {
        self.inputTokens = inputTokens
        self.sequences = sequences
    }
}

public struct DecisionResponse: Sendable {
    /// One answer per question, in request order.
    public let answers: [DecisionAnswer]
    public let usage: DecisionUsage

    public init(answers: [DecisionAnswer], usage: DecisionUsage) {
        self.answers = answers
        self.usage = usage
    }
}

public enum DecisionError: Error, LocalizedError, Equatable {
    /// The request breaks the schema: no questions, duplicate ids, bad option counts.
    case invalidRequest(String)
    /// A single question does not fit the length limit with its state and images.
    case tooLong(String)
    /// The checkpoint does not declare a format this runtime serves.
    case unsupportedCheckpoint(String)
    /// The model produced something the answer cannot be built from.
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let detail): "Invalid decision request: \(detail)"
        case .tooLong(let detail): "Decision request too long: \(detail)"
        case .unsupportedCheckpoint(let detail): "Unsupported decision checkpoint: \(detail)"
        case .invalidOutput(let detail): "Invalid decision output: \(detail)"
        }
    }
}
