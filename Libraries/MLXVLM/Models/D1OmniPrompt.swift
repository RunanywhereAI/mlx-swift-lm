import Foundation
import MLXLMCommon

public struct D1OmniQuestion: Sendable {
    public enum Kind: String, Sendable, Codable {
        case choice, score, noul
        var index: Int {
            switch self {
            case .choice: 0
            case .score: 1
            case .noul: 2
            }
        }
    }

    public struct Option: Sendable {
        public let name: String
        public let description: String
        public init(_ name: String, description: String = "") {
            self.name = name
            self.description = description
        }
    }

    public let kind: Kind
    public let instructions: String
    public let options: [Option]

    public init(kind: Kind, instructions: String, options: [Option] = []) throws {
        switch kind {
        case .choice:
            guard options.count >= 2, Set(options.map(\.name)).count == options.count else {
                throw D1OmniError.invalidQuestion("choice needs at least two distinct names")
            }
        case .score:
            guard (2 ... 10).contains(options.count) else {
                throw D1OmniError.invalidQuestion("score needs two to ten levels")
            }
        case .noul:
            guard options.allSatisfy({ ["false", "true", "no", "yes"].contains($0.name) }),
                Set(options.map(\.name)).count == options.count
            else {
                throw D1OmniError.invalidQuestion("noul criteria must use false/true or no/yes")
            }
        }
        self.kind = kind
        self.instructions = instructions
        self.options = options
    }

    public var optionCount: Int { kind == .noul ? 2 : options.count }
    public var probabilityLabels: [String] {
        switch kind {
        case .choice: options.map(\.name)
        case .score: options.indices.map(String.init)
        case .noul: ["true", "false"]
        }
    }

    public var temperatureKey: String {
        let count = optionCount
        let bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"
        return "\(kind.rawValue):\(bucket)"
    }

    func renderedOptions(image: Bool) -> [String] {
        switch kind {
        case .choice:
            return options.map {
                $0.description.isEmpty ? $0.name : "\($0.name): \($0.description)"
            }
        case .score:
            return options.enumerated().map { "level \($0.offset): \($0.element.description)" }
        case .noul:
            let criteria = Dictionary(
                uniqueKeysWithValues: options.map { ($0.name, $0.description) })
            let defaults = options.isEmpty && image
            let negative = criteria["false"] ?? criteria["no"] ?? (defaults ? "no" : "")
            let positive = criteria["true"] ?? criteria["yes"] ?? (defaults ? "yes" : "")
            return [
                "false: " + (negative.isEmpty ? "no, the statement does not hold" : negative),
                "true: " + (positive.isEmpty ? "yes, the statement holds" : positive),
            ]
        }
    }
}

public struct D1OmniEncodedQuestion: Sendable {
    public let tokens: [Int]
    public let markers: [Int]
}

public enum D1OmniPrompt {
    public static func escape(_ text: String) -> String {
        text.replacingOccurrences(
            of: "<\\|([A-Za-z0-9_]+)\\|>", with: "<¦$1¦>", options: .regularExpression)
    }

    public static func encode(
        state: String, question: D1OmniQuestion, tokenizer: any Tokenizer,
        maxLength: Int, bosTokenId: Int = 1, image: Bool = false
    ) throws -> D1OmniEncodedQuestion {
        guard maxLength >= 64 else { throw D1OmniError.contextExceeded }
        func token(_ name: String) throws -> Int {
            guard let token = tokenizer.convertTokenToId(name) else {
                throw D1OmniError.missingToken(name)
            }
            return token
        }
        func encode(_ text: String) -> [Int] {
            tokenizer.encode(text: escape(text), addSpecialTokens: false)
        }
        let options = question.renderedOptions(image: image)
        let budget = max(96, min(options.count * 24 + 32, maxLength / 2))
        let perOption = max(2, (budget - 3 * options.count) / options.count)
        var suffix = Array(
            (try [token("<|reserved_8|>")] + encode(question.instructions)).prefix(max(16, budget)))
        var markers = [Int]()
        for option in options {
            markers.append(suffix.count + 1)
            suffix += try [token("<|reserved_9|>"), token("<|mask|>")]
            suffix += encode(" " + option).prefix(perOption)
            suffix.append(try token("<|reserved_10|>"))
        }
        suffix.append(try token("<|reserved_11|>"))
        let room = max(0, maxLength - suffix.count - 2)
        let stateTokens = try [token("<|reserved_7|>")] + encode(state).prefix(room)
        markers = markers.map { $0 + 1 + stateTokens.count }
        guard let last = markers.last, last < maxLength else { throw D1OmniError.contextExceeded }
        return D1OmniEncodedQuestion(
            tokens: Array(([bosTokenId] + stateTokens + suffix).prefix(maxLength)), markers: markers
        )
    }
}
