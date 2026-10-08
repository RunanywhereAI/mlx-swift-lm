import Foundation
import MLXLMCommon

public enum D1Question: Sendable {
    case noul(instructions: String, yes: String? = nil, no: String? = nil)
    case choice(instructions: String, options: [D1Option])
    case score(instructions: String, levels: [String])
}

public struct D1Option: Sendable {
    public let label: String
    public let description: String?

    public init(_ label: String, description: String? = nil) {
        self.label = label
        self.description = description
    }
}

public enum D1Error: Error, LocalizedError {
    case invalidQuestion(String)
    case missingToken(String)
    case invalidInput(String)
    case incompatibleModel

    public var errorDescription: String? {
        switch self {
        case .invalidQuestion(let message), .missingToken(let message), .invalidInput(let message):
            message
        case .incompatibleModel:
            "D1 requires an LFM2VL checkpoint with the D1Model auto_map."
        }
    }
}

public struct D1Prompt: Sendable {
    public let text: String
    public let tokenGroups: [[Int]]
    public let labels: [String]

    public init(
        state: String?, question: D1Question, tokenizer: any Tokenizer,
        imageMarkup: String = ""
    ) throws {
        let body: String
        switch question {
        case .noul(let instructions, let yes, let no):
            labels = ["true", "false"]
            tokenGroups = [
                Self.singleTokens(["yes", "Yes", "YES"], tokenizer: tokenizer),
                Self.singleTokens(["no", "No", "NO"], tokenizer: tokenizer),
            ]
            let criteria =
                yes == nil && no == nil ? "" : "\nYes: \(yes ?? "None")\nNo: \(no ?? "None")"
            body = "\(instructions)\(criteria)\n\nReply with yes or no only."
        case .choice(let instructions, let options):
            guard !options.isEmpty, Set(options.map(\.label)).count == options.count else {
                throw D1Error.invalidQuestion(
                    "Choice options must be nonempty and have unique labels.")
            }
            labels = options.map(\.label)
            let aliases = try Self.aliases(labels, tokenizer: tokenizer)
            tokenGroups = aliases.map { alias in
                [alias.id]
                    + Self.singleTokens([" \(alias.code)"], tokenizer: tokenizer).filter {
                        $0 != alias.id
                    }
            }
            let lines = zip(options, aliases).map { option, alias in
                let description =
                    option.description.flatMap { $0.isEmpty ? nil : $0 }
                    ?? option.label.replacingOccurrences(of: "_", with: " ")
                return "\(alias.code) \(description)"
            }.joined(separator: "\n")
            body = "\(instructions)\n\nOptions:\n\(lines)\n\nReply with the option code only."
        case .score(let instructions, let levels):
            guard (2 ... 10).contains(levels.count) else {
                throw D1Error.invalidQuestion("Score requires 2 to 10 levels.")
            }
            labels = levels.indices.map(String.init)
            tokenGroups = labels.map { Self.singleTokens([$0], tokenizer: tokenizer) }
            let legend = levels.enumerated().map { "\($0.offset) \($0.element)" }.joined(
                separator: "\n")
            body =
                "\(instructions)\n\n\(legend)\n\nReply with a single digit 0-\(levels.count - 1) only."
        }
        guard tokenGroups.allSatisfy({ !$0.isEmpty }) else {
            throw D1Error.missingToken(
                "Each D1 option needs at least one single-token vocabulary form.")
        }
        let stateBody = state.map { "\($0)\n\n\nQUESTION:\n" } ?? ""
        text =
            "\(tokenizer.bosToken ?? "")<|im_start|>user\n\(imageMarkup)\(stateBody)\(body)<|im_end|>\n<|im_start|>assistant\n"
    }

    private static func singleTokens(_ forms: [String], tokenizer: any Tokenizer) -> [Int] {
        var seen = Set<Int>()
        return forms.compactMap { form in
            let encoded = D1TokenEncoding.encode(form, tokenizer: tokenizer)
            guard encoded.count == 1, seen.insert(encoded[0]).inserted else { return nil }
            return encoded[0]
        }
    }

    private static func aliases(_ labels: [String], tokenizer: any Tokenizer) throws
        -> [(code: String, id: Int)]
    {
        let trimmed = labels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let letters = (65 ... 90).map { String(UnicodeScalar($0)!) }
        let native = trimmed.allSatisfy {
            $0.unicodeScalars.count == 1
                && $0.unicodeScalars.allSatisfy(CharacterSet.letters.contains)
        }
        let codes =
            native
            ? trimmed
            : labels.count <= 26
                ? Array(letters.prefix(labels.count))
                : labels.indices.map { String(format: "%02d", $0) }
        let fallback =
            letters + (0 ..< 100).map { String(format: "%02d", $0) }
            + (97 ... 122).map { String(UnicodeScalar($0)!) }
            + (0 ..< 200).map { "#\($0)" }
            + letters.flatMap { first in letters.map { first + $0 } }
        var used = Set<Int>()
        return try codes.map { code in
            for candidate in [code] + fallback {
                let ids = D1TokenEncoding.encode(candidate, tokenizer: tokenizer)
                if ids.count == 1, used.insert(ids[0]).inserted {
                    return (candidate, ids[0])
                }
            }
            throw D1Error.missingToken(
                "No distinct single-token alias remains for the choice options.")
        }
    }
}
