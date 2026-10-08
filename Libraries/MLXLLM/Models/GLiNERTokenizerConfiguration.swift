import CoreFoundation
import Foundation

public struct GLiNERTokenizerConfiguration: Sendable {
    public let tokenizerConfig: Data
    public let tokenizerData: Data

    public static func load(from directory: URL) throws -> GLiNERTokenizerConfiguration {
        let configurationData = try Data(
            contentsOf: directory.appendingPathComponent("tokenizer_config.json"))
        let vocabularyData = try Data(
            contentsOf: directory.appendingPathComponent("tokenizer.json"))
        return try adapt(configurationData: configurationData, vocabularyData: vocabularyData)
    }

    static func adapt(configurationData: Data, vocabularyData: Data) throws
        -> GLiNERTokenizerConfiguration
    {
        guard
            var configuration = try JSONSerialization.jsonObject(with: configurationData)
                as? [String: Any],
            var tokenizer = try JSONSerialization.jsonObject(with: vocabularyData)
                as? [String: Any],
            let tokenizerClass = configuration["tokenizer_class"] as? String,
            ["DebertaV2Tokenizer", "DebertaV2TokenizerFast"].contains(tokenizerClass),
            var model = tokenizer["model"] as? [String: Any],
            model["type"] as? String == "Unigram",
            var vocabulary = model["vocab"] as? [[Any]],
            let addedTokens = tokenizer["added_tokens"] as? [[String: Any]]
        else { throw GLiNERError.invalidCheckpoint("expected DeBERTa Unigram tokenizer JSON") }
        var tokenNames = Set(vocabulary.compactMap { $0.first as? String })
        guard tokenNames.count == vocabulary.count, !tokenNames.contains("") else {
            throw GLiNERError.invalidCheckpoint(
                "Unigram vocabulary must have unique nonempty token names")
        }
        let sortedTokens = try addedTokens.map { token -> (Int, String) in
            guard let number = token["id"] as? NSNumber,
                CFGetTypeID(number) != CFBooleanGetTypeID(),
                number.doubleValue.isFinite, number.doubleValue >= 0,
                number.doubleValue < Double(Int.max),
                number.doubleValue.rounded(.towardZero) == number.doubleValue,
                let content = token["content"] as? String, !content.isEmpty
            else { throw GLiNERError.invalidCheckpoint("invalid added token ID or content") }
            return (number.intValue, content)
        }.sorted { $0.0 < $1.0 }
        guard Set(sortedTokens.map(\.0)).count == sortedTokens.count,
            Set(sortedTokens.map(\.1)).count == sortedTokens.count
        else { throw GLiNERError.invalidCheckpoint("duplicate added token ID or content") }
        for (identifier, content) in sortedTokens {
            if identifier < vocabulary.count {
                guard vocabulary[identifier].first as? String == content else {
                    throw GLiNERError.invalidCheckpoint(
                        "added token conflicts with Unigram vocabulary")
                }
            } else {
                guard identifier == vocabulary.count else {
                    throw GLiNERError.invalidCheckpoint(
                        "added token IDs must extend the vocabulary contiguously")
                }
                guard tokenNames.insert(content).inserted else {
                    throw GLiNERError.invalidCheckpoint(
                        "added token duplicates a different vocabulary ID")
                }
                vocabulary.append([content, 0.0])
            }
        }
        configuration["tokenizer_class"] = "XLMRobertaTokenizer"
        model["vocab"] = vocabulary
        tokenizer["model"] = model
        return GLiNERTokenizerConfiguration(
            tokenizerConfig: try JSONSerialization.data(withJSONObject: configuration),
            tokenizerData: try JSONSerialization.data(withJSONObject: tokenizer))
    }
}
