import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class GLiNERClassificationTests: XCTestCase {
    func testRelativePositionDirections() {
        let (content, position) = GLiNERRelativePositions.indices(
            length: 3, buckets: 8, maximum: 32)
        XCTAssertEqual(content.asArray(Int32.self), [8, 7, 6, 9, 8, 7, 10, 9, 8])
        XCTAssertEqual(position.asArray(Int32.self), [8, 9, 10, 7, 8, 9, 6, 7, 8])
    }

    func testLogBucketsAndClipping() {
        XCTAssertEqual(GLiNERRelativePositions.bucket(128, buckets: 256, maximum: 512), 128)
        XCTAssertEqual(GLiNERRelativePositions.bucket(256, buckets: 256, maximum: 512), 192)
        XCTAssertEqual(GLiNERRelativePositions.bucket(511, buckets: 256, maximum: 512), 255)
        XCTAssertEqual(GLiNERRelativePositions.bucket(-511, buckets: 256, maximum: 512), -255)
        let (content, position) = GLiNERRelativePositions.indices(
            length: 50, buckets: 8, maximum: 32)
        XCTAssertEqual(content.min().item(Int.self), 0)
        XCTAssertEqual(content.max().item(Int.self), 15)
        XCTAssertEqual(position.min().item(Int.self), 0)
        XCTAssertEqual(position.max().item(Int.self), 15)
    }

    func testWordSplitting() throws {
        XCTAssertEqual(
            try GLiNERPreprocessor.words(
                "Mail Test.User+tag@Example.com @Support cost-saving Café!"),
            ["mail", "test.user+tag@example.com", "@support", "cost-saving", "café", "!"])
        XCTAssertEqual(
            try GLiNERPreprocessor.words("HTTPS://Example.org/a?x=1, hi."),
            ["https://example.org/a?x=1,", "hi", "."])
    }

    func testStableSoftmaxAndTemperature() throws {
        let task = GLiNERClassificationTask(
            name: "test", labels: [.init("a"), .init("b")], temperature: 2)
        let result = try GLiNERClassificationScoring.result(logits: [1000, 1002], task: task)
        XCTAssertEqual(result.scores[0].probability, 0.26894143, accuracy: 1e-6)
        XCTAssertEqual(result.scores[1].probability, 0.7310586, accuracy: 1e-6)
        XCTAssertEqual(result.selectedLabels, ["b"])
    }

    func testAutomaticMultiLabelSigmoid() throws {
        let task = GLiNERClassificationTask(
            name: "test", labels: [.init("a"), .init("b"), .init("c")], multiLabel: true)
        let result = try GLiNERClassificationScoring.result(logits: [0, 2, -2], task: task)
        XCTAssertEqual(result.scores[0].probability, 0.5)
        XCTAssertEqual(result.scores[1].probability, 0.880797, accuracy: 1e-6)
        XCTAssertEqual(result.selectedLabels, ["a", "b"])
    }

    func testActivationOverridesAndFallback() throws {
        var task = GLiNERClassificationTask(
            name: "test", labels: [.init("a"), .init("b")], multiLabel: true, activation: .softmax,
            threshold: 1)
        let result = try GLiNERClassificationScoring.result(logits: [-1000, -1000], task: task)
        XCTAssertEqual(result.scores.map(\.probability), [0.5, 0.5])
        XCTAssertEqual(result.selectedLabels, ["a"])
        task.multiLabel = false
        task.activation = .sigmoid
        let sigmoidResult = try GLiNERClassificationScoring.result(
            logits: [-1000, 1000], task: task)
        XCTAssertEqual(sigmoidResult.scores.map(\.probability), [0, 1])
        XCTAssertEqual(sigmoidResult.selectedLabels, ["b"])
    }

    func testInvalidTasks() throws {
        var task = GLiNERClassificationTask(name: "test", labels: [.init("a")])
        task.labels = []
        XCTAssertThrowsError(try task.validate())
        task.labels = [.init("a"), .init("a")]
        XCTAssertThrowsError(try task.validate())
        task.labels = [.init(" ")]
        XCTAssertThrowsError(try task.validate())
        task.labels = [.init("a")]
        for temperature in [Float(0), -1, .nan, .infinity] {
            task.temperature = temperature
            XCTAssertThrowsError(try task.validate())
        }
        task.temperature = 1
        for threshold in [Float(-0.1), 1.1, .nan] {
            task.threshold = threshold
            XCTAssertThrowsError(try task.validate())
        }
        task.threshold = 0.5
        task.examples = [.init(text: "example", label: "missing")]
        XCTAssertThrowsError(try task.validate())
    }

    func testInvalidScoresAndReservedMarkers() {
        let task = GLiNERClassificationTask(name: "test", labels: [.init("a")])
        XCTAssertThrowsError(try GLiNERClassificationScoring.result(logits: [], task: task))
        XCTAssertThrowsError(try GLiNERClassificationScoring.result(logits: [.nan], task: task))
        for marker in GLiNERPreprocessor.markers {
            XCTAssertThrowsError(
                try GLiNERPreprocessor.validateField("data \(marker)", name: "field"))
        }
    }

    func testExtractionTensorValidation() throws {
        let hidden = 4
        let accepted = ["count_embed.gru.bias_ih_l0": MLXArray.zeros([12])]
        XCTAssertTrue(
            try GLiNERClassificationNetwork.classificationWeights(accepted, hidden: hidden).isEmpty)
        XCTAssertThrowsError(
            try GLiNERClassificationNetwork.classificationWeights(
                ["count_embed.gru.bias_ih_l0": MLXArray.zeros([11])], hidden: hidden))
        XCTAssertThrowsError(
            try GLiNERClassificationNetwork.classificationWeights(
                ["count_embed.gru.bias_ih_l0": MLXArray.zeros([12], dtype: .int32)], hidden: hidden)
        )
        let unknown = ["span_rep.unrecognized.weight": MLXArray.zeros([1])]
        XCTAssertNotNil(
            try GLiNERClassificationNetwork.classificationWeights(unknown, hidden: hidden)[
                "span_rep.unrecognized.weight"])
        let duplicate = [
            "classifier.0.bias": MLXArray.zeros([8]), "classifier.fc1.bias": MLXArray.zeros([8]),
        ]
        XCTAssertThrowsError(
            try GLiNERClassificationNetwork.classificationWeights(duplicate, hidden: hidden))
    }

    func testEncoderConfigurationValidation() throws {
        let config = try smallEncoderConfiguration()
        XCTAssertNoThrow(try config.validate())
        let data = smallEncoderJSON.replacingOccurrences(
            of: "\"share_att_key\":true", with: "\"share_att_key\":false")
        let invalid = try JSONDecoder().decode(
            GLiNEREncoderConfiguration.self, from: Data(data.utf8))
        XCTAssertThrowsError(try invalid.validate())
    }

    func testPaddingKeysDoNotAffectRealQueries() throws {
        let encoder = GLiNERDeberta(try smallEncoderConfiguration())
        let mask = MLXArray([Int32(1), 1, 0]).reshaped(1, 3)
        let first = encoder(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), mask: mask)
        let second = encoder(MLXArray([Int32(1), 2, 7]).reshaped(1, 3), mask: mask)
        XCTAssertEqual(first.shape, [1, 3, 8])
        XCTAssertTrue(first.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertLessThan((first - second).abs().max().item(Float.self), 1e-6)
        let unpadded = encoder(
            MLXArray([Int32(1), 2]).reshaped(1, 2),
            mask: MLXArray.ones([1, 2], dtype: .int32))
        XCTAssertLessThan(
            (first[0..., 0 ..< 2, 0...] - unpadded).abs().max().item(Float.self), 1e-5)
        let bias = GLiNERDeberta.keyBias(mask).asArray(Float.self)
        XCTAssertEqual(bias[0], 0)
        XCTAssertEqual(bias[1], 0)
        XCTAssertEqual(bias[2], -Float.infinity)
    }

    func testRegistryBuildsClassifier() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(smallEncoderJSON.utf8), modelType: "extractor")
        XCTAssertTrue(model is GLiNERClassificationNetwork)
    }

    func testTokenizerAdapterPreservesMarkerIds() throws {
        let config = Data(#"{"tokenizer_class":"DebertaV2Tokenizer"}"#.utf8)
        let tokenizer = Data(
            #"{"model":{"type":"Unigram","vocab":[["[UNK]",0]]},"added_tokens":[{"id":0,"content":"[UNK]"},{"id":1,"content":"[P]"}]}"#
                .utf8)
        let adapted = try GLiNERTokenizerConfiguration.adapt(
            configurationData: config, vocabularyData: tokenizer)
        let decoded = try XCTUnwrap(
            JSONSerialization.jsonObject(with: adapted.tokenizerData) as? [String: Any])
        let model = try XCTUnwrap(decoded["model"] as? [String: Any])
        let vocabulary = try XCTUnwrap(model["vocab"] as? [[Any]])
        XCTAssertEqual(vocabulary.count, 2)
        XCTAssertEqual(vocabulary[0][0] as? String, "[UNK]")
        XCTAssertEqual(vocabulary[1][0] as? String, "[P]")
        let configuration = try XCTUnwrap(
            JSONSerialization.jsonObject(with: adapted.tokenizerConfig) as? [String: Any])
        XCTAssertEqual(configuration["tokenizer_class"] as? String, "XLMRobertaTokenizer")
    }

    func testTokenizerAdapterRejectsConflictingAddedTokens() throws {
        let config = Data(#"{"tokenizer_class":"DebertaV2Tokenizer"}"#.utf8)
        for addedToken in [
            #"{"id":0,"content":"[P]"}"#,
            #"{"id":2,"content":"[P]"}"#,
            #"{"id":1,"content":"[UNK]"}"#,
            #"{"id":true,"content":"[P]"}"#,
            #"{"id":-1,"content":"[P]"}"#,
        ] {
            let tokenizer = Data(
                "{\"model\":{\"type\":\"Unigram\",\"vocab\":[[\"[UNK]\",0]]},\"added_tokens\":[\(addedToken)]}"
                    .utf8)
            XCTAssertThrowsError(
                try GLiNERTokenizerConfiguration.adapt(
                    configurationData: config, vocabularyData: tokenizer))
        }
    }
}

private let smallEncoderJSON = """
    {"hidden_size":8,"intermediate_size":16,"num_attention_heads":2,"num_hidden_layers":1,
     "vocab_size":10,"layer_norm_eps":1e-7,"position_buckets":8,"max_position_embeddings":32,
     "max_relative_positions":-1,"relative_attention":true,"share_att_key":true,
     "position_biased_input":false,"norm_rel_ebd":"layer_norm","type_vocab_size":0,
     "pos_att_type":["p2c","c2p"],"hidden_act":"gelu"}
    """

private func smallEncoderConfiguration() throws -> GLiNEREncoderConfiguration {
    try JSONDecoder().decode(GLiNEREncoderConfiguration.self, from: Data(smallEncoderJSON.utf8))
}
