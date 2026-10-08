import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

private let d1SmallConfiguration = """
    {"model_type":"d1_omni","max_length":512,"image_text_length":128,"head_layers":2,
     "projector_hidden_size":64,"temperatures":{"noul:2":2,"choice:2":2,"score":3},"bos_token_id":1,
     "text_config":{"vocab_size":64,"hidden_size":64,"intermediate_size":96,
       "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
       "layer_types":["conv","full_attention"],"norm_eps":0.00001,"conv_L_cache":3,
       "block_ffn_dim_multiplier":1,"block_multiple_of":8,"rope_theta":1000000},
     "vision_config":{"hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,
       "num_attention_heads":2,"num_channels":3,"num_patches":256,"patch_size":16,"layer_norm_eps":0.000001}}
    """

final class D1OmniTests: XCTestCase {
    private func configuration(_ json: String = d1SmallConfiguration) throws -> D1OmniConfiguration
    {
        try JSONDecoder().decode(D1OmniConfiguration.self, from: Data(json.utf8))
    }

    func testConfigurationAndFeedForwardWidth() throws {
        let config = try configuration()
        try config.validateModelConfiguration()
        XCTAssertEqual(config.text.feedForwardSize, 64)
        XCTAssertEqual(config.headLayers, 2)
        let model = D1Omni(config)
        XCTAssertEqual(
            model.parameters().flattened()["encoder.layers.0.conv.conv.weight"]?.shape,
            [64, 1, 3])
    }

    func testInvalidConfigurationRejectsUnsupportedConvolution() throws {
        let config = try configuration(
            d1SmallConfiguration.replacingOccurrences(
                of: "\"conv_L_cache\":3", with: "\"conv_L_cache\":5"))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testInvalidConfigurationRejectsTemperatureAndLayerLayout() throws {
        for json in [
            d1SmallConfiguration.replacingOccurrences(of: "\"noul:2\":2", with: "\"noul:2\":0"),
            d1SmallConfiguration.replacingOccurrences(
                of: "[\"conv\",\"full_attention\"]", with: "[\"conv\"]"),
        ] {
            XCTAssertThrowsError(try configuration(json).validateModelConfiguration())
        }
    }

    func testQuestionValidationAndTemperatureBuckets() throws {
        XCTAssertThrowsError(
            try D1OmniQuestion(kind: .choice, instructions: "pick", options: [.init("only")]))
        XCTAssertThrowsError(try D1OmniQuestion(kind: .score, instructions: "rate", options: []))
        XCTAssertThrowsError(
            try D1OmniQuestion(kind: .noul, instructions: "yes?", options: [.init("unknown")]))
        let question = try D1OmniQuestion(
            kind: .choice, instructions: "pick", options: (0 ..< 11).map { .init(String($0)) })
        XCTAssertEqual(question.temperatureKey, "choice:11+")
        XCTAssertEqual(
            try D1OmniQuestion(kind: .noul, instructions: "yes?").probabilityLabels,
            ["true", "false"])
    }

    func testPromptEscapingAndImageDefaults() throws {
        XCTAssertEqual(
            D1OmniPrompt.escape("<|mask|> <|reserved_8|> <|not-a-token|>"),
            "<¦mask¦> <¦reserved_8¦> <|not-a-token|>")
        let question = try D1OmniQuestion(kind: .noul, instructions: "yes?")
        XCTAssertEqual(question.renderedOptions(image: true), ["false: no", "true: yes"])
        XCTAssertEqual(
            question.renderedOptions(image: false),
            ["false: no, the statement does not hold", "true: yes, the statement holds"])
    }

    func testCalibrationAndNoulOrder() throws {
        let model = D1Omni(try configuration())
        let question = try D1OmniQuestion(kind: .noul, instructions: "yes?")
        let text = model.calibratedProbabilities(
            MLXArray([Float(0), 2]), question: question, image: false)
        let image = model.calibratedProbabilities(
            MLXArray([Float(0), 2]), question: question, image: true)
        XCTAssertEqual(text[0], 0.7310586, accuracy: 1e-6)
        XCTAssertEqual(image[0], 0.8807971, accuracy: 1e-6)
        XCTAssertEqual(text.reduce(0, +), 1, accuracy: 1e-6)
    }

    func testHeadSelectsOptionMarkers() throws {
        let model = D1Omni(try configuration())
        model.weight("head.scorer.0.weight")._updateInternal(MLXArray.ones([64]))
        model.weight("head.scorer.1.weight")._updateInternal(MLXArray.eye(64))
        var scorer = [Float](repeating: 0, count: 64)
        scorer[0] = 1
        model.weight("head.scorer.3.weight")._updateInternal(MLXArray(scorer).reshaped(1, 64))
        var input = [Float](repeating: 0, count: 3 * 64)
        input[0] = 1
        input[64] = -1
        input[128] = 2
        let states = MLXArray(input).reshaped(1, 3, 64)
        let all = model.decisionLogits(states, markers: [0, 1, 2], kind: .choice).asArray(
            Float.self)
        let selected = model.decisionLogits(states, markers: [2, 0], kind: .choice).asArray(
            Float.self)
        XCTAssertGreaterThan(all[0], 1)
        XCTAssertLessThan(abs(all[1]), 1e-4)
        XCTAssertEqual(selected[0], all[2], accuracy: 1e-6)
        XCTAssertEqual(selected[1], all[0], accuracy: 1e-6)
    }

    func testStrictWeightValidationAndAudioFiltering() throws {
        let model = D1Omni(try configuration())
        var weights = model.parameters().flattened()
        weights["audio.encoder.weight"] = MLXArray.zeros([1])
        let sanitized = model.sanitize(weights: weights)
        XCTAssertNil(sanitized["audio.encoder.weight"])
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        var missing = sanitized
        missing.removeValue(forKey: "encoder.embed_tokens.weight")
        XCTAssertThrowsError(
            try model.update(parameters: ModuleParameters.unflattened(missing), verify: [.all]))
        var extra = sanitized
        extra["unknown.weight"] = MLXArray.zeros([1])
        XCTAssertThrowsError(
            try model.update(parameters: ModuleParameters.unflattened(extra), verify: [.all]))
        var wrongShape = sanitized
        wrongShape["head.scorer.3.weight"] = MLXArray.zeros([2, 64])
        XCTAssertThrowsError(
            try model.update(parameters: ModuleParameters.unflattened(wrongShape), verify: [.all]))
    }

    func testGenerationAndAudioRejected() throws {
        let model = D1Omni(try configuration())
        XCTAssertThrowsError(try model.newCache(parameters: nil)) {
            XCTAssertEqual($0 as? D1OmniError, .generationUnsupported)
        }
        let input = LMInput(
            text: .init(tokens: MLXArray([1])), audio: .init(samples: MLXArray([Float(0)])))
        XCTAssertThrowsError(try model.prepare(input, cache: [], state: nil, prefill: .init())) {
            XCTAssertEqual($0 as? D1OmniError, .audioUnsupported)
        }
        XCTAssertThrowsError(
            try model.prepare(
                LMInput(tokens: MLXArray([1])), cache: [], state: nil, prefill: .init())
        ) {
            XCTAssertEqual($0 as? D1OmniError, .generationUnsupported)
        }
    }

    func testImageValidationAndLayout() throws {
        XCTAssertThrowsError(try D1OmniImage(width: 0, height: 1, rgb: []))
        XCTAssertThrowsError(try D1OmniImage(width: 1, height: 1, rgb: [0]))
        let layout = D1OmniProcessor.layout(width: 640, height: 480)
        XCTAssertEqual(layout.width, 576)
        XCTAssertEqual(layout.height, 416)
        XCTAssertFalse(layout.tiled)
        let large = D1OmniProcessor.layout(width: 2048, height: 1024)
        XCTAssertTrue(large.tiled)
        XCTAssertEqual(large.gridWidth, 4)
        XCTAssertEqual(large.gridHeight, 2)
    }
}
