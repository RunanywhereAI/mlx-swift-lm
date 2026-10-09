import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class D1Tests: XCTestCase {
    private func configuration(_ extra: String = "") throws -> LFM2VLConfiguration {
        try JSONDecoder().decode(
            LFM2VLConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "lfm2_vl",
                  "text_config": {
                    "model_type": "lfm2", "hidden_size": 32, "num_hidden_layers": 1,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64
                    \(extra)
                  },
                  "vision_config": {
                    "model_type": "siglip2_vision_model", "hidden_size": 16,
                    "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2
                  },
                  "projector_hidden_size": 32
                }
                """.utf8))
    }

    func testIntermediateSize10752() throws {
        let config = try configuration(
            ", \"intermediate_size\": 10752, \"block_auto_adjust_ff_dim\": false")
        XCTAssertEqual(config.textConfiguration.blockFFDim, 10752)
        let model = LFM2VL(config)
        XCTAssertEqual(
            model.parameters().flattened().first {
                $0.0 == "language_model.model.layers.0.feed_forward.w1.weight"
            }?.1.shape, [10752, 32])
    }

    func testBlockFFDimPrecedence() throws {
        let config = try configuration(
            ", \"intermediate_size\": 10752, \"block_ff_dim\": 96, \"block_auto_adjust_ff_dim\": false"
        )
        XCTAssertEqual(config.textConfiguration.blockFFDim, 96)
        let model = LFM2VL(config)
        XCTAssertEqual(
            model.parameters().flattened().first {
                $0.0 == "language_model.model.layers.0.feed_forward.w1.weight"
            }?.1.shape, [96, 32])
    }

    func testLegacyHiddenSizeFallbackAndAutoAdjust() throws {
        let config = try configuration()
        XCTAssertEqual(config.textConfiguration.blockFFDim, 32)
        XCTAssertTrue(config.textConfiguration.blockAutoAdjustFFDim)
        XCTAssertFalse(config.isD1)
        let model = LFM2VL(config)
        XCTAssertEqual(
            model.parameters().flattened().first {
                $0.0 == "language_model.model.layers.0.feed_forward.w1.weight"
            }?.1.shape, [256, 32])
        XCTAssertEqual(model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil).shape, [1, 3, 64])
    }

    func testIntermediateSizeKeepsExplicitAutoAdjust() throws {
        let config = try configuration(
            ", \"intermediate_size\": 96, \"block_auto_adjust_ff_dim\": true, \"block_multiple_of\": 16"
        )
        let model = LFM2VL(config)
        XCTAssertEqual(
            model.parameters().flattened().first {
                $0.0 == "language_model.model.layers.0.feed_forward.w1.weight"
            }?.1.shape, [64, 32])
    }

    func testCalibrationValidationAndFormula() throws {
        XCTAssertThrowsError(try D1Calibration(temperature: 0))
        XCTAssertThrowsError(try D1Calibration(biases: [.nan]))
        let calibration = try D1Calibration(temperature: 2, biases: [1, -1])
        XCTAssertEqual(try calibration.apply([-3, -4]), [-1, -2.5])
        XCTAssertThrowsError(try calibration.apply([1]))
    }

    func testLegacyProcessorDefaultsAndNestedConfigRemainCompatible() throws {
        let legacy = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy.tileSize, 512)
        XCTAssertEqual(legacy.encoderPatchSize, 16)
        XCTAssertEqual(legacy.imageMean, [0.5, 0.5, 0.5])
        XCTAssertNil(legacy.d1ImageConfiguration)
        let other = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self,
            from: Data(
                "{\"image_processor\": {\"image_processor_type\": \"Siglip2ImageProcessor\"}}".utf8)
        )
        XCTAssertNil(other.d1ImageConfiguration)
    }

    func testSmartResizePreservesAspectAndBudget() throws {
        let processor = D1ImageProcessor(config: try configuration())
        let size = processor.smartSize(width: 320, height: 256)
        XCTAssertEqual(size.width, 320)
        XCTAssertEqual(size.height, 256)
        let large = processor.smartSize(width: 960, height: 640)
        XCTAssertEqual(large.width, 608)
        XCTAssertEqual(large.height, 416)
    }

    func testBilinearPositionsPreserveConstantAndIdentity() {
        let input = MLXArray.ones([16, 16, 4]).asType(.bfloat16)
        let output = D1PositionInterpolation.resize(input, height: 8, width: 20)
        XCTAssertEqual(output.shape, [8, 20, 4])
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertEqual(output.min().item(Float.self), 1)
        XCTAssertEqual(output.max().item(Float.self), 1)
        XCTAssertEqual(
            D1PositionInterpolation.resize(input, height: 16, width: 16).shape, input.shape)
    }
}
