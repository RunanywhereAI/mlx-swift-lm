// Copyright © 2026 RunAnywhere AI.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class MapleTests: XCTestCase {
    private let configurationJSON = """
        {
          "model_type": "maple",
          "vocab_size": 32,
          "hidden_size": 8,
          "intermediate_size": 16,
          "moe_intermediate_size": 4,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "first_k_dense_replace": 0,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000,
          "partial_rotary_factor": 0.5,
          "max_position_embeddings": 128,
          "sliding_window": 16,
          "layer_types": ["sliding_attention", "full_attention"],
          "use_qk_norm": true,
          "use_bias": false,
          "tie_word_embeddings": false,
          "quantization": {"bits": 2, "group_size": 128, "mode": "affine"}
        }
        """

    func testConfigurationDecodesHybridAttentionLayout() throws {
        let configuration = try JSONDecoder().decode(
            MapleConfiguration.self, from: Data(configurationJSON.utf8))

        XCTAssertEqual(configuration.modelType, "maple")
        XCTAssertEqual(configuration.numExperts, 2)
        XCTAssertEqual(configuration.numExpertsPerToken, 1)
        XCTAssertEqual(configuration.partialRotaryFactor, 0.5)
        XCTAssertEqual(configuration.quantizationGroupSize, 128)
        XCTAssertEqual(configuration.layerTypes, ["sliding_attention", "full_attention"])
    }

    func testTypeRegistryConstructsMaple() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(configurationJSON.utf8), modelType: "maple")

        XCTAssertTrue(model is MapleModel)
    }

    func testSanitizeConvertsRowAlphaAndStacksExpertWeights() throws {
        let configuration = try JSONDecoder().decode(
            MapleConfiguration.self, from: Data(configurationJSON.utf8))
        let model = MapleModel(configuration)
        let prefix = "model.layers.0.mlp"
        let rowAlphaPrefix = "model.layers.0.self_attn.q_proj"

        let sanitized = model.sanitize(weights: [
            "lm_head_flash.weight": MLXArray.zeros([1]),
            "\(rowAlphaPrefix).weight": MLXArray.zeros([8, 8]),
            "\(rowAlphaPrefix).row_alpha": MLXArray([0.25, 0.5]),
            "\(prefix).experts.0.gate_proj.weight": MLXArray.zeros([4, 8]),
            "\(prefix).experts.1.gate_proj.weight": MLXArray.ones([4, 8]),
        ])

        XCTAssertNil(sanitized["lm_head_flash.weight"])
        XCTAssertNil(sanitized["\(rowAlphaPrefix).row_alpha"])
        XCTAssertEqual(sanitized["\(rowAlphaPrefix).scales"]?.shape, [2, 1])
        XCTAssertEqual(sanitized["\(rowAlphaPrefix).biases"]?.shape, [2, 1])
        XCTAssertNil(sanitized["\(prefix).experts.0.gate_proj.weight"])
        XCTAssertEqual(
            sanitized["\(prefix).switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8])
    }
}
