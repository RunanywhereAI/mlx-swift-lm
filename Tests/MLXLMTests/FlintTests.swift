// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class FlintTests: XCTestCase {
    private func json(_ overrides: [String: Any] = [:], removing: [String] = []) -> Data {
        var config: [String: Any] = [
            "model_type": "flint", "hidden_size": 64, "num_hidden_layers": 4,
            "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
            "mlp_only_layers": [0],
            "layer_types": [
                "full_attention", "sliding_attention", "sliding_attention", "full_attention",
            ],
            "no_rope_layer_indices": [0, 3],
            "sliding_window": 5, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "rope_parameters": ["rope_theta": 5000.0], "rope_theta": 10000.0,
            "router_score_function": "sigmoid", "softmax_type": "learnable",
            "tie_word_embeddings": false,
        ]
        for key in removing { config.removeValue(forKey: key) }
        config.merge(overrides) { _, new in new }
        return try! JSONSerialization.data(withJSONObject: config)
    }

    private func configuration(_ overrides: [String: Any] = [:], removing: [String] = [])
        throws -> FlintConfiguration
    {
        try JSONDecoder().decode(
            FlintConfiguration.self, from: json(overrides, removing: removing))
    }

    private func assertThrows(
        _ expected: FlintError, _ overrides: [String: Any] = [:], removing: [String] = [],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try configuration(overrides, removing: removing), file: file, line: line
        ) { XCTAssertEqual($0 as? FlintError, expected, file: file, line: line) }
    }

    private func model(_ overrides: [String: Any] = [:]) throws -> FlintModel {
        MLXRandom.seed(7)
        let model = FlintModel(try configuration(overrides))
        eval(model)
        return model
    }

    // MARK: Configuration

    func testDecodesConfiguration() throws {
        let config = try configuration()
        XCTAssertEqual(config.ropeTheta, 5000)
        XCTAssertEqual(config.layerTypes, [.full, .sliding, .sliding, .full])
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testRopeThetaFallsBackToTopLevelThenDefault() throws {
        XCTAssertEqual(try configuration(removing: ["rope_parameters"]).ropeTheta, 10000)
        XCTAssertEqual(
            try configuration(["rope_theta": 2000.0], removing: ["rope_parameters"]).ropeTheta,
            2000)
        XCTAssertEqual(
            try configuration(removing: ["rope_parameters", "rope_theta"]).ropeTheta, 10000)
    }

    func testOptionalKeysDefault() throws {
        let config = try configuration(removing: [
            "tie_word_embeddings", "no_rope_layer_indices", "router_score_function",
            "softmax_type",
        ])
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNoThrow(
            try configuration(["scoring_func": "sigmoid"], removing: ["router_score_function"]))
    }

    func testRejectsLayerTypeCountMismatch() {
        assertThrows(
            .invalidLayerTypes("3 entries for 4 layers"),
            ["layer_types": ["full_attention", "sliding_attention", "full_attention"]])
    }

    func testRejectsUnknownLayerType() {
        assertThrows(
            .invalidLayerTypes("unknown type linear_attention"),
            [
                "layer_types": [
                    "full_attention", "linear_attention", "sliding_attention", "full_attention",
                ]
            ])
    }

    func testRejectsMissingLayerKind() {
        assertThrows(
            .invalidLayerTypes("needs both sliding and full attention layers"),
            ["layer_types": Array(repeating: "full_attention", count: 4)],
            removing: ["no_rope_layer_indices"])
    }

    func testRejectsNoRopeDisagreement() {
        assertThrows(
            .invalidLayerTypes("no_rope_layer_indices disagrees with the full-attention layers"),
            ["no_rope_layer_indices": [0, 1]])
    }

    func testRejectsNonSigmoidRouter() {
        assertThrows(
            .unsupportedRouterScoreFunction("softmax"),
            ["scoring_func": "softmax"], removing: ["router_score_function"])
    }

    func testRejectsNonLearnableSinks() {
        assertThrows(.unsupportedSoftmaxType("vanilla"), ["softmax_type": "vanilla"])
    }

    func testErrorsDescribeThemselves() {
        let errors: [FlintError] = [
            .invalidLayerTypes("x"), .unsupportedRouterScoreFunction("x"),
            .unsupportedSoftmaxType("x"),
            .quantizedKVCacheUnsupported,
        ]
        for error in errors {
            XCTAssertFalse(error.errorDescription?.isEmpty ?? true)
        }
    }

    func testFactoryRegistersFlint() async throws {
        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: json(), modelType: "flint")
        XCTAssertTrue(created is FlintModel)
        XCTAssertTrue(LLMRegistry.shared.contains(id: "RecursiveSI/flint-int4"))
    }

    // MARK: Model structure

    func testLayersFollowConfiguration() throws {
        let model = try model()
        let layers = model.model.layers
        XCTAssertTrue(layers[0].mlp is FlintMLP)
        XCTAssertTrue(layers[1].mlp is FlintSparseMoE)
        XCTAssertNil(layers[0].selfAttn.rope)
        XCTAssertNotNil(layers[1].selfAttn.rope)
        XCTAssertEqual(model.kvHeads, [2, 2, 2, 2])
        XCTAssertEqual(model.loraLayers.count, 4)
    }

    // MARK: Forward pass

    private func tokens(_ count: Int) -> [Int32] {
        (0 ..< count).map { Int32(($0 * 7 + 3) % 64) }
    }

    private func decodeOneByOne(_ model: FlintModel, _ ids: [Int32]) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        var logits = MLXArray(0)
        for id in ids {
            logits = model(MLXArray([id])[.newAxis], cache: cache)[0..., -1, 0...]
        }
        return logits
    }

    func testCachedDecodeMatchesFullForwardBeyondWindow() throws {
        let model = try model()
        for count in [3, 12] {
            let ids = tokens(count)
            let full = model(MLXArray(ids)[.newAxis], cache: nil)[0..., -1, 0...]
            let cached = try decodeOneByOne(model, ids)
            XCTAssertTrue(allClose(full, cached, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }
    }

    func testCachedPrefillMatchesFullForward() throws {
        let model = try model()
        let ids = tokens(12)
        let cache = try model.newCache(parameters: nil)
        _ = model(MLXArray(Array(ids.prefix(9)))[.newAxis], cache: cache)
        let cached = model(MLXArray(Array(ids.suffix(3)))[.newAxis], cache: cache)[0..., -1, 0...]
        let full = model(MLXArray(ids)[.newAxis], cache: nil)[0..., -1, 0...]
        XCTAssertTrue(allClose(full, cached, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    func testSinksChangeAttention() throws {
        let model = try model()
        let input = MLXArray(tokens(6))[.newAxis]
        let before = model(input, cache: nil)
        for layer in model.model.layers {
            layer.selfAttn.update(
                parameters: ModuleParameters.unflattened(["softmax_offset": MLXArray.ones([4]) * 3])
            )
        }
        let after = model(input, cache: nil)
        XCTAssertFalse(allClose(before, after).item(Bool.self))
    }

    func testTiedEmbeddingsUseEmbeddingAsHead() throws {
        let model = try model(["tie_word_embeddings": true])
        XCTAssertNil(model.lmHead)
        let input = MLXArray(tokens(4))[.newAxis]
        let expected = model.model.embedTokens.asLinear(model.model(input, cache: nil))
        XCTAssertTrue(allClose(model(input, cache: nil), expected).item(Bool.self))
    }

    // MARK: Cache

    func testNewCacheUsesRotatingCacheForSlidingLayers() throws {
        let cache = try model().newCache(parameters: GenerateParameters())
        XCTAssertFalse(cache[0] is RotatingKVCache)
        XCTAssertTrue(cache[1] is RotatingKVCache)
        XCTAssertFalse(cache[3] is RotatingKVCache)
    }

    func testQuantizedKVCacheIsRefused() throws {
        XCTAssertThrowsError(try model().newCache(parameters: GenerateParameters(kvBits: 4))) {
            XCTAssertEqual($0 as? FlintError, .quantizedKVCacheUnsupported)
        }
    }

    // MARK: Checkpoint

    func testSanitizeStacksExpertsAndDropsTiedHead() throws {
        let model = try model(["tie_word_embeddings": true])
        var weights: [String: MLXArray] = ["lm_head.weight": MLXArray.zeros([64, 64])]
        for e in 0 ..< 8 {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                weights["model.layers.1.mlp.experts.\(e).\(p).weight"] = MLXArray.zeros([32, 64])
            }
        }
        let sanitized = model.sanitize(weights: weights)
        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape, [8, 32, 64])
    }

    private let quantization: [String: Any] = ["group_size": 32, "bits": 4]

    private func quantizedWeights(embeddingBits: Int, asBytes: Bool) -> [String: MLXArray] {
        func pack(_ name: String, _ w: MLXArray, bits: Int, into weights: inout [String: MLXArray])
        {
            let (q, scales, biases) = quantized(w, groupSize: 32, bits: bits)
            weights["\(name).weight"] = asBytes ? q.view(dtype: .uint8) : q
            weights["\(name).scales"] = scales
            weights["\(name).biases"] = biases!
        }
        var weights: [String: MLXArray] = ["model.norm.weight": MLXArray.ones([64])]
        pack("model.embed_tokens", MLXRandom.normal([64, 64]), bits: embeddingBits, into: &weights)
        pack("lm_head", MLXRandom.normal([64, 64]), bits: 4, into: &weights)
        weights["orphan.scales"] = MLXArray.ones([4, 2])
        return weights
    }

    func testRepacksByteCodesAndQuantizesEmbeddingAtItsRealWidth() throws {
        let model = try model(["quantization": quantization])
        let sanitized = model.sanitize(
            weights: quantizedWeights(embeddingBits: 8, asBytes: true))
        XCTAssertEqual(sanitized["model.embed_tokens.weight"]?.dtype, .uint32)
        XCTAssertEqual(sanitized["model.embed_tokens.weight"]?.shape, [64, 16])
        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [64, 8])
        XCTAssertEqual((model.model.embedTokens as? QuantizedEmbedding)?.bits, 8)
        XCTAssertFalse(model.lmHead is QuantizedLinear)
    }

    func testKeepsUInt32CodesAtConfiguredBits() throws {
        let model = try model(["quantization": quantization])
        let sanitized = model.sanitize(
            weights: quantizedWeights(embeddingBits: 4, asBytes: false))
        XCTAssertEqual(sanitized["model.embed_tokens.weight"]?.dtype, .uint32)
        XCTAssertFalse(model.model.embedTokens is QuantizedEmbedding)
    }

    func testLeavesUnsupportedPackedWidthForTheLoaderToReject() throws {
        let model = try model(["quantization": quantization])
        let sanitized = model.sanitize(
            weights: quantizedWeights(embeddingBits: 2, asBytes: true))
        XCTAssertEqual(sanitized["model.embed_tokens.weight"]?.dtype, .uint8)
        XCTAssertFalse(model.model.embedTokens is QuantizedEmbedding)
    }

    func testUnquantizedCheckpointPassesThrough() throws {
        let sanitized = try model().sanitize(
            weights: quantizedWeights(embeddingBits: 8, asBytes: true))
        XCTAssertEqual(sanitized["model.embed_tokens.weight"]?.dtype, .uint8)
    }

    func testQuantizedCheckpointLoadsAndRuns() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let source = try model()
        var weights = Dictionary(
            uniqueKeysWithValues: source.parameters().flattened().map { ($0.0, $0.1) })
        for (name, bits) in [("model.embed_tokens", 8), ("lm_head", 4)] {
            let (q, scales, biases) = quantized(
                weights["\(name).weight"]!, groupSize: 32, bits: bits)
            weights["\(name).weight"] = q.view(dtype: .uint8)
            weights["\(name).scales"] = scales
            weights["\(name).biases"] = biases!
        }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))

        let model = try self.model(["quantization": quantization])
        try loadWeights(
            modelDirectory: directory, model: model,
            quantization: .init(groupSize: 32, bits: 4))
        XCTAssertEqual((model.model.embedTokens as? QuantizedEmbedding)?.bits, 8)
        XCTAssertEqual((model.lmHead as? QuantizedLinear)?.bits, 4)
        let logits = model(MLXArray(tokens(4))[.newAxis], cache: nil)
        XCTAssertEqual(logits.shape, [1, 4, 64])
        XCTAssertFalse(any(isNaN(logits)).item(Bool.self))
    }
}
