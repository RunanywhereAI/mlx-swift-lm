// Copyright © 2026 Apple Inc.
//
// Port of mlx_lm/models/flint.py (ml-explore/mlx-lm#1998).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum FlintError: Error, Equatable, LocalizedError {
    case invalidLayerTypes(String)
    case unsupportedRouterScoreFunction(String)
    case unsupportedSoftmaxType(String)
    case quantizedKVCacheUnsupported

    public var errorDescription: String? {
        switch self {
        case .invalidLayerTypes(let reason):
            "Flint layer_types are invalid: \(reason)."
        case .unsupportedRouterScoreFunction(let name):
            "Flint supports only the sigmoid router; config asks for \(name)."
        case .unsupportedSoftmaxType(let name):
            "Flint supports only learnable attention sinks; config asks for \(name)."
        case .quantizedKVCacheUnsupported:
            "Flint's attention sinks have no quantized-attention path; use a full-precision KV cache."
        }
    }
}

// MARK: - Configuration

public struct FlintConfiguration: Codable, Sendable {
    public enum LayerType: String, Codable, Sendable {
        case sliding = "sliding_attention"
        case full = "full_attention"
    }

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var mlpOnlyLayers: [Int]
    var layerTypes: [LayerType]
    var slidingWindow: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var tieWordEmbeddings: Bool
    var quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
    }

    private enum ExtraKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
        case routerScoreFunction = "router_score_function"
        case scoringFunc = "scoring_func"
        case softmaxType = "softmax_type"
        case noRopeLayerIndices = "no_rope_layer_indices"
    }

    private struct RopeParameters: Codable {
        var ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let extra = try decoder.container(keyedBy: ExtraKeys.self)

        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        mlpOnlyLayers = try c.decode([Int].self, forKey: .mlpOnlyLayers)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)

        let ropeParameters = try extra.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta =
            try ropeParameters?.ropeTheta
            ?? c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000

        let rawLayerTypes = try c.decode([String].self, forKey: .layerTypes)
        guard rawLayerTypes.count == hiddenLayers else {
            throw FlintError.invalidLayerTypes(
                "\(rawLayerTypes.count) entries for \(hiddenLayers) layers")
        }
        layerTypes = try rawLayerTypes.map {
            guard let type = LayerType(rawValue: $0) else {
                throw FlintError.invalidLayerTypes("unknown type \($0)")
            }
            return type
        }
        guard layerTypes.contains(.sliding), layerTypes.contains(.full) else {
            throw FlintError.invalidLayerTypes("needs both sliding and full attention layers")
        }
        // RoPE placement is taken from layer_types, as the reference does; the explicit
        // list must agree or the checkpoint describes a different model.
        if let noRope = try extra.decodeIfPresent([Int].self, forKey: .noRopeLayerIndices) {
            let fullLayers = layerTypes.indices.filter { layerTypes[$0] == .full }
            guard Set(noRope) == Set(fullLayers) else {
                throw FlintError.invalidLayerTypes(
                    "no_rope_layer_indices disagrees with the full-attention layers")
            }
        }

        let scoreFunction =
            try extra.decodeIfPresent(String.self, forKey: .routerScoreFunction)
            ?? extra.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        guard scoreFunction == "sigmoid" else {
            throw FlintError.unsupportedRouterScoreFunction(scoreFunction)
        }
        let softmaxType =
            try extra.decodeIfPresent(String.self, forKey: .softmaxType) ?? "learnable"
        guard softmaxType == "learnable" else {
            throw FlintError.unsupportedSoftmaxType(softmaxType)
        }
    }
}

// MARK: - Layers

class FlintAttention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var softmaxOffset: MLXArray

    let rope: RoPE?

    init(_ config: FlintConfiguration, layerType: FlintConfiguration.LayerType) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        scale = pow(Float(config.headDim), -0.5)

        _wq.wrappedValue = Linear(config.hiddenSize, heads * config.headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * config.headDim, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * config.headDim, bias: false)
        _wo.wrappedValue = Linear(heads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _softmaxOffset.wrappedValue = MLXArray.zeros([heads])

        rope =
            layerType == .sliding
            ? RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
            : nil
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qNorm(wq(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        var k = kNorm(wk(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            q = applyRotaryPosition(rope, to: q, offset: offset)
            k = applyRotaryPosition(rope, to: k, offset: offset)
        }

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask, sinks: softmaxOffset)
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class FlintMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class FlintSparseMoE: Module, UnaryLayer {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP

    init(_ config: FlintConfiguration) {
        topK = config.numExpertsPerTok
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts)
        _sharedExpert.wrappedValue = FlintMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scores = sigmoid(gate(x.asType(.float32)))
        let kth = scores.dim(-1) - topK
        let inds = argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        var weights = takeAlong(scores, inds, axis: -1)
        weights = weights / weights.sum(axis: -1, keepDims: true)
        let y = switchMLP(x, inds)
        let routed = (y * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        return routed + sharedExpert(x)
    }
}

class FlintDecoderLayer: Module {
    let layerType: FlintConfiguration.LayerType

    @ModuleInfo(key: "self_attn") var selfAttn: FlintAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: FlintConfiguration, index: Int) {
        layerType = config.layerTypes[index]
        _selfAttn.wrappedValue = FlintAttention(config, layerType: layerType)
        mlp =
            config.mlpOnlyLayers.contains(index)
            ? FlintMLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            : FlintSparseMoE(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class FlintModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [FlintDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let fullIndex: Int
    let slidingIndex: Int
    let slidingWindow: Int

    init(_ config: FlintConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { FlintDecoderLayer(config, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        // The configuration guarantees both layer types are present.
        fullIndex = config.layerTypes.firstIndex(of: .full)!
        slidingIndex = config.layerTypes.firstIndex(of: .sliding)!
        slidingWindow = config.slidingWindow
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: cache?[fullIndex])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingIndex], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layer.layerType == .sliding ? slidingMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class FlintModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: FlintModelInner
    let configuration: FlintConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: FlintConfiguration) {
        configuration = config
        modelType = config.modelType
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        model = FlintModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights = weights.filter { !$0.key.hasPrefix("lm_head.") }
        }
        for layer in 0 ..< configuration.hiddenLayers
        where !configuration.mlpOnlyLayers.contains(layer) {
            let prefix = "model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for key in ["weight", "scales", "biases"]
                where weights["\(prefix).experts.0.\(projection).\(key)"] != nil {
                    let experts = (0 ..< configuration.numExperts).compactMap {
                        weights.removeValue(forKey: "\(prefix).experts.\($0).\(projection).\(key)")
                    }
                    weights["\(prefix).switch_mlp.\(projection).\(key)"] = stacked(experts)
                }
            }
        }
        repackQuantizedWeights(&weights)
        return weights
    }

    /// Recursive's INT4 export stores packed codes as `uint8` and quantizes the embedding at
    /// 8 bits while its config names only the 4-bit default. MLX kernels read `uint32` words,
    /// so the bytes are reinterpreted in place (both are little-endian packings), and each
    /// layer's bit width is recovered from the packed width rather than trusted from config.
    private func repackQuantizedWeights(_ weights: inout [String: MLXArray]) {
        guard let configured = configuration.quantization else { return }
        for (key, scales) in weights where key.hasSuffix(".scales") {
            let layer = String(key.dropLast(".scales".count))
            guard let packed = weights["\(layer).weight"] else { continue }
            let inputs = scales.dim(-1) * configured.groupSize
            let bytesPerRow = packed.dim(-1) * packed.dtype.size
            let bits = bytesPerRow * 8 / inputs
            // Any other width is left as is; the loader's shape check then rejects it.
            guard bits == 4 || bits == 8, bytesPerRow * 8 == bits * inputs else { continue }
            if packed.dtype == .uint8 {
                weights["\(layer).weight"] = packed.view(dtype: .uint32)
            }
            // Quantize the layer at its real width now; the loader skips modules that are
            // already quantized, so it cannot re-quantize this one at the config default.
            if bits != configured.bits {
                quantize(model: self) { path, _ in
                    path == layer ? (configured.groupSize, bits) : nil
                }
            }
        }
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        if let strategy = try parameters?.kvCachePlan().configuration?.strategy,
            strategy != .fullPrecision
        {
            throw FlintError.quantizedKVCacheUnsupported
        }
        return try model.layers.map { layer in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layer.layerType == .sliding)
        }
    }
}

extension FlintModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
