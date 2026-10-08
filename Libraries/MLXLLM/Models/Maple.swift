// Copyright © 2026 DeepGrove AI and RunAnywhere AI.
//
// Reference Swift port of deepgrove-ai/mlx-lm-deepgrove's Maple model. The
// upstream Python implementation contains optional fused Metal decode kernels;
// this implementation intentionally uses stock MLX operators first so every
// supported Apple target shares one correctness path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func mapleTopK(
    _ array: MLXArray, k: Int, axis: Int = -1
) -> (values: MLXArray, indices: MLXArray) {
    let partitionedIndices = argPartition(array, kth: -k, axis: axis)
    let topKIndices = partitionedIndices[.ellipsis, (-k)...]
    let topKValues = takeAlong(array, topKIndices, axis: axis)
    return (topKValues, topKIndices)
}

private struct MapleQuantizationConfiguration: Codable, Sendable {
    var groupSize: Int = 128

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
    }
}

public struct MapleConfiguration: Codable, Sendable {
    var modelType: String = "maple"
    var vocabularySize: Int = 151_936
    var hiddenSize: Int = 2048
    var intermediateSize: Int = 5120
    var moeIntermediateSize: Int = 512
    var hiddenLayers: Int = 24
    var attentionHeads: Int = 16
    var kvHeads: Int = 4
    var headDim: Int = 128
    var numExperts: Int = 256
    var numExpertsPerToken: Int = 8
    var firstDenseLayers: Int = 0
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 10_000
    var ropeScaling: [String: StringOrNumber]? = nil
    var partialRotaryFactor: Float = 0.5
    var maxPositionEmbeddings: Int = 128_000
    var slidingWindow: Int = 512
    var layerTypes: [String] = []
    var useQKNorm: Bool = true
    var useBias: Bool = false
    var tieWordEmbeddings: Bool = false
    private var quantization: MapleQuantizationConfiguration? = nil

    var quantizationGroupSize: Int { quantization?.groupSize ?? 128 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case firstDenseLayers = "first_k_dense_replace"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case useQKNorm = "use_qk_norm"
        case useBias = "use_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "maple"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        firstDenseLayers = try c.decodeIfPresent(Int.self, forKey: .firstDenseLayers) ?? 0
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 128_000
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        quantization = try c.decodeIfPresent(
            MapleQuantizationConfiguration.self, forKey: .quantization)
    }
}

private final class MapleRMSNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(
            x.asType(.float32), weight: weight.asType(.float32), eps: eps
        ).asType(x.dtype)
    }
}

private final class MapleAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let usesRoPE: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MapleRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: MapleRMSNorm?
    let rope: RoPELayer?

    init(_ args: MapleConfiguration, layerIndex: Int) {
        nHeads = args.attentionHeads
        nKVHeads = args.kvHeads
        headDim = args.headDim
        scale = pow(Float(args.headDim), -0.5)
        usesRoPE = args.layerTypes[layerIndex] == "sliding_attention"

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * args.headDim, bias: args.useBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * args.headDim, bias: args.useBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * args.headDim, bias: args.useBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * args.headDim, args.hiddenSize, bias: args.useBias)
        if args.useQKNorm {
            _qNorm.wrappedValue = MapleRMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = MapleRMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        }

        if usesRoPE {
            rope = initializeRope(
                dims: Int(Float(args.headDim) * args.partialRotaryFactor),
                base: args.ropeTheta,
                traditional: false,
                scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
        } else {
            rope = nil
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var q = qProj(x).reshaped(batch, length, nHeads, headDim)
        var k = kProj(x).reshaped(batch, length, nKVHeads, headDim)
        var v = vProj(x).reshaped(batch, length, nKVHeads, headDim)

        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        if let rope {
            let offset = cache?.ropeOffset
            q = applyRotaryPosition(rope, to: q, offset: offset)
            k = applyRotaryPosition(rope, to: k, offset: offset)
        }

        return oProj(
            attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1))
    }
}

private final class MapleDenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: MapleConfiguration) {
        _gateProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.useBias)
        _upProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.useBias)
        _downProj.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class MapleGate: Module {
    @ParameterInfo var weight: MLXArray
    let topK: Int

    init(_ args: MapleConfiguration) {
        _weight.wrappedValue = MLXArray.zeros([args.numExperts, args.hiddenSize])
        topK = args.numExpertsPerToken
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).transposed())
        let probabilities = softmax(logits, axis: -1)
        let (scores, indices) = mapleTopK(probabilities, k: topK, axis: -1)
        return (indices, scores / (scores.sum(axis: -1, keepDims: true) + 1e-20))
    }
}

private final class MapleSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(_ args: MapleConfiguration) {
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: args.hiddenSize, outputDims: args.moeIntermediateSize,
            numExperts: args.numExperts, bias: args.useBias)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: args.hiddenSize, outputDims: args.moeIntermediateSize,
            numExperts: args.numExperts, bias: args.useBias)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: args.moeIntermediateSize, outputDims: args.hiddenSize,
            numExperts: args.numExperts, bias: args.useBias)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, indices: MLXArray) -> MLXArray {
        var x = expandedDimensions(input, axes: [-2, -3])
        let shouldSort = indices.size >= 64
        var selected = indices
        var inverseOrder = MLXArray()
        if shouldSort {
            (x, selected, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gate = gateProj(x, selected, sortedIndices: shouldSort)
        let up = upProj(x, selected, sortedIndices: shouldSort)
        let activated =
            silu(minimum(gate, MLXArray(7.0)))
            * clip(up, min: MLXArray(-7.0), max: MLXArray(7.0))
        x = downProj(activated, selected, sortedIndices: shouldSort)
        if shouldSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }
        return squeezed(x, axis: -2)
    }
}

private final class MapleSparseMoE: Module, UnaryLayer {
    @ModuleInfo var gate: MapleGate
    @ModuleInfo(key: "switch_mlp") var switchMlp: MapleSwitchGLU

    init(_ args: MapleConfiguration) {
        _gate.wrappedValue = MapleGate(args)
        _switchMlp.wrappedValue = MapleSwitchGLU(args)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        let outputs = switchMlp(x, indices: indices)
        return (outputs.asType(.float32) * expandedDimensions(scores, axis: -1))
            .sum(axis: -2)
            .asType(outputs.dtype)
    }
}

private final class MapleDecoderLayer: Module {
    let usesSlidingWindow: Bool
    @ModuleInfo(key: "self_attn") var attention: MapleAttention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputNorm: MapleRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: MapleRMSNorm

    init(_ args: MapleConfiguration, layerIndex: Int) {
        usesSlidingWindow = args.layerTypes[layerIndex] == "sliding_attention"
        _attention.wrappedValue = MapleAttention(args, layerIndex: layerIndex)
        mlp = layerIndex >= args.firstDenseLayers ? MapleSparseMoE(args) : MapleDenseMLP(args)
        _inputNorm.wrappedValue = MapleRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionNorm.wrappedValue = MapleRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let hidden = x + attention(inputNorm(x), mask: mask, cache: cache)
        return hidden + mlp(postAttentionNorm(hidden))
    }
}

private final class MapleInnerModel: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo var layers: [MapleDecoderLayer]
    @ModuleInfo var norm: MapleRMSNorm
    let slidingWindow: Int
    let fullAttentionIndex: Int?
    let slidingAttentionIndex: Int?

    init(_ args: MapleConfiguration) {
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        layers = (0 ..< args.hiddenLayers).map { MapleDecoderLayer(args, layerIndex: $0) }
        _norm.wrappedValue = MapleRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        slidingWindow = args.slidingWindow
        fullAttentionIndex = args.layerTypes.firstIndex(of: "full_attention")
        slidingAttentionIndex = args.layerTypes.firstIndex(of: "sliding_attention")
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = wordEmbeddings(inputs)
        let fullMask =
            fullAttentionIndex.map {
                createAttentionMask(h: hidden, cache: cache?[$0])
            } ?? .none
        let slidingMask =
            slidingAttentionIndex.map {
                createAttentionMask(h: hidden, cache: cache?[$0], windowSize: slidingWindow)
            } ?? .none

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                mask: layer.usesSlidingWindow ? slidingMask : fullMask,
                cache: cache?[index])
        }
        return norm(hidden)
    }
}

public final class MapleModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    private let configuration: MapleConfiguration
    fileprivate let model: MapleInnerModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: MapleConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        model = MapleInnerModel(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead?(hidden) ?? model.wordEmbeddings.asLinear(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter {
            !$0.key.contains("rotary_emb.inv_freq") && !$0.key.hasPrefix("lm_head_flash.")
        }
        result = filterLMHeadWeights(
            from: result, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        let rowAlphaKeys = result.keys.filter { $0.hasSuffix(".row_alpha") }
        for key in rowAlphaKeys {
            guard let alpha = result.removeValue(forKey: key) else { continue }
            let prefix = String(key.dropLast(".row_alpha".count))
            guard let packed = result["\(prefix).weight"] else { continue }
            let groupCount = packed.dim(-1) * 16 / configuration.quantizationGroupSize
            let expanded = expandedDimensions(alpha, axis: -1)
            let scales = contiguous(broadcast(expanded, to: alpha.shape + [groupCount]))
            result["\(prefix).scales"] = scales
            result["\(prefix).biases"] = -scales
        }

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases", "bias"] {
                    let first = "\(prefix).experts.0.\(projection).\(suffix)"
                    guard result[first] != nil else { continue }
                    let values = (0 ..< configuration.numExperts).compactMap {
                        result.removeValue(
                            forKey: "\(prefix).experts.\($0).\(projection).\(suffix)")
                    }
                    if values.count == configuration.numExperts {
                        result["\(prefix).switch_mlp.\(projection).\(suffix)"] = stacked(values)
                    }
                }
            }
        }
        return result
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map {
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: $0.usesSlidingWindow)
        }
    }
}

extension MapleModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
