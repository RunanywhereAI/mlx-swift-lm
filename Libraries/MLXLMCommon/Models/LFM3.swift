// LFM3 text stack: sliding attention, a per-head sigmoid gate, sandwich RMSNorm,
// and partial RoPE whose frequencies are spaced over the full head and zeroed
// on the tail. LiquidAI/LFM3-Omni and LiquidAI/d1 both use this backbone.

import Foundation
import MLX
import MLXNN

public struct LFM3RopeSpec: Sendable, Equatable {
    public var theta: Float
    public var partialRotaryFactor: Float

    public init(theta: Float, partialRotaryFactor: Float) {
        self.theta = theta
        self.partialRotaryFactor = partialRotaryFactor
    }
}

public struct LFM3Configuration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let normEps: Float
    public let convBias: Bool
    public let convLCache: Int
    public let layerTypes: [String]
    public let slidingWindow: Int?
    public let useSandwichNorm: Bool
    public let autoAdjustFFN: Bool
    public let blockMultipleOf: Int
    public let blockFFNDimMultiplier: Float
    public let ropeByLayerType: [String: LFM3RopeSpec]
    public let maxPositionEmbeddings: Int?

    public var isLFM2Attention: Bool { modelType == "lfm2" }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case autoAdjustFFN = "block_auto_adjust_ff_dim"
        case blockMultipleOf = "block_multiple_of"
        case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3"
        self.vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 1
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        let decodedHead = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.headDim = decodedHead ?? (hiddenSize / attentionHeads)
        self.normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7
        self.layerTypes =
            try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.useSandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm)
            ?? (modelType == "lfm3")
        self.autoAdjustFFN =
            try container.decodeIfPresent(Bool.self, forKey: .autoAdjustFFN) ?? false
        self.blockMultipleOf =
            try container.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
        self.blockFFNDimMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .blockFFNDimMultiplier) ?? 1
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)

        let ropeJSON = try container.decodeIfPresent(JSONValue.self, forKey: .ropeParameters)
        let fallbackTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.ropeByLayerType = Self.ropeSpecs(ropeJSON, fallbackTheta: fallbackTheta, layerTypes: layerTypes)

        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "LFM3 layer_types (\(layerTypes.count)) must match num_hidden_layers (\(hiddenLayers))."
                ))
        }
        if layerTypes.contains("sliding_attention"), slidingWindow == nil {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "LFM3 sliding_attention layers require sliding_window."))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(normEps, forKey: .normEps)
        try container.encode(convBias, forKey: .convBias)
        try container.encode(convLCache, forKey: .convLCache)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encodeIfPresent(slidingWindow, forKey: .slidingWindow)
        try container.encode(useSandwichNorm, forKey: .useSandwichNorm)
        try container.encodeIfPresent(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
    }

    /// LFM2 audio towers store `intermediate_size` and then round it the way LFM2 does.
    public var mlpSize: Int {
        guard autoAdjustFFN else { return intermediateSize }
        let scaled = Int((Float(2 * intermediateSize) / 3) * blockFFNDimMultiplier)
        let multiple = max(blockMultipleOf, 1)
        return multiple * ((scaled + multiple - 1) / multiple)
    }

    public func rope(for layerType: String) -> LFM3RopeSpec {
        ropeByLayerType[layerType]
            ?? LFM3RopeSpec(theta: 1_000_000, partialRotaryFactor: 1)
    }

    public func isAttention(_ index: Int) -> Bool {
        let kind = layerTypes[index]
        return kind == "full_attention" || kind == "sliding_attention"
    }

    private static func ropeSpecs(
        _ value: JSONValue?, fallbackTheta: Float, layerTypes: [String]
    ) -> [String: LFM3RopeSpec] {
        guard let value else {
            return Dictionary(
                uniqueKeysWithValues: Set(layerTypes).filter { $0 != "conv" }.map {
                    ($0, LFM3RopeSpec(theta: fallbackTheta, partialRotaryFactor: 1))
                })
        }
        if case .object(let object) = value, object["rope_theta"] != nil || object["rope_type"] != nil {
            let spec = spec(from: object, fallbackTheta: fallbackTheta)
            return Dictionary(
                uniqueKeysWithValues: Set(layerTypes).filter { $0 != "conv" }.map { ($0, spec) })
        }
        guard case .object(let object) = value else { return [:] }
        var specs: [String: LFM3RopeSpec] = [:]
        for (name, nested) in object {
            guard case .object(let fields) = nested else { continue }
            specs[name] = spec(from: fields, fallbackTheta: fallbackTheta)
        }
        return specs
    }

    private static func spec(from fields: [String: JSONValue], fallbackTheta: Float) -> LFM3RopeSpec {
        LFM3RopeSpec(
            theta: float(fields["rope_theta"]) ?? fallbackTheta,
            partialRotaryFactor: float(fields["partial_rotary_factor"]) ?? 1)
    }

    private static func float(_ value: JSONValue?) -> Float? {
        switch value {
        case .int(let int): Float(int)
        case .double(let double): Float(double)
        default: nil
        }
    }
}

func lfm3ApplyRoPE(
    queries: MLXArray, keys: MLXArray, offset: Int, headDim: Int, spec: LFM3RopeSpec
) -> (MLXArray, MLXArray) {
    let half = headDim / 2
    let angles = Int(spec.partialRotaryFactor * Float(headDim)) / 2
    var inverse = [Float](repeating: 0, count: half)
    if angles > 0 {
        for index in 0 ..< angles {
            inverse[index] = 1 / pow(spec.theta, Float(2 * index) / Float(headDim))
        }
    }
    let length = queries.dim(2)
    let positions = MLXArray(Array(offset ..< (offset + length))).asType(.float32)
    let frequencies = positions.expandedDimensions(axis: -1) * MLXArray(inverse)
    let embedding = concatenated([frequencies, frequencies], axis: -1)
    let cos = MLX.cos(embedding).expandedDimensions(axes: [0, 1])
    let sin = MLX.sin(embedding).expandedDimensions(axes: [0, 1])

    func rotate(_ values: MLXArray) -> MLXArray {
        let floatValues = values.asType(.float32)
        let first = floatValues[.ellipsis, ..<half]
        let second = floatValues[.ellipsis, half...]
        let rotated = concatenated([-second, first], axis: -1)
        return ((floatValues * cos) + (rotated * sin)).asType(values.dtype)
    }
    return (rotate(queries), rotate(keys))
}

final class LFM3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let hasGate: Bool
    let rope: LFM3RopeSpec

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear?
    @ModuleInfo(key: "out_proj") var outProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm?
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm?
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?

    init(_ args: LFM3Configuration, layerType: String) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.hasGate = !args.isLFM2Attention
        self.rope = args.rope(for: layerType)
        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        if args.isLFM2Attention {
            _outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
            _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
            _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        } else {
            _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
            _gateProj.wrappedValue = Linear(dim, heads, bias: false)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(batch, length, heads, headDim)
        var keys = kProj(x).reshaped(batch, length, kvHeads, headDim)
        var values = vProj(x).reshaped(batch, length, kvHeads, headDim)
        if let qNorm {
            queries = qNorm(queries)
            keys = kNorm!(keys)
        } else {
            queries = qLayerNorm!(queries)
            keys = kLayerNorm!(keys)
        }
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset: Int
        if case .scalar(let value) = cache?.ropeOffset {
            offset = value
        } else {
            offset = cache?.offset ?? 0
        }
        (queries, keys) = lfm3ApplyRoPE(
            queries: queries, keys: keys, offset: offset, headDim: headDim, spec: rope)

        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        if let gateProj {
            let gate = sigmoid(gateProj(x)).expandedDimensions(axis: -1)
            output = output * gate
        }
        let flat = output.reshaped(batch, length, -1)
        if let oProj {
            return oProj(flat)
        }
        return outProj!(flat)
    }
}

final class LFM3ShortConv: Module {
    let hiddenSize: Int
    let kernel: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        self.hiddenSize = args.hiddenSize
        self.kernel = args.convLCache
        _conv.wrappedValue = Conv1d(
            inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: kernel,
            groups: hiddenSize, bias: args.convBias)
        _inProj.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let parts = inProj(x).split(parts: 3, axis: -1)
        let b = parts[0]
        let c = parts[1]
        let values = parts[2]
        var gated = b * values
        var state = cache?[0] ?? MLXArray.zeros([gated.dim(0), kernel - 1, hiddenSize], dtype: gated.dtype)
        gated = concatenated([state, gated], axis: -2)
        if let cache {
            let start = gated.dim(1) - (kernel - 1)
            cache[0] = contiguous(gated[0..., start..., 0...])
            cache.advance(values.dim(1))
        }
        return outProj(c * conv(gated))
    }
}

public final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(dim: Int, hidden: Int) {
        _w1.wrappedValue = Linear(dim, hidden, bias: false)
        _w3.wrappedValue = Linear(dim, hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

public final class LFM3Layer: Module {
    let kind: String
    let sandwich: Bool

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ args: LFM3Configuration, index: Int) {
        self.kind = args.layerTypes[index]
        self.sandwich = args.useSandwichNorm
        if kind == "conv" {
            _conv.wrappedValue = LFM3ShortConv(args)
        } else {
            _attention.wrappedValue = LFM3Attention(args, layerType: kind)
        }
        _feedForward.wrappedValue = LFM3MLP(dim: args.hiddenSize, hidden: args.mlpSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if sandwich {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var mixed: MLXArray
        if let attention {
            mixed = attention(operatorNorm(x), mask: mask, cache: cache)
        } else {
            mixed = conv!(operatorNorm(x), cache: cache as? MambaCache)
        }
        if let postOperatorNorm {
            mixed = postOperatorNorm(mixed)
        }
        let hidden = x + mixed
        var feed = feedForward(ffnNorm(hidden))
        if let postFFNNorm {
            feed = postFFNNorm(feed)
        }
        return hidden + feed
    }
}

public final class LFM3Backbone: Module {
    public let configuration: LFM3Configuration
    public let includesEmbeddings: Bool

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding?
    @ModuleInfo(key: "layers") public var layers: [LFM3Layer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    public init(_ configuration: LFM3Configuration, includesEmbeddings: Bool = true) {
        self.configuration = configuration
        self.includesEmbeddings = includesEmbeddings
        if includesEmbeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        }
        self._layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            LFM3Layer(configuration, index: $0)
        }
        _embeddingNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.normEps)
    }

    public var kvHeads: [Int] {
        configuration.layerTypes.map { $0 == "conv" ? 0 : configuration.kvHeads }
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.layerTypes.map { kind in
            switch kind {
            case "conv":
                MambaCache()
            case "sliding_attention":
                RotatingKVCache(maxSize: configuration.slidingWindow ?? 256)
            default:
                try makeAttentionKVCache(parameters: parameters)
            }
        }
    }

    public func callAsFunction(
        tokens: MLXArray? = nil, embeddings: MLXArray? = nil, cache: [KVCache]? = nil
    ) -> MLXArray {
        var hidden = embeddings ?? embedTokens!(tokens!)
        let fullMask = createAttentionMask(
            h: hidden,
            cache: cache?.first { !($0 is MambaCache) && !($0 is RotatingKVCache) })
        let slidingMask = createAttentionMask(
            h: hidden,
            cache: cache?.first { $0 is RotatingKVCache },
            windowSize: configuration.slidingWindow)
        for (index, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                layer.kind == "sliding_attention" ? slidingMask : fullMask
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return embeddingNorm(hidden)
    }

    public func logits(_ hidden: MLXArray) -> MLXArray {
        embedTokens!.asLinear(hidden)
    }
}

public func lfm3Sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]
    for (name, value) in weights {
        if name.contains("conv.conv.weight"), value.ndim == 3, value.dim(-1) > value.dim(1) {
            sanitized[name] = value.transposed(0, 2, 1)
        } else if value.ndim == 4, value.dim(2) <= 3, value.dim(3) <= 3,
            value.dim(1) > 3 || (value.dim(1) == 1 && value.dim(2) > 1)
        {
            sanitized[name] = value.transposed(0, 2, 3, 1)
        } else {
            sanitized[name] = value
        }
    }
    return sanitized
}
