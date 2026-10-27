import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct LFM3VisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let layerTypes: [String]
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let inChannels: Int
    public let spatialMergeSize: Int
    public let convKernelSize: Int
    public let normEps: Float
    public let ropeTheta: Float

    public var headDim: Int { hiddenSize / attentionHeads }
    public var patchDim: Int { inChannels * temporalPatchSize * patchSize * patchSize }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case layerTypes = "layer_types"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case inChannels = "in_channels"
        case spatialMergeSize = "spatial_merge_size"
        case convKernelSize = "conv_kernel_size"
        case normEps = "norm_eps"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize = try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        spatialMergeSize = try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 3
        normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        let rope = try container.decodeIfPresent(JSONValue.self, forKey: .ropeParameters)
        if case .object(let fields) = rope, case .double(let theta) = fields["rope_theta"] {
            ropeTheta = Float(theta)
        } else if case .object(let fields) = rope, case .int(let theta) = fields["rope_theta"] {
            ropeTheta = Float(theta)
        } else {
            ropeTheta = 1_000_000
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(layerTypes, forKey: .layerTypes)
    }
}

public struct LFM3VLConfiguration: Codable, Sendable {
    public let modelType: String
    public let text: LFM3Configuration
    public let vision: LFM3VisionConfiguration
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let projectorHiddenSize: Int
    public let projectorBias: Bool
    public let projectorNormEps: Float
    public let downsampleFactor: Int
    public let d1Codes: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorNormEps = "projector_layer_norm_eps"
        case downsampleFactor = "downsample_factor"
        case d1Codes = "d1_codes"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3_vl"
        text = try container.decode(LFM3Configuration.self, forKey: .text)
        vision = try container.decode(LFM3VisionConfiguration.self, forKey: .vision)
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125_030
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125_029
        projectorHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        projectorBias = try container.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorNormEps = try container.decodeIfPresent(Float.self, forKey: .projectorNormEps) ?? 1e-6
        downsampleFactor = try container.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        d1Codes = try container.decodeIfPresent(Int.self, forKey: .d1Codes) ?? 2
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(text, forKey: .text)
        try container.encode(vision, forKey: .vision)
        try container.encode(imageTokenId, forKey: .imageTokenId)
        try container.encode(videoTokenId, forKey: .videoTokenId)
    }
}

final class LFM3Projector: Module {
    let merge: Int
    let joined: Int
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(vision: Int, hidden: Int, projector: Int, merge: Int, bias: Bool, eps: Float) {
        self.merge = merge
        self.joined = vision * merge * merge
        _layerNorm.wrappedValue = LayerNorm(dimensions: vision, eps: eps)
        _linear1.wrappedValue = Linear(joined, projector, bias: bias)
        _linear2.wrappedValue = Linear(projector, hidden, bias: bias)
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let normed = layerNorm(tokens).reshaped(-1, joined)
        return linear2(gelu(linear1(normed)))
    }
}

final class LFM3VisionAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

    init(_ config: LFM3VisionConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _outProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
    }

    func callAsFunction(_ hidden: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = hidden.dim(0)
        var queries = qNorm(qProj(hidden).reshaped(length, heads, headDim))
        var keys = kNorm(kProj(hidden).reshaped(length, kvHeads, headDim))
        let values = vProj(hidden).reshaped(length, kvHeads, headDim)
        queries = lfm3Rotate(queries, cos: cos, sin: sin)
        keys = lfm3Rotate(keys, cos: cos, sin: sin)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: keys.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: values.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: scale, mask: .none
        )
        return outProj(output.squeezed(axis: 0).transposed(1, 0, 2).reshaped(length, -1))
    }
}

final class LFM3VisionConv: Module {
    let merge: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: LFM3VisionConfiguration) {
        merge = config.spatialMergeSize
        _inProj.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: false)
        _spatialConv.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize, outputChannels: config.hiddenSize,
            kernelSize: .init(config.convKernelSize), padding: .init(config.convKernelSize / 2),
            groups: config.hiddenSize, bias: false)
        _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ hidden: MLXArray, height: Int, width: Int) -> MLXArray {
        let parts = inProj(hidden).split(parts: 3, axis: -1)
        let gated = parts[0] * parts[2]
        let channels = gated.dim(-1)
        var grid = gated.reshaped(height / merge, width / merge, merge, merge, channels)
        grid = grid.transposed(0, 2, 1, 3, 4).reshaped(1, height, width, channels)
        grid = spatialConv(grid).reshaped(height / merge, merge, width / merge, merge, channels)
        grid = grid.transposed(0, 2, 1, 3, 4)
        return outProj(parts[1] * grid.reshaped(height * width, channels))
    }
}

final class LFM3VisionLayer: Module {
    let isAttention: Bool
    @ModuleInfo(key: "self_attn") var attention: LFM3VisionAttention?
    @ModuleInfo(key: "conv") var conv: LFM3VisionConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ config: LFM3VisionConfiguration, index: Int) {
        isAttention = config.layerTypes[index] == "full_attention"
        if isAttention {
            _attention.wrappedValue = LFM3VisionAttention(config)
        } else {
            _conv.wrappedValue = LFM3VisionConv(config)
        }
        _feedForward.wrappedValue = LFM3MLP(dim: config.hiddenSize, hidden: config.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }
}

final class LFM3VisionTower: Module {
    let config: LFM3VisionConfiguration
    @ModuleInfo(key: "patchifier") var patchifier: Linear
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [LFM3VisionLayer]

    init(_ config: LFM3VisionConfiguration) {
        self.config = config
        _patchifier.wrappedValue = Linear(config.patchDim, config.hiddenSize, bias: false)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { LFM3VisionLayer(config, index: $0) }
    }

    func callAsFunction(_ patches: MLXArray, frames: [THW]) -> MLXArray {
        var hidden = embeddingNorm(patchifier(patches))
        let (cos, sin) = lfm3VisionFrequencies(
            frames: frames, merge: config.spatialMergeSize, headDim: config.headDim,
            theta: config.ropeTheta)
        for layer in layers {
            let normed = layer.operatorNorm(hidden)
            let mixed: MLXArray
            if layer.isAttention {
                mixed = layer.attention!(normed, cos: cos, sin: sin)
            } else {
                var parts: [MLXArray] = []
                var offset = 0
                for frame in frames {
                    let count = frame.t * frame.h * frame.w
                    for _ in 0 ..< frame.t {
                        let slice = normed[offset ..< (offset + frame.h * frame.w)]
                        parts.append(layer.conv!(slice, height: frame.h, width: frame.w))
                        offset += frame.h * frame.w
                    }
                    if count != frame.t * frame.h * frame.w { break }
                }
                mixed = concatenated(parts, axis: 0)
            }
            hidden = hidden + mixed
            hidden = hidden + layer.feedForward(layer.ffnNorm(hidden))
        }
        return hidden
    }
}

final class LFM3Language: Module {
    @ModuleInfo(key: "model") var model: LFM3Backbone
    init(_ configuration: LFM3Configuration) {
        _model.wrappedValue = LFM3Backbone(configuration)
    }
}

public final class LFM3VL: Module, VLMModel, KVCacheDimensionProvider {
    public let config: LFM3VLConfiguration
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: LFM3Language
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3VisionTower
    @ModuleInfo(key: "multi_modal_projector") var projector: LFM3Projector

    public init(_ config: LFM3VLConfiguration) {
        self.config = config
        vocabularySize = config.text.vocabularySize
        kvHeads = config.text.layerTypes.map { $0 == "conv" ? 0 : config.text.kvHeads }
        _languageModel.wrappedValue = LFM3Language(config.text)
        _visionTower.wrappedValue = LFM3VisionTower(config.vision)
        _projector.wrappedValue = LFM3Projector(
            vision: config.vision.hiddenSize, hidden: config.text.hiddenSize,
            projector: config.projectorHiddenSize, merge: config.downsampleFactor,
            bias: config.projectorBias, eps: config.projectorNormEps)
    }

    public var loraLayers: [Module] { languageModel.model.layers }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.model.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        lfm3Sanitize(weights)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return languageModel.model.logits(languageModel.model(tokens: tokens, cache: cache))
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil else {
            return try prepareMultimodal(input, cache: cache, prefill: prefill)
        }
        let stepSize = prefill.resolvedStepSize()
        let text = input.text
        let total = text.tokens.size
        guard total > stepSize else { return .tokens(text) }
        var processed = 0
        try withPreparedCache(cache, lengths: text.sequenceLengths) {
            processed = try prefill.forEachChunk(total: total, reserving: 1) { range in
                _ = self(text.tokens[range], cache: cache)
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
        }
        return .tokens(text[processed...])
    }

    private func prepareMultimodal(
        _ input: LMInput, cache: [KVCache], prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        var embeds = languageModel.model.embedTokens!(tokens)
        if let image = input.image, let frames = image.frames {
            let features = projector(visionTower(image.pixels, frames: frames))
            embeds = try scatter(embeds, tokens: tokens, features: features, tokenId: config.imageTokenId)
        }
        let total = embeds.dim(1)
        var processed = 0
        let result = try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            processed = try prefill.forEachChunk(total: total, reserving: 1) { range in
                _ = languageModel.model(embeddings: embeds[0..., range, 0...], cache: cache)
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            let logits = languageModel.model.logits(
                languageModel.model(embeddings: embeds[0..., processed..., 0...], cache: cache))
            prefill.progress?(total, total)
            return logits
        }
        return .logits(.init(logits: result))
    }

    /// One forward pass. The last position is the answer slot.
    public func score(_ tokenIds: [Int]) -> MLXArray {
        let tokens = MLXArray(tokenIds.map(Int32.init)).expandedDimensions(axis: 0)
        return callAsFunction(tokens, cache: nil)[0, -1]
    }
}

public struct LFM3OmniConfiguration: Codable, Sendable {
    public let modelType: String
    public let text: LFM3Configuration
    public let vision: LFM3VisionConfiguration
    public let audioText: LFM3Configuration
    public let depth: LFM3Configuration
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let projectorHiddenSize: Int
    public let projectorBias: Bool
    public let projectorNormEps: Float
    public let downsampleFactor: Int
    public let numCodebooks: Int
    public let codebookSize: Int
    public let audioChannels: Int
    public let audioHidden: Int
    public let melBins: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_encoder_config"
        case depth = "depth_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorNormEps = "projector_layer_norm_eps"
        case downsampleFactor = "downsample_factor"
        case numCodebooks = "num_codebooks"
        case codebookSize = "codebook_size"
    }

    enum AudioKeys: String, CodingKey {
        case text = "text_config"
        case channels = "conv_channels"
        case hidden = "hidden_size"
        case mel = "num_mel_bins"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3_omni"
        text = try container.decode(LFM3Configuration.self, forKey: .text)
        vision = try container.decode(LFM3VisionConfiguration.self, forKey: .vision)
        depth = try container.decode(LFM3Configuration.self, forKey: .depth)
        let audio = try container.nestedContainer(keyedBy: AudioKeys.self, forKey: .audio)
        audioText = try audio.decode(LFM3Configuration.self, forKey: .text)
        audioChannels = try audio.decode(Int.self, forKey: .channels)
        audioHidden = try audio.decode(Int.self, forKey: .hidden)
        melBins = try audio.decode(Int.self, forKey: .mel)
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125_030
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125_029
        projectorHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        projectorBias = try container.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorNormEps = try container.decodeIfPresent(Float.self, forKey: .projectorNormEps) ?? 1e-6
        downsampleFactor = try container.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        numCodebooks = try container.decode(Int.self, forKey: .numCodebooks)
        codebookSize = try container.decode(Int.self, forKey: .codebookSize)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
    }
}

final class LFM3AudioDownsample: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "conv_mid") var convMid: Conv2d
    @ModuleInfo(key: "lin_mid") var linMid: Conv2d
    @ModuleInfo(key: "conv_end") var convEnd: Conv2d
    @ModuleInfo(key: "lin_end") var linEnd: Conv2d
    @ModuleInfo(key: "lin_out") var linOut: Linear

    init(channels: Int, melBins: Int, hidden: Int) {
        let kernel = IntOrPair(3)
        let stride = IntOrPair(2)
        let pad = IntOrPair(1)
        _convIn.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: channels, kernelSize: kernel, stride: stride,
            padding: pad, bias: true)
        _convMid.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernel, stride: stride,
            padding: pad, groups: channels, bias: true)
        _linMid.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: .init(1), bias: true)
        _convEnd.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernel, stride: stride,
            padding: pad, groups: channels, bias: true)
        _linEnd.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: .init(1), bias: true)
        _linOut.wrappedValue = Linear(melBins * channels / 8, hidden, bias: true)
    }
}

final class LFM3AudioTower: Module {
    @ModuleInfo(key: "downsample") var downsample: LFM3AudioDownsample
    @ModuleInfo(key: "lfm") var lfm: LFM3Backbone

    init(_ config: LFM3OmniConfiguration) {
        _downsample.wrappedValue = LFM3AudioDownsample(
            channels: config.audioChannels, melBins: config.melBins, hidden: config.audioHidden)
        _lfm.wrappedValue = LFM3Backbone(config.audioText, includesEmbeddings: false)
    }
}

final class LFM3AudioProjector: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(audio: Int, text: Int) {
        _layerNorm.wrappedValue = LayerNorm(dimensions: audio, eps: 1e-5)
        _linear1.wrappedValue = Linear(audio, text, bias: true)
        _linear2.wrappedValue = Linear(text, text, bias: true)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        linear2(gelu(linear1(layerNorm(hidden))))
    }
}

final class LFM3Codebook: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(count: Int, dim: Int, eps: Float) {
        _embedding.wrappedValue = Embedding(embeddingCount: count, dimensions: dim)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
    }
}

final class LFM3DepthDecoder: Module {
    @ModuleInfo(key: "depth_linear") var depthLinear: Linear
    @ModuleInfo(key: "depth_embeddings") var depthEmbeddings: [LFM3Codebook]
    @ModuleInfo(key: "model") var model: LFM3Backbone
    @ParameterInfo(key: "mask_embedding") var maskEmbedding: MLXArray

    init(_ config: LFM3OmniConfiguration) {
        _depthLinear.wrappedValue = Linear(
            config.text.hiddenSize, config.numCodebooks * config.depth.hiddenSize, bias: true)
        _depthEmbeddings.wrappedValue = (0 ..< config.numCodebooks).map { _ in
            LFM3Codebook(
                count: config.codebookSize, dim: config.depth.hiddenSize, eps: config.depth.normEps)
        }
        _model.wrappedValue = LFM3Backbone(config.depth, includesEmbeddings: false)
        _maskEmbedding.wrappedValue = MLXArray.zeros([config.depth.hiddenSize])
    }
}

public final class LFM3Omni: Module, VLMModel, KVCacheDimensionProvider {
    public let config: LFM3OmniConfiguration
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: LFM3Language
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3VisionTower
    @ModuleInfo(key: "multi_modal_projector") var projector: LFM3Projector
    @ModuleInfo(key: "audio_tower") var audioTower: LFM3AudioTower
    @ModuleInfo(key: "audio_projector") var audioProjector: LFM3AudioProjector
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding
    @ModuleInfo(key: "depth_decoder") var depthDecoder: LFM3DepthDecoder

    public init(_ config: LFM3OmniConfiguration) {
        self.config = config
        vocabularySize = config.text.vocabularySize
        kvHeads = config.text.layerTypes.map { $0 == "conv" ? 0 : config.text.kvHeads }
        _languageModel.wrappedValue = LFM3Language(config.text)
        _visionTower.wrappedValue = LFM3VisionTower(config.vision)
        _projector.wrappedValue = LFM3Projector(
            vision: config.vision.hiddenSize, hidden: config.text.hiddenSize,
            projector: config.projectorHiddenSize, merge: config.downsampleFactor,
            bias: config.projectorBias, eps: config.projectorNormEps)
        _audioTower.wrappedValue = LFM3AudioTower(config)
        _audioProjector.wrappedValue = LFM3AudioProjector(
            audio: config.audioHidden, text: config.text.hiddenSize)
        _codebookEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numCodebooks * config.codebookSize,
            dimensions: config.text.hiddenSize)
        _depthDecoder.wrappedValue = LFM3DepthDecoder(config)
    }

    public var loraLayers: [Module] { languageModel.model.layers }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.model.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        lfm3Sanitize(weights)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return languageModel.model.logits(languageModel.model(tokens: tokens, cache: cache))
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let stepSize = prefill.resolvedStepSize()
        let text = input.text
        let total = text.tokens.size
        guard input.image == nil, input.video == nil, input.audio == nil, total > stepSize else {
            return .tokens(text)
        }
        var processed = 0
        try withPreparedCache(cache, lengths: text.sequenceLengths) {
            processed = try prefill.forEachChunk(total: total, reserving: 1) { range in
                _ = self(text.tokens[range], cache: cache)
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
        }
        return .tokens(text[processed...])
    }
}

public struct LFM3ProcessorConfiguration: Codable, Sendable {
    public init(from decoder: Decoder) throws {}
    public func encode(to encoder: Encoder) throws {}
}

public struct LFM3MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        var content: [[String: String]] = []
        content += message.images.map { _ in ["type": "image"] }
        content += message.videos.map { _ in ["type": "video"] }
        content.append(["type": "text", "text": message.content])
        return ["role": message.role.rawValue, "content": content]
    }
}

public struct LFM3Processor: UserInputProcessor {
    private let tokenizer: any Tokenizer
    public init(_ config: LFM3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = messages(from: input)
        let tokens: [Int]
        if let templated = try? tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext),
            !templated.isEmpty
        {
            tokens = templated
        } else {
            let text = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
            let bos = tokenizer.bosToken ?? ""
            tokens = D1TokenEncoding.encode(
                "\(bos)<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n",
                tokenizer: tokenizer)
        }
        guard !tokens.isEmpty else {
            throw VLMError.processing("LFM3 prompt produced no tokens.")
        }
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }

    private func messages(from input: UserInput) -> [[String: any Sendable]] {
        switch input.prompt {
        case .messages(let messages):
            return messages
        case .chat(let messages):
            return messages.map { LFM3MessageGenerator().generate(message: $0) }
        case .text(let text):
            return [["role": "user", "content": text]]
        }
    }
}

public struct LFM3Calibration: Sendable {
    public let temperature: Double

    public init(directory: URL) throws {
        let url = directory.appendingPathComponent("calibration.json")
        let data = try Data(contentsOf: url)
        let file = try JSONDecoder().decode(File.self, from: data)
        guard file.defaultTemperature.isFinite, file.defaultTemperature > 0 else {
            throw DecisionError.unsupportedCheckpoint("calibration temperature must be positive")
        }
        temperature = file.defaultTemperature
    }

    private struct File: Decodable {
        let defaultTemperature: Double
        enum CodingKeys: String, CodingKey { case defaultTemperature = "default" }
    }
}

public final class LFM3D1 {
    public let model: LFM3VL
    public let tokenizer: any Tokenizer
    public let calibration: LFM3Calibration

    public init(model: LFM3VL, tokenizer: any Tokenizer, calibration: LFM3Calibration) {
        self.model = model
        self.tokenizer = tokenizer
        self.calibration = calibration
    }

    public static func load(from directory: URL, using tokenizerLoader: any TokenizerLoader)
        async throws -> LFM3D1
    {
        let context = try await VLMModelFactory.shared.load(from: directory, using: tokenizerLoader)
        guard let model = context.model as? LFM3VL else {
            throw DecisionError.unsupportedCheckpoint("d1 requires model_type lfm3_vl")
        }
        return LFM3D1(
            model: model, tokenizer: context.tokenizer,
            calibration: try LFM3Calibration(directory: directory))
    }

    public func decide(_ request: DecisionRequest) throws -> DecisionResponse {
        guard !request.questions.isEmpty else {
            throw DecisionError.invalidRequest("at least one question is required")
        }
        guard request.images.isEmpty else {
            throw DecisionError.invalidRequest("this forward scores text; pass images through the processor")
        }
        var answers: [DecisionAnswer] = []
        var inputTokens = 0
        for question in request.questions {
            let prompt = try encode(state: request.state, question: question)
            inputTokens += prompt.tokens.count
            let row = model.score(prompt.tokens).asType(.float32)
            let logProbabilities = row - logSumExp(row, axis: -1)
            let values = logProbabilities.asArray(Float.self)
            let scores = prompt.groups.map { group in
                group.map { Double(values[$0]) }.max()! / calibration.temperature
            }
            let maximum = scores.max()!
            let weights = scores.map { exp($0 - maximum) }
            let denominator = weights.reduce(0, +)
            let probabilities = weights.map { $0 / denominator }
            answers.append(try DecisionAnswer(question: question, probabilities: probabilities))
        }
        return DecisionResponse(
            answers: answers,
            usage: DecisionUsage(inputTokens: inputTokens, sequences: request.questions.count))
    }

    private struct Encoded {
        let tokens: [Int]
        let groups: [[Int]]
    }

    private func encode(state: String, question: DecisionQuestion) throws -> Encoded {
        let instructions = question.instructions ?? ""
        let body: String
        let groups: [[Int]]
        switch question.criteria {
        case .noul(let whenTrue, let whenFalse):
            let extra =
                whenTrue == nil && whenFalse == nil
                ? "" : "\nYes: \(whenTrue ?? "None")\nNo: \(whenFalse ?? "None")"
            body = "\(instructions)\(extra)\n\nReply with yes or no only."
            let yes = singleTokens(["yes", "Yes", "YES"])
            let no = singleTokens(["no", "No", "NO"])
            guard !yes.isEmpty, !no.isEmpty else {
                throw DecisionError.invalidOutput("yes and no must be single tokens")
            }
            groups = [no, yes]
        case .choice(let options):
            guard !options.isEmpty, Set(options.map(\.key)).count == options.count else {
                throw DecisionError.invalidRequest("choice options must be unique")
            }
            let codes = optionCodes(options.map(\.key), scheme: model.config.d1Codes)
            var lines: [String] = []
            var built: [[Int]] = []
            for (option, code) in zip(options, codes) {
                let description = option.description ?? option.key.replacingOccurrences(of: "_", with: " ")
                lines.append("\(code) \(description)")
                let id = singleTokens([code])
                guard let first = id.first else {
                    throw DecisionError.invalidOutput("option code \(code) is not one token")
                }
                let spaced = singleTokens([" \(code)"]).filter { $0 != first }
                built.append([first] + spaced)
            }
            body = "\(instructions)\n\nOptions:\n\(lines.joined(separator: "\n"))\n\nReply with the option code only."
            groups = built
        case .score(let levels):
            guard (2 ... 10).contains(levels.count) else {
                throw DecisionError.invalidRequest("score needs 2 to 10 levels")
            }
            let legend = levels.enumerated().map { "\($0.offset) \($0.element)" }.joined(separator: "\n")
            body =
                "\(instructions)\n\n\(legend)\n\nReply with a single digit 0-\(levels.count - 1) only."
            groups = try levels.indices.map { index in
                let ids = singleTokens([String(index)])
                guard !ids.isEmpty else {
                    throw DecisionError.invalidOutput("score digit \(index) is not one token")
                }
                return ids
            }
        }
        let bos = tokenizer.bosToken ?? ""
        let text =
            "\(bos)<|im_start|>user\n\(state)\n\nQUESTION:\n\(body)<|im_end|>\n<|im_start|>assistant\n"
        let tokens = D1TokenEncoding.encode(text, tokenizer: tokenizer)
        guard !tokens.isEmpty else {
            throw DecisionError.invalidOutput("decision prompt produced no tokens")
        }
        return Encoded(tokens: tokens, groups: groups)
    }

    private func singleTokens(_ forms: [String]) -> [Int] {
        var seen = Set<Int>()
        return forms.compactMap { form in
            let encoded = D1TokenEncoding.encode(form, tokenizer: tokenizer)
            guard encoded.count == 1, seen.insert(encoded[0]).inserted else { return nil }
            return encoded[0]
        }
    }

    private func optionCodes(_ labels: [String], scheme: Int) -> [String] {
        if labels.allSatisfy({ $0.count == 1 && $0.unicodeScalars.allSatisfy(CharacterSet.letters.contains) }) {
            return labels
        }
        let letters = (65 ... 90).map { String(UnicodeScalar($0)!) }
        if labels.count <= 26 { return Array(letters.prefix(labels.count)) }
        if scheme == 2 {
            let pairs = letters.flatMap { a in letters.map { a + $0 } }
            return Array((letters + pairs).prefix(labels.count))
        }
        return labels.indices.map { String(format: "%02d", $0) }
    }
}

private func scatter(_ embeds: MLXArray, tokens: MLXArray, features: MLXArray, tokenId: Int) throws
    -> MLXArray
{
    let flat = tokens.reshaped(-1).asArray(Int32.self)
    let indices = flat.enumerated().compactMap { $0.element == Int32(tokenId) ? Int32($0.offset) : nil }
    guard indices.count == features.dim(0) else {
        throw VLMError.processing(
            "image tokens \(indices.count) do not match features \(features.dim(0))")
    }
    guard !indices.isEmpty else { return embeds }
    let hidden = embeds.dim(-1)
    let rows = embeds.reshaped([-1, hidden])
    rows[MLXArray(indices), 0...] = features.asType(embeds.dtype)
    return rows.reshaped(embeds.shape)
}

private func lfm3Rotate(_ values: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let half = values.dim(-1) / 2
    let floatValues = values.asType(.float32)
    let first = floatValues[.ellipsis, ..<half]
    let second = floatValues[.ellipsis, half...]
    let rotated = concatenated([-second, first], axis: -1)
    return ((floatValues * cos) + (rotated * sin)).asType(values.dtype)
}

private func lfm3VisionFrequencies(frames: [THW], merge: Int, headDim: Int, theta: Float)
    -> (MLXArray, MLXArray)
{
    let spatial = headDim / 2
    let angles = spatial / 2
    var inverse = [Float](repeating: 0, count: angles)
    for index in 0 ..< angles {
        inverse[index] = 1 / pow(theta, Float(2 * index) / Float(spatial))
    }
    var heights: [Float] = []
    var widths: [Float] = []
    for frame in frames {
        for _ in 0 ..< frame.t {
            for blockH in stride(from: 0, to: frame.h, by: merge) {
                for blockW in stride(from: 0, to: frame.w, by: merge) {
                    for innerH in 0 ..< merge {
                        for innerW in 0 ..< merge {
                            heights.append(Float(blockH + innerH))
                            widths.append(Float(blockW + innerW))
                        }
                    }
                }
            }
        }
    }
    let inv = MLXArray(inverse)
    let freqH = MLXArray(heights).expandedDimensions(axis: -1) * inv
    let freqW = MLXArray(widths).expandedDimensions(axis: -1) * inv
    let pair = concatenated([freqH, freqW], axis: -1)
    let full = concatenated([pair, pair], axis: -1)
    let cos = MLX.cos(full).expandedDimensions(axis: 1)
    let sin = MLX.sin(full).expandedDimensions(axis: 1)
    return (cos, sin)
}
