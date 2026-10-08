import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum GLiNERError: Error, LocalizedError {
    case classificationOnly
    case invalidConfiguration(String)
    case invalidInput(String)
    case invalidCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .classificationOnly:
            "GLiNER is a label classifier, with no token-generation head. Use GLiNERClassifier.load(from:using:) and classify(text:tasks:)."
        case .invalidConfiguration(let detail): "Invalid GLiNER configuration: \(detail)"
        case .invalidInput(let detail): "Invalid GLiNER input: \(detail)"
        case .invalidCheckpoint(let detail): "Invalid GLiNER checkpoint: \(detail)"
        }
    }
}

struct GLiNERConfiguration: Decodable {
    let modelType: String
    let architecture: String
    let countingLayer: String
    let tokenPooling: String
    let useMoE: Bool?
    let encoderConfig: GLiNEREncoderConfiguration
    let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architecture
        case countingLayer = "counting_layer"
        case tokenPooling = "token_pooling"
        case useMoE = "use_moe"
        case encoderConfig = "encoder_config"
        case quantization
    }

    func validate() throws {
        guard modelType == "extractor", architecture == "span", countingLayer == "count_lstm",
            tokenPooling == "first", useMoE != true
        else { throw GLiNERError.invalidConfiguration("unsupported extractor architecture") }
        try encoderConfig.validate()
        if let quantization {
            guard quantization.bits == 4, quantization.groupSize == 64,
                quantization.mode == .affine
            else { throw GLiNERError.invalidConfiguration("expected affine 4-bit groups of 64") }
        }
    }
}

public struct GLiNEREncoderConfiguration: Decodable {
    let hiddenSize: Int
    let intermediateSize: Int
    let heads: Int
    let layers: Int
    let vocabularySize: Int
    let epsilon: Float
    let positionBuckets: Int
    let maxPositionEmbeddings: Int
    let maxRelativePositions: Int
    let relativeAttention: Bool
    let shareAttentionKey: Bool
    let positionBiasedInput: Bool
    let relativeNormalization: String
    let typeVocabularySize: Int
    let positionAttentionTypes: [String]
    let convolutionKernelSize: Int?
    let hiddenActivation: String

    enum CodingKeys: String, CodingKey {
        case encoderConfig = "encoder_config"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case heads = "num_attention_heads"
        case layers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case epsilon = "layer_norm_eps"
        case positionBuckets = "position_buckets"
        case maxPositionEmbeddings = "max_position_embeddings"
        case maxRelativePositions = "max_relative_positions"
        case relativeAttention = "relative_attention"
        case shareAttentionKey = "share_att_key"
        case positionBiasedInput = "position_biased_input"
        case relativeNormalization = "norm_rel_ebd"
        case typeVocabularySize = "type_vocab_size"
        case positionAttentionTypes = "pos_att_type"
        case convolutionKernelSize = "conv_kernel_size"
        case hiddenActivation = "hidden_act"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if container.contains(.encoderConfig) {
            self = try container.decode(GLiNEREncoderConfiguration.self, forKey: .encoderConfig)
            return
        }
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        heads = try container.decode(Int.self, forKey: .heads)
        layers = try container.decode(Int.self, forKey: .layers)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        epsilon = try container.decode(Float.self, forKey: .epsilon)
        positionBuckets = try container.decode(Int.self, forKey: .positionBuckets)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        maxRelativePositions = try container.decode(Int.self, forKey: .maxRelativePositions)
        relativeAttention = try container.decode(Bool.self, forKey: .relativeAttention)
        shareAttentionKey = try container.decode(Bool.self, forKey: .shareAttentionKey)
        positionBiasedInput = try container.decode(Bool.self, forKey: .positionBiasedInput)
        relativeNormalization = try container.decode(String.self, forKey: .relativeNormalization)
        typeVocabularySize = try container.decode(Int.self, forKey: .typeVocabularySize)
        positionAttentionTypes = try container.decode([String].self, forKey: .positionAttentionTypes)
        convolutionKernelSize = try container.decodeIfPresent(Int.self, forKey: .convolutionKernelSize)
        hiddenActivation = try container.decode(String.self, forKey: .hiddenActivation)
    }

    var relativeLimit: Int {
        maxRelativePositions > 0 ? maxRelativePositions : maxPositionEmbeddings
    }

    func validate() throws {
        guard hiddenSize > 0, intermediateSize > 0, heads > 0, hiddenSize % heads == 0,
            layers > 0, vocabularySize > 0, epsilon.isFinite, epsilon > 0,
            positionBuckets >= 4, positionBuckets % 2 == 0,
            relativeLimit > positionBuckets / 2 + 1,
            relativeAttention, shareAttentionKey, !positionBiasedInput,
            relativeNormalization == "layer_norm", typeVocabularySize == 0,
            positionAttentionTypes.sorted() == ["c2p", "p2c"],
            (convolutionKernelSize ?? 0) == 0, hiddenActivation == "gelu"
        else { throw GLiNERError.invalidConfiguration("unsupported DeBERTa relative attention") }
    }
}

enum GLiNERRelativePositions {
    static func bucket(_ position: Int, buckets: Int, maximum: Int) -> Int {
        let midpoint = buckets / 2
        let magnitude = abs(position)
        guard magnitude > midpoint else { return position }
        let logarithmic = ceil(
            log(Float(magnitude) / Float(midpoint))
                / log(Float(maximum - 1) / Float(midpoint)) * Float(midpoint - 1))
        return (Int(logarithmic) + midpoint) * (position < 0 ? -1 : 1)
    }

    static func indices(length: Int, buckets: Int, maximum: Int) -> (MLXArray, MLXArray) {
        var contentToPosition = [Int32]()
        var positionToContent = [Int32]()
        for query in 0 ..< length {
            for key in 0 ..< length {
                let relative = bucket(query - key, buckets: buckets, maximum: maximum)
                contentToPosition.append(Int32(min(max(relative + buckets, 0), 2 * buckets - 1)))
                positionToContent.append(Int32(min(max(-relative + buckets, 0), 2 * buckets - 1)))
            }
        }
        return (
            MLXArray(contentToPosition).reshaped(length, length),
            MLXArray(positionToContent).reshaped(length, length)
        )
    }
}

final class GLiNERDisentangledAttention: Module {
    @ModuleInfo(key: "query_proj") var query: Linear
    @ModuleInfo(key: "key_proj") var key: Linear
    @ModuleInfo(key: "value_proj") var value: Linear
    let heads: Int

    init(_ config: GLiNEREncoderConfiguration) {
        heads = config.heads
        _query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func split(_ input: MLXArray) -> MLXArray {
        input.reshaped(input.dim(0), input.dim(1), heads, -1).transposed(0, 2, 1, 3)
    }

    func callAsFunction(
        _ input: MLXArray, keyBias: MLXArray, relative: MLXArray,
        contentIndices: MLXArray, positionIndices: MLXArray
    ) -> MLXArray {
        let queries = split(query(input))
        let keys = split(key(input))
        let values = split(value(input))
        let positionQueries = split(query(relative))
        let positionKeys = split(key(relative))
        let scale = 1 / sqrt(Float(queries.dim(-1) * 3))
        let shape = [input.dim(0), heads, input.dim(1), input.dim(1)]
        let contentBias = takeAlong(
            matmul(queries, positionKeys.transposed(0, 1, 3, 2)),
            broadcast(contentIndices, to: shape), axis: -1)
        let positionBias = takeAlong(
            matmul(keys, positionQueries.transposed(0, 1, 3, 2)),
            broadcast(positionIndices, to: shape), axis: -1)
        let bias = (contentBias + positionBias.transposed(0, 1, 3, 2)) * scale + keyBias
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: .array(bias.asType(queries.dtype)))
        return output.transposed(0, 2, 1, 3).reshaped(input.dim(0), input.dim(1), -1)
    }
}

final class GLiNERDenseOutput: Module {
    @ModuleInfo var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(input: Int, output: Int, epsilon: Float) {
        _dense.wrappedValue = Linear(input, output)
        _norm.wrappedValue = LayerNorm(dimensions: output, eps: epsilon)
    }

    func callAsFunction(_ input: MLXArray, residual: MLXArray) -> MLXArray {
        norm(dense(input) + residual)
    }
}

final class GLiNERAttention: Module {
    @ModuleInfo(key: "self") var attention: GLiNERDisentangledAttention
    let output: GLiNERDenseOutput

    init(_ config: GLiNEREncoderConfiguration) {
        _attention.wrappedValue = GLiNERDisentangledAttention(config)
        output = GLiNERDenseOutput(
            input: config.hiddenSize, output: config.hiddenSize, epsilon: config.epsilon)
    }

    func callAsFunction(
        _ input: MLXArray, keyBias: MLXArray, relative: MLXArray,
        contentIndices: MLXArray, positionIndices: MLXArray
    ) -> MLXArray {
        output(
            attention(
                input, keyBias: keyBias, relative: relative,
                contentIndices: contentIndices, positionIndices: positionIndices), residual: input)
    }
}

final class GLiNERIntermediate: Module {
    @ModuleInfo var dense: Linear
    init(_ config: GLiNEREncoderConfiguration) {
        _dense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray { gelu(dense(input)) }
}

final class GLiNERDebertaLayer: Module {
    let attention: GLiNERAttention
    let intermediate: GLiNERIntermediate
    let output: GLiNERDenseOutput

    init(_ config: GLiNEREncoderConfiguration) {
        attention = GLiNERAttention(config)
        intermediate = GLiNERIntermediate(config)
        output = GLiNERDenseOutput(
            input: config.intermediateSize, output: config.hiddenSize, epsilon: config.epsilon)
    }

    func callAsFunction(
        _ input: MLXArray, keyBias: MLXArray, relative: MLXArray,
        contentIndices: MLXArray, positionIndices: MLXArray
    ) -> MLXArray {
        let attended = attention(
            input, keyBias: keyBias, relative: relative,
            contentIndices: contentIndices, positionIndices: positionIndices)
        return output(intermediate(attended), residual: attended)
    }
}

final class GLiNEREmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var words: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ config: GLiNEREncoderConfiguration) {
        _words.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.epsilon)
    }

    func callAsFunction(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        norm(words(ids)) * mask[.ellipsis, .newAxis]
    }
}

final class GLiNEREncoderLayers: Module {
    let layer: [GLiNERDebertaLayer]
    @ModuleInfo(key: "rel_embeddings") var relative: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ config: GLiNEREncoderConfiguration) {
        layer = (0 ..< config.layers).map { _ in GLiNERDebertaLayer(config) }
        _relative.wrappedValue = Embedding(
            embeddingCount: config.positionBuckets * 2, dimensions: config.hiddenSize)
        _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.epsilon)
    }
}

final class GLiNERDeberta: Module {
    let embeddings: GLiNEREmbeddings
    let encoder: GLiNEREncoderLayers
    let config: GLiNEREncoderConfiguration

    init(_ config: GLiNEREncoderConfiguration) {
        self.config = config
        embeddings = GLiNEREmbeddings(config)
        encoder = GLiNEREncoderLayers(config)
    }

    func callAsFunction(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = embeddings(ids, mask: mask.asType(encoder.norm.weight!.dtype))
        let keyBias = Self.keyBias(mask)
        let relative = encoder.norm(encoder.relative.weight)[.newAxis]
        let (contentIndices, positionIndices) = GLiNERRelativePositions.indices(
            length: ids.dim(1), buckets: config.positionBuckets, maximum: config.relativeLimit)
        for layer in encoder.layer {
            hidden = layer(
                hidden, keyBias: keyBias, relative: relative,
                contentIndices: contentIndices, positionIndices: positionIndices)
        }
        return hidden
    }

    static func keyBias(_ mask: MLXArray) -> MLXArray {
        MLX.where(
            mask[0..., .newAxis, .newAxis, 0...] .> 0,
            MLXArray(Float(0)), MLXArray(-Float.infinity))
    }
}

final class GLiNERClassificationHead: Module {
    let fc1: Linear
    let fc2: Linear

    init(hidden: Int) {
        fc1 = Linear(hidden, hidden * 2)
        fc2 = Linear(hidden * 2, 1)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray { fc2(relu(fc1(input))) }
}

final class GLiNERClassificationNetwork: Module, BaseLanguageModel {
    let encoder: GLiNERDeberta
    let classifier: GLiNERClassificationHead

    init(_ config: GLiNEREncoderConfiguration) {
        encoder = GLiNERDeberta(config)
        classifier = GLiNERClassificationHead(hidden: config.hiddenSize)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        (try? Self.classificationWeights(weights, hidden: encoder.config.hiddenSize)) ?? [:]
    }

    static func classificationWeights(
        _ weights: [String: MLXArray], hidden: Int
    ) throws -> [String: MLXArray] {
        let ignoredShapes = extractionShapes(hidden: hidden)
        var result = [String: MLXArray]()
        for (name, value) in weights {
            if let shape = ignoredShapes[name] {
                guard value.shape == shape,
                    [DType.float32, .float16, .bfloat16].contains(value.dtype)
                else {
                    throw GLiNERError.invalidCheckpoint(
                        "extraction tensor \(name) has invalid shape or dtype")
                }
                continue
            }
            var mapped = name
            for (index, layer) in [("0", "fc1"), ("2", "fc2")] {
                let prefix = "classifier.\(index)."
                if name.hasPrefix(prefix) {
                    mapped = "classifier.\(layer)." + name.dropFirst(prefix.count)
                }
            }
            guard result[mapped] == nil else {
                throw GLiNERError.invalidCheckpoint("duplicate tensor \(mapped)")
            }
            result[mapped] = value
        }
        return result
    }

    static func extractionShapes(hidden: Int) -> [String: [Int]] {
        var shapes = [String: [Int]]()
        func mlp(_ name: String, input: Int, middle: Int, output: Int, second: String) {
            for first in ["0", "fc1"] {
                shapes["\(name).\(first).weight"] = [middle, input]
                shapes["\(name).\(first).bias"] = [middle]
            }
            for last in [second, "fc2"] {
                shapes["\(name).\(last).weight"] = [output, middle]
                shapes["\(name).\(last).bias"] = [output]
            }
        }
        mlp("count_pred", input: hidden, middle: hidden * 2, output: 20, second: "2")
        mlp(
            "count_embed.projector", input: hidden * 2, middle: hidden * 4, output: hidden,
            second: "2")
        for name in ["project_start", "project_end", "out_project"] {
            mlp(
                "span_rep.span_rep_layer.\(name)",
                input: name == "out_project" ? hidden * 2 : hidden,
                middle: hidden * 4, output: hidden, second: "3")
        }
        shapes["count_embed.pos_embedding.weight"] = [20, hidden]
        for name in ["weight_ih_l0", "weight_hh_l0"] {
            shapes["count_embed.gru.\(name)"] = [3 * hidden, hidden]
        }
        for name in ["bias_ih_l0", "bias_hh_l0"] {
            shapes["count_embed.gru.\(name)"] = [3 * hidden]
        }
        return shapes
    }
}
