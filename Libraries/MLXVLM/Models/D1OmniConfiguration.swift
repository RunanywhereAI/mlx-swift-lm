import Foundation
import MLXLMCommon

public struct D1OmniConfiguration: Codable, Sendable, ModelConfigurationValidating {
    public struct Text: Codable, Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let layerTypes: [String]
        public let normEps: Float
        public let convLCache: Int
        public let blockFFNDimMultiplier: Float
        public let blockMultipleOf: Int
        public let ropeTheta: Float

        public var feedForwardSize: Int {
            let reduced = Int(blockFFNDimMultiplier * Float(Int(Float(2 * intermediateSize) / 3)))
            return blockMultipleOf * ((reduced + blockMultipleOf - 1) / blockMultipleOf)
        }

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case layerTypes = "layer_types"
            case normEps = "norm_eps"
            case convLCache = "conv_L_cache"
            case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case blockMultipleOf = "block_multiple_of"
            case ropeTheta = "rope_theta"
        }
    }

    public struct Vision: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numChannels: Int
        public let numPatches: Int
        public let patchSize: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numChannels = "num_channels"
            case numPatches = "num_patches"
            case patchSize = "patch_size"
            case layerNormEps = "layer_norm_eps"
        }
    }

    public let modelType: String
    public let maxLength: Int
    public let imageTextLength: Int
    public let headLayers: Int
    public let projectorHiddenSize: Int
    public let temperatures: [String: Float]
    public let bosTokenId: Int
    public let text: Text
    public let vision: Vision

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case maxLength = "max_length"
        case imageTextLength = "image_text_length"
        case headLayers = "head_layers"
        case projectorHiddenSize = "projector_hidden_size"
        case temperatures
        case bosTokenId = "bos_token_id"
        case text = "text_config"
        case vision = "vision_config"
    }

    public func validateModelConfiguration() throws {
        guard modelType == "d1_omni", maxLength >= 64, imageTextLength >= 64,
            imageTextLength <= maxLength, headLayers > 0, projectorHiddenSize > 0,
            text.hiddenSize > 0, text.hiddenSize % 64 == 0, text.vocabSize > bosTokenId,
            bosTokenId >= 0, text.numHiddenLayers > 0,
            text.layerTypes.count == text.numHiddenLayers,
            text.layerTypes.allSatisfy({ $0 == "conv" || $0 == "full_attention" }),
            text.numAttentionHeads > 0, text.numKeyValueHeads > 0,
            text.hiddenSize % text.numAttentionHeads == 0,
            (text.hiddenSize / text.numAttentionHeads) % 2 == 0,
            text.numAttentionHeads % text.numKeyValueHeads == 0, text.convLCache == 3,
            text.intermediateSize > 0, text.blockMultipleOf > 0,
            text.blockFFNDimMultiplier.isFinite, text.blockFFNDimMultiplier > 0,
            text.normEps.isFinite, text.normEps > 0, text.ropeTheta.isFinite, text.ropeTheta > 0,
            vision.hiddenSize > 0, vision.numAttentionHeads > 0,
            vision.hiddenSize % vision.numAttentionHeads == 0, vision.intermediateSize > 0,
            vision.numHiddenLayers > 0, vision.numChannels == 3, vision.patchSize == 16,
            vision.numPatches > 0,
            Int(sqrt(Double(vision.numPatches))) * Int(sqrt(Double(vision.numPatches)))
                == vision.numPatches,
            vision.layerNormEps.isFinite, vision.layerNormEps > 0,
            temperatures.values.allSatisfy({ $0.isFinite && $0 > 0 })
        else { throw D1OmniError.invalidConfiguration }
    }
}

public enum D1OmniError: Error, LocalizedError, Equatable {
    case invalidConfiguration
    case invalidQuestion(String)
    case missingToken(String)
    case contextExceeded
    case invalidImage
    case audioUnsupported
    case generationUnsupported
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration: "Unsupported or invalid D1 Omni configuration."
        case .invalidQuestion(let reason): "Invalid D1 Omni question: \(reason)"
        case .missingToken(let token): "D1 Omni tokenizer is missing \(token)."
        case .contextExceeded: "D1 Omni options or media exceed the available context."
        case .invalidImage: "D1 Omni requires a nonempty RGB image."
        case .audioUnsupported: "D1 Omni audio scoring is not supported."
        case .generationUnsupported:
            "D1 Omni scores decisions; use loadD1Omni and probabilities instead of generation."
        case .invalidInput(let reason): "Invalid D1 Omni input: \(reason)"
        }
    }
}
