import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class D1Omni: Module, LanguageModel {
    public let config: D1OmniConfiguration
    @ModuleInfo(key: "encoder") var encoder: D1OmniEncoder
    @ModuleInfo(key: "head") var decision: D1OmniDecision
    @ModuleInfo(key: "vision") var visionModel: D1OmniVision

    public init(_ config: D1OmniConfiguration) {
        self.config = config
        self._encoder.wrappedValue = D1OmniEncoder(config.text)
        self._decision.wrappedValue = D1OmniDecision(config: config)
        self._visionModel.wrappedValue = D1OmniVision(config: config)
    }

    func parameter(_ name: String) -> MLXArray? {
        Dictionary(uniqueKeysWithValues: parameters().flattened())[name]
    }

    func weight(_ name: String) -> MLXArray {
        guard let value = parameter(name) else {
            preconditionFailure("D1 Omni is missing \(name)")
        }
        return value
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (name, value) in weights {
            if name.hasPrefix("audio.") { continue }
            let canonical = name.replacingOccurrences(
                of: "vision.tower.vision_model.", with: "vision.tower."
            )
            .replacingOccurrences(of: "head.scorer.0.", with: "head.scorer.norm.")
            .replacingOccurrences(of: "head.scorer.1.", with: "head.scorer.hidden.")
            .replacingOccurrences(of: "head.scorer.3.", with: "head.scorer.output.")
            result[canonical] = value
        }
        return result
    }

    func linear(_ input: MLXArray, _ name: String) -> MLXArray {
        let output = matmul(input, weight(name + ".weight").T)
        if let bias = parameter(name + ".bias") { return output + bias }
        return output
    }

    func norm(_ input: MLXArray, _ name: String, eps: Float = 1e-5, rms: Bool = false) -> MLXArray {
        if rms {
            let precise = input.asType(.float32)
            let normalized =
                precise * rsqrt(mean(precise * precise, axis: -1, keepDims: true) + eps)
            return normalized.asType(input.dtype) * weight(name + ".weight")
        }
        return MLXFast.layerNorm(
            input, weight: weight(name + ".weight"), bias: parameter(name + ".bias"),
            eps: eps)
    }

    func attention(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, mask: MLXArray)
        -> MLXArray
    {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: pow(Float(queries.dim(-1)), -0.5), mask: .array(mask))
    }

    func trunk(_ input: MLXArray, prefix: Int) -> MLXArray {
        let text = config.text
        let length = input.dim(1)
        let hidden = text.hiddenSize
        let headDim = hidden / text.numAttentionHeads
        let positions = MLXArray(0 ..< length)
        let blocked =
            (positions .< prefix).expandedDimensions(axis: 1)
            .&& (positions .>= prefix).expandedDimensions(axis: 0)
        let mask = which(
            blocked, MLXArray(input.dtype == .float16 ? Float(-65504) : Float(-1e9)),
            MLXArray(Float(0))
        )
        .asType(input.dtype).reshaped(1, 1, length, length)
        let keep = (positions .!= (prefix - 1)).asType(input.dtype).reshaped(1, length, 1)
        let inverse = MLXArray(
            (0 ..< headDim / 2).map { index in
                1 / pow(text.ropeTheta, Float(index * 2) / Float(headDim))
            })
        let angles =
            positions.asType(.float32).expandedDimensions(axis: 1)
            * inverse.expandedDimensions(axis: 0)
        let doubled = concatenated([angles, angles], axis: -1).reshaped(1, 1, length, headDim)
        let cosine = cos(doubled).asType(input.dtype)
        let sine = sin(doubled).asType(input.dtype)
        func rotate(_ value: MLXArray) -> MLXArray {
            let halves = value.split(parts: 2, axis: -1)
            return concatenated([-halves[1], halves[0]], axis: -1)
        }
        var states = input
        for (index, kind) in text.layerTypes.enumerated() {
            let layer = "encoder.layers.\(index)"
            let normalized = norm(states, layer + ".operator_norm", eps: text.normEps, rms: true)
            let mixed: MLXArray
            if kind == "full_attention" {
                let name = layer + ".self_attn"
                let query = norm(
                    linear(normalized, name + ".q_proj").reshaped(
                        1, length, text.numAttentionHeads, headDim), name + ".q_layernorm",
                    eps: text.normEps, rms: true
                ).transposed(0, 2, 1, 3)
                let key = norm(
                    linear(normalized, name + ".k_proj").reshaped(
                        1, length, text.numKeyValueHeads, headDim), name + ".k_layernorm",
                    eps: text.normEps, rms: true
                ).transposed(0, 2, 1, 3)
                let value = linear(normalized, name + ".v_proj").reshaped(
                    1, length, text.numKeyValueHeads, headDim
                ).transposed(0, 2, 1, 3)
                let output = attention(
                    query * cosine + rotate(query) * sine, key * cosine + rotate(key) * sine, value,
                    mask: mask)
                mixed = linear(
                    output.transposed(0, 2, 1, 3).reshaped(1, length, hidden), name + ".out_proj")
            } else {
                let name = layer + ".conv"
                let parts = linear(normalized, name + ".in_proj").split(parts: 3, axis: -1)
                let gated = parts[0] * parts[2]
                let padded = concatenated(
                    [
                        MLXArray.zeros([1, 1, hidden], dtype: gated.dtype), gated,
                        MLXArray.zeros([1, 1, hidden], dtype: gated.dtype),
                    ], axis: 1)
                let kernel = weight(name + ".conv.weight")[0..., 0, 0...]
                let output =
                    padded[0..., 0 ..< length, 0...] * kernel[0..., 0]
                    + padded[0..., 1 ..< (length + 1), 0...] * kernel[0..., 1]
                    + padded[0..., 2 ..< (length + 2), 0...] * keep * kernel[0..., 2]
                mixed = linear(parts[1] * output, name + ".out_proj")
            }
            states = states + mixed
            let normalizedFFN = norm(states, layer + ".ffn_norm", eps: text.normEps, rms: true)
            states =
                states
                + linear(
                    silu(linear(normalizedFFN, layer + ".feed_forward.w1"))
                        * linear(normalizedFFN, layer + ".feed_forward.w3"),
                    layer + ".feed_forward.w2")
            eval(states)
        }
        return norm(states, "encoder.embedding_norm", eps: text.normEps, rms: true)
    }

    public func probabilities(
        state: String, questions: [D1OmniQuestion], tokenizer: any Tokenizer,
        images: [D1OmniImage] = [], audio: MLXArray? = nil
    ) throws -> [[Float]] {
        if audio != nil { throw D1OmniError.audioUnsupported }
        if questions.isEmpty { return [] }
        let media = images.isEmpty ? nil : try visionPrefix(images)
        let offset = media?.dim(1) ?? 0
        let limit = min(
            images.isEmpty ? config.maxLength : config.imageTextLength, config.maxLength - offset)
        guard limit >= 64 else { throw D1OmniError.contextExceeded }
        return try questions.map { question in
            let encoded = try D1OmniPrompt.encode(
                state: state, question: question, tokenizer: tokenizer, maxLength: limit,
                bosTokenId: config.bosTokenId, image: !images.isEmpty)
            guard encoded.tokens.allSatisfy({ (0 ..< config.text.vocabSize).contains($0) }) else {
                throw D1OmniError.invalidInput("token outside vocabulary")
            }
            let embedded = weight("encoder.embed_tokens.weight")[MLXArray(encoded.tokens)]
                .expandedDimensions(axis: 0)
            let input = media.map { concatenated([$0, embedded], axis: 1) } ?? embedded
            let states = trunk(input, prefix: offset)[0..., offset..., 0...]
            let logits = decisionLogits(states, markers: encoded.markers, kind: question.kind)
            return calibratedProbabilities(logits, question: question, image: !images.isEmpty)
        }
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        if input.audio != nil { throw D1OmniError.audioUnsupported }
        throw D1OmniError.generationUnsupported
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        throw D1OmniError.generationUnsupported
    }

}

extension VLMModelFactory {
    public func loadD1Omni(from directory: URL) async throws -> sending D1Omni {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let model = try await typeRegistry.createModel(configuration: data, modelType: "d1_omni")
        guard let model = model as? D1Omni else { throw D1OmniError.invalidConfiguration }
        try loadWeights(modelDirectory: directory, model: model)
        return model
    }
}
