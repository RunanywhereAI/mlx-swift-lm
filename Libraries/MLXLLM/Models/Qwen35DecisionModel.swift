// Copyright © 2026 RunAnywhere AI.
//
// Decision checkpoint runtime: a Qwen3.5 backbone plus the joint decision head.
//
// `load` reuses the standard LLM factory for the backbone (config validation,
// quantization-aware weight loading, tokenizer), then attaches the head from
// the sibling `joint_head.safetensors`. `decide` encodes the request, runs a
// single backbone forward pass, and softmaxes the per-option scores.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Results

/// One answered field.
public struct ClefDecisionAnswer: Sendable {
    public let id: String
    public let kind: ClefQuestionKind
    /// Option keys in prompt order.
    public let optionIds: [String]
    /// Probability per option key.
    public let probabilities: [String: Float]

    /// Winning option of a `choice` field.
    public var topOption: String? {
        optionIds.max { probabilities[$0, default: 0] < probabilities[$1, default: 0] }
    }

    /// Probability mass on `true` for a `noul` field.
    public var probabilityOfTrue: Float? {
        kind == .noul ? probabilities["true"] : nil
    }

    /// Probability-weighted level for a `score` field, on the `0 ..< n` scale.
    public var expectedLevel: Float? {
        guard kind == .score else { return nil }
        return optionIds.enumerated().reduce(into: Float(0)) { sum, entry in
            sum += Float(entry.offset) * probabilities[entry.element, default: 0]
        }
    }
}

/// All answered fields of one request.
public struct ClefDecisionResult: Sendable {
    public let answers: [ClefDecisionAnswer]
}

// MARK: - Errors

public enum ClefDecisionModelError: Error, LocalizedError {
    case unsupportedBackbone(String)
    case missingHeadConfiguration(URL)
    case missingHeadWeights(URL)

    public var errorDescription: String? {
        switch self {
        case .unsupportedBackbone(let type):
            "Decision checkpoints require a Qwen3.5 backbone, found '\(type)'."
        case .missingHeadConfiguration(let url):
            "Missing head configuration at \(url.path)."
        case .missingHeadWeights(let url):
            "Missing head weights at \(url.path)."
        }
    }
}

// MARK: - Model

/// A decision checkpoint: Qwen3.5 backbone plus joint decision head.
public final class ClefDecisionModel {
    public let backbone: Qwen35Model
    public let tokenizer: any Tokenizer
    public let head: ClefJointSchemaHead

    init(backbone: Qwen35Model, tokenizer: any Tokenizer, head: ClefJointSchemaHead) {
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head = head
    }

    /// Whether `directory` contains a decision checkpoint (head config next to
    /// a standard model directory).
    public static func isDecisionCheckpoint(at directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appending(component: "joint_head_config.json").path)
    }

    /// Loads a decision checkpoint from a local model directory.
    public static func load(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader
    ) async throws -> ClefDecisionModel {
        let context = try await LLMModelFactory.shared.load(
            from: directory, using: tokenizerLoader)
        guard let backbone = context.model as? Qwen35Model else {
            throw ClefDecisionModelError.unsupportedBackbone(
                String(describing: type(of: context.model)))
        }

        let configurationURL = directory.appending(component: "joint_head_config.json")
        guard FileManager.default.fileExists(atPath: configurationURL.path) else {
            throw ClefDecisionModelError.missingHeadConfiguration(configurationURL)
        }
        let configuration = try JSONDecoder().decode(
            ClefDecisionHeadConfiguration.self,
            from: Data(contentsOf: configurationURL))

        let head = ClefJointSchemaHead(configuration)

        let weightsURL = directory.appending(component: "joint_head.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw ClefDecisionModelError.missingHeadWeights(weightsURL)
        }
        let (weights, _) = try loadArraysAndMetadata(url: weightsURL)
        let sanitized = ClefJointSchemaHead.sanitize(weights)
        try head.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        return ClefDecisionModel(
            backbone: backbone, tokenizer: context.tokenizer, head: head)
    }

    /// Answers every field of `request` in one forward pass.
    public func decide(
        _ request: ClefDecisionRequest,
        maxLength: Int = ClefDecisionEncoder.defaultMaxLength
    ) throws -> ClefDecisionResult {
        let encoding = try ClefDecisionEncoder.encode(
            request, tokenizer: tokenizer, maxLength: maxLength)

        let ids = MLXArray(encoding.inputIds)
        let hidden = backbone.languageModel.model.forward(
            ids.reshaped(1, -1), cache: nil, applyFinalNorm: true)[0]

        let logits = head(
            hidden, inputIds: ids, record: encoding,
            lexicalLookup: { [backbone] tokenIds in
                Self.lexicalRows(for: backbone, tokenIds: tokenIds)
            })

        let answers = zip(encoding.questions, logits).map { question, scores in
            let probabilities = MLX.softmax(scores.asType(.float32))
            let values = probabilities.asArray(Float.self)
            var byKey = [String: Float]()
            for (id, value) in zip(question.optionIds, values) {
                byKey[id] = value
            }
            return ClefDecisionAnswer(
                id: question.id, kind: question.kind,
                optionIds: question.optionIds, probabilities: byKey)
        }

        return ClefDecisionResult(answers: answers)
    }

    // MARK: - Lexical lookup

    /// Rows of the output embedding used by the head's lexical features,
    /// dequantized when the checkpoint is quantized.
    private static func lexicalRows(for backbone: Qwen35Model, tokenIds: MLXArray) -> MLXArray {
        if let lmHead = backbone.languageModel.lmHead {
            return rows(of: lmHead, tokenIds: tokenIds)
        }
        return rows(of: backbone.languageModel.model.embedTokens, tokenIds: tokenIds)
    }

    private static func rows(of module: Module, tokenIds: MLXArray) -> MLXArray {
        if let quantized = module as? QuantizedLinear {
            return dequantized(
                quantized.weight[tokenIds], scales: quantized.scales[tokenIds],
                biases: quantized.biases.map { $0[tokenIds] },
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
        }
        if let quantized = module as? QuantizedEmbedding {
            return dequantized(
                quantized.weight[tokenIds], scales: quantized.scales[tokenIds],
                biases: quantized.biases.map { $0[tokenIds] },
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
        }
        if let linear = module as? Linear {
            return linear.weight[tokenIds]
        }
        if let embedding = module as? Embedding {
            return embedding.weight[tokenIds]
        }
        preconditionFailure("Unsupported output projection type \(type(of: module))")
    }
}