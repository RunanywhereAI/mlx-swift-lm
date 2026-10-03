// Copyright © 2026 RunAnywhere AI.
//
// Joint decision head for decision checkpoints built on a Qwen3.5 backbone.
//
// The head consumes pooled hidden states over question and option spans and
// produces one score per option in a single forward pass. This is a Swift port
// of the reference `JointSchemaHead` shipped with the model checkpoint
// (`joint_head.safetensors` / `joint_head_config.json`), preserving module
// topology and math so probabilities match the reference bit-for-bit up to
// floating point associativity.

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// Configuration decoded from `joint_head_config.json`.
public struct ClefDecisionHeadConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 4096
    public var width: Int = 1024
    public var routingLayers: Int = 2
    public var layers: Int = 4
    public var heads: Int = 16
    public var feedforward: Int = 4096

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case width
        case routingLayers = "routing_layers"
        case layers
        case heads
        case feedforward
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.width = try container.decode(Int.self, forKey: .width)
        self.routingLayers = try container.decode(Int.self, forKey: .routingLayers)
        self.layers = try container.decode(Int.self, forKey: .layers)
        self.heads = try container.decode(Int.self, forKey: .heads)
        self.feedforward = try container.decode(Int.self, forKey: .feedforward)
    }
}

// MARK: - Shared helpers

/// Row-wise L2 normalization with a guarded denominator.
func clefL2Normalized(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
    let norm = sqrt((x * x).sum(axis: -1, keepDims: true))
    return x / maximum(norm, MLXArray(eps))
}

// MARK: - Attention

/// Packed multi-head attention with Q, K and V packed in one weight, matching
/// `torch.nn.MultiheadAttention` with `batch_first=True`.
final class ClefMultiheadAttention: Module {
    let heads: Int

    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(width: Int, heads: Int) {
        self.heads = heads
        _inProjWeight.wrappedValue = MLXArray.zeros([3 * width, width])
        _inProjBias.wrappedValue = MLXArray.zeros([3 * width])
        _outProj.wrappedValue = Linear(width, width, bias: true)
        super.init()
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let d = inProjWeight.dim(1)
        let weight = inProjWeight
        let bias = inProjBias

        let queries = q.matmul(weight[0 ..< d].T) + bias[0 ..< d]
        let keys = k.matmul(weight[d ..< 2 * d].T) + bias[d ..< 2 * d]
        let values = v.matmul(weight[(2 * d)...].T) + bias[(2 * d)...]

        let batch = queries.dim(0)
        let queryLength = queries.dim(1)
        let keyLength = keys.dim(1)
        let headDim = d / heads

        let reshapedQueries = queries.reshaped(batch, queryLength, heads, headDim)
            .transposed(0, 2, 1, 3)
        let reshapedKeys = keys.reshaped(batch, keyLength, heads, headDim)
            .transposed(0, 2, 1, 3)
        let reshapedValues = values.reshaped(batch, keyLength, heads, headDim)
            .transposed(0, 2, 1, 3)

        let attention = MLXFast.scaledDotProductAttention(
            queries: reshapedQueries, keys: reshapedKeys, values: reshapedValues,
            scale: pow(Float(headDim), -0.5), mask: .none)

        let merged = attention.transposed(0, 2, 1, 3)
            .reshaped(batch, queryLength, d)
        return outProj(merged)
    }
}

// MARK: - Feed-forward

final class ClefDecisionFeedForward: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(width: Int, feedforward: Int) {
        _fc1.wrappedValue = Linear(width, feedforward)
        _fc2.wrappedValue = Linear(feedforward, width)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

// MARK: - Evidence routing

/// Cross-attention block that lets the option queries read the prompt memory.
final class ClefEvidenceRoutingLayer: Module {
    @ModuleInfo(key: "query_norm") var queryNorm: LayerNorm
    @ModuleInfo(key: "memory_norm") var memoryNorm: LayerNorm
    @ModuleInfo(key: "attention") var attention: ClefMultiheadAttention
    @ModuleInfo(key: "feedforward_norm") var feedforwardNorm: LayerNorm
    @ModuleInfo(key: "feedforward") var feedforward: ClefDecisionFeedForward

    init(width: Int, heads: Int, feedforward: Int) {
        _queryNorm.wrappedValue = LayerNorm(dimensions: width)
        _memoryNorm.wrappedValue = LayerNorm(dimensions: width)
        _attention.wrappedValue = ClefMultiheadAttention(width: width, heads: heads)
        _feedforwardNorm.wrappedValue = LayerNorm(dimensions: width)
        _feedforward.wrappedValue = ClefDecisionFeedForward(width: width, feedforward: feedforward)
        super.init()
    }

    func callAsFunction(_ queries: MLXArray, _ memory: MLXArray) -> MLXArray {
        let m = memoryNorm(memory)
        let routed = queries + attention(queryNorm(queries), m, m)
        return routed + feedforward(feedforwardNorm(routed))
    }
}

// MARK: - Joint decoder

/// Pre-norm transformer decoder block (`norm_first=True`) used by the joint pass.
final class ClefJointDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: ClefMultiheadAttention
    @ModuleInfo(key: "multihead_attn") var multiheadAttention: ClefMultiheadAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    init(width: Int, heads: Int, feedforward: Int) {
        _selfAttention.wrappedValue = ClefMultiheadAttention(width: width, heads: heads)
        _multiheadAttention.wrappedValue = ClefMultiheadAttention(width: width, heads: heads)
        _linear1.wrappedValue = Linear(width, feedforward)
        _linear2.wrappedValue = Linear(feedforward, width)
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _norm2.wrappedValue = LayerNorm(dimensions: width)
        _norm3.wrappedValue = LayerNorm(dimensions: width)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ memory: MLXArray) -> MLXArray {
        let h = norm1(x)
        var out = x + selfAttention(h, h, h)
        out = out + multiheadAttention(norm2(out), memory, memory)
        return out + linear2(gelu(linear1(norm3(out))))
    }
}

// MARK: - Joint schema head

/// Scores every option of every question in one pass from backbone hidden
/// states and span annotations.
public final class ClefJointSchemaHead: Module {
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: LayerNorm
    @ModuleInfo(key: "memory_projection") var memoryProjection: Linear
    @ModuleInfo(key: "question_projection") var questionProjection: Linear
    @ModuleInfo(key: "option_question_projection") var optionQuestionProjection: Linear
    @ModuleInfo(key: "global_projection") var globalProjection: Linear
    @ModuleInfo(key: "option_context_projection") var optionContextProjection: Linear
    @ModuleInfo(key: "option_lexical_projection") var optionLexicalProjection: Linear
    @ModuleInfo(key: "type_embedding") var typeEmbedding: Embedding
    @ModuleInfo(key: "evidence_layers") var evidenceLayers: [ClefEvidenceRoutingLayer]
    @ModuleInfo(key: "option_summary_norm") var optionSummaryNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [ClefJointDecoderLayer]
    @ModuleInfo(key: "field_norm") var fieldNorm: LayerNorm
    @ModuleInfo(key: "option_norm") var optionNorm: LayerNorm
    @ModuleInfo(key: "scorer1") var scorer1: Linear
    @ModuleInfo(key: "scorer2") var scorer2: Linear
    @ParameterInfo(key: "prior_logit_scale") var priorLogitScale: MLXArray
    @ParameterInfo(key: "joint_logit_scale") var jointLogitScale: MLXArray
    @ParameterInfo(key: "residual_gate") var residualGate: MLXArray

    public init(_ configuration: ClefDecisionHeadConfiguration) {
        let width = configuration.width
        _hiddenNorm.wrappedValue = LayerNorm(dimensions: configuration.hiddenSize)
        _memoryProjection.wrappedValue = Linear(configuration.hiddenSize, width, bias: false)
        _questionProjection.wrappedValue = Linear(configuration.hiddenSize, width, bias: false)
        _optionQuestionProjection.wrappedValue = Linear(
            configuration.hiddenSize, width, bias: false)
        _globalProjection.wrappedValue = Linear(configuration.hiddenSize, width, bias: false)
        _optionContextProjection.wrappedValue = Linear(
            configuration.hiddenSize, width, bias: false)
        _optionLexicalProjection.wrappedValue = Linear(
            configuration.hiddenSize, width, bias: false)
        _typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
        _evidenceLayers.wrappedValue = (0 ..< configuration.routingLayers).map { _ in
            ClefEvidenceRoutingLayer(
                width: width, heads: configuration.heads, feedforward: configuration.feedforward)
        }
        _optionSummaryNorm.wrappedValue = LayerNorm(dimensions: width)
        _layers.wrappedValue = (0 ..< configuration.layers).map { _ in
            ClefJointDecoderLayer(
                width: width, heads: configuration.heads, feedforward: configuration.feedforward)
        }
        _fieldNorm.wrappedValue = LayerNorm(dimensions: width)
        _optionNorm.wrappedValue = LayerNorm(dimensions: width)
        _scorer1.wrappedValue = Linear(4 * width, width)
        _scorer2.wrappedValue = Linear(width, 1)
        _priorLogitScale.wrappedValue = MLXArray.zeros([])
        _jointLogitScale.wrappedValue = MLXArray.zeros([])
        _residualGate.wrappedValue = MLXArray.zeros([])
        super.init()
    }

    /// Renames checkpoint parameter paths to the module names used here.
    ///
    /// The checkpoint stores sequential `nn.Sequential` blocks by index
    /// (`feedforward.0`, `feedforward.3`, `residual_scorer.0`,
    /// `residual_scorer.3`); this module uses named layers.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        result.reserveCapacity(weights.count)
        for (key, value) in weights {
            var key = key
            key = key.replacingOccurrences(of: ".feedforward.0.", with: ".feedforward.fc1.")
            key = key.replacingOccurrences(of: ".feedforward.3.", with: ".feedforward.fc2.")
            key = key.replacingOccurrences(of: "residual_scorer.0.", with: "scorer1.")
            key = key.replacingOccurrences(of: "residual_scorer.3.", with: "scorer2.")
            result[key] = value
        }
        return result
    }

    /// Scores every option of every question.
    ///
    /// - Parameters:
    ///   - hidden: Final backbone hidden states `[nTokens, hiddenSize]`.
    ///   - inputIds: Token ids `[nTokens]` matching the spans in `record`.
    ///   - record: Encoded record with question and option spans.
    ///   - lexicalLookup: Maps token ids to dequantized output-embedding rows.
    /// - Returns: One `[nOptions]` score vector per question.
    public func callAsFunction(
        _ hidden: MLXArray,
        inputIds: MLXArray,
        record: ClefDecisionEncoding,
        lexicalLookup: (MLXArray) -> MLXArray
    ) -> [MLXArray] {
        let questions = record.questions

        let h = hiddenNorm(hidden)
        let memory = memoryProjection(h).expandedDimensions(axis: 0)
        let globalVector = h[h.dim(0) - 1]

        let questionVectors = stacked(
            questions.map { h[$0.questionSpan.lowerBound ..< $0.questionSpan.upperBound]
                .mean(axis: 0)
            })
        let typeIds = MLXArray(questions.map { Int32($0.typeIndex) })

        var optionContexts: [MLXArray] = []
        var lexicalOptions: [MLXArray] = []
        var counts: [Int] = []
        for question in questions {
            optionContexts.append(
                stacked(question.optionSpans.map { h[$0].mean(axis: 0) }))
            lexicalOptions.append(
                stacked(question.optionSpans.map { lexicalLookup(inputIds[$0]).mean(axis: 0) }))
            counts.append(question.optionSpans.count)
        }

        var optionQueries: [MLXArray] = []
        for index in questions.indices {
            let context = optionContextProjection(optionContexts[index])
            let lexical = optionLexicalProjection(lexicalOptions[index])
            let question = optionQuestionProjection(questionVectors[index])
                .expandedDimensions(axis: 0)
            optionQueries.append(context + lexical + question)
        }

        var routed = concatenated(optionQueries, axis: 0).expandedDimensions(axis: 0)
        for layer in evidenceLayers {
            routed = layer(routed, memory)
        }
        let routedOptions = routed[0]

        var splitOffsets: [Int] = []
        var running = 0
        for count in counts.dropLast() {
            running += count
            splitOffsets.append(running)
        }
        let splitOptions =
            splitOffsets.isEmpty ? [routedOptions] : routedOptions.split(indices: splitOffsets)

        let baseFields = questionProjection(questionVectors)

        var summaries: [MLXArray] = []
        for (index, options) in splitOptions.enumerated() {
            let scores = options.matmul(baseFields[index]) / sqrt(Float(options.dim(-1)))
            let weights = MLX.softmax(scores, axis: 0).expandedDimensions(axis: -1)
            summaries.append((weights * options).sum(axis: 0))
        }

        var fields =
            baseFields
            + optionSummaryNorm(stacked(summaries))
            + globalProjection(globalVector).expandedDimensions(axis: 0)
            + typeEmbedding(typeIds)
        fields = fields.expandedDimensions(axis: 0)
        for layer in layers {
            fields = layer(fields, memory)
        }
        let fieldVectors = fieldNorm(fields[0])

        let priorScale = exp(minimum(priorLogitScale, MLXArray(Float(log(100.0)))))
        let jointScale = exp(minimum(jointLogitScale, MLXArray(Float(log(100.0)))))
        let gate = sigmoid(residualGate)

        var logits: [MLXArray] = []
        for index in questions.indices {
            let lexical = lexicalOptions[index]
            let options = splitOptions[index]

            let anchor = clefL2Normalized(questionVectors[index] + globalVector)
            let prior = priorScale * clefL2Normalized(lexical).matmul(anchor)

            let normalizedOptions = optionNorm(options)
            let fieldRow = fieldVectors[index].expandedDimensions(axis: 0)
            let fieldBroadcast = broadcast(fieldRow, to: normalizedOptions.shape)
            let denominator = maximum(
                sqrt((fieldBroadcast * fieldBroadcast).sum(axis: -1))
                    * sqrt((normalizedOptions * normalizedOptions).sum(axis: -1)),
                MLXArray(Float(1e-8)))
            let cosine = (fieldBroadcast * normalizedOptions).sum(axis: -1) / denominator

            let features = concatenated(
                [
                    fieldBroadcast, normalizedOptions, fieldBroadcast * normalizedOptions,
                    abs(fieldBroadcast - normalizedOptions),
                ], axis: -1)
            let residual = scorer2(gelu(scorer1(features))).squeezed(axis: -1)

            logits.append(prior + gate * (jointScale * cosine + residual))
        }

        return logits
    }
}