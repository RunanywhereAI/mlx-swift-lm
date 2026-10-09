import MLX
import MLXNN

final class D1OmniScale: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ size: Int) {
        self._weight.wrappedValue = MLXArray.ones([size])
    }
}

final class D1OmniNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(_ size: Int) {
        self._weight.wrappedValue = MLXArray.ones([size])
        self._bias.wrappedValue = MLXArray.zeros([size])
    }
}

final class D1OmniConvWeight: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(hidden: Int) {
        self._weight.wrappedValue = MLXArray.zeros([hidden, 1, 3])
    }
}

final class D1OmniConv: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "conv") var kernel: D1OmniConvWeight

    init(hidden: Int) {
        self._inProj.wrappedValue = Linear(hidden, hidden * 3, bias: false)
        self._outProj.wrappedValue = Linear(hidden, hidden, bias: false)
        self._kernel.wrappedValue = D1OmniConvWeight(hidden: hidden)
    }
}

final class D1OmniAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: D1OmniScale
    @ModuleInfo(key: "k_layernorm") var keyNorm: D1OmniScale

    init(hidden: Int, keyValue: Int, headDim: Int) {
        self._query.wrappedValue = Linear(hidden, hidden, bias: false)
        self._key.wrappedValue = Linear(hidden, keyValue, bias: false)
        self._value.wrappedValue = Linear(hidden, keyValue, bias: false)
        self._output.wrappedValue = Linear(hidden, hidden, bias: false)
        self._queryNorm.wrappedValue = D1OmniScale(headDim)
        self._keyNorm.wrappedValue = D1OmniScale(headDim)
    }
}

final class D1OmniFeedForward: Module {
    @ModuleInfo(key: "w1") var up: Linear
    @ModuleInfo(key: "w2") var down: Linear
    @ModuleInfo(key: "w3") var gate: Linear

    init(hidden: Int, feedForward: Int) {
        self._up.wrappedValue = Linear(hidden, feedForward, bias: false)
        self._down.wrappedValue = Linear(feedForward, hidden, bias: false)
        self._gate.wrappedValue = Linear(hidden, feedForward, bias: false)
    }
}

final class D1OmniLayer: Module {
    @ModuleInfo(key: "operator_norm") var operatorNorm: D1OmniScale
    @ModuleInfo(key: "ffn_norm") var ffnNorm: D1OmniScale
    @ModuleInfo(key: "feed_forward") var feedForward: D1OmniFeedForward
    @ModuleInfo(key: "self_attn") var attention: D1OmniAttention?
    @ModuleInfo(key: "conv") var conv: D1OmniConv?

    init(text: D1OmniConfiguration.Text, kind: String) {
        let hidden = text.hiddenSize
        let headDim = hidden / text.numAttentionHeads
        self._operatorNorm.wrappedValue = D1OmniScale(hidden)
        self._ffnNorm.wrappedValue = D1OmniScale(hidden)
        self._feedForward.wrappedValue = D1OmniFeedForward(
            hidden: hidden, feedForward: text.feedForwardSize)
        if kind == "full_attention" {
            self._attention.wrappedValue = D1OmniAttention(
                hidden: hidden, keyValue: text.numKeyValueHeads * headDim, headDim: headDim)
            self._conv.wrappedValue = nil
        } else {
            self._attention.wrappedValue = nil
            self._conv.wrappedValue = D1OmniConv(hidden: hidden)
        }
    }
}

final class D1OmniEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: D1OmniScale
    @ModuleInfo(key: "layers") var layers: [D1OmniLayer]

    init(_ text: D1OmniConfiguration.Text) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: text.vocabSize, dimensions: text.hiddenSize)
        self._embeddingNorm.wrappedValue = D1OmniScale(text.hiddenSize)
        self._layers.wrappedValue = text.layerTypes.map { D1OmniLayer(text: text, kind: $0) }
    }
}

final class D1OmniFusedAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inputWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inputBias: MLXArray
    @ModuleInfo(key: "out_proj") var output: Linear

    init(hidden: Int) {
        self._inputWeight.wrappedValue = MLXArray.zeros([3 * hidden, hidden])
        self._inputBias.wrappedValue = MLXArray.zeros([3 * hidden])
        self._output.wrappedValue = Linear(hidden, hidden, bias: true)
    }
}

final class D1OmniHeadLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: D1OmniNorm
    @ModuleInfo(key: "norm2") var norm2: D1OmniNorm
    @ModuleInfo(key: "self_attn") var attention: D1OmniFusedAttention
    @ModuleInfo(key: "linear1") var up: Linear
    @ModuleInfo(key: "linear2") var down: Linear

    init(hidden: Int) {
        self._norm1.wrappedValue = D1OmniNorm(hidden)
        self._norm2.wrappedValue = D1OmniNorm(hidden)
        self._attention.wrappedValue = D1OmniFusedAttention(hidden: hidden)
        self._up.wrappedValue = Linear(hidden, hidden * 4, bias: true)
        self._down.wrappedValue = Linear(hidden * 4, hidden, bias: true)
    }
}

final class D1OmniHeadStack: Module {
    @ModuleInfo(key: "layers") var layers: [D1OmniHeadLayer]

    init(count: Int, hidden: Int) {
        self._layers.wrappedValue = (0 ..< count).map { _ in D1OmniHeadLayer(hidden: hidden) }
    }
}

final class D1OmniScorer: Module {
    @ModuleInfo(key: "norm") var norm: D1OmniNorm
    @ModuleInfo(key: "hidden") var hidden: Linear
    @ModuleInfo(key: "output") var output: Linear

    init(hidden: Int) {
        self._norm.wrappedValue = D1OmniNorm(hidden)
        self._hidden.wrappedValue = Linear(hidden, hidden, bias: true)
        self._output.wrappedValue = Linear(hidden, 1, bias: true)
    }
}

final class D1OmniDecision: Module {
    @ModuleInfo(key: "type_emb") var typeEmbedding: Embedding
    @ModuleInfo(key: "head") var stack: D1OmniHeadStack
    @ModuleInfo(key: "scorer") var scorer: D1OmniScorer

    init(config: D1OmniConfiguration) {
        let hidden = config.text.hiddenSize
        self._typeEmbedding.wrappedValue = Embedding(embeddingCount: 3, dimensions: hidden)
        self._stack.wrappedValue = D1OmniHeadStack(count: config.headLayers, hidden: hidden)
        self._scorer.wrappedValue = D1OmniScorer(hidden: hidden)
    }
}

final class D1OmniVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear

    init(hidden: Int) {
        self._query.wrappedValue = Linear(hidden, hidden, bias: true)
        self._key.wrappedValue = Linear(hidden, hidden, bias: true)
        self._value.wrappedValue = Linear(hidden, hidden, bias: true)
        self._output.wrappedValue = Linear(hidden, hidden, bias: true)
    }
}

final class D1OmniVisionMLP: Module {
    @ModuleInfo(key: "fc1") var up: Linear
    @ModuleInfo(key: "fc2") var down: Linear

    init(hidden: Int, intermediate: Int) {
        self._up.wrappedValue = Linear(hidden, intermediate, bias: true)
        self._down.wrappedValue = Linear(intermediate, hidden, bias: true)
    }
}

final class D1OmniVisionLayer: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: D1OmniNorm
    @ModuleInfo(key: "layer_norm2") var norm2: D1OmniNorm
    @ModuleInfo(key: "self_attn") var attention: D1OmniVisionAttention
    @ModuleInfo(key: "mlp") var mlp: D1OmniVisionMLP

    init(hidden: Int, intermediate: Int) {
        self._norm1.wrappedValue = D1OmniNorm(hidden)
        self._norm2.wrappedValue = D1OmniNorm(hidden)
        self._attention.wrappedValue = D1OmniVisionAttention(hidden: hidden)
        self._mlp.wrappedValue = D1OmniVisionMLP(hidden: hidden, intermediate: intermediate)
    }
}

final class D1OmniVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [D1OmniVisionLayer]

    init(config: D1OmniConfiguration.Vision) {
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            D1OmniVisionLayer(hidden: config.hiddenSize, intermediate: config.intermediateSize)
        }
    }
}

final class D1OmniPatchEmbed: Module {
    @ModuleInfo(key: "patch_embedding") var patch: Linear

    init(config: D1OmniConfiguration) {
        let vision = config.vision
        self._patch.wrappedValue = Linear(
            3 * vision.patchSize * vision.patchSize, vision.hiddenSize, bias: true)
    }
}

final class D1OmniPositionEmbed: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(count: Int, dimensions: Int) {
        self._weight.wrappedValue = MLXArray.zeros([count, dimensions])
    }
}

final class D1OmniVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patch: Linear
    @ModuleInfo(key: "position_embedding") var position: D1OmniPositionEmbed

    init(config: D1OmniConfiguration) {
        let vision = config.vision
        self._patch.wrappedValue = Linear(
            3 * vision.patchSize * vision.patchSize, vision.hiddenSize, bias: true)
        self._position.wrappedValue = D1OmniPositionEmbed(
            count: vision.numPatches, dimensions: vision.hiddenSize)
    }
}

final class D1OmniVisionTower: Module {
    @ModuleInfo(key: "embeddings") var embeddings: D1OmniVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: D1OmniVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postNorm: D1OmniNorm

    init(config: D1OmniConfiguration) {
        self._embeddings.wrappedValue = D1OmniVisionEmbeddings(config: config)
        self._encoder.wrappedValue = D1OmniVisionEncoder(config: config.vision)
        self._postNorm.wrappedValue = D1OmniNorm(config.vision.hiddenSize)
    }
}

final class D1OmniProjector: Module {
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear

    init(config: D1OmniConfiguration) {
        self._first.wrappedValue = Linear(
            config.vision.hiddenSize * 4, config.projectorHiddenSize, bias: true)
        self._second.wrappedValue = Linear(
            config.projectorHiddenSize, config.text.hiddenSize, bias: true)
    }
}

final class D1OmniVision: Module {
    @ModuleInfo(key: "tower") var tower: D1OmniVisionTower
    @ModuleInfo(key: "projector") var projector: D1OmniProjector

    init(config: D1OmniConfiguration) {
        self._tower.wrappedValue = D1OmniVisionTower(config: config)
        self._projector.wrappedValue = D1OmniProjector(config: config)
    }
}
