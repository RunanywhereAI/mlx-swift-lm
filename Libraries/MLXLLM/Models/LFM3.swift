import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: LFM3Configuration

    @ModuleInfo(key: "model") var model: LFM3Backbone

    public init(_ configuration: LFM3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0 ..< configuration.hiddenLayers).map { index in
            configuration.isAttention(index) ? configuration.kvHeads : 0
        }
        self._model.wrappedValue = LFM3Backbone(configuration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.logits(model(tokens: inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        lfm3Sanitize(weights)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.newCache(parameters: parameters)
    }

    public var loraLayers: [Module] { model.layers }
}
