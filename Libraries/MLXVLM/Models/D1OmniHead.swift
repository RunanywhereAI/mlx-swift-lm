import MLX
import MLXNN

extension D1Omni {
    func decisionLogits(_ input: MLXArray, markers: [Int], kind: D1OmniQuestion.Kind) -> MLXArray {
        let hidden = config.text.hiddenSize
        let length = input.dim(1)
        var states = input + weight("head.type_emb.weight")[kind.index].reshaped(1, 1, hidden)
        let mask = MLXArray.zeros([1, 1, 1, length], dtype: input.dtype)
        for index in 0 ..< config.headLayers {
            let name = "head.head.layers.\(index)"
            let normalized = norm(states, name + ".norm1")
            let projected =
                matmul(normalized, weight(name + ".self_attn.in_proj_weight").T)
                + weight(name + ".self_attn.in_proj_bias")
            let parts = projected.split(parts: 3, axis: -1).map {
                $0.reshaped(1, length, hidden / 64, 64).transposed(0, 2, 1, 3)
            }
            let attended = attention(parts[0], parts[1], parts[2], mask: mask).transposed(
                0, 2, 1, 3
            ).reshaped(1, length, hidden)
            states = states + linear(attended, name + ".self_attn.out_proj")
            states =
                states
                + linear(
                    relu(linear(norm(states, name + ".norm2"), name + ".linear1")),
                    name + ".linear2")
        }
        let selected = states[0..., MLXArray(markers), 0...]
        return linear(
            gelu(linear(norm(selected, "head.scorer.0"), "head.scorer.1")), "head.scorer.3")[
                0, 0..., 0
            ].asType(.float32)
    }

    func calibratedProbabilities(_ logits: MLXArray, question: D1OmniQuestion, image: Bool)
        -> [Float]
    {
        let temperature =
            image
            ? 1
            : config.temperatures[question.temperatureKey] ?? config.temperatures[
                question.kind.rawValue] ?? 1
        let values = softmax(logits / temperature).asArray(Float.self)
        return question.kind == .noul ? Array(values.reversed()) : values
    }
}
