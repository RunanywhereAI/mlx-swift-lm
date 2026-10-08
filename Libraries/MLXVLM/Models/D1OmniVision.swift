import Foundation
import MLX
import MLXNN

extension D1Omni {
    func visionPrefix(_ images: [D1OmniImage]) throws -> MLXArray {
        let vision = config.vision
        let hidden = vision.hiddenSize
        let headDim = hidden / vision.numAttentionHeads
        let dtype = weight("encoder.embed_tokens.weight").dtype
        let positions = weight("vision.tower.embeddings.position_embedding.weight").asType(.float32)
            .asArray(Float.self)
        let side = Int(sqrt(Double(vision.numPatches)))
        var prefixes = [MLXArray]()
        var tokenCount = 0
        for image in images {
            for crop in try D1OmniProcessor.preprocess(image, dtype: dtype) {
                let count = crop.height * crop.width
                tokenCount += count / 4
                guard tokenCount <= config.maxLength - 64 else { throw D1OmniError.contextExceeded }
                var resized = D1OmniProcessor.resizePositions(
                    positions, side: side, dimensions: hidden, height: crop.height,
                    width: crop.width)
                let first = Array(resized.prefix(hidden))
                for _ in count ..< 1024 { resized += first }
                var states =
                    linear(crop.pixels, "vision.tower.embeddings.patch_embedding")
                    + MLXArray(resized).reshaped(1, 1024, hidden).asType(dtype)
                let mask = which(
                    MLXArray(0 ..< 1024) .< count, MLXArray(Float(0)), MLXArray(-Float.infinity)
                ).asType(dtype).reshaped(1, 1, 1, 1024)
                for index in 0 ..< vision.numHiddenLayers {
                    let name = "vision.tower.encoder.layers.\(index)"
                    let normalized = norm(states, name + ".layer_norm1", eps: vision.layerNormEps)
                    let projections = ["q", "k", "v"].map {
                        linear(normalized, name + ".self_attn.\($0)_proj").reshaped(
                            1, 1024, vision.numAttentionHeads, headDim
                        ).transposed(0, 2, 1, 3)
                    }
                    let attended = attention(
                        projections[0], projections[1], projections[2], mask: mask
                    ).transposed(0, 2, 1, 3).reshaped(1, 1024, hidden)
                    states = states + linear(attended, name + ".self_attn.out_proj")
                    states =
                        states
                        + linear(
                            geluApproximate(
                                linear(
                                    norm(states, name + ".layer_norm2", eps: vision.layerNormEps),
                                    name + ".mlp.fc1")), name + ".mlp.fc2")
                    eval(states)
                }
                states = norm(states, "vision.tower.post_layernorm", eps: vision.layerNormEps)[
                    0..., 0 ..< count, 0...
                ].reshaped(1, crop.height, crop.width, hidden)
                states = states.reshaped(1, crop.height, crop.width / 2, hidden * 2).transposed(
                    0, 2, 1, 3
                ).reshaped(1, crop.width / 2, crop.height / 2, hidden * 4).transposed(0, 2, 1, 3)
                let projected = linear(
                    gelu(linear(states, "vision.projector.linear_1")), "vision.projector.linear_2"
                ).reshaped(1, -1, config.text.hiddenSize)
                eval(projected)
                prefixes.append(projected)
            }
        }
        return concatenated(prefixes, axis: 1)
    }
}
