import CoreImage
import Foundation
import MLX
import MLXLMCommon

public final class D1Model {
    public let backbone: LFM2VL
    public let tokenizer: any Tokenizer

    public init(backbone: LFM2VL, tokenizer: any Tokenizer) throws {
        guard backbone.config.isD1 else { throw D1Error.incompatibleModel }
        self.backbone = backbone
        self.tokenizer = tokenizer
    }

    public static func load(from directory: URL, using tokenizerLoader: any TokenizerLoader)
        async throws -> D1Model
    {
        let context = try await VLMModelFactory.shared.load(from: directory, using: tokenizerLoader)
        guard let backbone = context.model as? LFM2VL else { throw D1Error.incompatibleModel }
        return try D1Model(backbone: backbone, tokenizer: context.tokenizer)
    }

    public func decide(
        state: String?, question: D1Question, images: [CIImage] = [],
        calibration: D1Calibration? = nil
    ) throws -> D1Answer {
        let prepared = try prepare(state: state, question: question, images: images)
        let cache = try backbone.newCache(parameters: nil)
        let result = try backbone.prepare(
            prepared.input, cache: cache, state: nil, prefill: .init(chunking: .unchunked))
        guard case .logits(let output) = result else {
            throw D1Error.invalidInput("D1 did not return logits at the answer slot.")
        }
        let row = output.logits[0, -1]
        let logProbabilities = row - logSumExp(row, axis: -1)
        return try D1Readout.answer(
            logProbabilities: logProbabilities, prompt: prepared.prompt, question: question,
            inputTokens: prepared.input.text.tokens.dim(-1), calibration: calibration)
    }

    public func prepare(state: String?, question: D1Question, images: [CIImage] = []) throws
        -> (prompt: D1Prompt, input: LMInput)
    {
        let processed = try D1ImageProcessor(config: backbone.config).prepare(images)
        let prompt = try D1Prompt(
            state: state, question: question, tokenizer: tokenizer, imageMarkup: processed.markup)
        let tokens = D1TokenEncoding.encode(prompt.text, tokenizer: tokenizer)
        guard !tokens.isEmpty, tokens.count <= 32768 else {
            throw D1Error.invalidInput("D1 prompt must contain 1 to 32768 tokens.")
        }
        let expectedImageTokens = processed.frames.reduce(0) {
            $0 + (($1.h + backbone.config.downsampleFactor - 1) / backbone.config.downsampleFactor)
                * (($1.w + backbone.config.downsampleFactor - 1) / backbone.config.downsampleFactor)
        }
        guard tokens.filter({ $0 == backbone.config.imageTokenIndex }).count == expectedImageTokens
        else {
            throw D1Error.invalidInput(
                "D1 image placeholders do not match the processed image features.")
        }
        let image = processed.pixels.map {
            LMInput.ProcessedImage(pixels: $0, frames: processed.frames)
        }
        return (
            prompt,
            LMInput(text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)), image: image)
        )
    }
}
