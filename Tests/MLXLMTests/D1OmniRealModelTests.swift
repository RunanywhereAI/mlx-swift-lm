import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import XCTest

@testable import MLXVLM

private struct D1Tokenizer: MLXLMCommon.Tokenizer {
    let inner: any Tokenizers.Tokenizer
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        inner.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { inner.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner.convertIdToToken(id) }
    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw D1OmniError.generationUnsupported
    }
}

private struct D1Reference: Decodable {
    struct Case: Decodable {
        struct Question: Decodable {
            struct Option: Decodable {
                let name: String
                let description: String
            }
            let kind: D1OmniQuestion.Kind
            let instructions: String
            let options: [Option]
            func typed() throws -> D1OmniQuestion {
                try D1OmniQuestion(
                    kind: kind, instructions: instructions,
                    options: options.map { .init($0.name, description: $0.description) })
            }
        }
        struct Sequence: Decodable {
            let tokens: [Int]
            let markers: [Int]
        }
        let name: String
        let state: String
        let questions: [Question]
        let sequences: [Sequence]
        let probabilities: [[Float]]
        let image: String?
    }
    let revision: String
    let dtype: String
    let cases: [Case]
}

final class D1OmniRealModelTests: XCTestCase {
    private func paths() throws -> (model: URL, reference: URL) {
        let environment = ProcessInfo.processInfo.environment
        guard environment["RUN_D1_OMNI_REAL_TESTS"] == "1" else {
            throw XCTSkip("Set RUN_D1_OMNI_REAL_TESTS=1 to run checkpoint parity")
        }
        let model = URL(
            fileURLWithPath: environment["D1_OMNI_MODEL_PATH"] ?? NSHomeDirectory()
                + "/.local/share/runanywhere/Models/MLX/mlx-d1-omni-600m-fp16")
        guard
            FileManager.default.fileExists(
                atPath: model.appendingPathComponent("model.safetensors").path)
        else {
            throw D1OmniError.invalidInput("opted-in checkpoint is missing at \(model.path)")
        }
        guard let reference = environment["D1_OMNI_REFERENCE_PATH"],
            FileManager.default.fileExists(atPath: reference)
        else {
            throw D1OmniError.invalidInput(
                "D1_OMNI_REFERENCE_PATH must point to the Python reference JSON")
        }
        return (model, URL(fileURLWithPath: reference))
    }

    private func compare(_ names: [String]) async throws {
        let paths = try paths()
        let reference = try JSONDecoder().decode(
            D1Reference.self, from: Data(contentsOf: paths.reference))
        XCTAssertEqual(reference.revision, "87dbe563c4ddd3a6b521bce21ecb43af55964ef3")
        XCTAssertEqual(reference.dtype, "float16")
        let model = try await VLMModelFactory.shared.loadD1Omni(from: paths.model)
        XCTAssertEqual(model.weight("encoder.embed_tokens.weight").dtype, .float16)
        XCTAssertEqual(model.weight("encoder.layers.0.feed_forward.w1.weight").shape, [4608, 1024])
        let tokenizer = D1Tokenizer(inner: try await AutoTokenizer.from(modelFolder: paths.model))
        for name in names {
            let fixture = try XCTUnwrap(reference.cases.first { $0.name == name })
            let questions = try fixture.questions.map { try $0.typed() }
            let images =
                try fixture.image.map { [try D1OmniImage(contentsOf: URL(fileURLWithPath: $0))] }
                ?? []
            for (index, question) in questions.enumerated() {
                let encoded = try D1OmniPrompt.encode(
                    state: fixture.state, question: question, tokenizer: tokenizer,
                    maxLength: images.isEmpty
                        ? model.config.maxLength : model.config.imageTextLength,
                    image: !images.isEmpty)
                XCTAssertEqual(
                    encoded.tokens, fixture.sequences[index].tokens,
                    "\(name) question \(index) tokens")
                XCTAssertEqual(
                    encoded.markers, fixture.sequences[index].markers,
                    "\(name) question \(index) markers")
            }
            let actual = try model.probabilities(
                state: fixture.state, questions: questions, tokenizer: tokenizer, images: images)
            XCTAssertEqual(actual.count, fixture.probabilities.count)
            for (index, pair) in zip(actual, fixture.probabilities).enumerated() {
                XCTAssertEqual(pair.0.count, pair.1.count)
                let error = zip(pair.0, pair.1).map { abs($0 - $1) }.max()!
                print(
                    "D1_PARITY \(name) question=\(index) fp16 max_error=\(error) actual=\(pair.0) python=\(pair.1) tolerance=0.01"
                )
                XCTAssertLessThanOrEqual(error, 0.01)
                XCTAssertEqual(
                    pair.0.indices.max(by: { pair.0[$0] < pair.0[$1] }),
                    pair.1.indices.max(by: { pair.1[$0] < pair.1[$1] }))
                XCTAssertEqual(pair.0.reduce(0, +), 1, accuracy: 1e-5)
            }
        }
    }

    func testDoubleChargeRoutesToBillingAndScoresAnger() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["RUN_D1_OMNI_REAL_TESTS"] == "1" else {
            throw XCTSkip("Set RUN_D1_OMNI_REAL_TESTS=1 to score the real checkpoint")
        }
        let modelURL = URL(
            fileURLWithPath: environment["D1_OMNI_MODEL_PATH"] ?? NSHomeDirectory()
                + "/.local/share/runanywhere/Models/MLX/mlx-d1-omni-600m-fp16")
        guard
            FileManager.default.fileExists(
                atPath: modelURL.appendingPathComponent("model.safetensors").path)
        else {
            throw D1OmniError.invalidInput("opted-in checkpoint is missing at \(modelURL.path)")
        }
        let model = try await VLMModelFactory.shared.loadD1Omni(from: modelURL)
        let tokenizer = D1Tokenizer(inner: try await AutoTokenizer.from(modelFolder: modelURL))
        let state = "I was charged twice for my order last week and nobody has replied."
        let questions = [
            try D1OmniQuestion(
                kind: .choice, instructions: "Which team should handle this?",
                options: [
                    .init("billing", description: "payments and refunds"),
                    .init("shipping"),
                    .init("technical"),
                ]),
            try D1OmniQuestion(kind: .noul, instructions: "Is the customer angry?"),
        ]
        let actual = try model.probabilities(
            state: state, questions: questions, tokenizer: tokenizer)
        let expected: [[Float]] = [
            [0.9973800778388977, 0.000014903380360919982, 0.0026050019077956676],
            [0.7306188941001892, 0.26938116550445557],
        ]
        XCTAssertEqual(actual.count, expected.count)
        for (index, pair) in zip(actual, expected).enumerated() {
            XCTAssertEqual(pair.0.count, pair.1.count)
            let error = zip(pair.0, pair.1).map { abs($0 - $1) }.max()!
            print(
                "D1_DECISION question=\(index) labels=\(questions[index].probabilityLabels) actual=\(pair.0) python=\(pair.1) max_error=\(error)"
            )
            XCTAssertLessThanOrEqual(error, 0.01)
            XCTAssertEqual(pair.0.reduce(0, +), 1, accuracy: 1e-5)
        }
        XCTAssertEqual(actual[0].indices.max(by: { actual[0][$0] < actual[0][$1] }), 0)
        XCTAssertEqual(questions[0].probabilityLabels, ["billing", "shipping", "technical"])
        XCTAssertEqual(questions[1].probabilityLabels, ["true", "false"])
    }

    func testTextScoringParity() async throws {
        try await compare(["refund", "negative", "escaped"])
    }
    func testImageScoringParity() async throws { try await compare(["cats"]) }

    func testImagePreprocessingParity() throws {
        let paths = try paths()
        let root = paths.reference.deletingLastPathComponent()
        let image = try D1OmniImage(contentsOf: root.appendingPathComponent("cats.png"))
        XCTAssertEqual(
            image.rgb, Array(try Data(contentsOf: root.appendingPathComponent("cats-rgb.bin"))))
        let reference = try loadArrays(
            url: root.appendingPathComponent("image-reference.safetensors"))
        let crops = try D1OmniProcessor.preprocess(image, dtype: .float32)
        XCTAssertEqual(crops.count, reference["pixel_values"]!.dim(0))
        for (index, crop) in crops.enumerated() {
            XCTAssertEqual(
                [crop.height, crop.width], reference["spatial_shapes"]![index].asArray(Int.self))
            let error = abs(crop.pixels[0] - reference["pixel_values"]![index]).max().item(
                Float.self)
            print("D1_IMAGE_PREPROCESS max_error=\(error) tolerance=0.000001")
            XCTAssertLessThanOrEqual(error, 1e-6)
        }
    }
}
