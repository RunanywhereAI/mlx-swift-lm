import CoreImage
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
        try inner.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: additionalContext)
    }
}

private struct D1TokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        D1Tokenizer(inner: try await Tokenizers.AutoTokenizer.from(modelFolder: directory))
    }
}

final class D1RealModelTests: XCTestCase {
    private struct Reference: Decodable {
        struct Question: Decodable {
            let type: String
            let instructions: String
            let criteria: Criteria?
            enum Criteria: Decodable {
                case options([String: String])
                case levels([String])
                init(from decoder: any Swift.Decoder) throws {
                    let container = try decoder.singleValueContainer()
                    if let levels = try? container.decode([String].self) {
                        self = .levels(levels)
                    } else {
                        self = .options(try container.decode([String: String].self))
                    }
                }
            }
        }
        let state: String?
        let question: Question
        let image: String?
        let tokens: [Int]
        let groups: [[Int]]
        let scores: [Double]
        let probabilities: [Double]
        let calibratedProbabilities: [Double]
        let dtype: String
        let frames: [[Int]]?
        let pixelFile: String?
    }

    private func modelDirectory() throws -> URL {
        guard ProcessInfo.processInfo.environment["MLX_D1_REAL_MODEL"] == "1" else {
            throw XCTSkip("Set MLX_D1_REAL_MODEL=1 to load and score the pinned D1 checkpoint.")
        }
        let directory = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["MLX_D1_MODEL_DIR"]
                ?? "\(NSHomeDirectory())/.local/share/runanywhere/Models/MLX/mlx-d1-3b-8bit")
        for file in ["model.safetensors", "config.json", "tokenizer.json"] {
            guard
                FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path)
            else {
                throw D1Error.invalidInput(
                    "Opted-in D1 test is missing \(directory.appendingPathComponent(file).path)")
            }
        }
        return directory
    }

    private func referenceDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MLX_D1_REFERENCE_DIR"] else {
            throw XCTSkip(
                "Set MLX_D1_REFERENCE_DIR to compare against Python reference outputs.")
        }
        return URL(fileURLWithPath: path)
    }

    func testDoubleChargeRouteAndAnger() async throws {
        let model = try await D1Model.load(from: modelDirectory(), using: D1TokenizerLoader())
        let state = "Customer message: I was charged twice for my order last week."
        let route = try model.decide(
            state: state,
            question: .choice(
                instructions: "Which team should handle this?",
                options: [
                    D1Option("billing", description: "payments, refunds"),
                    D1Option("shipping", description: "delivery"),
                    D1Option("technical", description: "bugs, login"),
                ]))
        XCTAssertEqual(route.labels, ["billing", "shipping", "technical"])
        XCTAssertEqual(route.probabilities.count, 3)
        XCTAssertEqual(route.probabilities.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertEqual(route.outputTokens, 0)
        XCTAssertTrue(route.scores.allSatisfy(\.isFinite))
        XCTAssertEqual(route.selectedLabel, "billing")
        XCTAssertEqual(route.probabilities[0], 0.9846983809029197, accuracy: 0.02)
        XCTAssertEqual(route.probabilities[1], 0.009068758564058832, accuracy: 0.02)
        XCTAssertEqual(route.probabilities[2], 0.006232860533021447, accuracy: 0.02)

        let angry = try model.decide(
            state: state, question: .noul(instructions: "Is the customer angry?"))
        XCTAssertEqual(angry.labels, ["true", "false"])
        XCTAssertEqual(angry.probabilities.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertEqual(angry.outputTokens, 0)
        let probabilityOfTrue = try XCTUnwrap(angry.probabilityOfTrue)
        XCTAssertEqual(probabilityOfTrue, 0.10669059394565118, accuracy: 0.02)
        XCTAssertEqual(angry.scores[1], -0.125, accuracy: 1e-6)

        let image = CIImage(color: CIColor(red: 0.75, green: 0.15, blue: 0.1)).cropped(
            to: CGRect(x: 0, y: 0, width: 128, height: 96))
        let withImage = try model.decide(
            state: state, question: .noul(instructions: "Is the customer angry?"), images: [image])
        XCTAssertEqual(withImage.probabilities.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertGreaterThan(withImage.inputTokens, angry.inputTokens)
        let imageAnger = try XCTUnwrap(withImage.probabilityOfTrue)
        XCTAssertEqual(imageAnger, 0.1480471980316895, accuracy: 0.02)
        print(
            "D1 route \(route.selectedLabel) probabilities=\(route.probabilities) scores=\(route.scores) angry=\(probabilityOfTrue) scores=\(angry.scores) imageAngry=\(imageAnger)"
        )
    }

    func testFactoryLoadsActualQuantizedTiedCheckpoint() async throws {
        let directory = try modelDirectory()
        let context = try await VLMModelFactory.shared.load(
            from: directory, using: D1TokenizerLoader())
        let model = try D1Model(
            backbone: XCTUnwrap(context.model as? LFM2VL), tokenizer: context.tokenizer)
        XCTAssertEqual(D1TokenEncoding.encode("\n\n", tokenizer: context.tokenizer), [8])
        XCTAssertEqual(model.backbone.config.textConfiguration.blockFFDim, 10752)
        XCTAssertFalse(model.backbone.config.textConfiguration.blockAutoAdjustFFDim)
        XCTAssertEqual(model.backbone.config.imageTokenIndex, 124907)
        let weights = Dictionary(uniqueKeysWithValues: model.backbone.parameters().flattened())
        XCTAssertEqual(
            weights["language_model.model.layers.0.feed_forward.w1.weight"]?.shape, [10752, 512])
        XCTAssertEqual(
            weights["language_model.model.layers.0.feed_forward.w2.weight"]?.shape, [2048, 2688])
        XCTAssertEqual(weights["language_model.model.embed_tokens.weight"]?.shape, [128000, 512])
        XCTAssertEqual(weights["language_model.model.embed_tokens.scales"]?.dtype, .bfloat16)
        XCTAssertFalse(weights.keys.contains { $0.contains("lm_head") })
        let answer = try model.decide(
            state: "I was charged twice.",
            question: .noul(instructions: "Is this a refund request?"))
        XCTAssertEqual(answer.probabilities.count, 2)
        XCTAssertEqual(answer.probabilities.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertEqual(answer.outputTokens, 0)
        guard ProcessInfo.processInfo.environment["MLX_D1_REFERENCE_DIR"] != nil else { return }
        let imageURL = try referenceDirectory().appendingPathComponent("cats-small.png")
        let image = try XCTUnwrap(CIImage(contentsOf: imageURL))
        let input = try await context.processor.prepare(
            input: UserInput(prompt: "Describe the image.", images: [.ciImage(image)]))
        let imageTokens = input.text.tokens.asArray(Int.self).filter { $0 == 124907 }
        XCTAssertEqual(imageTokens.count, 80)
        XCTAssertEqual(input.image?.pixels.shape, [1, 320, 768])
        let result = try model.backbone.prepare(
            input, cache: model.backbone.newCache(parameters: nil), state: nil,
            prefill: .init(chunking: .unchunked))
        guard case .logits(let output) = result else {
            return XCTFail("Factory-prepared D1 image did not produce logits.")
        }
        XCTAssertEqual(output.logits.dtype, .bfloat16)
        XCTAssertEqual(output.logits.dim(-1), 128000)
        XCTAssertTrue(output.logits[0, -1].asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testRealTextNoulChoiceScoreParity() async throws {
        try await compare(images: false)
    }

    func testRealTokenizerVocabularyShortcutsAndUnicode() async throws {
        let tokenizer = try await D1TokenizerLoader().load(from: modelDirectory())
        let cases: [(String, [Int])] = [
            ("Hello\n\nworld", [35808, 8, 27735]),
            (
                "Café नमस्ते 12345\n\tYES",
                [
                    43, 1804, 359, 72967, 72948, 72929, 73086, 49507, 229, 9792, 2136, 207, 206, 65,
                    2020,
                ]
            ),
            ("I'm excited!\n\nA cat", [49, 5716, 18548, 65653, 41, 5140]),
            ("<|not_a_token|>\n\n<image>", [36, 100, 2396, 18352, 39483, 100, 64012, 124907]),
        ]
        for (text, expected) in cases {
            XCTAssertEqual(D1TokenEncoding.encode(text, tokenizer: tokenizer), expected)
        }
    }

    func testRealImageParity() async throws {
        try await compare(images: true)
    }

    private func compare(images: Bool) async throws {
        let directory = try modelDirectory()
        let referenceDirectory = try referenceDirectory()
        let fixtures = try JSONDecoder().decode(
            [Reference].self,
            from: Data(contentsOf: referenceDirectory.appendingPathComponent("reference.json")))
        let model = try await D1Model.load(from: directory, using: D1TokenizerLoader())
        let cases = fixtures.filter { ($0.image != nil) == images }
        XCTAssertEqual(cases.count, images ? 2 : 6)
        for fixture in cases {
            let question: D1Question
            switch fixture.question.type {
            case "noul":
                question = .noul(instructions: fixture.question.instructions)
            case "choice":
                guard case .options(let options) = fixture.question.criteria else {
                    throw D1Error.invalidQuestion("Missing reference choice criteria.")
                }
                let order = images ? ["cat", "dog", "bird"] : ["refund", "shipping", "praise"]
                question = .choice(
                    instructions: fixture.question.instructions,
                    options: try order.map {
                        D1Option($0, description: try XCTUnwrap(options[$0]))
                    })
            case "score":
                guard case .levels(let levels) = fixture.question.criteria else {
                    throw D1Error.invalidQuestion("Missing reference score levels.")
                }
                question = .score(instructions: fixture.question.instructions, levels: levels)
            default:
                throw D1Error.invalidQuestion("Unknown reference question type.")
            }
            let pictures =
                try fixture.image.map { filename in
                    [
                        try XCTUnwrap(
                            CIImage(contentsOf: referenceDirectory.appendingPathComponent(filename))
                        )
                    ]
                } ?? []
            let prepared = try model.prepare(
                state: fixture.state, question: question, images: pictures)
            XCTAssertEqual(prepared.prompt.tokenGroups, fixture.groups)
            XCTAssertEqual(prepared.input.text.tokens.asArray(Int.self), fixture.tokens)
            if let pixelFile = fixture.pixelFile {
                let pixels = try XCTUnwrap(prepared.input.image?.pixels)
                let referencePixels = try MLX.loadArray(
                    url: referenceDirectory.appendingPathComponent(pixelFile))
                XCTAssertEqual(pixels.shape[0], referencePixels.shape[0])
                let cropped = referencePixels[0..., 0 ..< pixels.dim(1), 0...]
                let error = abs(pixels - cropped).max().item(Float.self)
                XCTAssertLessThanOrEqual(error, 2.0 / 255.0 + 1e-6)
                XCTAssertEqual(prepared.input.image?.frames?.map { [$0.h, $0.w] }, fixture.frames)
                print("D1 image pixel maximum error \(error)")
            }
            let answer = try model.decide(
                state: fixture.state, question: question, images: pictures)
            XCTAssertEqual(answer.inputTokens, fixture.tokens.count)
            XCTAssertEqual(answer.outputTokens, 0)
            switch question {
            case .noul:
                XCTAssertEqual(
                    try XCTUnwrap(answer.probabilityOfTrue), fixture.probabilities[0],
                    accuracy: 0.02)
            case .choice:
                let best = fixture.probabilities.firstIndex(of: fixture.probabilities.max()!)!
                XCTAssertEqual(answer.selectedLabel, prepared.prompt.labels[best])
                XCTAssertEqual(answer.confidence, fixture.probabilities[best], accuracy: 0.02)
            case .score:
                let expected = fixture.probabilities.enumerated().reduce(0) {
                    $0 + Double($1.offset) * $1.element
                }
                XCTAssertEqual(try XCTUnwrap(answer.expectedScore), expected, accuracy: 0.04)
            }
            XCTAssertEqual(fixture.dtype, "mlx.core.bfloat16")
            for index in fixture.probabilities.indices {
                XCTAssertEqual(answer.scores[index], fixture.scores[index], accuracy: 0.25)
                XCTAssertEqual(
                    answer.probabilities[index], fixture.probabilities[index], accuracy: 0.02)
            }
            print(
                "D1 \(fixture.image ?? fixture.question.type) scores=\(answer.scores) probabilities=\(answer.probabilities) reference=\(fixture.probabilities)"
            )
            let calibration = try D1Calibration(
                temperature: 1.5, biases: Array(repeating: 0.1, count: answer.scores.count))
            let calibrated = try model.decide(
                state: fixture.state, question: question, images: pictures, calibration: calibration
            )
            let expectedScores = try calibration.apply(answer.scores)
            let maximum = expectedScores.max()!
            let exponentials = expectedScores.map { exp($0 - maximum) }
            for index in exponentials.indices {
                XCTAssertEqual(
                    calibrated.probabilities[index], fixture.calibratedProbabilities[index],
                    accuracy: 0.02)
                XCTAssertEqual(
                    calibrated.probabilities[index],
                    exponentials[index] / exponentials.reduce(0, +), accuracy: 1e-12)
            }
            Memory.clearCache()
        }
    }
}
