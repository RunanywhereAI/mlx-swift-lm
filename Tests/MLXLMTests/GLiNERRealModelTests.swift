import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

@testable import MLXLLM

private struct GLiNERTestTokenizer: MLXLMCommon.Tokenizer {
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
        throw GLiNERError.classificationOnly
    }
}

private struct GLiNERTestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let files = try GLiNERTokenizerConfiguration.load(from: directory)
        let config = try JSONDecoder().decode(Config.self, from: files.tokenizerConfig)
        let data = try JSONDecoder().decode(Config.self, from: files.tokenizerData)
        return GLiNERTestTokenizer(
            inner: try AutoTokenizer.from(
                tokenizerConfig: config, tokenizerData: data))
    }
}

final class GLiNERRealModelTests: XCTestCase {
    private func directory() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        guard environment["GLINER_REAL_MODEL_TESTS"] == "1" else {
            throw XCTSkip("Set GLINER_REAL_MODEL_TESTS=1 to opt into real checkpoint parity tests")
        }
        let directory = URL(
            fileURLWithPath: environment["GLINER_REAL_MODEL_PATH"]
                ?? NSHomeDirectory()
                + "/.local/share/runanywhere/Models/MLX/mlx-gliner25-decide-4bit")
        for file in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
        {
            guard
                FileManager.default.fileExists(atPath: directory.appendingPathComponent(file).path)
            else {
                throw GLiNERError.invalidCheckpoint(
                    "opted-in real test requires \(directory.path)/\(file)")
            }
        }
        return directory
    }

    func testCheckpointBFloat16Parity() async throws {
        try await checkParity(
            precision: "bfloat16", dtype: nil, logitTolerance: 0.03125, probabilityTolerance: 0.006)
    }

    func testCheckpointFloat32Parity() async throws {
        try await checkParity(
            precision: "float32", dtype: .float32, logitTolerance: 0.0005,
            probabilityTolerance: 0.0001)
    }

    private func checkParity(
        precision: String, dtype: DType?, logitTolerance: Float, probabilityTolerance: Float
    ) async throws {
        let model = try await GLiNERClassifier.load(
            from: directory(), using: GLiNERTestTokenizerLoader(), floatingPointType: dtype)
        XCTAssertEqual(model.hiddenSize, 1024)
        XCTAssertEqual(model.layerCount, 24)
        XCTAssertTrue(model.network.encoder.embeddings.words is QuantizedEmbedding)
        XCTAssertTrue(
            model.network.encoder.encoder.layer[0].attention.attention.query is QuantizedLinear)
        XCTAssertFalse(model.network.encoder.encoder.relative is QuantizedEmbedding)
        XCTAssertFalse(model.network.classifier.fc1 is QuantizedLinear)
        XCTAssertEqual(model.network.classifier.fc1.weight.dtype, dtype ?? .bfloat16)
        let reference = try GLiNERReference.load()
        XCTAssertEqual(reference.revision, "6023a32c2e696529decacfcaeb683b62bc535bdf")
        let cases = try XCTUnwrap(reference.runs[precision])
        XCTAssertEqual(cases.count, 7)
        var maximumLogitError: Float = 0
        var maximumProbabilityError: Float = 0
        var comparedScores = 0
        for (index, testCase) in cases.enumerated() {
            let encoding = try GLiNERPreprocessor.encode(
                text: testCase.text, tasks: testCase.tasks, tokenizer: model.tokenizer,
                maxTokens: 2048, maxWords: testCase.maxWords)
            XCTAssertEqual(encoding.inputIds, testCase.inputIds, "case \(index) tokens")
            XCTAssertEqual(
                encoding.markerPositions, testCase.markerPositions, "case \(index) markers")
            let results = try model.classify(
                text: testCase.text, tasks: testCase.tasks, maxWords: testCase.maxWords)
            XCTAssertEqual(results.count, testCase.expected.count)
            for (result, expected) in zip(results, testCase.expected) {
                XCTAssertEqual(result.task, expected.task)
                XCTAssertEqual(result.selectedLabels, expected.selected)
                XCTAssertEqual(result.scores.count, expected.logits.count)
                for (scoreIndex, score) in result.scores.enumerated() {
                    let logitError = abs(score.logit - expected.logits[scoreIndex])
                    let probabilityError = abs(
                        score.probability - expected.probabilities[scoreIndex])
                    maximumLogitError = max(maximumLogitError, logitError)
                    maximumProbabilityError = max(maximumProbabilityError, probabilityError)
                    comparedScores += 1
                    XCTAssertLessThanOrEqual(
                        logitError, logitTolerance, "case \(index) \(result.task) \(score.label)")
                    XCTAssertLessThanOrEqual(
                        probabilityError, probabilityTolerance,
                        "case \(index) \(result.task) \(score.label)")
                }
                print(
                    "GLiNER \(precision) case=\(index) task=\(result.task) logits=\(result.scores.map(\.logit)) probabilities=\(result.scores.map(\.probability))"
                )
            }
        }
        print(
            "GLiNER \(precision): cases=\(cases.count) scores=\(comparedScores) maxLogitError=\(maximumLogitError) maxProbabilityError=\(maximumProbabilityError) tolerances=\(logitTolerance),\(probabilityTolerance)"
        )
        let task = GLiNERClassificationTask(
            name: "intent", labels: [.init("refund"), .init("cancel")])
        XCTAssertThrowsError(try model.classify(text: "text", tasks: []))
        XCTAssertThrowsError(try model.classify(text: "text", tasks: [task, task]))
        XCTAssertThrowsError(try model.classify(text: "text", tasks: [task], maxTokens: 1))
        XCTAssertThrowsError(try model.classify(text: "text", tasks: [task], maxWords: 0))
        XCTAssertThrowsError(try model.classify(text: "[L] injected", tasks: [task]))
    }
}
