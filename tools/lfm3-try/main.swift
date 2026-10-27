import Foundation
import MLXLMCommon
import MLXVLM
import Tokenizers

struct HubTokenizer: MLXLMCommon.Tokenizer {
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
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

struct Loader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        HubTokenizer(inner: try await Tokenizers.AutoTokenizer.from(modelFolder: directory))
    }
}

func usage() -> Never {
    fputs(
        """
        usage: lfm3-try --omni <model-dir> [--prompt text] --d1 <model-dir>
          either flag may be omitted

        """, stderr)
    exit(2)
}

var omni: String?
var d1: String?
var prompt = "Reply with one word: hello"
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let flag = args.removeFirst()
    guard !args.isEmpty else { usage() }
    let value = args.removeFirst()
    switch flag {
    case "--omni": omni = value
    case "--d1": d1 = value
    case "--prompt": prompt = value
    default: usage()
    }
}
guard omni != nil || d1 != nil else { usage() }

let loader = Loader()
if let omni {
    let container = try await VLMModelFactory.shared.loadContainer(
        from: URL(fileURLWithPath: omni), using: loader)
    let input = try await container.prepare(input: UserInput(prompt: prompt))
    var text = ""
    let stream = try await container.generate(
        input: input, parameters: GenerateParameters(maxTokens: 16, temperature: 0))
    for await event in stream {
        if let chunk = event.chunk { text += chunk }
    }
    print("omni: \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
}
if let d1 {
    let model = try await LFM3D1.load(from: URL(fileURLWithPath: d1), using: loader)
    let response = try model.decide(
        DecisionRequest(
            state: "The capital of France is Paris.",
            questions: [
                DecisionQuestion(
                    id: "q",
                    instructions: "The capital of France is Paris.",
                    criteria: .noul())
            ]))
    let answer = response.answers[0]
    print("d1 noul P(true): \(answer.noul ?? -1) probabilities: \(answer.probabilities)")
}
