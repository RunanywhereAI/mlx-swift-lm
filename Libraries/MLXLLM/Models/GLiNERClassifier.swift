import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum GLiNERClassActivation: String, Codable, Sendable {
    case auto
    case softmax
    case sigmoid
}

public struct GLiNERClassLabel: Codable, Sendable, Equatable {
    public var name: String
    public var description: String?

    public init(_ name: String, description: String? = nil) {
        self.name = name
        self.description = description
    }
}

public struct GLiNERClassExample: Codable, Sendable, Equatable {
    public var text: String
    public var label: String

    public init(text: String, label: String) {
        self.text = text
        self.label = label
    }
}

public struct GLiNERClassificationTask: Codable, Sendable {
    public var name: String
    public var labels: [GLiNERClassLabel]
    public var multiLabel: Bool
    public var activation: GLiNERClassActivation
    public var temperature: Float
    public var threshold: Float
    public var prompt: String?
    public var examples: [GLiNERClassExample]

    public init(
        name: String, labels: [GLiNERClassLabel], multiLabel: Bool = false,
        activation: GLiNERClassActivation = .auto, temperature: Float = 1,
        threshold: Float = 0.5, prompt: String? = nil, examples: [GLiNERClassExample] = []
    ) {
        self.name = name
        self.labels = labels
        self.multiLabel = multiLabel
        self.activation = activation
        self.temperature = temperature
        self.threshold = threshold
        self.prompt = prompt
        self.examples = examples
    }

    func validate() throws {
        try GLiNERPreprocessor.validateField(name, name: "task name", allowEmpty: false)
        guard !labels.isEmpty, Set(labels.map(\.name)).count == labels.count else {
            throw GLiNERError.invalidInput("labels must be nonempty and unique")
        }
        guard temperature.isFinite, temperature > 0, threshold.isFinite,
            (0 ... 1).contains(threshold)
        else {
            throw GLiNERError.invalidInput("temperature must be positive and threshold in [0, 1]")
        }
        for label in labels {
            try GLiNERPreprocessor.validateField(label.name, name: "label", allowEmpty: false)
            if let description = label.description {
                try GLiNERPreprocessor.validateField(description, name: "description")
            }
        }
        if let prompt { try GLiNERPreprocessor.validateField(prompt, name: "prompt") }
        for example in examples {
            try GLiNERPreprocessor.validateField(example.text, name: "example")
            guard labels.contains(where: { $0.name == example.label }) else {
                throw GLiNERError.invalidInput("example label is outside the task label set")
            }
        }
    }
}

public struct GLiNERLabelScore: Sendable, Equatable {
    public let label: String
    public let logit: Float
    public let probability: Float
}

public struct GLiNERClassificationResult: Sendable {
    public let task: String
    public let scores: [GLiNERLabelScore]
    public let selectedLabels: [String]
}

struct GLiNERClassificationEncoding: Sendable {
    let inputIds: [Int]
    let markerPositions: [[Int]]
}

enum GLiNERPreprocessor {
    static let markers = [
        "[P]", "[L]", "[E]", "[C]", "[R]", "[SEP_STRUCT]", "[SEP_TEXT]", "[DESCRIPTION]",
        "[EXAMPLE]", "[OUTPUT]",
    ]

    static func validateField(_ value: String, name: String, allowEmpty: Bool = true) throws {
        guard allowEmpty || !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GLiNERError.invalidInput("\(name) is empty")
        }
        guard !markers.contains(where: value.contains) else {
            throw GLiNERError.invalidInput("\(name) contains a reserved schema marker")
        }
    }

    static func words(_ text: String) throws -> [String] {
        let pattern =
            #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S"#
        let expression = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        return expression.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap
        {
            guard let range = Range($0.range, in: text) else { return nil }
            return String(text[range]).lowercased()
        }
    }

    static func encode(
        text: String, tasks: [GLiNERClassificationTask], tokenizer: any Tokenizer,
        maxTokens: Int, maxWords: Int? = nil
    ) throws -> GLiNERClassificationEncoding {
        guard !tasks.isEmpty, Set(tasks.map(\.name)).count == tasks.count else {
            throw GLiNERError.invalidInput("tasks must be nonempty and have unique names")
        }
        guard maxTokens > 0, maxWords == nil || maxWords! > 0 else {
            throw GLiNERError.invalidInput("token and word limits must be positive")
        }
        try validateField(text, name: "text")
        for task in tasks { try task.validate() }
        for marker in markers {
            guard let identifier = tokenizer.convertTokenToId(marker),
                identifier != tokenizer.unknownTokenId,
                tokenizer.encode(text: marker, addSpecialTokens: false) == [identifier]
            else { throw GLiNERError.invalidCheckpoint("tokenizer lacks atomic marker \(marker)") }
        }
        var ids = [Int]()
        var positions = [[Int]]()
        func append(_ value: String) throws {
            let tokens = tokenizer.encode(text: value, addSpecialTokens: false)
            guard !tokens.isEmpty else {
                throw GLiNERError.invalidInput("schema or word tokenizes to an empty sequence")
            }
            guard tokens.count <= maxTokens - ids.count else {
                throw GLiNERError.invalidInput("encoded request exceeds \(maxTokens) tokens")
            }
            ids += tokens
        }
        for (index, task) in tasks.enumerated() {
            if index > 0 { try append("[SEP_STRUCT]") }
            var taskPositions = [Int]()
            var parent = task.name
            if let prompt = task.prompt, !prompt.isEmpty { parent += ": \(prompt)" }
            for label in task.labels {
                if let description = label.description {
                    parent += " [DESCRIPTION] \(label.name): \(description)"
                }
            }
            for example in task.examples {
                parent += " [EXAMPLE] \(example.text) [OUTPUT] \(example.label)"
            }
            try append("(")
            taskPositions.append(ids.count)
            try append("[P]")
            try append(parent)
            try append("(")
            for label in task.labels {
                taskPositions.append(ids.count)
                try append("[L]")
                try append(label.name)
            }
            try append(")")
            try append(")")
            positions.append(taskPositions)
        }
        try append("[SEP_TEXT]")
        var punctuated = text
        if text.isEmpty {
            punctuated = "."
        } else if ![".", "!", "?"].contains(where: text.hasSuffix) {
            punctuated += "."
        }
        let textWords = try words(punctuated)
        for word in textWords.prefix(maxWords ?? textWords.count) { try append(word) }
        return GLiNERClassificationEncoding(inputIds: ids, markerPositions: positions)
    }
}

enum GLiNERClassificationScoring {
    static func result(
        logits: [Float], task: GLiNERClassificationTask
    ) throws -> GLiNERClassificationResult {
        try task.validate()
        guard logits.count == task.labels.count, logits.allSatisfy(\.isFinite) else {
            throw GLiNERError.invalidInput("classifier logits must be finite and match labels")
        }
        let scaled = logits.map { $0 / task.temperature }
        guard scaled.allSatisfy(\.isFinite) else {
            throw GLiNERError.invalidInput("temperature overflows scaled logits")
        }
        let probabilities: [Float]
        if task.activation == .sigmoid || (task.activation == .auto && task.multiLabel) {
            probabilities = scaled.map { value in
                if value >= 0 { return 1 / (1 + exp(-value)) }
                let exponential = exp(value)
                return exponential / (1 + exponential)
            }
        } else {
            let maximum = scaled.max()!
            let exponentials = scaled.map { exp($0 - maximum) }
            let total = exponentials.reduce(0, +)
            probabilities = exponentials.map { $0 / total }
        }
        let scores = task.labels.indices.map {
            GLiNERLabelScore(
                label: task.labels[$0].name, logit: logits[$0], probability: probabilities[$0])
        }
        let best = probabilities.indices.reduce(0) {
            probabilities[$1] > probabilities[$0] ? $1 : $0
        }
        var selected =
            task.multiLabel ? scores.filter { $0.probability >= task.threshold }.map(\.label) : []
        if selected.isEmpty { selected = [scores[best].label] }
        return GLiNERClassificationResult(task: task.name, scores: scores, selectedLabels: selected)
    }
}

public final class GLiNERClassifier {
    let network: GLiNERClassificationNetwork
    public let tokenizer: any Tokenizer
    public let hiddenSize: Int
    public let layerCount: Int

    init(network: GLiNERClassificationNetwork, tokenizer: any Tokenizer) {
        self.network = network
        self.tokenizer = tokenizer
        hiddenSize = network.encoder.config.hiddenSize
        layerCount = network.encoder.config.layers
    }

    public static func load(
        from directory: URL, using tokenizerLoader: any TokenizerLoader,
        floatingPointType: DType? = nil
    ) async throws -> GLiNERClassifier {
        if let floatingPointType,
            ![DType.float32, .float16, .bfloat16].contains(floatingPointType)
        {
            throw GLiNERError.invalidInput("floatingPointType must be a floating point dtype")
        }
        let config = try JSONDecoder().decode(
            GLiNERConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        var weights = [String: MLXArray]()
        let files = try safetensorWeightURLs(in: directory)
        guard !files.isEmpty else {
            throw GLiNERError.invalidCheckpoint("no safetensor weights found")
        }
        for file in files {
            for (name, value) in try loadArrays(url: file) {
                guard weights[name] == nil else {
                    throw GLiNERError.invalidCheckpoint("duplicate tensor \(name)")
                }
                weights[name] = value
            }
        }
        weights = try GLiNERClassificationNetwork.classificationWeights(
            weights, hidden: config.encoderConfig.hiddenSize)
        let network = GLiNERClassificationNetwork(config.encoderConfig)
        let modules = Dictionary(uniqueKeysWithValues: network.leafModules().flattened())
        for name in weights.keys where name.hasSuffix(".scales") {
            let path = String(name.dropLast(".scales".count))
            guard config.quantization != nil, path.hasPrefix("encoder."),
                !path.contains("rel_embeddings"),
                modules[path] is Linear || modules[path] is Embedding
            else { throw GLiNERError.invalidCheckpoint("unsupported quantized module \(path)") }
        }
        if let quantization = config.quantization {
            quantize(model: network) { path, _ in
                weights["\(path).scales"] == nil ? nil : quantization.asTuple
            }
        }
        if let floatingPointType {
            weights = weights.mapValues { value in
                [DType.float32, .float16, .bfloat16].contains(value.dtype)
                    ? value.asType(floatingPointType) : value
            }
        }
        try network.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        network.train(false)
        eval(network)
        let tokenizer = try await tokenizerLoader.load(from: directory)
        return GLiNERClassifier(network: network, tokenizer: tokenizer)
    }

    public func classify(
        text: String, tasks: [GLiNERClassificationTask], maxTokens: Int = 2048,
        maxWords: Int? = nil
    ) throws -> [GLiNERClassificationResult] {
        let encoding = try GLiNERPreprocessor.encode(
            text: text, tasks: tasks, tokenizer: tokenizer, maxTokens: maxTokens, maxWords: maxWords
        )
        guard
            encoding.inputIds.allSatisfy({
                (0 ..< network.encoder.config.vocabularySize).contains($0)
            })
        else {
            throw GLiNERError.invalidCheckpoint("tokenizer ID outside encoder vocabulary")
        }
        let ids = MLXArray(encoding.inputIds).reshaped(1, -1)
        let hidden = network.encoder(ids, mask: MLXArray.ones(ids.shape, dtype: .int32))[0]
        return try zip(tasks, encoding.markerPositions).map { task, positions in
            let indices = MLXArray(positions.dropFirst().map(Int32.init))
            let logits = network.classifier(hidden[indices])[0..., 0].asType(.float32).asArray(
                Float.self)
            return try GLiNERClassificationScoring.result(logits: logits, task: task)
        }
    }
}
