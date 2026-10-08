import Foundation
import MLX

public struct D1Calibration: Sendable {
    public let temperature: Double
    public let biases: [Double]

    public init(temperature: Double = 1, biases: [Double] = []) throws {
        guard temperature.isFinite, temperature > 0, biases.allSatisfy(\.isFinite) else {
            throw D1Error.invalidInput(
                "Calibration requires a positive finite temperature and finite biases.")
        }
        self.temperature = temperature
        self.biases = biases
    }

    func apply(_ scores: [Double]) throws -> [Double] {
        guard biases.isEmpty || biases.count == scores.count else {
            throw D1Error.invalidInput("Calibration bias count must match the option count.")
        }
        return scores.enumerated().map { index, score in
            (score + (biases.isEmpty ? 0 : biases[index])) / temperature
        }
    }
}

public struct D1Answer: Sendable {
    public let labels: [String]
    public let probabilities: [Double]
    public let scores: [Double]
    public let selectedLabel: String
    public let confidence: Double
    public let probabilityOfTrue: Double?
    public let expectedScore: Double?
    public let inputTokens: Int
    public var outputTokens: Int { 0 }
}

public enum D1Readout {
    public static func answer(
        logProbabilities: MLXArray, prompt: D1Prompt, question: D1Question,
        inputTokens: Int, calibration: D1Calibration? = nil
    ) throws -> D1Answer {
        guard logProbabilities.ndim == 1,
            prompt.tokenGroups.joined().allSatisfy({ $0 >= 0 && $0 < logProbabilities.size })
        else {
            throw D1Error.invalidInput(
                "D1 readout needs a vocabulary row containing every option token.")
        }
        let rawScores = prompt.tokenGroups.map { group in
            Double(logProbabilities[MLXArray(group)].max().item(Float.self))
        }
        let scores = try calibration?.apply(rawScores) ?? rawScores
        guard scores.allSatisfy(\.isFinite) else {
            throw D1Error.invalidInput("D1 option scores must be finite.")
        }
        let maximum = scores.max()!
        let exponentials = scores.map { exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        let probabilities = exponentials.map { $0 / denominator }
        let best = probabilities.firstIndex(of: probabilities.max()!)!
        let probabilityOfTrue: Double?
        let expectedScore: Double?
        switch question {
        case .noul:
            probabilityOfTrue = probabilities[0]
            expectedScore = nil
        case .choice:
            probabilityOfTrue = nil
            expectedScore = nil
        case .score:
            probabilityOfTrue = nil
            expectedScore = probabilities.enumerated().reduce(0) {
                $0 + Double($1.offset) * $1.element
            }
        }
        return D1Answer(
            labels: prompt.labels, probabilities: probabilities, scores: scores,
            selectedLabel: prompt.labels[best], confidence: probabilities[best],
            probabilityOfTrue: probabilityOfTrue, expectedScore: expectedScore,
            inputTokens: inputTokens)
    }
}
