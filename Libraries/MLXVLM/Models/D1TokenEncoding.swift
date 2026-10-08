import Foundation
import MLXLMCommon

enum D1TokenEncoding {
    private static let specialPattern = try! NSRegularExpression(
        pattern: #"<\|[^>]+\|>|<image>"#)
    private static let textPattern = try! NSRegularExpression(
        pattern:
            #"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"#
    )
    private static let byteCharacters: [String] = {
        let visible = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var scalars = Array(0 ... 255)
        var extensionIndex = 256
        for byte in 0 ... 255 where !visible.contains(byte) {
            scalars[byte] = extensionIndex
            extensionIndex += 1
        }
        return scalars.map { String(UnicodeScalar($0)!) }
    }()

    static func encode(_ text: String, tokenizer: any Tokenizer) -> [Int] {
        let source = text as NSString
        var result = [Int]()
        var offset = 0
        for match in specialPattern.matches(
            in: text, range: NSRange(location: 0, length: source.length))
        {
            let special = source.substring(with: match.range)
            guard let token = tokenizer.convertTokenToId(special),
                tokenizer.decode(tokenIds: [token], skipSpecialTokens: false) == special
            else { continue }
            result += encodeText(
                source.substring(
                    with: NSRange(location: offset, length: match.range.location - offset)),
                tokenizer: tokenizer)
            result.append(token)
            offset = NSMaxRange(match.range)
        }
        result += encodeText(source.substring(from: offset), tokenizer: tokenizer)
        return result
    }

    private static func encodeText(_ text: String, tokenizer: any Tokenizer) -> [Int] {
        let source = text as NSString
        return textPattern.matches(in: text, range: NSRange(location: 0, length: source.length))
            .flatMap { match in
                let piece = source.substring(with: match.range)
                let vocabularyForm = piece.utf8.map { byteCharacters[Int($0)] }.joined()
                if let token = tokenizer.convertTokenToId(vocabularyForm),
                    tokenizer.decode(tokenIds: [token], skipSpecialTokens: false) == piece
                {
                    return [token]
                }
                return tokenizer.encode(text: piece, addSpecialTokens: false)
            }
    }
}
