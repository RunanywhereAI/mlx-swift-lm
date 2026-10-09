import CoreGraphics
import Foundation
import ImageIO
import MLX

public struct D1OmniImage: Sendable {
    public let width: Int
    public let height: Int
    public let rgb: [UInt8]

    public init(width: Int, height: Int, rgb: [UInt8]) throws {
        guard width > 0, height > 0, width <= Int.max / height / 3,
            rgb.count == width * height * 3
        else { throw D1OmniError.invalidImage }
        self.width = width
        self.height = height
        self.rgb = rgb
    }

    public init(contentsOf url: URL) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw D1OmniError.invalidImage
        }
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let succeeded = rgba.withUnsafeMutableBytes { buffer in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard succeeded else { throw D1OmniError.invalidImage }
        var rgb = [UInt8]()
        rgb.reserveCapacity(width * height * 3)
        for index in stride(from: 0, to: rgba.count, by: 4) {
            let alpha = Int(rgba[index + 3])
            for channel in 0 ..< 3 {
                rgb.append(
                    alpha == 0
                        ? 0
                        : UInt8(min(255, (Int(rgba[index + channel]) * 255 + alpha / 2) / alpha)))
            }
        }
        try self.init(width: width, height: height, rgb: rgb)
    }
}

enum D1OmniProcessor {
    struct Crop {
        let pixels: MLXArray
        let height: Int
        let width: Int
    }

    static func layout(width: Int, height: Int) -> (
        height: Int, width: Int, gridWidth: Int, gridHeight: Int, tiled: Bool
    ) {
        let factor = 32.0
        let maximum = 256.0 * 1024
        let minimum = 64.0 * 1024
        var targetHeight = max(32, Int((Double(height) / factor).rounded(.toNearestOrEven)) * 32)
        var targetWidth = max(32, Int((Double(width) / factor).rounded(.toNearestOrEven)) * 32)
        if Double(targetHeight * targetWidth) > maximum {
            let beta = sqrt(Double(height) * Double(width) / maximum)
            targetHeight = max(32, Int(floor(Double(height) / beta / factor)) * 32)
            targetWidth = max(32, Int(floor(Double(width) / beta / factor)) * 32)
        } else if Double(targetHeight * targetWidth) < minimum {
            let beta = sqrt(minimum / (Double(height) * Double(width)))
            targetHeight = Int(ceil(Double(height) * beta / factor)) * 32
            targetWidth = Int(ceil(Double(width) * beta / factor)) * 32
        }
        let tiled =
            max(16, Int((Double(height) / factor).rounded(.toNearestOrEven)) * 32)
            * max(16, Int((Double(width) / factor).rounded(.toNearestOrEven)) * 32)
            > Int(maximum * 2)
        var grid = (width: 1, height: 1)
        if tiled {
            var ratios = [(width: Int, height: Int)]()
            for gridWidth in 1 ... 10 {
                for gridHeight in 1 ... 10 where (2 ... 10).contains(gridWidth * gridHeight) {
                    ratios.append((gridWidth, gridHeight))
                }
            }
            ratios.sort { left, right in
                let leftArea = left.width * left.height
                let rightArea = right.width * right.height
                return leftArea == rightArea ? left.width < right.width : leftArea < rightArea
            }
            var best = Double.infinity
            for ratio in ratios {
                let difference = abs(
                    Double(width) / Double(height) - Double(ratio.width) / Double(ratio.height))
                if difference < best
                    || (difference == best
                        && Double(width) * Double(height) > 0.5 * 512 * 512
                            * Double(ratio.width * ratio.height))
                {
                    grid = ratio
                    best = difference
                }
            }
        }
        return (targetHeight, targetWidth, grid.width, grid.height, tiled)
    }

    static func resizeCoefficients(source: Int, target: Int) -> [(start: Int, values: [Double])] {
        let scale = Double(source) / Double(target)
        let support = max(scale, 1)
        return (0 ..< target).map { index in
            let center = scale * (Double(index) + 0.5)
            let low = max(0, Int(center - support + 0.5))
            let high = min(source, Int(center + support + 0.5))
            let values = (low ..< high).map {
                max(0, 1 - abs((Double($0) - center + 0.5) * (scale >= 1 ? 1 / scale : 1)))
            }
            let total = values.reduce(0, +)
            return (low, values.map { $0 / total })
        }
    }

    static func resize(_ image: D1OmniImage, height: Int, width: Int) -> [UInt8] {
        var pixels = image.rgb
        var sourceWidth = image.width
        for horizontal in [true, false] {
            let source = horizontal ? sourceWidth : image.height
            let target = horizontal ? width : height
            if source == target { continue }
            let coefficients = resizeCoefficients(source: source, target: target)
            let maximum = coefficients.flatMap(\.values).max()!
            var precision = 0
            while precision < 22 && Int(0.5 + maximum * Double(1 << (precision + 1))) < 32768 {
                precision += 1
            }
            let fixed = coefficients.map { row in
                (
                    start: row.start,
                    values: row.values.map { Int(0.5 + $0 * Double(1 << precision)) }
                )
            }
            let outputWidth = horizontal ? width : sourceWidth
            let outputHeight = horizontal ? image.height : height
            var output = [UInt8](repeating: 0, count: outputWidth * outputHeight * 3)
            for row in 0 ..< outputHeight {
                for column in 0 ..< outputWidth {
                    let coefficients = fixed[horizontal ? column : row]
                    for channel in 0 ..< 3 {
                        var value = 0
                        for (offset, coefficient) in coefficients.values.enumerated() {
                            let sourceRow = horizontal ? row : coefficients.start + offset
                            let sourceColumn = horizontal ? coefficients.start + offset : column
                            value +=
                                Int(pixels[(sourceRow * sourceWidth + sourceColumn) * 3 + channel])
                                * coefficient
                        }
                        output[(row * outputWidth + column) * 3 + channel] = UInt8(
                            clamping: (value + (1 << (precision - 1))) >> precision)
                    }
                }
            }
            pixels = output
            sourceWidth = outputWidth
        }
        return pixels
    }

    static func preprocess(_ image: D1OmniImage, dtype: DType) throws -> [Crop] {
        let plan = layout(width: image.width, height: image.height)
        var images = [D1OmniImage]()
        if plan.tiled {
            let resized = resize(image, height: plan.gridHeight * 512, width: plan.gridWidth * 512)
            for row in 0 ..< plan.gridHeight {
                for column in 0 ..< plan.gridWidth {
                    var pixels = [UInt8]()
                    pixels.reserveCapacity(512 * 512 * 3)
                    for innerRow in 0 ..< 512 {
                        let start =
                            ((row * 512 + innerRow) * plan.gridWidth * 512 + column * 512) * 3
                        pixels += resized[start ..< (start + 512 * 3)]
                    }
                    images.append(try D1OmniImage(width: 512, height: 512, rgb: pixels))
                }
            }
        }
        images.append(
            try D1OmniImage(
                width: plan.width, height: plan.height,
                rgb: resize(image, height: plan.height, width: plan.width)))
        return try images.map { crop in
            let patchHeight = crop.height / 16
            let patchWidth = crop.width / 16
            guard patchHeight * patchWidth <= 1024 else { throw D1OmniError.invalidImage }
            var patches = [Float](repeating: 0, count: 1024 * 768)
            for patchRow in 0 ..< patchHeight {
                for patchColumn in 0 ..< patchWidth {
                    let patch = (patchRow * patchWidth + patchColumn) * 768
                    for innerRow in 0 ..< 16 {
                        for innerColumn in 0 ..< 16 {
                            for channel in 0 ..< 3 {
                                let pixel =
                                    ((patchRow * 16 + innerRow) * crop.width + patchColumn * 16
                                        + innerColumn) * 3 + channel
                                patches[patch + (innerRow * 16 + innerColumn) * 3 + channel] =
                                    (Float(crop.rgb[pixel]) - 127.5) / 127.5
                            }
                        }
                    }
                }
            }
            return Crop(
                pixels: MLXArray(patches).reshaped(1, 1024, 768).asType(dtype), height: patchHeight,
                width: patchWidth)
        }
    }

    static func resizePositions(
        _ positions: [Float], side: Int, dimensions: Int, height: Int, width: Int
    ) -> [Float] {
        func coefficients(_ source: Int, _ target: Int) -> [(start: Int, values: [Float])] {
            let scale = Float(source) / Float(target)
            let support = max(scale, 1)
            return (0 ..< target).map { index in
                let center = scale * (Float(index) + 0.5)
                let low = max(0, Int(center - support + 0.5))
                let high = min(source, Int(center + support + 0.5))
                let values = (low ..< high).map {
                    max(Float(0), 1 - abs((Float($0) - center + 0.5) / support))
                }
                let total = values.reduce(Float(0), +)
                return (low, values.map { $0 / total })
            }
        }
        var intermediate = [Float](repeating: 0, count: side * width * dimensions)
        let horizontal = coefficients(side, width)
        for row in 0 ..< side {
            for column in 0 ..< width {
                let weights = horizontal[column]
                for channel in 0 ..< dimensions {
                    var value: Float = 0
                    for (index, coefficient) in weights.values.enumerated() {
                        value +=
                            positions[(row * side + weights.start + index) * dimensions + channel]
                            * coefficient
                    }
                    intermediate[(row * width + column) * dimensions + channel] = value
                }
            }
        }
        var output = [Float](repeating: 0, count: height * width * dimensions)
        let vertical = coefficients(side, height)
        for row in 0 ..< height {
            let weights = vertical[row]
            for column in 0 ..< width {
                for channel in 0 ..< dimensions {
                    var value: Float = 0
                    for (index, coefficient) in weights.values.enumerated() {
                        value +=
                            intermediate[
                                ((weights.start + index) * width + column) * dimensions + channel]
                            * coefficient
                    }
                    output[(row * width + column) * dimensions + channel] = value
                }
            }
        }
        return output
    }
}
