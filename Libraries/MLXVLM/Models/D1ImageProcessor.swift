import CoreImage
import Foundation
import MLX
import MLXLMCommon

struct D1ImageProcessor {
    static let imageTokenText = "<image>"

    static func expand(prompt: String, markup: String, imageCount: Int) throws -> String {
        let pieces = prompt.components(separatedBy: imageTokenText)
        guard pieces.count == imageCount + 1 else {
            throw D1Error.invalidInput("Image placeholders must match the number of D1 images.")
        }
        let imageSections = markup.components(separatedBy: "<|image_end|>")
        var expanded = pieces[0]
        for index in 0 ..< imageCount {
            expanded += imageSections[index] + "<|image_end|>" + pieces[index + 1]
        }
        return expanded
    }

    let config: D1ImageConfiguration

    init(config: LFM2VLConfiguration) {
        self.config = D1ImageConfiguration(config)
    }

    init(config: D1ImageConfiguration) {
        self.config = config
    }

    func prepare(_ images: [CIImage]) throws -> (markup: String, pixels: MLXArray?, frames: [THW]) {
        guard !images.isEmpty else { return ("", nil, []) }
        guard config.patchSize > 0, config.downsampleFactor > 0,
            config.minImageTokens > 0, config.maxImageTokens >= config.minImageTokens,
            config.tileSize > 0, config.tileSize % config.patchSize == 0,
            config.minTiles > 0, config.maxTiles >= config.minTiles, config.maxTiles <= 10,
            config.maxPixelsTolerance.isFinite, config.maxPixelsTolerance > 0
        else {
            throw D1Error.invalidInput("Invalid D1 image processor dimensions or token budget.")
        }
        var markup = ""
        var views = [MLXArray]()
        var frames = [THW]()
        for source in images {
            var image = try RGBImage(source)
            if image.width * image.height > 1024 * 1024 {
                let scale = sqrt(Double(1024 * 1024) / Double(image.width * image.height))
                image = image.resized(
                    width: max(1, Int(Double(image.width) * scale)),
                    height: max(1, Int(Double(image.height) * scale)))
            }
            let size = smartSize(width: image.width, height: image.height)
            let factor = config.patchSize * config.downsampleFactor
            let roundedWidth = max(
                config.patchSize,
                Int((Double(image.width) / Double(factor)).rounded(.toNearestOrEven)) * factor)
            let roundedHeight = max(
                config.patchSize,
                Int((Double(image.height) / Double(factor)).rounded(.toNearestOrEven)) * factor)
            let split =
                config.doImageSplitting
                && Double(roundedWidth * roundedHeight) > Double(
                    config.maxImageTokens * factor * factor) * config.maxPixelsTolerance
            markup += "<|image_start|>"
            if split {
                let grid = tileGrid(width: image.width, height: image.height)
                let tiled = image.resized(
                    width: grid.width * config.tileSize, height: grid.height * config.tileSize)
                for row in 0 ..< grid.height {
                    for column in 0 ..< grid.width {
                        markup += "<|img_row_\(row + 1)_col_\(column + 1)|>"
                        append(
                            tiled.cropped(
                                x: column * config.tileSize, y: row * config.tileSize,
                                width: config.tileSize, height: config.tileSize), markup: &markup,
                            views: &views, frames: &frames)
                    }
                }
                if config.useThumbnail {
                    markup += "<|img_thumbnail|>"
                    append(
                        image.resized(width: size.width, height: size.height), markup: &markup,
                        views: &views, frames: &frames)
                }
            } else {
                append(
                    image.resized(width: size.width, height: size.height), markup: &markup,
                    views: &views, frames: &frames)
            }
            markup += "<|image_end|>"
        }
        let maximumPatches = frames.map { $0.h * $0.w }.max()!
        let paddedViews = views.map { view in
            view.dim(0) == maximumPatches
                ? view
                : concatenated(
                    [
                        view,
                        MLXArray.zeros(
                            [maximumPatches - view.dim(0), view.dim(1)], dtype: view.dtype),
                    ], axis: 0)
        }
        return (markup, stacked(paddedViews), frames)
    }

    func smartSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let factor = config.patchSize * config.downsampleFactor
        let minimum = config.minImageTokens * factor * factor
        let maximum = config.maxImageTokens * factor * factor
        var targetWidth = max(
            factor, Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        var targetHeight = max(
            factor, Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor)
        if targetWidth * targetHeight > maximum {
            let scale = sqrt(Double(width * height) / Double(maximum))
            targetWidth = max(factor, Int(floor(Double(width) / scale / Double(factor))) * factor)
            targetHeight = max(factor, Int(floor(Double(height) / scale / Double(factor))) * factor)
        } else if targetWidth * targetHeight < minimum {
            let scale = sqrt(Double(minimum) / Double(width * height))
            targetWidth = Int(ceil(Double(width) * scale / Double(factor))) * factor
            targetHeight = Int(ceil(Double(height) * scale / Double(factor))) * factor
        }
        return (targetWidth, targetHeight)
    }

    private func tileGrid(width: Int, height: Int) -> (width: Int, height: Int) {
        var candidates = [(width: Int, height: Int)]()
        for columns in 1 ... config.maxTiles {
            for rows in 1 ... config.maxTiles
            where (config.minTiles ... config.maxTiles).contains(columns * rows) {
                candidates.append((columns, rows))
            }
        }
        candidates.sort { $0.width * $0.height < $1.width * $1.height }
        let aspect = Double(width) / Double(height)
        var best = candidates[0]
        var difference = Double.infinity
        for candidate in candidates {
            let current = abs(aspect - Double(candidate.width) / Double(candidate.height))
            if current < difference
                || current == difference
                    && width * height > config.tileSize * config.tileSize * candidate.width
                        * candidate.height / 2
            {
                best = candidate
                difference = current
            }
        }
        return best
    }

    private func append(
        _ image: RGBImage, markup: inout String, views: inout [MLXArray], frames: inout [THW]
    ) {
        let patch = config.patchSize
        let rows = image.height / patch
        let columns = image.width / patch
        var pixels = [Float]()
        pixels.reserveCapacity(rows * columns * patch * patch * 3)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                for pixelRow in 0 ..< patch {
                    for pixelColumn in 0 ..< patch {
                        let start =
                            ((row * patch + pixelRow) * image.width + column * patch + pixelColumn)
                            * 3
                        for channel in 0 ..< 3 {
                            pixels.append(
                                (Float(image.bytes[start + channel]) * Float(1.0 / 255.0) - 0.5)
                                    / 0.5)
                        }
                    }
                }
            }
        }
        let tokens =
            ((rows + config.downsampleFactor - 1) / config.downsampleFactor)
            * ((columns + config.downsampleFactor - 1) / config.downsampleFactor)
        markup += String(repeating: "<image>", count: tokens)
        views.append(MLXArray(pixels, [rows * columns, patch * patch * 3]))
        frames.append(THW(1, rows, columns))
    }

    private struct RGBImage {
        let width: Int
        let height: Int
        let bytes: [UInt8]

        init(width: Int, height: Int, bytes: [UInt8]) {
            self.width = width
            self.height = height
            self.bytes = bytes
        }

        init(_ image: CIImage) throws {
            width = Int(image.extent.width)
            height = Int(image.extent.height)
            guard width > 0, height > 0, width <= 32768, height <= 32768 else {
                throw D1Error.invalidInput(
                    "D1 image dimensions must be positive and at most 32768.")
            }
            let context = CIContext(options: [
                .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!
            ])
            guard let cgImage = context.createCGImage(image, from: image.extent) else {
                throw D1Error.invalidInput("Could not decode D1 image pixels.")
            }
            var rgba = [UInt8](repeating: 0, count: width * height * 4)
            let renderWidth = width
            let renderHeight = height
            let decoded = rgba.withUnsafeMutableBytes { buffer in
                guard
                    let bitmap = CGContext(
                        data: buffer.baseAddress, width: renderWidth, height: renderHeight,
                        bitsPerComponent: 8,
                        bytesPerRow: renderWidth * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                else { return false }
                bitmap.draw(
                    cgImage, in: CGRect(x: 0, y: 0, width: renderWidth, height: renderHeight))
                return true
            }
            guard decoded else { throw D1Error.invalidInput("Could not render D1 image pixels.") }
            bytes = stride(from: 0, to: rgba.count, by: 4).flatMap { Array(rgba[$0 ..< $0 + 3]) }
        }

        func cropped(x: Int, y: Int, width: Int, height: Int) -> RGBImage {
            var result = [UInt8]()
            result.reserveCapacity(width * height * 3)
            for row in y ..< y + height {
                let start = (row * self.width + x) * 3
                let end = start + width * 3
                result.append(contentsOf: bytes[start ..< end])
            }
            return RGBImage(width: width, height: height, bytes: result)
        }

        func resized(width: Int, height: Int) -> RGBImage {
            if width == self.width && height == self.height { return self }
            let horizontal = resizeAxis(
                bytes, width: self.width, height: self.height, output: width, horizontal: true)
            let vertical = resizeAxis(
                horizontal, width: width, height: self.height, output: height, horizontal: false)
            return RGBImage(width: width, height: height, bytes: vertical)
        }

        private func resizeAxis(
            _ input: [UInt8], width: Int, height: Int, output: Int, horizontal: Bool
        ) -> [UInt8] {
            let inputSize = horizontal ? width : height
            if inputSize == output { return input }
            let scale = Double(inputSize) / Double(output)
            let filterScale = max(1, scale)
            let support = 2 * filterScale
            let coefficients = (0 ..< output).map { position -> (start: Int, weights: [Int]) in
                let center = (Double(position) + 0.5) * scale
                let start = max(0, Int(center - support + 0.5))
                let end = min(inputSize, Int(center + support + 0.5))
                let values = (start ..< end).map { sample -> Double in
                    let distance = abs((Double(sample) - center + 0.5) / filterScale)
                    if distance < 1 { return ((1.5 * distance - 2.5) * distance) * distance + 1 }
                    if distance < 2 {
                        return ((-0.5 * distance + 2.5) * distance - 4) * distance + 2
                    }
                    return 0
                }
                let total = values.reduce(0, +)
                return (start, values.map { Int(($0 / total * Double(1 << 22)).rounded()) })
            }
            let outputWidth = horizontal ? output : width
            let outputHeight = horizontal ? height : output
            var result = [UInt8](repeating: 0, count: outputWidth * outputHeight * 3)
            for row in 0 ..< outputHeight {
                for column in 0 ..< outputWidth {
                    let coefficient = coefficients[horizontal ? column : row]
                    for channel in 0 ..< 3 {
                        var value = 1 << 21
                        for (offset, weight) in coefficient.weights.enumerated() {
                            let sourceRow = horizontal ? row : coefficient.start + offset
                            let sourceColumn = horizontal ? coefficient.start + offset : column
                            value +=
                                Int(input[(sourceRow * width + sourceColumn) * 3 + channel])
                                * weight
                        }
                        result[(row * outputWidth + column) * 3 + channel] = UInt8(
                            clamping: value >> 22)
                    }
                }
            }
            return result
        }
    }
}

enum D1PositionInterpolation {
    static func batch(
        embeddings: MLXArray, spatialShapes: MLXArray, length: Int, dimensions: Int
    ) -> MLXArray {
        let resized = MLXArray.zeros(
            [spatialShapes.dim(0), length, dimensions], dtype: embeddings.dtype)
        for index in 0 ..< spatialShapes.dim(0) {
            let height = spatialShapes[index, 0].item(Int.self)
            let width = spatialShapes[index, 1].item(Int.self)
            let positions = resize(embeddings, height: height, width: width).reshaped(
                height * width, dimensions)
            resized[index, 0 ..< height * width] = positions
        }
        return resized
    }

    static func resize(_ embeddings: MLXArray, height: Int, width: Int) -> MLXArray {
        var result = embeddings.asType(.float32)
        result = resizeAxis(result, output: width, axis: 1)
        result = resizeAxis(result, output: height, axis: 0)
        return result.asType(embeddings.dtype)
    }

    private static func resizeAxis(_ input: MLXArray, output: Int, axis: Int) -> MLXArray {
        let size = input.dim(axis)
        guard size != output else { return input }
        let scale = Double(size) / Double(output)
        let rows = (0 ..< output).map { position -> MLXArray in
            let source = (Double(position) + 0.5) * scale - 0.5
            let start: Int
            let end: Int
            if scale > 1 {
                start = max(0, Int(floor(source - scale + 1)))
                end = min(size - 1, Int(floor(source + scale)))
            } else {
                start = max(0, Int(floor(max(0, source))))
                end = min(size - 1, start + 1)
            }
            let indices = Array(start ... end)
            var weights = indices.map {
                Float(max(0, 1 - abs(Double($0) - max(0, source)) / max(1, scale)))
            }
            let total = weights.reduce(0, +)
            weights = weights.map { $0 / total }
            let selected = take(input, MLXArray(indices), axis: axis)
            var shape = Array(repeating: 1, count: input.ndim)
            shape[axis] = weights.count
            return sum(selected * MLXArray(weights).reshaped(shape), axis: axis)
        }
        return stacked(rows, axis: axis)
    }
}
