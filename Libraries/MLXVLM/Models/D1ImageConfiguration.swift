import Foundation

struct D1ImageConfiguration: Codable, Sendable {
    let patchSize: Int
    let downsampleFactor: Int
    let minImageTokens: Int
    let maxImageTokens: Int
    let doImageSplitting: Bool
    let useThumbnail: Bool
    let tileSize: Int
    let maxTiles: Int
    private let minimumTiles: Int?
    var minTiles: Int { minimumTiles ?? 2 }
    let maxPixelsTolerance: Double

    init(_ model: LFM2VLConfiguration) {
        patchSize = model.visionConfiguration.patchSize
        downsampleFactor = model.downsampleFactor
        minImageTokens = model.minImageTokens
        maxImageTokens = model.maxImageTokens
        doImageSplitting = model.doImageSplitting
        useThumbnail = model.useThumbnail
        tileSize = 512
        maxTiles = 10
        minimumTiles = model.minTiles
        maxPixelsTolerance = 2
    }

    enum CodingKeys: String, CodingKey {
        case patchSize = "encoder_patch_size"
        case downsampleFactor = "downsample_factor"
        case minImageTokens = "min_image_tokens"
        case maxImageTokens = "max_image_tokens"
        case doImageSplitting = "do_image_splitting"
        case useThumbnail = "use_thumbnail"
        case tileSize = "tile_size"
        case maxTiles = "max_tiles"
        case minimumTiles = "min_tiles"
        case maxPixelsTolerance = "max_pixels_tolerance"
    }
}
