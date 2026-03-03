import Foundation
import Vision

/// Cached newline byte — avoids allocating "\n".data(using: .utf8)! per frame.
let jsonlNewline = Data([0x0A])

/// Count non-empty lines in a file (used to report total frames in JSONL output).
func countLines(in url: URL) -> Int {
    guard let data = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
    return data.components(separatedBy: "\n").filter { !$0.isEmpty }.count
}

/// Write a single frame's landmarks as a JSONL line.
///
/// Shared between PoseExtractor and ConcurrentPoseExtractor to avoid
/// format divergence — any JSONL schema change only needs to happen here.
func writeFrameJSONL(
    to fileHandle: FileHandle,
    frame: Int,
    timestamp: Double,
    landmarks: [LandmarkMapper.Landmark],
    width: Int,
    height: Int
) {
    // Build landmarks array: [[x, y, confidence], ...]
    var landmarkArrays: [[Double]] = []
    landmarkArrays.reserveCapacity(LandmarkMapper.totalSlots)
    for lm in landmarks {
        landmarkArrays.append([
            round(lm.x * 1_000_000) / 1_000_000,
            round(lm.y * 1_000_000) / 1_000_000,
            round(lm.confidence * 1_000_000) / 1_000_000,
        ])
    }

    let frameDict: [String: Any] = [
        "frame": frame,
        "timestamp": round(timestamp * 1_000_000) / 1_000_000,
        "landmarks": landmarkArrays,
        "width": width,
        "height": height,
    ]

    guard let data = try? JSONSerialization.data(withJSONObject: frameDict, options: []) else {
        return
    }
    fileHandle.write(data)
    fileHandle.write(jsonlNewline)
}

/// Write an empty frame (all zeros) when pixel buffer is unavailable.
func writeEmptyFrameJSONL(
    to fileHandle: FileHandle,
    frame: Int,
    timestamp: Double,
    width: Int,
    height: Int
) {
    let emptyLandmarks = Array(
        repeating: LandmarkMapper.zeroLandmark, count: LandmarkMapper.totalSlots)
    writeFrameJSONL(
        to: fileHandle, frame: frame, timestamp: timestamp,
        landmarks: emptyLandmarks, width: width, height: height
    )
}
