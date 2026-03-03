import AVFoundation
import Foundation
import Vision

/// Extracts pose landmarks from video files using Apple Vision framework.
///
/// For each frame, runs three Vision requests:
/// 1. Body pose detection (up to 19 joints)
/// 2. Hand pose detection (up to 2 hands, 21 joints each)
/// 3. Face landmark detection (76 points)
///
/// Results are mapped to a MediaPipe-compatible 543-slot landmark array
/// and written as JSONL (one JSON object per frame).
final class PoseExtractor {

    /// Target FPS for sampling. 0 means process every frame.
    let targetFPS: Int

    /// Progress reporter for stderr output.
    let reporter: ProgressReporter

    /// Optional filename context for progress reporting (batch mode).
    let fileName: String?

    init(targetFPS: Int = 0, reporter: ProgressReporter = ProgressReporter(), fileName: String? = nil) {
        self.targetFPS = targetFPS
        self.reporter = reporter
        self.fileName = fileName
    }

    /// Extract landmarks from a video file and write JSONL to the output path.
    /// - Parameters:
    ///   - inputURL: Path to the input video file.
    ///   - outputURL: Path to the output JSONL file.
    /// - Throws: If the video cannot be read or Vision requests fail catastrophically.
    func extract(inputURL: URL, outputURL: URL) throws {
        let asset = AVURLAsset(url: inputURL)

        // Load video properties using modern async API, bridged to sync via semaphore
        let semaphore = DispatchSemaphore(value: 0)
        var videoTrack: AVAssetTrack?
        var loadError: Error?
        var naturalSize: CGSize = .zero
        var nominalFrameRate: Float = 0
        var assetDuration: CMTime = .zero

        Task {
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else {
                    loadError = PoseExtractionError.noVideoTrack(inputURL.path)
                    semaphore.signal()
                    return
                }
                videoTrack = track
                naturalSize = try await track.load(.naturalSize)
                nominalFrameRate = try await track.load(.nominalFrameRate)
                assetDuration = try await asset.load(.duration)
            } catch {
                loadError = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        if let loadError = loadError { throw loadError }
        guard let videoTrack = videoTrack else {
            throw PoseExtractionError.noVideoTrack(inputURL.path)
        }

        // Determine video properties
        let videoWidth = Int(naturalSize.width)
        let videoHeight = Int(naturalSize.height)
        let duration = CMTimeGetSeconds(assetDuration)
        let totalFrames: Int

        // Calculate frame interval for sampling
        let frameInterval: Int
        if targetFPS > 0 && Float(targetFPS) < nominalFrameRate {
            frameInterval = max(1, Int(round(nominalFrameRate / Float(targetFPS))))
            totalFrames = Int(ceil(duration * Double(targetFPS)))
        } else {
            frameInterval = 1
            totalFrames = Int(ceil(duration * Double(nominalFrameRate)))
        }

        // Set up asset reader
        let reader = try AVAssetReader(asset: asset)

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ]

        let trackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        trackOutput.alwaysCopiesSampleData = false

        guard reader.canAdd(trackOutput) else {
            throw PoseExtractionError.cannotAddTrackOutput
        }
        reader.add(trackOutput)

        guard reader.startReading() else {
            throw PoseExtractionError.cannotStartReading(
                reader.error?.localizedDescription ?? "unknown error"
            )
        }

        // Create Vision requests
        let bodyRequest = VNDetectHumanBodyPoseRequest()
        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 2
        let faceRequest = VNDetectFaceLandmarksRequest()

        // Open output file
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        guard let fileHandle = FileHandle(forWritingAtPath: outputURL.path) else {
            throw PoseExtractionError.cannotOpenOutput(outputURL.path)
        }
        defer { fileHandle.closeFile() }

        var frameIndex = 0
        var outputFrameIndex = 0
        let reportInterval = max(1, totalFrames / 100)  // Report ~100 times

        // Process frames
        while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            // Check if we should process this frame (FPS sampling)
            if frameIndex % frameInterval != 0 {
                frameIndex += 1
                continue
            }

            let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

            // Get pixel buffer
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                // Write empty frame if we can't get pixel buffer
                writeEmptyFrameJSONL(
                    to: fileHandle, frame: outputFrameIndex, timestamp: timestamp,
                    width: videoWidth, height: videoHeight
                )
                frameIndex += 1
                outputFrameIndex += 1
                continue
            }

            // Run Vision requests
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: .up,
                options: [:]
            )

            var bodyObservation: VNHumanBodyPoseObservation?
            var handObservations: [VNHumanHandPoseObservation] = []
            var faceLandmarks: VNFaceLandmarks2D?

            do {
                try handler.perform([bodyRequest, handRequest, faceRequest])

                // Extract body
                bodyObservation = bodyRequest.results?.first

                // Extract hands
                if let hands = handRequest.results {
                    handObservations = hands
                }

                // Extract face
                if let faceObs = faceRequest.results?.first {
                    faceLandmarks = faceObs.landmarks
                }
            } catch {
                // Vision request failed for this frame — write zeros
                reporter.reportError(
                    "Vision request failed at frame \(frameIndex): \(error.localizedDescription)",
                    file: fileName
                )
            }

            // Assemble 543-slot landmark array
            let landmarks = LandmarkMapper.assemble(
                bodyObservation: bodyObservation,
                handObservations: handObservations,
                faceLandmarks: faceLandmarks
            )

            // Write JSONL line
            writeFrameJSONL(
                to: fileHandle,
                frame: outputFrameIndex,
                timestamp: timestamp,
                landmarks: landmarks,
                width: videoWidth,
                height: videoHeight
            )

            // Report progress
            if outputFrameIndex % reportInterval == 0 || outputFrameIndex == totalFrames - 1 {
                reporter.report(frame: outputFrameIndex, total: totalFrames, file: fileName)
            }

            frameIndex += 1
            outputFrameIndex += 1
        }

        // Final progress report
        reporter.report(frame: outputFrameIndex, total: outputFrameIndex, file: fileName)

        // Check reader status
        if reader.status == .failed {
            throw PoseExtractionError.readingFailed(
                reader.error?.localizedDescription ?? "unknown error"
            )
        }
    }

}

// MARK: - Errors

enum PoseExtractionError: LocalizedError {
    case noVideoTrack(String)
    case cannotAddTrackOutput
    case cannotStartReading(String)
    case cannotOpenOutput(String)
    case readingFailed(String)

    var errorDescription: String? {
        switch self {
        case .noVideoTrack(let path):
            return "No video track found in '\(path)'"
        case .cannotAddTrackOutput:
            return "Cannot add track output to asset reader"
        case .cannotStartReading(let detail):
            return "Cannot start reading video: \(detail)"
        case .cannotOpenOutput(let path):
            return "Cannot open output file '\(path)' for writing"
        case .readingFailed(let detail):
            return "Video reading failed: \(detail)"
        }
    }
}
