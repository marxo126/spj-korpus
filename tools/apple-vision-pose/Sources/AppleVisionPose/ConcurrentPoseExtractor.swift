import AVFoundation
import Foundation
import Vision

/// Concurrent pose extractor using GCD for parallel Vision inference.
///
/// Reads frames sequentially from AVAssetReader (required by the API),
/// dispatches Vision inference to a concurrent GCD queue bounded by a semaphore,
/// and writes results to JSONL in correct frame order.
///
/// Key design decisions:
/// - `trackOutput.alwaysCopiesSampleData = true` so each worker gets its own pixel buffer
/// - Fresh VNDetect*Request objects per async block (request objects store results, NOT thread-safe)
/// - LandmarkMapper.assemble() is all-static with no mutable state — safe for concurrent use
/// - Results stored in partner-dictnary keyed by frame index, drained in order by the main thread
/// - Memory bounded: semaphore limits in-flight buffers (concurrency × frame_size)
final class ConcurrentPoseExtractor {

    /// Target FPS for sampling. 0 means process every frame.
    let targetFPS: Int

    /// Maximum number of concurrent Vision inference workers.
    let concurrency: Int

    /// Progress reporter for stderr output.
    let reporter: ProgressReporter

    /// Optional filename context for progress reporting (batch mode).
    let fileName: String?

    // Ordered result drain state (protected by lock)
    private var pendingResults: [Int: (Double, [LandmarkMapper.Landmark])] = [:]
    private let lock = NSLock()
    private var nextWriteIndex = 0

    init(
        targetFPS: Int = 0,
        concurrency: Int = 8,
        reporter: ProgressReporter = ProgressReporter(),
        fileName: String? = nil
    ) {
        self.targetFPS = targetFPS
        self.concurrency = max(1, concurrency)
        self.reporter = reporter
        self.fileName = fileName
    }

    /// Extract landmarks from a video file and write JSONL to the output path.
    /// - Parameters:
    ///   - inputURL: Path to the input video file.
    ///   - outputURL: Path to the output JSONL file.
    /// - Throws: If the video cannot be read or processing fails.
    func extract(inputURL: URL, outputURL: URL) throws {
        let asset = AVURLAsset(url: inputURL)

        // Load video properties using modern async API, bridged to sync via semaphore
        let loadSem = DispatchSemaphore(value: 0)
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
                    loadSem.signal()
                    return
                }
                videoTrack = track
                naturalSize = try await track.load(.naturalSize)
                nominalFrameRate = try await track.load(.nominalFrameRate)
                assetDuration = try await asset.load(.duration)
            } catch {
                loadError = error
            }
            loadSem.signal()
        }
        loadSem.wait()

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

        // CRITICAL: Each concurrent worker needs its own pixel buffer.
        // Without copying, the reader may recycle the underlying memory before
        // the worker finishes processing.
        trackOutput.alwaysCopiesSampleData = true

        guard reader.canAdd(trackOutput) else {
            throw PoseExtractionError.cannotAddTrackOutput
        }
        reader.add(trackOutput)

        guard reader.startReading() else {
            throw PoseExtractionError.cannotStartReading(
                reader.error?.localizedDescription ?? "unknown error"
            )
        }

        // Open output file
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        guard let fileHandle = FileHandle(forWritingAtPath: outputURL.path) else {
            throw PoseExtractionError.cannotOpenOutput(outputURL.path)
        }
        defer { fileHandle.closeFile() }

        // Reset drain state
        pendingResults = [:]
        nextWriteIndex = 0

        let sem = DispatchSemaphore(value: concurrency)
        let workerQueue = DispatchQueue(
            label: "com.spj.pose.concurrent", attributes: .concurrent)
        let group = DispatchGroup()

        let emptyLandmarks = Array(
            repeating: LandmarkMapper.zeroLandmark, count: LandmarkMapper.totalSlots)
        let reportInterval = max(1, totalFrames / 100)

        var frameIndex = 0
        var outputFrameIndex = 0

        // Process frames: read sequentially, infer concurrently, write in order
        while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
            // Check if we should process this frame (FPS sampling)
            if frameIndex % frameInterval != 0 {
                frameIndex += 1
                continue
            }

            let timestamp = CMTimeGetSeconds(
                CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

            // Get pixel buffer
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                // No pixel buffer — store empty frame directly
                storeResult(
                    index: outputFrameIndex, timestamp: timestamp,
                    landmarks: emptyLandmarks)
                drainAndWrite(
                    to: fileHandle, width: videoWidth, height: videoHeight,
                    totalFrames: totalFrames, reportInterval: reportInterval)
                frameIndex += 1
                outputFrameIndex += 1
                continue
            }

            let capturedIndex = outputFrameIndex
            let capturedTimestamp = timestamp
            // Closure captures pixelBuffer — ARC retains it, keeping the copied
            // pixel data alive until the worker finishes.

            sem.wait()
            group.enter()
            workerQueue.async { [self] in
                defer {
                    sem.signal()
                    group.leave()
                }

                // Fresh request objects per task — they store results internally
                // and are NOT thread-safe for concurrent use.
                let bodyReq = VNDetectHumanBodyPoseRequest()
                let handReq = VNDetectHumanHandPoseRequest()
                handReq.maximumHandCount = 2
                let faceReq = VNDetectFaceLandmarksRequest()

                let handler = VNImageRequestHandler(
                    cvPixelBuffer: pixelBuffer,
                    orientation: .up,
                    options: [:]
                )

                var bodyObs: VNHumanBodyPoseObservation?
                var handObs: [VNHumanHandPoseObservation] = []
                var faceLm: VNFaceLandmarks2D?

                do {
                    try handler.perform([bodyReq, handReq, faceReq])
                    bodyObs = bodyReq.results?.first
                    handObs = handReq.results ?? []
                    faceLm = faceReq.results?.first?.landmarks
                } catch {
                    self.reporter.reportError(
                        "Vision request failed at frame \(capturedIndex): \(error.localizedDescription)",
                        file: self.fileName
                    )
                }

                let landmarks = LandmarkMapper.assemble(
                    bodyObservation: bodyObs,
                    handObservations: handObs,
                    faceLandmarks: faceLm
                )

                self.storeResult(
                    index: capturedIndex, timestamp: capturedTimestamp,
                    landmarks: landmarks)
            }

            // Try to drain completed frames in order (non-blocking)
            drainAndWrite(
                to: fileHandle, width: videoWidth, height: videoHeight,
                totalFrames: totalFrames, reportInterval: reportInterval)

            frameIndex += 1
            outputFrameIndex += 1
        }

        // Wait for all in-flight workers to complete
        group.wait()

        // Drain remaining results
        drainAndWrite(
            to: fileHandle, width: videoWidth, height: videoHeight,
            totalFrames: totalFrames, reportInterval: reportInterval)

        // Final progress report
        reporter.report(frame: nextWriteIndex, total: nextWriteIndex, file: fileName)

        // Check reader status
        if reader.status == .failed {
            throw PoseExtractionError.readingFailed(
                reader.error?.localizedDescription ?? "unknown error"
            )
        }
    }

    // MARK: - Thread-safe result storage

    private func storeResult(
        index: Int, timestamp: Double, landmarks: [LandmarkMapper.Landmark]
    ) {
        lock.lock()
        pendingResults[index] = (timestamp, landmarks)
        lock.unlock()
    }

    /// Drain contiguous completed frames from pendingResults and write to file in order.
    /// Called from the main (producer) thread only — fileHandle writes are single-threaded.
    private func drainAndWrite(
        to fileHandle: FileHandle,
        width: Int,
        height: Int,
        totalFrames: Int,
        reportInterval: Int
    ) {
        lock.lock()
        while let (timestamp, landmarks) = pendingResults[nextWriteIndex] {
            pendingResults.removeValue(forKey: nextWriteIndex)
            let idx = nextWriteIndex
            nextWriteIndex += 1
            // WARNING: must not hold lock during writeFrameJSONL — it does I/O
            lock.unlock()

            writeFrameJSONL(
                to: fileHandle, frame: idx, timestamp: timestamp,
                landmarks: landmarks, width: width, height: height)

            if idx % reportInterval == 0 || idx == totalFrames - 1 {
                reporter.report(frame: idx, total: totalFrames, file: fileName)
            }

            lock.lock()
        }
        lock.unlock()
    }
}
