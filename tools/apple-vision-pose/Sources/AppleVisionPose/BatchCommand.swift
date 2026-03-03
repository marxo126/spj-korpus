import ArgumentParser
import Foundation

struct BatchCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "batch",
        abstract: "Extract pose landmarks from all videos in a directory"
    )

    @Option(name: .long, help: "Input directory containing video files")
    var inputDir: String

    @Option(name: .long, help: "Output directory for JSONL files")
    var outputDir: String

    @Option(name: .long, help: "Maximum concurrent video extractions")
    var workers: Int = 4

    @Option(name: .long, help: "Target FPS for sampling (0 = every frame)")
    var fps: Int = 0

    @Option(name: .long, help: "Concurrent frames per video (1 = sequential, >1 = GCD concurrent)")
    var frameConcurrent: Int = 1

    /// Supported video file extensions.
    private static let videoExtensions: Set<String> = ["mp4", "mkv", "mov"]

    func run() throws {
        let inputDirURL = URL(fileURLWithPath: inputDir)
        let outputDirURL = URL(fileURLWithPath: outputDir)

        // Validate input directory
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: inputDirURL.path, isDirectory: &isDir),
              isDir.boolValue
        else {
            throw ValidationError("Input directory does not exist: \(inputDir)")
        }

        // Create output directory
        try FileManager.default.createDirectory(
            at: outputDirURL, withIntermediateDirectories: true)

        // Find video files
        let contents = try FileManager.default.contentsOfDirectory(
            at: inputDirURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        let videoFiles = contents.filter { url in
            Self.videoExtensions.contains(url.pathExtension.lowercased())
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !videoFiles.isEmpty else {
            throw ValidationError(
                "No video files (.mp4, .mkv, .mov) found in \(inputDir)")
        }

        let reporter = ProgressReporter()

        // Report batch start
        let batchInfo: [String: Any] = [
            "status": "batch_start",
            "total_files": videoFiles.count,
            "workers": workers,
            "frame_concurrent": frameConcurrent,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: batchInfo) {
            FileHandle.standardError.write(data)
            FileHandle.standardError.write("\n".data(using: .utf8)!)
        }

        // Process files concurrently with worker limit
        let semaphore = DispatchSemaphore(value: workers)
        let group = DispatchGroup()
        let queue = DispatchQueue(
            label: "com.spj.pose.batch", attributes: .concurrent)

        var errors: [(String, String)] = []
        let errorsLock = NSLock()

        for videoURL in videoFiles {
            let outputFileName =
                videoURL.deletingPathExtension().lastPathComponent + ".jsonl"
            let outputFileURL = outputDirURL.appendingPathComponent(outputFileName)

            // Skip if output already exists
            if FileManager.default.fileExists(atPath: outputFileURL.path) {
                let skipInfo: [String: Any] = [
                    "status": "skipped",
                    "file": videoURL.lastPathComponent,
                    "reason": "output exists",
                ]
                if let data = try? JSONSerialization.data(withJSONObject: skipInfo) {
                    FileHandle.standardError.write(data)
                    FileHandle.standardError.write("\n".data(using: .utf8)!)
                }
                continue
            }

            group.enter()
            semaphore.wait()

            queue.async {
                defer {
                    semaphore.signal()
                    group.leave()
                }

                let startTime = Date()
                do {
                    if self.frameConcurrent > 1 {
                        let extractor = ConcurrentPoseExtractor(
                            targetFPS: self.fps,
                            concurrency: self.frameConcurrent,
                            reporter: reporter,
                            fileName: videoURL.lastPathComponent
                        )
                        try extractor.extract(
                            inputURL: videoURL, outputURL: outputFileURL)
                    } else {
                        let extractor = PoseExtractor(
                            targetFPS: self.fps,
                            reporter: reporter,
                            fileName: videoURL.lastPathComponent
                        )
                        try extractor.extract(
                            inputURL: videoURL, outputURL: outputFileURL)
                    }

                    let elapsed = Date().timeIntervalSince(startTime)
                    reporter.reportComplete(
                        file: videoURL.lastPathComponent,
                        totalFrames: countLines(in: outputFileURL),
                        duration: elapsed
                    )
                } catch {
                    errorsLock.lock()
                    errors.append(
                        (videoURL.lastPathComponent,
                         error.localizedDescription))
                    errorsLock.unlock()
                    reporter.reportError(
                        error.localizedDescription,
                        file: videoURL.lastPathComponent
                    )
                }
            }
        }

        group.wait()

        // Report batch completion
        let batchComplete: [String: Any] = [
            "status": "batch_complete",
            "total_files": videoFiles.count,
            "errors": errors.count,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: batchComplete) {
            FileHandle.standardError.write(data)
            FileHandle.standardError.write("\n".data(using: .utf8)!)
        }

        if !errors.isEmpty {
            for (file, msg) in errors {
                reporter.reportError("Failed: \(msg)", file: file)
            }
        }
    }

}
