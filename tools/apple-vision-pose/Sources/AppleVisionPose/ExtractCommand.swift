import ArgumentParser
import Foundation

struct ExtractCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "extract",
        abstract: "Extract pose landmarks from a single video file"
    )

    @Option(name: .long, help: "Input video file path (.mp4, .mkv, .mov)")
    var input: String

    @Option(name: .long, help: "Output JSONL file path")
    var output: String

    @Option(name: .long, help: "Target FPS for sampling (0 = every frame)")
    var fps: Int = 0

    @Option(name: .long, help: "Concurrent frame workers (1 = sequential, >1 = GCD concurrent)")
    var concurrent: Int = 1

    func run() throws {
        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: output)

        // Validate input exists
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input file does not exist: \(input)")
        }

        // Create output directory if needed
        let outputDir = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true)

        let reporter = ProgressReporter()
        let startTime = Date()

        if concurrent > 1 {
            let extractor = ConcurrentPoseExtractor(
                targetFPS: fps,
                concurrency: concurrent,
                reporter: reporter,
                fileName: inputURL.lastPathComponent
            )
            try extractor.extract(inputURL: inputURL, outputURL: outputURL)
        } else {
            let extractor = PoseExtractor(
                targetFPS: fps,
                reporter: reporter,
                fileName: inputURL.lastPathComponent
            )
            try extractor.extract(inputURL: inputURL, outputURL: outputURL)
        }

        let elapsed = Date().timeIntervalSince(startTime)

        reporter.reportComplete(
            file: inputURL.lastPathComponent,
            totalFrames: countLines(in: outputURL),
            duration: elapsed
        )
    }

}
