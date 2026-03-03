import Foundation

/// Reports extraction progress as JSON lines to stderr.
/// This allows the Python consumer to parse progress without interfering with stdout JSONL output.
struct ProgressReporter {

    /// Report frame-level progress.
    /// - Parameters:
    ///   - frame: Current frame index (0-based).
    ///   - total: Total number of frames to process.
    ///   - file: Optional filename (used in batch mode).
    func report(frame: Int, total: Int, file: String? = nil) {
        var msg: [String: Any] = ["frame": frame, "total": total]
        if let file = file {
            msg["file"] = file
        }
        guard let data = try? JSONSerialization.data(withJSONObject: msg) else { return }
        FileHandle.standardError.write(data)
        FileHandle.standardError.write("\n".data(using: .utf8)!)
    }

    /// Report an error message to stderr as JSON.
    /// - Parameters:
    ///   - message: Human-readable error description.
    ///   - file: Optional filename context.
    func reportError(_ message: String, file: String? = nil) {
        var msg: [String: Any] = ["error": message]
        if let file = file {
            msg["file"] = file
        }
        guard let data = try? JSONSerialization.data(withJSONObject: msg) else { return }
        FileHandle.standardError.write(data)
        FileHandle.standardError.write("\n".data(using: .utf8)!)
    }

    /// Report completion for a file (batch mode).
    /// - Parameters:
    ///   - file: Filename that finished processing.
    ///   - totalFrames: Number of frames processed.
    ///   - duration: Wall-clock time in seconds.
    func reportComplete(file: String, totalFrames: Int, duration: Double) {
        let msg: [String: Any] = [
            "status": "complete",
            "file": file,
            "total_frames": totalFrames,
            "duration_seconds": round(duration * 100) / 100,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: msg) else { return }
        FileHandle.standardError.write(data)
        FileHandle.standardError.write("\n".data(using: .utf8)!)
    }
}
