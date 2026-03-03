import ArgumentParser

@main
struct AppleVisionPose: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "apple-vision-pose",
        abstract: "Extract pose landmarks from video using Apple Vision framework",
        subcommands: [ExtractCommand.self, BatchCommand.self]
    )
}
