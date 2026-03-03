import Foundation
import Vision

/// Maps Apple Vision landmark detections to a MediaPipe-compatible 543-slot array.
///
/// MediaPipe landmark layout (543 total):
///   - Indices 0–32:   Body (33 landmarks)
///   - Indices 33–53:  Left hand (21 landmarks)
///   - Indices 54–74:  Right hand (21 landmarks)
///   - Indices 75–542: Face mesh (468 landmarks)
///
/// Apple Vision provides fewer landmarks than MediaPipe in some regions.
/// Unmapped slots are filled with [0, 0, 0].
struct LandmarkMapper {

    /// Total number of slots in the MediaPipe-compatible output array.
    static let totalSlots = 543

    /// A single landmark with normalized coordinates and confidence.
    typealias Landmark = (x: Double, y: Double, confidence: Double)

    /// Empty landmark (used for unmapped slots).
    static let zeroLandmark: Landmark = (0.0, 0.0, 0.0)

    // MARK: - Body Mapping

    /// Maps VNHumanBodyPose2DObservation joint names to MediaPipe body indices (0-32).
    static let bodyJointMap: [(VNHumanBodyPoseObservation.JointName, Int)] = [
        (.nose, 0),
        (.leftEye, 1),
        (.rightEye, 5),
        (.leftEar, 7),
        (.rightEar, 8),
        (.leftShoulder, 11),
        (.rightShoulder, 12),
        (.leftElbow, 13),
        (.rightElbow, 14),
        (.leftWrist, 15),
        (.rightWrist, 16),
        (.leftHip, 23),
        (.rightHip, 24),
        (.leftKnee, 25),
        (.rightKnee, 26),
        (.leftAnkle, 27),
        (.rightAnkle, 28),
    ]

    /// Map body pose observation to landmark array slots.
    /// - Parameter observation: Detected body pose.
    /// - Returns: Array of (index, landmark) pairs to place into the 543-slot array.
    static func mapBody(_ observation: VNHumanBodyPoseObservation) -> [(Int, Landmark)] {
        var result: [(Int, Landmark)] = []
        for (jointName, mpIndex) in bodyJointMap {
            guard let point = try? observation.recognizedPoint(jointName),
                  point.confidence > 0.0
            else { continue }
            // Vision: origin bottom-left, y goes up. MediaPipe: origin top-left, y goes down.
            let landmark: Landmark = (
                x: Double(point.location.x),
                y: 1.0 - Double(point.location.y),
                confidence: Double(point.confidence)
            )
            result.append((mpIndex, landmark))
        }
        return result
    }

    // MARK: - Hand Mapping

    /// Ordered list of hand joint names matching MediaPipe hand model (0-20).
    static let handJointOrder: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip,
    ]

    /// Map hand pose observation to landmark array slots.
    /// - Parameters:
    ///   - observation: Detected hand pose.
    ///   - isLeft: Whether this is the left hand (offset 33) or right hand (offset 54).
    /// - Returns: Array of (index, landmark) pairs.
    static func mapHand(_ observation: VNHumanHandPoseObservation, isLeft: Bool) -> [(Int, Landmark)] {
        let baseOffset = isLeft ? 33 : 54
        var result: [(Int, Landmark)] = []
        for (i, jointName) in handJointOrder.enumerated() {
            guard let point = try? observation.recognizedPoint(jointName),
                  point.confidence > 0.0
            else { continue }
            let landmark: Landmark = (
                x: Double(point.location.x),
                y: 1.0 - Double(point.location.y),
                confidence: Double(point.confidence)
            )
            result.append((baseOffset + i, landmark))
        }
        return result
    }

    /// Determine if a hand is left or right based on chirality or wrist position.
    /// - Parameters:
    ///   - handObservation: The hand pose observation.
    ///   - bodyObservation: Optional body pose for wrist-position fallback.
    /// - Returns: True if the hand is likely the left hand.
    static func isLeftHand(
        _ handObservation: VNHumanHandPoseObservation,
        bodyObservation: VNHumanBodyPoseObservation?
    ) -> Bool {
        // Use chirality if available (macOS 14+)
        if #available(macOS 14.0, *) {
            switch handObservation.chirality {
            case .left: return true
            case .right: return false
            default: break
            }
        }

        // Fallback: compare hand wrist position to body center
        guard let handWrist = try? handObservation.recognizedPoint(.wrist),
              handWrist.confidence > 0.0,
              let body = bodyObservation
        else {
            // No body reference — default to left for first hand
            return true
        }

        let leftShoulder = try? body.recognizedPoint(.leftShoulder)
        let rightShoulder = try? body.recognizedPoint(.rightShoulder)

        if let ls = leftShoulder, ls.confidence > 0.0,
           let rs = rightShoulder, rs.confidence > 0.0
        {
            let centerX = (ls.location.x + rs.location.x) / 2.0
            // In Vision coordinates: leftShoulder is actually on the person's left
            // (screen-right). Hand on screen-left of center is the right hand.
            return handWrist.location.x > centerX
        }

        return true
    }

    // MARK: - Face Mapping

    /// MediaPipe face mesh indices (relative to face mesh, i.e., 0-467) for each region.
    /// These get offset by 75 when placed in the 543-slot array.
    static let faceMeshOffset = 75

    /// Lips outer contour — MediaPipe face mesh indices.
    static let lipsOuterIndices: [Int] = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0
    ]

    /// Lips inner contour — MediaPipe face mesh indices.
    static let lipsInnerIndices: [Int] = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13
    ]

    /// Left eye contour — MediaPipe face mesh indices.
    static let leftEyeIndices: [Int] = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]

    /// Right eye contour — MediaPipe face mesh indices.
    static let rightEyeIndices: [Int] = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]

    /// Left eyebrow — MediaPipe face mesh indices.
    static let leftEyebrowIndices: [Int] = [
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    ]

    /// Right eyebrow — MediaPipe face mesh indices.
    static let rightEyebrowIndices: [Int] = [
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276
    ]

    /// Nose — MediaPipe face mesh indices.
    static let noseIndices: [Int] = [
        1, 2, 98, 327, 4, 5, 195
    ]

    /// Map face landmarks observation to the 543-slot array.
    ///
    /// Apple Vision provides face landmarks as region-based point groups.
    /// We map available points from each region to the corresponding MediaPipe
    /// face mesh vertex indices.
    ///
    /// - Parameter observation: Detected face landmarks.
    /// - Returns: Array of (index, landmark) pairs.
    static func mapFace(_ observation: VNFaceLandmarks2D) -> [(Int, Landmark)] {
        var result: [(Int, Landmark)] = []

        // Helper: map a Vision region's points to MediaPipe indices
        func mapRegion(_ region: VNFaceLandmarkRegion2D?, to mpIndices: [Int]) {
            guard let region = region else { return }
            let points = region.normalizedPoints
            let count = min(points.count, mpIndices.count)
            for i in 0..<count {
                let p = points[i]
                let globalIndex = faceMeshOffset + mpIndices[i]
                guard globalIndex < totalSlots else { continue }
                let landmark: Landmark = (
                    x: Double(p.x),
                    y: 1.0 - Double(p.y),  // Flip Y axis
                    confidence: 1.0  // Face landmarks don't have per-point confidence
                )
                result.append((globalIndex, landmark))
            }
        }

        // Map outer lips
        mapRegion(observation.outerLips, to: lipsOuterIndices)
        // Map inner lips
        mapRegion(observation.innerLips, to: lipsInnerIndices)
        // Map left eye
        mapRegion(observation.leftEye, to: leftEyeIndices)
        // Map right eye
        mapRegion(observation.rightEye, to: rightEyeIndices)
        // Map left eyebrow
        mapRegion(observation.leftEyebrow, to: leftEyebrowIndices)
        // Map right eyebrow
        mapRegion(observation.rightEyebrow, to: rightEyebrowIndices)
        // Map nose crest + median line to nose indices
        mapRegion(observation.noseCrest, to: noseIndices)

        return result
    }

    // MARK: - Combined Assembly

    /// Assemble a full 543-slot landmark array from all detections for a single frame.
    /// - Parameters:
    ///   - bodyObservation: Optional body pose detection.
    ///   - handObservations: Array of hand pose detections (0-2 hands).
    ///   - faceLandmarks: Optional face landmarks from face detection.
    /// - Returns: Array of exactly 543 Landmark tuples.
    static func assemble(
        bodyObservation: VNHumanBodyPoseObservation?,
        handObservations: [VNHumanHandPoseObservation],
        faceLandmarks: VNFaceLandmarks2D?
    ) -> [Landmark] {
        var slots = Array(repeating: zeroLandmark, count: totalSlots)

        // Body
        if let body = bodyObservation {
            for (idx, lm) in mapBody(body) {
                slots[idx] = lm
            }
        }

        // Hands — assign left/right
        var leftAssigned = false
        var rightAssigned = false
        for hand in handObservations {
            let isLeft = isLeftHand(hand, bodyObservation: bodyObservation)
            if isLeft && !leftAssigned {
                for (idx, lm) in mapHand(hand, isLeft: true) {
                    slots[idx] = lm
                }
                leftAssigned = true
            } else if !isLeft && !rightAssigned {
                for (idx, lm) in mapHand(hand, isLeft: false) {
                    slots[idx] = lm
                }
                rightAssigned = true
            } else if !leftAssigned {
                // Fallback: if both detected as right, assign second to left
                for (idx, lm) in mapHand(hand, isLeft: true) {
                    slots[idx] = lm
                }
                leftAssigned = true
            } else if !rightAssigned {
                for (idx, lm) in mapHand(hand, isLeft: false) {
                    slots[idx] = lm
                }
                rightAssigned = true
            }
        }

        // Face
        if let face = faceLandmarks {
            for (idx, lm) in mapFace(face) {
                slots[idx] = lm
            }
        }

        return slots
    }
}
