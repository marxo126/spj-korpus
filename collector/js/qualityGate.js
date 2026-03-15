/**
 * SPJ Collector — Post-recording quality gate
 * Analyzes 5 frames from recorded video using IMAGE-mode MediaPipe detectors.
 * Returns pass/fail with per-check scores.
 */

const QualityGate = {
    _handDetector: null,
    _faceDetector: null,
    _loaded: false,

    /** Load IMAGE-mode detectors (separate from VIDEO-mode live checker) */
    async loadModels() {
        if (this._loaded) return true;
        try {
            const vision = await import(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
            );
            const { HandLandmarker, FaceDetector, FilesetResolver } = vision;

            const filesetResolver = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
            );

            this._handDetector = await HandLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
                    delegate: 'GPU'
                },
                runningMode: 'IMAGE',
                numHands: 2
            });

            this._faceDetector = await FaceDetector.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
                    delegate: 'GPU'
                },
                runningMode: 'IMAGE'
            });

            this._loaded = true;
            return true;
        } catch (err) {
            console.warn('QualityGate: MediaPipe load failed, skipping gate:', err);
            return false;
        }
    },

    /**
     * Analyze a recorded video blob.
     * Extracts 5 evenly-spaced frames and runs detection on each.
     * @param {Blob} blob - recorded video
     * @returns {Object} { passed, face: {score, required}, hands: {score, required}, brightness: {score, required} }
     */
    async analyze(blob) {
        // If models didn't load, pass by default
        if (!this._loaded) {
            return { passed: true, skipped: true };
        }

        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        video.src = URL.createObjectURL(blob);

        // Wait for metadata
        await new Promise((resolve, reject) => {
            video.onloadedmetadata = resolve;
            video.onerror = reject;
            setTimeout(() => reject(new Error('Video metadata timeout')), 10000);
        });

        const duration = video.duration;
        if (!duration || duration < 0.5) {
            URL.revokeObjectURL(video.src);
            return { passed: true, skipped: true };
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        // Extract 5 frames
        const positions = [];
        for (let i = 0; i < 5; i++) {
            positions.push((duration * (i + 0.5)) / 5);
        }

        let faceCount = 0;
        let handsCount = 0;
        let brightnessOkCount = 0;

        for (const time of positions) {
            // Seek to position
            video.currentTime = time;
            await new Promise(resolve => {
                video.onseeked = resolve;
                setTimeout(resolve, 2000); // timeout
            });

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Face detection
            try {
                const faceResult = this._faceDetector.detect(canvas);
                if (faceResult.detections && faceResult.detections.length > 0) {
                    faceCount++;
                }
            } catch (e) { /* ignore */ }

            // Hand detection
            try {
                const handResult = this._handDetector.detect(canvas);
                if (handResult.landmarks && handResult.landmarks.length > 0) {
                    handsCount++;
                }
            } catch (e) { /* ignore */ }

            // Brightness check
            const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            const step = 16;
            let sum = 0;
            const count = Math.floor(pixels.length / step);
            for (let i = 0; i < pixels.length; i += step) {
                sum += 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
            }
            const avg = sum / count;
            if (avg >= 60 && avg <= 220) brightnessOkCount++;
        }

        URL.revokeObjectURL(video.src);

        const facePass = faceCount >= 4;
        const handsPass = handsCount >= 3;
        const brightnessPass = brightnessOkCount >= 4;

        return {
            passed: facePass && handsPass && brightnessPass,
            skipped: false,
            face: { score: faceCount, required: 4, passed: facePass },
            hands: { score: handsCount, required: 3, passed: handsPass },
            brightness: { score: brightnessOkCount, required: 4, passed: brightnessPass }
        };
    }
};
