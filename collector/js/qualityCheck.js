/**
 * SPJ Collector — Real-time quality checks on camera preview
 * Runs MediaPipe HandLandmarker + FaceDetector + brightness analysis.
 * All checks happen in the browser — zero server cost.
 */

class QualityChecker {
    constructor(videoElement, onStatusChange) {
        this.video = videoElement;
        this.onStatusChange = onStatusChange;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        this.running = false;
        this.handDetector = null;
        this.faceDetector = null;
        this.frameCount = 0;
        this.checkInterval = 5; // check every 5th frame (~6 fps)
    }

    async start() {
        await this.loadModels();
        this.running = true;
        this.loop();
    }

    stop() {
        this.running = false;
    }

    async loadModels() {
        try {
            // Import MediaPipe tasks-vision from CDN
            const vision = await import(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
            );

            const { HandLandmarker, FaceDetector, FilesetResolver } = vision;

            const filesetResolver = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
            );

            this.handDetector = await HandLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
                    delegate: 'GPU'
                },
                runningMode: 'VIDEO',
                numHands: 2
            });

            this.faceDetector = await FaceDetector.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
                    delegate: 'GPU'
                },
                runningMode: 'VIDEO'
            });

        } catch (err) {
            console.warn('MediaPipe load failed, quality checks disabled:', err);
            // Still work without MediaPipe — just no hand/face detection
        }
    }

    loop() {
        if (!this.running) return;

        this.frameCount++;
        if (this.frameCount % this.checkInterval === 0) {
            this.check();
        }

        requestAnimationFrame(() => this.loop());
    }

    check() {
        if (this.video.readyState < 2) return; // video not ready

        const w = this.video.videoWidth;
        const h = this.video.videoHeight;
        if (!w || !h) return;

        this.canvas.width = w;
        this.canvas.height = h;
        this.ctx.drawImage(this.video, 0, 0, w, h);

        const status = {
            hands: true,
            handsEdge: false,
            face: true,
            faceWarning: null,
            light: 'ok',
            contrast: true
        };

        // 1. Brightness check
        status.light = this.checkBrightness();

        // 2. Low contrast check
        status.contrast = this.checkContrast();

        // 3. Hand detection
        if (this.handDetector) {
            try {
                const handResult = this.handDetector.detectForVideo(this.video, performance.now());
                status.hands = handResult.landmarks && handResult.landmarks.length > 0;
                // Check if hands are near frame edge (<5% from border)
                if (status.hands && handResult.landmarks) {
                    status.handsEdge = handResult.landmarks.some(hand =>
                        hand.some(lm => lm.x < 0.05 || lm.x > 0.95 || lm.y < 0.05 || lm.y > 0.95)
                    );
                }
            } catch (e) {
                // ignore detection errors
            }
        }

        // 3. Face detection
        if (this.faceDetector) {
            try {
                const faceResult = this.faceDetector.detectForVideo(this.video, performance.now());
                if (!faceResult.detections || faceResult.detections.length === 0) {
                    status.face = false;
                } else {
                    const det = faceResult.detections[0];
                    const bbox = det.boundingBox;
                    if (bbox) {
                        const topY = bbox.originY / h;       // head top relative to frame
                        const centerY = (bbox.originY + bbox.height / 2) / h;
                        const widthRatio = bbox.width / w;

                        if (topY < 0.03) status.faceWarning = 'Hlava je príliš hore — posuňte sa nižšie';
                        else if (centerY < 0.15) status.faceWarning = 'Príliš vysoko';
                        else if (centerY > 0.60) status.faceWarning = 'Príliš nízko';
                        else if (widthRatio > 0.50) status.faceWarning = 'Príliš blízko';
                        else if (widthRatio < 0.10) status.faceWarning = 'Príliš ďaleko';
                    }
                }
            } catch (e) {
                // ignore detection errors
            }
        }

        this.onStatusChange(status);
    }

    checkBrightness() {
        const pixels = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
        let sum = 0;
        let clipped = 0;
        const step = 16; // sample every 4th pixel (RGBA = 4 bytes × 4 skip)
        const count = Math.floor(pixels.length / step);

        for (let i = 0; i < pixels.length; i += step) {
            const lum = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
            sum += lum;
            if (lum > 250) clipped++;
        }

        const avg = sum / count;
        const clippedRatio = clipped / count;

        // Store for contrast check
        this._lastLumValues = { sum, count, pixels, step };

        if (avg < 60) return 'dark';
        if (avg > 220 || clippedRatio > 0.3) return 'bright';
        return 'ok';
    }

    checkContrast() {
        const pixels = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
        const step = 16;
        const count = Math.floor(pixels.length / step);
        let sum = 0;
        let sumSq = 0;

        for (let i = 0; i < pixels.length; i += step) {
            const lum = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
            sum += lum;
            sumSq += lum * lum;
        }

        const mean = sum / count;
        const stddev = Math.sqrt(sumSq / count - mean * mean);
        return stddev >= 20; // false = low contrast
    }
}
