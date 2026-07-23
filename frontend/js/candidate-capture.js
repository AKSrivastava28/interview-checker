class CandidateCapture {
    constructor(wsClient, voiceEngine) {
        this.wsClient = wsClient;
        this.voiceEngine = voiceEngine;
        this.consentGiven = false;
        
        this.faceMesh = null;
        this.speechRecognizer = null;
        this.videoElem = null;
        this.stream = null;
        this.gazeSampleInterval = null;
        
        this.latestGazeCoords = { x: 0.5, y: 0.5 };
        
        // Gaze Calibration properties
        this.calibrationSamples = [];
        this.calibratedX = 0.5;
        this.calibratedY = 0.5;
        this.isCalibrated = false;
        this.isCalibrating = false;
        this.onCalibrationProgress = null;
        this.onCalibrationComplete = null;
        
        // Silence detection trackers
        this.isAnswering = false;
        this.hasSpokenInWindow = false;
        this.lastSpeechTimestamp = 0;
        this.silenceCheckInterval = null;
    }

    startCalibration(onProgress, onComplete) {
        this.calibrationSamples = [];
        this.calibratedX = 0.5;
        this.calibratedY = 0.5;
        this.isCalibrated = false;
        this.onCalibrationProgress = onProgress;
        this.onCalibrationComplete = onComplete;
        this.isCalibrating = true;
        console.log("[Gaze Calibration] Started manual calibration baseline gathering...");
    }

    resetCalibration() {
        this.calibrationSamples = [];
        this.isCalibrated = false;
        console.log("[Gaze Calibration] Baseline reset. Recalibrating...");
    }

    initConsentModal(onConsentCallback, onCaptureInitialized) {
        const modalHtml = `
            <div id="consent-modal" class="modal-overlay">
                <div class="modal-content">
                    <div class="modal-icon">🔒</div>
                    <h3>Interview Integrity Notice</h3>
                    <p>
                        This session analyzes candidate speech timing, gaze direction patterns, and window focus state 
                        to support interview integrity review. 
                        <strong>Camera and microphone data are processed in real-time in your browser and are NEVER stored as raw video files.</strong>
                    </p>
                    <button id="btn-consent" class="btn btn-primary" style="width:100%;">
                        I Consent & Join Room
                    </button>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        document.getElementById('btn-consent').addEventListener('click', async () => {
            document.getElementById('consent-modal').remove();
            this.consentGiven = true;
            if (onConsentCallback) await onConsentCallback();
            await this.startCapture();
            if (onCaptureInitialized) {
                await onCaptureInitialized();
            }
        });
    }

    async startCapture() {
        console.log("[Candidate Capture] Starting passive capture streams...");

        // 1. Setup Camera for MediaPipe Gaze Tracking
        await this.setupMediaPipeGaze();

        // 2. Setup Web Speech Recognition
        this.setupSpeechRecognition();

        // 3. Setup Browser Window Event Listeners
        this.setupWindowListeners();
    }

    async setupMediaPipeGaze() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            
            // Create hidden video element to feed MediaPipe
            this.videoElem = document.createElement("video");
            this.videoElem.srcObject = this.stream;
            this.videoElem.autoplay = true;
            this.videoElem.playsInline = true;
            this.videoElem.style.display = "none";
            document.body.appendChild(this.videoElem);

            // Also stream to video bubble on UI
            const previewVideo = document.getElementById("preview-video");
            if (previewVideo) {
                previewVideo.srcObject = this.stream;
            }

            await new Promise((resolve) => {
                this.videoElem.onloadedmetadata = () => resolve();
            });

            if (window.FaceMesh) {
                this.faceMesh = new window.FaceMesh({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
                });

                this.faceMesh.setOptions({
                    maxNumFaces: 1,
                    refineLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });

                this.faceMesh.onResults((results) => {
                    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                        const landmarks = results.multiFaceLandmarks[0];
                        const leftCorner = landmarks[362];
                        const rightCorner = landmarks[263];
                        const topBoundary = landmarks[386];
                        const bottomBoundary = landmarks[374];
                        const iris = landmarks[468];

                        if (leftCorner && rightCorner && topBoundary && bottomBoundary && iris) {
                            const eyeWidth = Math.abs(rightCorner.x - leftCorner.x);
                            if (eyeWidth > 0) {
                                const centerX = (leftCorner.x + rightCorner.x) / 2;
                                const centerY = (topBoundary.y + bottomBoundary.y) / 2;

                                // Normalize both X and Y offsets strictly by the stable eye width (ignores blink height noise)
                                const relX = 0.5 + (iris.x - centerX) / eyeWidth;
                                const relY = 0.5 + (iris.y - centerY) / eyeWidth;

                                if (this.isCalibrating) {
                                    this.calibrationSamples.push({ x: relX, y: relY });
                                    if (this.onCalibrationProgress) {
                                        this.onCalibrationProgress(this.calibrationSamples.length);
                                    }
                                    if (this.calibrationSamples.length >= 60) {
                                        const sumX = this.calibrationSamples.reduce((sum, s) => sum + s.x, 0);
                                        const sumY = this.calibrationSamples.reduce((sum, s) => sum + s.y, 0);
                                        this.calibratedX = sumX / this.calibrationSamples.length;
                                        this.calibratedY = sumY / this.calibrationSamples.length;
                                        this.isCalibrating = false;
                                        this.isCalibrated = true;
                                        if (this.onCalibrationComplete) {
                                            this.onCalibrationComplete();
                                        }
                                    }
                                }

                                // Apply calibration offsets to map the straight gaze exactly to 0.5
                                const mappedX = 0.5 + (relX - this.calibratedX);
                                const mappedY = 0.5 + (relY - this.calibratedY);

                                this.latestGazeCoords = {
                                    x: Math.round(Math.max(0.0, Math.min(1.0, mappedX)) * 1000) / 1000,
                                    y: Math.round(Math.max(0.0, Math.min(1.0, mappedY)) * 1000) / 1000
                                };
                                return;
                            }
                        }
                        
                        // Fallback to absolute nose landmark if eye sockets cannot be resolved
                        const gazePoint = landmarks[1] || { x: 0.5, y: 0.5 };
                        this.latestGazeCoords = {
                            x: Math.round(gazePoint.x * 1000) / 1000,
                            y: Math.round(gazePoint.y * 1000) / 1000
                        };
                    }
                });

                // Frame processing loop
                const processFrame = async () => {
                    if (this.videoElem && !this.videoElem.paused && !this.videoElem.ended) {
                        await this.faceMesh.send({ image: this.videoElem });
                    }
                    requestAnimationFrame(processFrame);
                };
                processFrame();

                // Send gaze samples ~5x per second
                this.gazeSampleInterval = setInterval(() => {
                    if (this.isAnswering) {
                        this.wsClient.send({
                            type: "gaze",
                            x: this.latestGazeCoords.x,
                            y: this.latestGazeCoords.y,
                            ts: Date.now() / 1000.0
                        });
                    }
                }, 200);

                console.log("[Candidate Capture] MediaPipe FaceMesh active at ~5Hz.");
            } else {
                console.warn("[Candidate Capture] MediaPipe FaceMesh library not found. Running gaze fallback.");
            }

        } catch (err) {
            console.error("[Candidate Capture] Failed to initialize camera for MediaPipe:", err);
        }
    }

    setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn("[Candidate Capture] SpeechRecognition API not supported in this browser.");
            return;
        }

        this.speechRecognizer = new SpeechRecognition();
        this.speechRecognizer.continuous = true;
        this.speechRecognizer.interimResults = true;
        this.speechRecognizer.lang = 'en-US';

        this.speechRecognizer.onresult = (event) => {
            let fullText = "";
            for (let i = 0; i < event.results.length; ++i) {
                fullText += event.results[i][0].transcript + " ";
            }
            fullText = fullText.trim();

            // Update transcript UI
            const transcriptBox = document.getElementById("live-transcript");
            if (transcriptBox) {
                transcriptBox.textContent = fullText;
            }

            // If candidate is answering, stream chunks & update silence trackers
            if (this.isAnswering) {
                this.hasSpokenInWindow = true;
                this.lastSpeechTimestamp = Date.now();
                
                this.wsClient.send({
                    type: "transcript",
                    text: fullText,
                    is_final: event.results[event.results.length - 1].isFinal,
                    ts: Date.now() / 1000.0
                });
            }
        };

        this.speechRecognizer.onerror = (event) => {
            console.warn("[Candidate Capture] SpeechRecognition error:", event.error);
        };

        this.speechRecognizer.onend = () => {
            // Only restart if candidate is actively answering to prevent background noise buildup
            if (this.consentGiven && this.isAnswering) {
                try { this.speechRecognizer.start(); } catch(e){}
            }
        };
    }

    startSpeechRecognition() {
        if (this.speechRecognizer) {
            try {
                this.speechRecognizer.start();
                console.log("[Candidate Capture] SpeechRecognition started for active answer window.");
            } catch (e) {
                console.warn("[Candidate Capture] SpeechRecognition start failed or already active:", e);
            }
        }
    }

    stopSpeechRecognition() {
        if (this.speechRecognizer) {
            try {
                this.speechRecognizer.stop();
                console.log("[Candidate Capture] SpeechRecognition stopped.");
            } catch (e) {
                console.warn("[Candidate Capture] SpeechRecognition stop failed:", e);
            }
        }
    }

    setupWindowListeners() {
        window.addEventListener('blur', () => {
            this.wsClient.send({
                type: "event",
                name: "tab_blur",
                ts: Date.now() / 1000.0
            });
        });

        window.addEventListener('focus', () => {
            this.wsClient.send({
                type: "event",
                name: "tab_focus",
                ts: Date.now() / 1000.0
            });
        });

        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.wsClient.send({
                    type: "event",
                    name: "visibility_hidden",
                    ts: Date.now() / 1000.0
                });
            }
        });

        window.addEventListener('resize', () => {
            this.wsClient.send({
                type: "event",
                name: "window_resize",
                ts: Date.now() / 1000.0
            });
        });

        console.log("[Candidate Capture] Window behavior listeners registered.");
    }

    setupSilenceDetection() {
        // Run check every 500ms
        this.silenceCheckInterval = setInterval(() => {
            if (this.isAnswering && this.hasSpokenInWindow) {
                const silenceDuration = Date.now() - this.lastSpeechTimestamp;
                if (silenceDuration > 3000) { // 3.0 seconds of silence
                    console.log("[Silence Detector] Silence exceeded 3.0s. Autocomplete answer.");
                    this.triggerDoneAnswering();
                }
            }
        }, 500);
    }

    startAnswerWindow() {
        this.isAnswering = true;
        this.hasSpokenInWindow = false;
        this.lastSpeechTimestamp = Date.now();
        
        // Clear text field
        const transcriptBox = document.getElementById("live-transcript");
        if (transcriptBox) transcriptBox.textContent = "Listening to your response...";

        this.startSpeechRecognition();
    }

    triggerDoneAnswering() {
        if (!this.isAnswering) return;
        this.isAnswering = false;

        this.stopSpeechRecognition();

        this.wsClient.send({
            type: "done_answering",
            ts: Date.now() / 1000.0
        });

        const transcriptBox = document.getElementById("live-transcript");
        if (transcriptBox) transcriptBox.textContent = "Analyzing response integrity...";
    }
}

window.CandidateCapture = CandidateCapture;
