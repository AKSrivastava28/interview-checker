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
        
        // Silence detection trackers
        this.isAnswering = false;
        this.hasSpokenInWindow = false;
        this.lastSpeechTimestamp = 0;
        this.silenceCheckInterval = null;
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
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480, facingMode: "user" },
                audio: false 
            });
            
            const previewVideo = document.getElementById("preview-video");
            const overlayCanvas = document.getElementById("camera-overlay");
            const gazeBadge = document.getElementById("gaze-badge");
            const gazeText = document.getElementById("gaze-status-text");

            if (previewVideo) {
                previewVideo.srcObject = this.stream;
                await previewVideo.play().catch(e => console.warn("previewVideo autoplay:", e));
            }

            if (overlayCanvas) {
                overlayCanvas.width = 640;
                overlayCanvas.height = 480;
            }
            const ctx = overlayCanvas ? overlayCanvas.getContext("2d") : null;

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
                    if (ctx && overlayCanvas) {
                        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                    }

                    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                        const landmarks = results.multiFaceLandmarks[0];

                        // 1. Compute Face Bounding Box
                        let minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
                        for (let i = 0; i < landmarks.length; i++) {
                            const p = landmarks[i];
                            if (p.x < minX) minX = p.x;
                            if (p.x > maxX) maxX = p.x;
                            if (p.y < minY) minY = p.y;
                            if (p.y > maxY) maxY = p.y;
                        }

                        // 2. Physical Head & Gaze Metrics
                        // 1: Nose tip, 168: Nose bridge, 10: Forehead, 152: Chin
                        // 234: Right cheek, 454: Left cheek, 468: Left iris, 473: Right iris
                        const nose = landmarks[1];
                        const noseBridge = landmarks[168];
                        const forehead = landmarks[10];
                        const chin = landmarks[152];
                        const cheekR = landmarks[234];
                        const cheekL = landmarks[454];

                        const faceW = Math.abs(cheekL.x - cheekR.x) || 0.1;
                        const midCheekX = (cheekR.x + cheekL.x) / 2;

                        // Head Yaw: horizontal rotation (normal: -0.12 to +0.12)
                        const headYaw = (nose.x - midCheekX) / faceW;

                        // Head Pitch: forehead-to-nose vs nose-to-chin ratio (normal: 0.75 - 1.55)
                        const dForehead = Math.abs(nose.y - forehead.y);
                        const dChin = Math.abs(chin.y - nose.y);
                        const pitchRatio = dForehead / (dChin + 0.001);

                        // Eye Drop: iris distance below nose bridge relative to face size
                        let eyeDrop = 0;
                        if (landmarks[468] && noseBridge) {
                            eyeDrop = (landmarks[468].y - noseBridge.y) / faceW;
                        }

                        // Gaze decisions
                        const isYawOk = Math.abs(headYaw) < 0.16;
                        const isPitchOk = pitchRatio >= 0.70 && pitchRatio <= 1.65;
                        const isEyeDown = eyeDrop > 0.36;

                        const isFocusedOnScreen = isYawOk && isPitchOk && !isEyeDown;

                        // 3. Draw Proctoring Overlay on Canvas
                        if (ctx && overlayCanvas) {
                            const cw = overlayCanvas.width;
                            const ch = overlayCanvas.height;
                            const pad = 16;
                            const boxX = Math.max(0, minX * cw - pad);
                            const boxY = Math.max(0, minY * ch - pad);
                            const boxW = Math.min(cw - boxX, (maxX - minX) * cw + pad * 2);
                            const boxH = Math.min(ch - boxY, (maxY - minY) * ch + pad * 2);

                            const color = isFocusedOnScreen ? '#10b981' : '#f43f5e';
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 3;

                            // Draw high-tech corner brackets around face
                            const cornerLen = Math.min(25, boxW / 4, boxH / 4);
                            ctx.beginPath();
                            // Top-left
                            ctx.moveTo(boxX, boxY + cornerLen);
                            ctx.lineTo(boxX, boxY);
                            ctx.lineTo(boxX + cornerLen, boxY);
                            // Top-right
                            ctx.moveTo(boxX + boxW - cornerLen, boxY);
                            ctx.lineTo(boxX + boxW, boxY);
                            ctx.lineTo(boxX + boxW, boxY + cornerLen);
                            // Bottom-left
                            ctx.moveTo(boxX, boxY + boxH - cornerLen);
                            ctx.lineTo(boxX, boxY + boxH);
                            ctx.lineTo(boxX + cornerLen, boxY + boxH);
                            // Bottom-right
                            ctx.moveTo(boxX + boxW - cornerLen, boxY + boxH);
                            ctx.lineTo(boxX + boxW, boxY + boxH);
                            ctx.lineTo(boxX + boxW, boxY + cornerLen);
                            ctx.stroke();

                            // Draw pupil tracking markers
                            ctx.fillStyle = color;
                            if (landmarks[468]) {
                                ctx.beginPath();
                                ctx.arc(landmarks[468].x * cw, landmarks[468].y * ch, 4, 0, Math.PI * 2);
                                ctx.fill();
                            }
                            if (landmarks[473]) {
                                ctx.beginPath();
                                ctx.arc(landmarks[473].x * cw, landmarks[473].y * ch, 4, 0, Math.PI * 2);
                                ctx.fill();
                            }
                        }

                        // 4. Update UI Status Badge & Coordinates for Backend
                        if (isFocusedOnScreen) {
                            if (gazeBadge) gazeBadge.className = "gaze-status-badge looking-screen";
                            if (gazeText) gazeText.textContent = "🟢 Screen Focused";
                            this.latestGazeCoords = { x: 0.50, y: 0.50 };
                        } else if (pitchRatio > 1.65 || isEyeDown) {
                            if (gazeBadge) gazeBadge.className = "gaze-status-badge looking-away";
                            if (gazeText) gazeText.textContent = "🔴 Gaze Alert: Phone / Lap";
                            this.latestGazeCoords = { x: 0.50, y: 0.95 };
                        } else {
                            if (gazeBadge) gazeBadge.className = "gaze-status-badge looking-away";
                            if (gazeText) gazeText.textContent = "🔴 Gaze Alert: Side Display";
                            const sideX = headYaw > 0 ? 0.95 : 0.05;
                            this.latestGazeCoords = { x: sideX, y: 0.50 };
                        }

                    } else {
                        // No face detected in frame
                        if (gazeBadge) gazeBadge.className = "gaze-status-badge no-face";
                        if (gazeText) gazeText.textContent = "⚠️ No Face Detected";
                        this.latestGazeCoords = { x: 0.0, y: 0.0 };
                    }
                });

                // Frame processing with MediaPipe Camera or smooth animation loop
                if (window.Camera && previewVideo) {
                    const camera = new window.Camera(previewVideo, {
                        onFrame: async () => {
                            if (previewVideo && !previewVideo.paused && !previewVideo.ended) {
                                await this.faceMesh.send({ image: previewVideo });
                            }
                        },
                        width: 640,
                        height: 480
                    });
                    camera.start();
                    console.log("[Candidate Capture] MediaPipe Camera controller active.");
                } else if (previewVideo) {
                    const processFrame = async () => {
                        if (previewVideo && !previewVideo.paused && !previewVideo.ended) {
                            await this.faceMesh.send({ image: previewVideo });
                        }
                        requestAnimationFrame(processFrame);
                    };
                    processFrame();
                }

                // Stream gaze samples ~5x per second during answering window
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

                console.log("[Candidate Capture] FaceMesh + Live Proctoring overlay initialized.");
            } else {
                console.warn("[Candidate Capture] MediaPipe FaceMesh library not found. Running fallback.");
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
