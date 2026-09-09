class CandidateCapture {
    constructor(wsClient, voiceEngine, sessionId = "") {
        this.wsClient = wsClient;
        this.voiceEngine = voiceEngine;
        this.sessionId = sessionId || wsClient?.sessionId || wsClient?.roomId || "";
        this.audioStream = null;
        this.consentGiven = false;
        
        this.faceMesh = null;
        this.speechRecognizer = null;
        this.videoElem = null;
        this.stream = null;
        this.gazeSampleInterval = null;
        
        this.latestGazeCoords = { x: 0.5, y: 0.5, reading_detected: false, reading_type: "" };
        this.isReadingDetected = false;
        this.readingDetail = "";
        this.gazeHistory = [];
        this.isFaceInFrame = false;
        this.onFaceStatusChanged = null;
        
        // Silence detection trackers
        this.isInterviewActive = false;
        this.isAnswering = false;
        this.hasSpokenInWindow = false;
        this.lastSpeechTimestamp = 0;
        this.silenceCheckInterval = null;

        // Acoustic Prosody (F0 Pitch) Tracking
        this.audioContext = null;
        this.analyserNode = null;
        this.audioSource = null;
        this.pitchSamples = [];
        this.pitchSampleInterval = null;

        // Head pose dynamism / teleprompter freeze tracking
        this.headPoseSamples = [];
        this.eyebrowSamples = [];

        // Audio recording for Whisper STT
        this.mediaRecorder = null;
        this.audioChunks = [];
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

        // 1. Setup Camera & Mic for MediaPipe Gaze & Acoustic Prosody Tracking
        await this.setupMediaPipeGaze();

        // 2. Setup Web Speech Recognition
        this.setupSpeechRecognition();

        // 3. Setup Browser Window Event Listeners
        this.setupWindowListeners();

        // 4. Setup Acoustic Prosody Tracker
        this.setupAcousticTracker();

        // 5. Setup Audio MediaRecorder for Whisper STT
        this.setupMediaRecorder();
    }

    async setupMediaPipeGaze() {
        try {
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: "user" },
                    audio: true 
                });
            } catch (mediaErr) {
                console.warn("[Candidate Capture] getUserMedia with audio failed, falling back to video only:", mediaErr);
                this.stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: "user" },
                    audio: false 
                });
            }
            
            const previewVideo = document.getElementById("preview-video");
            const overlayCanvas = document.getElementById("camera-overlay");
            const gazeBadge = document.getElementById("gaze-badge");
            const gazeText = document.getElementById("gaze-status-text");

            if (previewVideo) {
                previewVideo.srcObject = this.stream;
                try {
                    await previewVideo.play();
                } catch (e) {
                    console.warn("[Candidate Capture] previewVideo play warning:", e);
                }
                let attempts = 0;
                while (previewVideo.videoWidth === 0 && attempts < 20) {
                    await new Promise(r => setTimeout(r, 50));
                    attempts++;
                }
            }

            if (overlayCanvas && previewVideo) {
                overlayCanvas.width = previewVideo.videoWidth || 640;
                overlayCanvas.height = previewVideo.videoHeight || 480;
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

                    const cameraBubble = document.getElementById("camera-bubble-container");

                    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                        this.isFaceInFrame = true;
                        if (cameraBubble) {
                            cameraBubble.classList.add("face-aligned");
                            cameraBubble.classList.remove("face-lost");
                        }

                        const landmarks = results.multiFaceLandmarks[0];
                        const cw = overlayCanvas ? overlayCanvas.width : 640;
                        const ch = overlayCanvas ? overlayCanvas.height : 480;

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

                        // 1. Head Yaw: horizontal rotation (normal screen: -0.06 to +0.06; iPad/side screen: > 0.085 or < -0.085)
                        const headYaw = (nose.x - midCheekX) / faceW;

                        // 2. Head Pitch: forehead-to-nose vs nose-to-chin (normal: 0.80 - 1.25; nodding down at iPad/desk: > 1.30)
                        const dForehead = Math.abs(nose.y - forehead.y);
                        const dChin = Math.abs(chin.y - nose.y);
                        const pitchRatio = dForehead / (dChin + 0.001);

                        // 3. Head Roll (lateral tilt): difference in height between eye outer corners
                        const eyeOuterR = landmarks[33];
                        const eyeOuterL = landmarks[263];
                        const headRoll = (eyeOuterL.y - eyeOuterR.y) / faceW;

                        // 4. Eye Drop: iris distance below nose bridge (normal: 0.18 - 0.28; looking down at desk: > 0.30)
                        let eyeDrop = 0;
                        if (landmarks[468] && noseBridge) {
                            eyeDrop = (landmarks[468].y - noseBridge.y) / faceW;
                        }

                        // Eye Pupil Offset within Socket (Micro-Gaze Tracking)
                        // Right Eye (MediaPipe coordinates: 33 outer, 133 inner, 159 top, 145 bottom, 468 iris center)
                        const eye1W = Math.abs(landmarks[133].x - landmarks[33].x) || 0.01;
                        const eye1H = Math.abs(landmarks[145].y - landmarks[159].y) || 0.01;
                        const eye1CenterX = (landmarks[33].x + landmarks[133].x) / 2;
                        const eye1CenterY = (landmarks[159].y + landmarks[145].y) / 2;
                        const disp1X = landmarks[468] ? (landmarks[468].x - eye1CenterX) / eye1W : 0;
                        const disp1Y = landmarks[468] ? (landmarks[468].y - eye1CenterY) / eye1H : 0;

                        // Left Eye (MediaPipe coordinates: 362 inner, 263 outer, 386 top, 374 bottom, 473 iris center)
                        const eye2W = Math.abs(landmarks[263].x - landmarks[362].x) || 0.01;
                        const eye2H = Math.abs(landmarks[374].y - landmarks[386].y) || 0.01;
                        const eye2CenterX = (landmarks[362].x + landmarks[263].x) / 2;
                        const eye2CenterY = (landmarks[386].y + landmarks[374].y) / 2;
                        const disp2X = landmarks[473] ? (landmarks[473].x - eye2CenterX) / eye2W : 0;
                        const disp2Y = landmarks[473] ? (landmarks[473].y - eye2CenterY) / eye2H : 0;

                        const pupilGazeX = (disp1X + disp2X) / 2;
                        const pupilGazeY = (disp1Y + disp2Y) / 2;

                        // 5. Combined Horizontal Gaze: head rotation + iris displacement within eye sockets
                        // When looking at an iPad on the left: head yaw is negative and irises shift left
                        const totalHorizGaze = headYaw + (pupilGazeX * 0.25);

                        // Calibrated physical bounds:
                        const isYawOk = Math.abs(headYaw) <= 0.085;
                        const isPitchOk = pitchRatio >= 0.65 && pitchRatio <= 1.30;
                        const isEyeDown = eyeDrop > 0.30;
                        const isRollOk = Math.abs(headRoll) <= 0.075;
                        const isGazeCentered = Math.abs(totalHorizGaze) <= 0.095;
                        const isFocusedOnScreen = isYawOk && isPitchOk && !isEyeDown && isRollOk && isGazeCentered;

                        // Track head pose & eyebrow dynamism during answer window
                        if (this.isAnswering) {
                            this.headPoseSamples.push(headYaw);
                            const brow1Y = landmarks[105] ? landmarks[105].y : 0;
                            const eye1Y = landmarks[159] ? landmarks[159].y : 0;
                            this.eyebrowSamples.push(Math.abs(brow1Y - eye1Y));
                        }

                        // 3. Draw Proctoring Overlay on Canvas
                        if (ctx && overlayCanvas) {
                            const pad = 16;
                            const boxX = Math.max(0, minX * cw - pad);
                            const boxY = Math.max(0, minY * ch - pad);
                            const boxW = Math.min(cw - boxX, (maxX - minX) * cw + pad * 2);
                            const boxH = Math.min(ch - boxY, (maxY - minY) * ch + pad * 2);

                            const color = isFocusedOnScreen ? '#10b981' : '#f43f5e';

                            ctx.strokeStyle = color;
                            ctx.lineWidth = 3;

                            // Draw corner brackets around face
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

                            // Draw pupil & nose tracking markers
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
                            ctx.beginPath();
                            ctx.arc(nose.x * cw, nose.y * ch, 3, 0, Math.PI * 2);
                            ctx.fill();
                        }

                        // 4. Update UI Status Badge & Coordinates for Backend
                        if (!isFocusedOnScreen) {
                            let alertText = "🔴 Gaze Alert: Looking Away";
                            let sideX = 0.50;
                            let sideY = 0.50;

                            if (pitchRatio > 1.30 || isEyeDown) {
                                alertText = "🔴 Gaze Alert: Looking Down / Desk Device (iPad/Phone)";
                                sideY = 0.95;
                            } else if (!isRollOk) {
                                const tiltDir = headRoll > 0 ? "Left" : "Right";
                                alertText = `🔴 Gaze Alert: Head Tilted Sideways (${tiltDir})`;
                                sideX = headRoll > 0 ? 0.05 : 0.95;
                            } else if (!isYawOk || !isGazeCentered) {
                                const turnDir = totalHorizGaze < 0 ? "Left (iPad/Side Device)" : "Right (2nd Screen)";
                                alertText = `🔴 Gaze Alert: Looking ${turnDir}`;
                                sideX = totalHorizGaze < 0 ? 0.05 : 0.95;
                            }

                            if (gazeBadge) gazeBadge.className = "gaze-status-badge looking-away";
                            if (gazeText) gazeText.textContent = alertText;
                            this.latestGazeCoords = { x: sideX, y: sideY, reading_detected: false, reading_type: "" };

                            if (cameraBubble) {
                                cameraBubble.className = "camera-bubble face-aligned face-lost";
                            }
                        } else {
                            // Looking at the screen, at the interviewer, or at the camera: 100% CLEAN
                            if (gazeBadge) gazeBadge.className = "gaze-status-badge looking-screen";
                            if (gazeText) gazeText.textContent = "🟢 Face Detected: Screen Focused";
                            this.latestGazeCoords = { x: 0.50, y: 0.50, reading_detected: false, reading_type: "" };
                            if (cameraBubble) {
                                cameraBubble.className = "camera-bubble face-aligned";
                            }
                        }

                        if (this.onFaceStatusChanged) {
                            this.onFaceStatusChanged(true, isFocusedOnScreen);
                        }

                    } else {
                        // No face detected in frame (e.g. camera pointed at ceiling!)
                        this.isFaceInFrame = false;
                        if (cameraBubble) {
                            cameraBubble.classList.remove("face-aligned");
                            cameraBubble.classList.add("face-lost");
                        }
                        if (gazeBadge) {
                            gazeBadge.className = "gaze-status-badge no-face";
                            if (gazeText) gazeText.textContent = "❌ No Face in Camera Frame";
                        }
                        this.latestGazeCoords = { x: 0.0, y: 0.0, reading_detected: false, reading_type: "" };

                        if (this.onFaceStatusChanged) {
                            this.onFaceStatusChanged(false, false);
                        }
                    }
                });

                // 5. Native Frame Processing Loop (No duplicate camera streams!)
                let isProcessingFrame = false;
                const processFrame = async () => {
                    if (previewVideo && previewVideo.readyState >= 2 && !previewVideo.paused && !previewVideo.ended) {
                        if (!isProcessingFrame) {
                            isProcessingFrame = true;
                            try {
                                await this.faceMesh.send({ image: previewVideo });
                            } catch (err) {
                                console.warn("[Candidate Capture] FaceMesh frame send warning:", err);
                            } finally {
                                isProcessingFrame = false;
                            }
                        }
                    }
                    if ('requestVideoFrameCallback' in previewVideo) {
                        previewVideo.requestVideoFrameCallback(processFrame);
                    } else {
                        requestAnimationFrame(processFrame);
                    }
                };

                if ('requestVideoFrameCallback' in previewVideo) {
                    previewVideo.requestVideoFrameCallback(processFrame);
                } else {
                    requestAnimationFrame(processFrame);
                }

                // Stream gaze samples ~5x per second during answering window
                this.gazeSampleInterval = setInterval(() => {
                    if (this.isAnswering) {
                        this.wsClient.send({
                            type: "gaze",
                            x: this.latestGazeCoords.x,
                            y: this.latestGazeCoords.y,
                            reading_detected: Boolean(this.latestGazeCoords.reading_detected),
                            reading_type: this.latestGazeCoords.reading_type || "",
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
                const wordCount = fullText.split(/\s+/).filter(Boolean).length;
                
                this.wsClient.send({
                    type: "transcript",
                    text: fullText,
                    is_final: event.results[event.results.length - 1].isFinal,
                    word_count: wordCount,
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

    requestFullscreen() {
        const elem = document.documentElement;
        if (elem.requestFullscreen) {
            elem.requestFullscreen().catch(err => console.warn("[Fullscreen] Failed to enter fullscreen:", err));
        } else if (elem.webkitRequestFullscreen) {
            elem.webkitRequestFullscreen();
        }
    }

    setupWindowListeners() {
        window.addEventListener('blur', () => {
            this.wsClient.send({
                type: "event",
                name: "tab_blur",
                ts: Date.now() / 1000.0
            });
            const blurAlert = document.getElementById("focus-alert-banner");
            if (blurAlert && this.isInterviewActive) {
                blurAlert.style.display = "flex";
            }
        });

        window.addEventListener('focus', () => {
            this.wsClient.send({
                type: "event",
                name: "tab_focus",
                ts: Date.now() / 1000.0
            });
            const blurAlert = document.getElementById("focus-alert-banner");
            if (blurAlert) {
                setTimeout(() => { blurAlert.style.display = "none"; }, 1500);
            }
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

        document.addEventListener('fullscreenchange', () => {
            const isFs = Boolean(document.fullscreenElement);
            this.wsClient.send({
                type: "event",
                name: isFs ? "fullscreen_enter" : "fullscreen_exit",
                ts: Date.now() / 1000.0
            });

            const fsModal = document.getElementById("fullscreen-modal");
            if (fsModal) {
                if (!isFs && this.isInterviewActive) {
                    fsModal.style.display = "flex";
                } else {
                    fsModal.style.display = "none";
                }
            }
        });

        window.addEventListener('resize', () => {
            this.wsClient.send({
                type: "event",
                name: "window_resize",
                ts: Date.now() / 1000.0
            });
        });

        console.log("[Candidate Capture] Window & Fullscreen behavior listeners registered.");
    }

    setupAcousticTracker() {
        try {
            const audioTracks = this.stream ? this.stream.getAudioTracks() : [];
            if (audioTracks.length === 0) {
                console.warn("[Candidate Capture] No audio track available for acoustic prosody tracking.");
                return;
            }
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) return;

            this.audioContext = new AudioContextClass();
            const audioStream = new MediaStream([audioTracks[0]]);
            this.audioSource = this.audioContext.createMediaStreamSource(audioStream);
            this.analyserNode = this.audioContext.createAnalyser();
            this.analyserNode.fftSize = 2048;
            this.audioSource.connect(this.analyserNode);
            console.log("[Candidate Capture] Acoustic pitch analyzer connected successfully.");
        } catch (e) {
            console.warn("[Candidate Capture] setupAcousticTracker error:", e);
        }
    }

    setupMediaRecorder() {
        try {
            const audioTracks = this.stream ? this.stream.getAudioTracks() : [];
            if (audioTracks.length === 0) return;
            this.audioStream = new MediaStream([audioTracks[0]]);
            console.log("[Candidate Capture] audioStream configured for Whisper STT recording.");
        } catch (e) {
            console.warn("[Candidate Capture] setupMediaRecorder warning:", e);
        }
    }

    detectFundamentalFrequency(buffer, sampleRate) {
        const bufferSize = buffer.length;
        
        // 1. RMS Energy check to ignore silence or background hum
        let sumSquares = 0;
        for (let i = 0; i < bufferSize; i++) {
            sumSquares += buffer[i] * buffer[i];
        }
        const rms = Math.sqrt(sumSquares / bufferSize);
        if (rms < 0.015) {
            return null;
        }

        // Voice fundamental frequency search range: 80 Hz to 450 Hz
        const minPeriod = Math.floor(sampleRate / 450);
        const maxPeriod = Math.floor(sampleRate / 80);

        let bestCorrelation = -1;
        let bestPeriod = -1;

        let energy0 = 0;
        for (let i = 0; i < bufferSize - maxPeriod; i++) {
            energy0 += buffer[i] * buffer[i];
        }
        if (energy0 < 1e-4) return null;

        for (let lag = minPeriod; lag <= maxPeriod; lag++) {
            let crossCorr = 0;
            let energyLag = 0;
            const len = bufferSize - lag;

            for (let i = 0; i < len; i++) {
                crossCorr += buffer[i] * buffer[i + lag];
                energyLag += buffer[i + lag] * buffer[i + lag];
            }

            const denom = Math.sqrt(energy0 * energyLag);
            if (denom > 1e-5) {
                const normCorr = crossCorr / denom;
                if (normCorr > bestCorrelation) {
                    bestCorrelation = normCorr;
                    bestPeriod = lag;
                }
            }
        }

        // Check if periodic signal peak meets confidence (tuned to 0.65 to capture dynamic speech transitions)
        if (bestCorrelation > 0.65 && bestPeriod > 0) {
            return sampleRate / bestPeriod;
        }
        return null;
    }

    startPitchTracking() {
        if (this.pitchSampleInterval) clearInterval(this.pitchSampleInterval);
        this.pitchSamples = [];
        if (!this.analyserNode || !this.audioContext) return;

        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        const buffer = new Float32Array(this.analyserNode.fftSize);

        this.pitchSampleInterval = setInterval(() => {
            if (!this.isAnswering) return;
            this.analyserNode.getFloatTimeDomainData(buffer);
            const pitch = this.detectFundamentalFrequency(buffer, this.audioContext.sampleRate);
            if (pitch !== null && pitch >= 75 && pitch <= 450) {
                this.pitchSamples.push(pitch);
            }
        }, 100);
    }

    stopPitchTracking() {
        if (this.pitchSampleInterval) {
            clearInterval(this.pitchSampleInterval);
            this.pitchSampleInterval = null;
        }

        const samples = this.pitchSamples || [];
        if (samples.length < 5) {
            return {
                pitch_std: 0.0,
                mean_pitch: 0.0,
                pitch_samples_count: samples.length,
                head_motion_std: 0.015,
                is_rigid_head: false
            };
        }

        // Outlier Rejection: Gentle 2.5% trim to filter mic clicks while preserving 95% of human pitch inflection
        const sorted = [...samples].sort((a, b) => a - b);
        const trimCount = Math.floor(sorted.length * 0.025);
        const coreSamples = trimCount > 0 ? sorted.slice(trimCount, sorted.length - trimCount) : sorted;

        const mean = coreSamples.reduce((a, b) => a + b, 0) / coreSamples.length;
        const variance = coreSamples.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / coreSamples.length;
        const stdDev = Math.sqrt(variance);

        // Head pose stillness / teleprompter freeze calculation
        let headStd = 0.015;
        let isRigid = false;
        if (this.headPoseSamples && this.headPoseSamples.length >= 25) {
            const hMean = this.headPoseSamples.reduce((a, b) => a + b, 0) / this.headPoseSamples.length;
            const hVar = this.headPoseSamples.reduce((acc, v) => acc + Math.pow(v - hMean, 2), 0) / this.headPoseSamples.length;
            headStd = Math.sqrt(hVar);
            // Conversational speaking has head yaw std dev >= 0.008; reading freeze has std dev < 0.0045
            if (headStd < 0.0045) {
                isRigid = true;
                console.log("[Candidate Capture] Rigid Teleprompter Freeze detected. Head motion std dev:", headStd);
            }
        }

        return {
            pitch_std: Math.round(stdDev * 10) / 10,
            mean_pitch: Math.round(mean * 10) / 10,
            pitch_samples_count: coreSamples.length,
            head_motion_std: Math.round(headStd * 10000) / 10000,
            is_rigid_head: isRigid
        };
    }

    setupSilenceDetection() {
        // Run check every 500ms; generous 6.0s thinking window so candidates aren't cut off
        this.silenceCheckInterval = setInterval(() => {
            if (this.isAnswering && this.hasSpokenInWindow) {
                const silenceDuration = Date.now() - this.lastSpeechTimestamp;
                if (silenceDuration > 6000) { // 6.0 seconds of silence
                    console.log("[Silence Detector] Silence exceeded 6.0s. Completing answer.");
                    this.triggerDoneAnswering();
                }
            }
        }, 500);
    }

    startAnswerWindow() {
        this.isAnswering = true;
        this.hasSpokenInWindow = false;
        this.lastSpeechTimestamp = Date.now();
        this.headPoseSamples = [];
        this.eyebrowSamples = [];
        
        // Start fresh audio recording for Whisper STT
        this.audioChunks = [];
        if (this.audioStream) {
            try {
                let mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/webm';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = '';
                    }
                }
                const options = mimeType ? { mimeType } : {};
                this.mediaRecorder = new MediaRecorder(this.audioStream, options);
                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data && event.data.size > 0) {
                        this.audioChunks.push(event.data);
                    }
                };
                this.mediaRecorder.start(250);
                console.log("[Candidate Capture] Fresh MediaRecorder started for answer window. MimeType:", mimeType || "default");
            } catch (e) {
                console.warn("[Candidate Capture] MediaRecorder start error:", e);
            }
        }

        // Clear text field
        const transcriptBox = document.getElementById("live-transcript");
        if (transcriptBox) transcriptBox.textContent = "Listening to your response...";

        this.startSpeechRecognition();
        this.startPitchTracking();
    }

    async triggerDoneAnswering() {
        if (!this.isAnswering) return;
        this.isAnswering = false;

        this.stopSpeechRecognition();
        const acousticFeatures = this.stopPitchTracking();

        const transcriptBox = document.getElementById("live-transcript");
        if (transcriptBox) transcriptBox.textContent = "Transcribing response with Groq Whisper...";

        // Stop MediaRecorder and request Groq Whisper transcription
        let whisperTranscript = "";
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                await new Promise((resolve) => {
                    const timer = setTimeout(resolve, 800);
                    this.mediaRecorder.onstop = () => {
                        clearTimeout(timer);
                        resolve();
                    };
                    try {
                        this.mediaRecorder.stop();
                    } catch (e) {
                        clearTimeout(timer);
                        resolve();
                    }
                });

                if (this.audioChunks && this.audioChunks.length > 0) {
                    const audioBlob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType || 'audio/webm' });
                    console.log(`[Candidate Capture] Captured audio size: ${audioBlob.size} bytes across ${this.audioChunks.length} chunks.`);

                    if (audioBlob.size > 500) {
                        let targetSessionId = this.sessionId || this.wsClient?.sessionId || this.wsClient?.roomId || "";
                        if (!targetSessionId || targetSessionId === 'undefined') {
                            targetSessionId = new URLSearchParams(window.location.search).get('session_id') || "";
                        }

                        const formData = new FormData();
                        formData.append("file", audioBlob, "answer.webm");

                        console.log(`[Candidate Capture] Uploading audio to /api/transcribe/${targetSessionId}...`);
                        const controller = new AbortController();
                        const fetchTimer = setTimeout(() => controller.abort(), 5000);

                        try {
                            const resp = await fetch(`/api/transcribe/${targetSessionId}`, {
                                method: "POST",
                                body: formData,
                                signal: controller.signal
                            });
                            clearTimeout(fetchTimer);

                            if (resp.ok) {
                                const resData = await resp.json();
                                whisperTranscript = resData.transcript || "";
                                console.log("[Candidate Capture] Groq Whisper transcript success:", whisperTranscript);
                                if (transcriptBox && whisperTranscript) {
                                    transcriptBox.textContent = `"${whisperTranscript}"`;
                                }
                            } else {
                                console.warn(`[Candidate Capture] /api/transcribe returned HTTP ${resp.status}`);
                            }
                        } catch (fErr) {
                            clearTimeout(fetchTimer);
                            console.warn("[Candidate Capture] Whisper transcription fetch timed out or failed:", fErr);
                        }
                    } else {
                        console.warn(`[Candidate Capture] Audio too small (${audioBlob.size} bytes) for Whisper.`);
                    }
                }
            } catch (err) {
                console.warn("[Candidate Capture] Whisper transcription error, falling back to browser STT:", err);
            }
        }

        console.log("[Candidate Capture] Acoustic & Posture summary:", acousticFeatures);

        this.wsClient.send({
            type: "done_answering",
            ts: Date.now() / 1000.0,
            acoustic_features: acousticFeatures,
            whisper_transcript: whisperTranscript
        });

        if (transcriptBox && !whisperTranscript) {
            transcriptBox.textContent = "Analyzing response integrity...";
        }
    }
}

window.CandidateCapture = CandidateCapture;
