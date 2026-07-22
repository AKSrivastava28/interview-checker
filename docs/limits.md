# System Limitations & Safeguards Guide

## 1. Gaze Tracking Limitations
- **Environmental Dependencies**: MediaPipe FaceMesh performance depends on good lighting, front-facing camera positioning, and stable head positions. Eyeglass glare or dim environments can reduce landmark accuracy.
- **Physical Glances**: Legitimate glances away (e.g., looking down at a notebook, looking at keyboard, thinking) can be flagged as offscreen gaze.
- **Safeguard**: Offscreen gaze only increases risk when correlated with high speech pause and textbook transcript AI scores.

## 2. Speech recognition & Silence Detection Limitations
- **Accents & Background Noise**: Native Web `SpeechRecognition` is susceptible to background noise, regional accents, or microphone quality, which can cause minor transcript inaccuracies.
- **Natural Hesitations**: Articulate candidates may take long pauses to formulate thoughts legitimately, which might be flagged by the timing analyzer.
- **Safeguard**: The automatic silence detector (3.0s of silence) can be overridden by clicking the manual "Done Answering" button.

## 3. Transcript AI Scoring Limitations
- **Highly Rehearsed Candidates**: Candidates who speak very formally or have memorized answers may get high AI-likeness scores.
- **Safeguard**: Grok's evaluation prompt specifically values conversational elements (e.g., filler words like "um", "well", self-corrections) as human speech indicators.
- **Fallback**: If the Grok API is unconfigured, the system falls back to a structural grammar/marker heuristic that highlights formal structure indicators.

## 4. Fundamental Design Philosophy
The system **never automates decisions or rejections**. It serves strictly as a telemetry console for human reviewers to investigate potential signals of real-time overlay or assistant usage.
