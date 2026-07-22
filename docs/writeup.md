# Hackathon Submission Writeup — Catch the Invisible AI Cheater

## 1. Problem Statement
The proliferation of real-time AI interview assistants (e.g., Cluely, desktop screen overlays, second-monitor text readers) allows candidates to receive instant transcripts and answer prompts during remote technical interviews. Traditional invasive proctoring software requires desktop agent installations or heavy browser extensions, which are met with user resistance and fail to detect offscreen monitors.

## 2. Solution Overview
**IntegrityGuard AI** is a zero-install mock AI interviewer and live proctoring prototype designed to run entirely in-browser. By combining native web APIs and lightweight WebAssembly machine learning, it extracts multi-modal telemetry to evaluate the integrity of candidate responses in real time.

### Key Innovations
1. **Interactive AI Interviewer (TTS/STT Loop)**: A fully automated interview conductor using native `speechSynthesis` (Text-to-Speech) and `SpeechRecognition` (Speech-to-Text).
2. **MediaPipe Gaze Tracking**: Passive client-side eye tracking (~5Hz) using MediaPipe FaceMesh to calculate off-screen gaze ratio without storing raw frames.
3. **Speech Latency Profiling**: Calculates response delay between the end of TTS speaking and the start of candidate speech to measure prompt generation windows.
4. **Grok API Evaluation**: Scores transcript phrasing for AI-likeness (0-100) vs. natural spontaneous human speech, including detailed text structures analysis.
5. **Multi-Modal Risk Aggregator**: Combines latency, gaze %, focus events, and AI score into a weighted risk classification.
6. **Consent & Privacy Guardrails**: A transparent consent overlay gates all sensor captures, preserving privacy by sending only metadata.

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   CANDIDATE PORTAL (Tab 1)             │
│                                                        │
│  - FaceMesh gaze coordinates (~5Hz)                    │
│  - SpeechRecognition transcription                    │
│  - SpeechSynthesis (TTS) playback                      │
│  - Silence detection & focus event listeners           │
└───────────────────────────┬────────────────────────────┘
                            │ WebSocket
                            ▼
┌────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                    │
│                                                        │
│  /ws/{session_id}          ← Ingests signals           │
│  /ws/dashboard/{session_id}  → Streams scored risk     │
│                                                        │
│  Analyzers:                                            │
│   - TimingAnalyzer (pause delay)                       │
│   - GazeAnalyzer (off-center gaze %)                   │
│   - AILikenessScorer (Grok transcript evaluation)      │
│   - RiskAggregator (weighted risk classification)       │
└───────────────────────────┬────────────────────────────┘
                            │ WebSocket (Scored updates)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   REVIEWER CONSOLE (Tab 2)             │
│                                                        │
│  - Live risk level gauge (Clean / Suspicious / High)   │
│  - Detailed telemetry cards with numerical evidence    │
│  - Grok AI evaluation rationale logs                   │
└────────────────────────────────────────────────────────┘
```
