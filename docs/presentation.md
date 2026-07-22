# 3-5 Minute Pitch Presentation Outline

This document provides a slide-by-slide structure and speaking script outline for your 3-5 minute hackathon presentation.

---

## Slide 1: The Problem — The Invisible Cheater
- **Slide Visuals**: Graphic of a video call next to an invisible floating text overlay. Bold text: "Invisible to Screen Sharing."
- **Talking Points**:
  - The rise of LLMs has enabled "AI Interview Copilots" (like Cluely, Parakeet, and FinalRound AI).
  - These tools run as transparent, invisible desktop overlays or on secondary devices.
  - Traditional proctoring (screen sharing checks, browser extension lockouts) is completely blind to them because they leave no system footprints.
  - Our goal: Detect them from the browser without installing anything on the candidate's machine.

---

## Slide 2: The Solution — Multi-Modal Passive Telemetry
- **Slide Visuals**: Diagram showing the 3 signals (Timing, Gaze, Phrasing) fusing into a single Risk Score.
- **Talking Points**:
  - We don't try to scan the candidate's screen or run heavy object detection (which requires expensive GPUs and violates privacy).
  - Instead, we detect the *behavioral consequences* of using an assistant.
  - We track three passive browser signals:
    1. **Speech Latency**: The exact pause duration between the end of the question and the first word (capturing the AI text generation window).
    2. **Calibrated Gaze Deviation**: Tracking iris movement relative to eye corners using client-side MediaPipe FaceMesh to spot phone/overlay reading.
    3. **AI Phrasing Likeness**: Using linguistic structure models to determine if speech phrasing matches textbook AI patterns.

---

## Slide 3: Live Demo Flow
- **Slide Visuals**: Side-by-side screenshots of the Candidate Terminal and the Reviewer Console.
- **Talking Points**:
  - Run the demo:
    - **Step 1**: The candidate gives explicit consent before any camera or mic access.
    - **Step 2**: The AI Interviewer speaks a question.
    - **Step 3 (Honest Run)**: The candidate answers naturally, looking at the screen. The dashboard flags it as "CLEAN".
    - **Step 4 (Cheating Run)**: The candidate glances down at a phone or second screen, pauses, and reads formal text. The dashboard immediately fires a "HIGH RISK" alert.

---

## Slide 4: Defensible Limits (Honesty & Rigor)
- **Slide Visuals**: Bullet points showing "What it catches" vs "What it doesn't."
- **Talking Points**:
  - **Human-in-the-Loop**: We never auto-reject. The tool provides transparent evidence numbers for a human reviewer.
  - **Privacy First**: No raw video or audio is ever transmitted or stored on the backend.
  - **False Positive Protection**: Normal on-screen reading movement is accommodated by a widened gaze bounding box and a 15-frame startup self-calibration step.

---

## Slide 5: Future Roadmap & Impact
- **Slide Visuals**: Timeline: Style Profiling $\rightarrow$ Pacing Dynamics $\rightarrow$ Enterprise Integration.
- **Talking Points**:
  - **Pacing Analysis**: Detecting flat word-per-minute reading rates versus natural human speech fluctuations.
  - **Warmup Baselines**: Gathering conversational baseline patterns during a warm-up phase to personalize the evaluation.
  - **Summary**: An invisible copilot can hide from screen recorders, but it cannot hide from human biology and timing.
