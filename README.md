# Catch the Invisible AI Cheater 🛡️

A zero-install mock AI interviewer application prototype designed to detect candidates secretly using real-time AI overlay tools (like Cluely) during live interviews.

---

## 🚀 Quickstart (One-Command Setup)

### 1. Prerequisites
- Python 3.9+ installed
- Google Chrome or Microsoft Edge (recommended for native SpeechRecognition and SpeechSynthesis support)

### 2. Installation & Configuration

1. **Clone repository and create virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   *(Add `GROK_API_KEY` from x.ai. If left unconfigured, the application runs with robust local fallback heuristics and default questions).*

4. **Launch the FastAPI Server**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

5. **Open Candidate Portal**:
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🧪 Testing the Simulation (Single Player Mode)

1. Open `http://localhost:8000` in **Tab 1** (Candidate View).
2. Click **Create Mock Interview Session**.
3. Copy the **Reviewer Dashboard Link** and open it in **Tab 2** (Reviewer Console).
4. In Tab 1, click **I Consent & Join Room** to grant camera/mic permissions.
5. In Tab 1, click **Start Interview**.
6. Listen to the AI ask Question 1 (audio via TTS). Speak your answer:
   - **Honest Test**: Answer naturally with standard filler words ("um", "like", "you know") while looking directly at the camera.
   - **Cheating Simulation**: Look away at a secondary monitor / screen, pause for 5-8 seconds before answering, and read structured, textbook-style answers.
7. Click **Done Answering** (or wait 3 seconds for the automatic silence detector to submit).
8. Go to **Tab 2 (Reviewer Console)** and watch the live risk telemetry update instantly with pause latency, gaze offscreen %, tab blurs, and AI likeness scoring with Grok's rationale.

---

## 🛠️ Multi-Modal Detection Telemetry

- **Speech Latency**: Captures the exact millisecond delay between the end of AI speech playback and the candidate's first spoken sentence.
- **Off-Center Gaze Tracking**: Client-side MediaPipe FaceMesh tracks face and iris position relative to center screen coordinates ~5x/sec, calculating the percentage of time looking away.
- **AI-Likeness Scorer**: Grok API scores transcript fluency (0-100), flagging textbook sentence structure and lack of conversational hesitations.
- **Tab Blur Listeners**: Captures window focus loss events when clicking out of the tab.
- **Multi-Modal Risk Fusion**: Aggregates signals into a `clean`, `suspicious`, or `high_risk` summary with detailed breakdown cards.
