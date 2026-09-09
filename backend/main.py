import time
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import BACKEND_PORT
from backend.models import CreateSessionResponse, QuestionAnalysisResult
from backend.session_store import session_store, SessionState, QuestionWindow
from backend.websocket_manager import ws_manager
from backend.question_engine import question_engine
from backend.transcription_service import whisper_service
from backend.analyzers.timing_analyzer import TimingAnalyzer
from backend.analyzers.gaze_analyzer import GazeAnalyzer
from backend.analyzers.speech_analyzer import SpeechAnalyzer
from backend.analyzers.acoustic_analyzer import AcousticAnalyzer
from backend.analyzers.ai_likeness_scorer import AILikenessScorer
from backend.analyzers.risk_aggregator import RiskAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="Catch the Invisible AI Cheater - Simplified")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gaze_analyzer = GazeAnalyzer()
ai_scorer = AILikenessScorer()

# Configure total questions limit (2 questions requested by user)
MAX_QUESTIONS = 2

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")


@app.get("/dashboard/{session_id}")
def dashboard_page(session_id: str):
    return RedirectResponse(url=f"/static/dashboard.html?session_id={session_id}")


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session():
    session_id = str(uuid.uuid4())[:8]
    session_store.create_session(session_id)

    candidate_url = f"/?session_id={session_id}"
    dashboard_url = f"/dashboard/{session_id}"

    return CreateSessionResponse(
        session_id=session_id,
        candidate_url=candidate_url,
        dashboard_url=dashboard_url
    )


@app.get("/api/report/{session_id}")
def get_session_report(session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "total_questions_analyzed": len(session.question_results),
        "results": [res.model_dump() for res in session.question_results]
    }


@app.post("/api/transcribe/{session_id}")
async def transcribe_candidate_audio(
    session_id: str,
    file: UploadFile = File(...)
):
    session = session_store.get_session(session_id)
    if not session and session_store.sessions:
        logger.info(f"Session '{session_id}' not found directly, falling back to latest session.")
        session = list(session_store.sessions.values())[-1]

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    audio_bytes = await file.read()
    logger.info(f"Received audio upload for transcription: {len(audio_bytes)} bytes, filename: {file.filename}")
    transcript = await whisper_service.transcribe_audio(audio_bytes, filename=file.filename or "answer.webm")
    logger.info(f"Groq Whisper transcribed text: '{transcript}'")

    # Record into current question window if active
    if session.current_window and transcript:
        session.current_window.transcript_chunks.append({
            "text": transcript,
            "is_final": True,
            "ts": time.time(),
            "word_count": len(transcript.split())
        })

    return {"status": "ok", "transcript": transcript}


@app.get("/report/{session_id}", response_class=HTMLResponse)
def get_session_report_view(session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        return HTMLResponse(content="<h2>Session not found</h2>", status_code=404)

    rows_html = ""
    for res in session.question_results:
        badge_class = "risk-clean" if res.risk == "clean" else ("risk-suspicious" if res.risk == "suspicious" else "risk-high")
        delivery_str = res.speech_delivery.replace('_', ' ').title()
        vocal_badge = "style='color:#10b981;'" if res.vocal_style == "expressive" else ("style='color:#f43f5e;'" if res.vocal_style == "monotone_drone" else "style='color:#94a3b8;'")
        latency_str = f"{res.pause_s}s"
        if res.stall_detected and res.effective_latency_s > res.pause_s:
            latency_str = f"{res.pause_s}s<div style='color:#f43f5e;font-size:0.75rem;font-weight:600;'>Eff: {res.effective_latency_s}s (Stalled)</div>"
        stall_note = f"<span style='color:#f59e0b;font-weight:bold;'>[Stall: \"{res.stall_buffer}\"] </span>" if res.stall_detected else ""
        rows_html += f"""
        <tr class="{badge_class}">
            <td>Q{res.question_n}</td>
            <td><strong>{res.question_text}</strong></td>
            <td>{latency_str}</td>
            <td><strong>{delivery_str}</strong> ({res.speech_reading_score}/100)</td>
            <td><strong {vocal_badge}>{res.vocal_style.upper().replace('_', ' ')}</strong> ({res.pitch_std:.1f} Hz)</td>
            <td>{res.blur_count} Blurs / {res.fullscreen_exit_count} Exits</td>
            <td>{res.gaze_offscreen_pct}%</td>
            <td>{res.ai_likeness_score}/100</td>
            <td>
                <span class="badge {badge_class}">{res.risk.upper()}</span>
                <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;font-weight:600;">{getattr(res, 'confidence_pct', 85.0):.0f}% Conf.</div>
            </td>
            <td><p class="rationale">{res.ai_rationale}</p></td>
        </tr>
        <tr>
            <td colspan="10" class="transcript-cell"><em>Transcript:</em> {stall_note}"{res.transcript_text or 'No speech recorded'}"</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Interview Integrity Report - Session {session_id}</title>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0b0f19; color: #f8fafc; padding: 2rem; margin: 0; }}
            .container {{ max-width: 1100px; margin: 0 auto; background: #151c2e; border-radius: 12px; padding: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.08); }}
            h1 {{ color: #38bdf8; margin-top: 0; }}
            .meta {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1rem; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
            th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 0.9rem; }}
            th {{ background: #0b0f19; color: #cbd5e1; font-weight: 600; }}
            .badge {{ padding: 4px 10px; border-radius: 9999px; font-weight: bold; font-size: 0.8rem; text-transform: uppercase; }}
            .risk-clean .badge {{ background: rgba(16, 185, 129, 0.15); color: #10b981; }}
            .risk-suspicious .badge {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; }}
            .risk-high .badge {{ background: rgba(244, 63, 94, 0.15); color: #f43f5e; }}
            .transcript-cell {{ background: #0b0f19; color: #94a3b8; font-size: 0.85rem; font-style: italic; padding-left: 2rem; }}
            .rationale {{ margin: 0; font-size: 0.85rem; color: #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interview Integrity Audit Report</h1>
            <div class="meta">
                <span>Session ID: <strong>{session_id}</strong></span> |
                <span>Questions Evaluated: <strong>{len(session.question_results)}</strong></span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Question</th>
                        <th>Latency</th>
                        <th>Speech Delivery</th>
                        <th>Vocal Prosody (F₀)</th>
                        <th>Focus & Fullscreen</th>
                        <th>Offscreen Gaze</th>
                        <th>AI Score</th>
                        <th>Risk</th>
                        <th>Integrity Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html if rows_html else '<tr><td colspan="10">No question analysis data recorded for this session.</td></tr>'}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


async def evaluate_question_window(session: SessionState, win: QuestionWindow, full_transcript: str = ""):
    # 1. Transcript Aggregation (Establish authoritative full transcript first)
    if not full_transcript:
        final_transcripts = [c.get("text", "") for c in win.transcript_chunks if c.get("is_final")]
        full_transcript = " ".join(final_transcripts).strip()
        if not full_transcript and win.transcript_chunks:
            full_transcript = win.transcript_chunks[-1].get("text", "").strip()

    word_count = len(full_transcript.split()) if full_transcript else 0

    # 2. Speech Delivery & Timing Analysis (Reading cadence vs spontaneous flow & Stall Detection)
    speech_metrics = SpeechAnalyzer.analyze_speech_delivery(
        question_start_ts=win.start_ts,
        transcript_chunks=win.transcript_chunks,
        window_end_ts=win.end_ts or time.time(),
        full_transcript=full_transcript,
        question_text=win.question_text
    )
    pause_s = speech_metrics.get("prompt_latency_s", 0.0)
    effective_latency_s = speech_metrics.get("effective_latency_s", pause_s)
    speech_reading_score = speech_metrics.get("reading_cadence_score", 0)
    speech_delivery_label = speech_metrics.get("delivery_style", "spontaneous")
    speech_rationale = speech_metrics.get("rationale", "")
    stall_detected = speech_metrics.get("stall_detected", False)
    stall_buffer = speech_metrics.get("stall_buffer", "")

    # 3. Vocal Prosody & Acoustic Disfluency Analysis
    acoustic_metrics = AcousticAnalyzer.analyze_prosody_and_disfluencies(
        transcript_text=full_transcript,
        acoustic_features=win.acoustic_features
    )
    pitch_std = acoustic_metrics.get("pitch_std", 0.0)
    vocal_style = acoustic_metrics.get("vocal_style", "unmeasured")
    prosody_score = acoustic_metrics.get("prosody_score", 0)
    prosody_rationale = acoustic_metrics.get("rationale", "")

    # Keep speech cadence and acoustic prosody as independent observation channels
    # Append prosody rationale to speech delivery rationale without overwriting metrics
    speech_rationale += f" | {prosody_rationale}"

    # 4. Gaze Analysis (Offscreen: Phone in lap / 2nd monitor only)
    gaze_offscreen_pct = gaze_analyzer.calculate_offscreen_percentage(win.gaze_samples)

    # 5. Events Analysis (Tab blurs & Fullscreen exits)
    blur_count = sum(1 for e in win.event_samples if e.get("name") in ["tab_blur", "visibility_hidden"])
    fullscreen_exit_count = sum(1 for e in win.event_samples if e.get("name") == "fullscreen_exit")

    # 6. AI Likeness Scoring with Two-Phase Stall Slicing
    ai_score_val = 0

    if word_count < 4:
        speech_delivery_label = "non_responsive"
        speech_reading_score = 0
        ai_rationale = "[Non-Responsive Alert] Candidate provided no audible technical answer for this question."
        if gaze_offscreen_pct > 5.0:
            ai_rationale += f" [Integrity Violation] Candidate was looking away at an external device ({gaze_offscreen_pct}% offscreen gaze) while failing to answer."
    else:
        ai_result = await ai_scorer.analyze_transcript(full_transcript, win.question_text)
        ai_score_val = ai_result.score
        ai_rationale = ai_result.rationale

        # If AI scorer detected a stall, merge signals
        if ai_result.stall_detected:
            stall_detected = True
            stall_buffer = stall_buffer or ai_result.stall_buffer
            if ai_result.effective_latency_offset_s > 0:
                effective_latency_s = max(effective_latency_s, round(pause_s + ai_result.effective_latency_offset_s, 1))

        ai_rationale += f" [Vocal & Cadence Delivery] {speech_rationale}"
        if blur_count > 0 or fullscreen_exit_count > 0:
            ai_rationale += f" [Focus Alert] {blur_count} window blur(s), {fullscreen_exit_count} fullscreen exit(s)."

    # 7. Multi-Signal Risk Aggregator
    risk_label, composite_score, breakdown = RiskAggregator.compute_risk(
        pause_s=pause_s,
        gaze_offscreen_pct=gaze_offscreen_pct,
        blur_count=blur_count,
        ai_likeness_score=ai_score_val,
        reading_pct=float(speech_reading_score),
        speech_reading_score=speech_reading_score,
        prosody_score=prosody_score,
        vocal_style=vocal_style,
        fullscreen_exit_count=fullscreen_exit_count,
        total_words=word_count,
        stall_detected=stall_detected,
        effective_latency_s=effective_latency_s,
        stall_buffer=stall_buffer
    )

    confidence_pct = breakdown.get("confidence_pct", 85.0)

    analysis_res = QuestionAnalysisResult(
        question_n=win.question_n,
        question_text=win.question_text,
        pause_s=pause_s,
        effective_latency_s=effective_latency_s,
        gaze_offscreen_pct=gaze_offscreen_pct,
        reading_pct=float(speech_reading_score),
        speech_reading_score=speech_reading_score,
        speech_delivery=speech_delivery_label,
        pitch_std=pitch_std,
        vocal_style=vocal_style,
        prosody_score=prosody_score,
        blur_count=blur_count,
        fullscreen_exit_count=fullscreen_exit_count,
        ai_likeness_score=ai_score_val,
        ai_rationale=ai_rationale,
        stall_detected=stall_detected,
        stall_buffer=stall_buffer,
        confidence_pct=confidence_pct,
        risk=risk_label,
        transcript_text=full_transcript,
        ts=time.time()
    )

    session.question_results.append(analysis_res)

    # Broadcast results
    await ws_manager.send_to_dashboard(session.session_id, {
        "type": "question_result",
        "result": analysis_res.model_dump()
    })
    await ws_manager.send_to_candidate(session.session_id, {
        "type": "question_evaluated",
        "result": analysis_res.model_dump()
    })


async def trigger_next_question(session: SessionState):
    next_idx = session.current_question_n

    # Limit to configured questions count
    if next_idx >= MAX_QUESTIONS:
        await ws_manager.send_to_candidate(session.session_id, {"type": "interview_complete"})
        await ws_manager.send_to_dashboard(session.session_id, {"type": "interview_complete"})
        return

    # Check if this is Question 2 (index 1) or later, and candidate answered previous question
    question_text = ""
    if next_idx >= 1 and session.question_history and session.answer_history:
        last_q = session.question_history[-1]
        last_a = session.answer_history[-1]
        if last_a and len(last_a.split()) >= 6:
            logger.info(f"Generating contextual drill-down question based on candidate answer: {last_a[:60]}...")
            drilldown = await question_engine.generate_drilldown_question(last_q, last_a)
            if drilldown:
                question_text = drilldown

    if not question_text:
        # Fallback to hardcoded list
        question_text = question_engine.get_question(next_idx)

    session.start_new_question(question_text=question_text)

    await ws_manager.send_to_candidate(session.session_id, {
        "type": "new_question",
        "question_n": session.current_question_n,
        "max_questions": MAX_QUESTIONS,
        "question_text": question_text
    })
    await ws_manager.send_to_dashboard(session.session_id, {
        "type": "new_question",
        "question_n": session.current_question_n,
        "max_questions": MAX_QUESTIONS,
        "question_text": question_text
    })


@app.websocket("/ws/{session_id}")
async def websocket_candidate(websocket: WebSocket, session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        session = session_store.create_session(session_id)

    await ws_manager.connect_candidate(session_id, websocket)
    session.candidate_connected = True
    await ws_manager.send_to_dashboard(session_id, {"type": "candidate_status", "connected": True})

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            ts = data.get("ts", time.time())

            if msg_type == "start_interview":
                await trigger_next_question(session)
            elif msg_type == "question_speaking_finished":
                # Start timing analysis from the moment TTS completes speaking
                if session.current_window:
                    session.current_window.start_ts = time.time()
            elif msg_type == "gaze":
                session.add_gaze(
                    x=float(data.get("x", 0.5)),
                    y=float(data.get("y", 0.5)),
                    ts=ts,
                    reading_detected=bool(data.get("reading_detected", False)),
                    reading_type=str(data.get("reading_type", ""))
                )
            elif msg_type == "transcript":
                session.add_transcript(
                    text=str(data.get("text", "")),
                    is_final=bool(data.get("is_final", False)),
                    ts=ts,
                    word_count=int(data.get("word_count", 0))
                )
            elif msg_type == "event":
                session.add_event(name=str(data.get("name", "")), ts=ts)
            elif msg_type == "done_answering":
                if session.current_window:
                    win = session.current_window
                    win.end_ts = time.time()
                    win.acoustic_features = data.get("acoustic_features", {})

                    # Extract full transcript (prioritizing Groq Whisper authoritative transcript)
                    whisper_text = data.get("whisper_transcript", "").strip()
                    if whisper_text:
                        full_transcript = whisper_text
                    else:
                        final_transcripts = [c.get("text", "") for c in win.transcript_chunks if c.get("is_final")]
                        full_transcript = " ".join(final_transcripts).strip()
                        if not full_transcript and win.transcript_chunks:
                            full_transcript = win.transcript_chunks[-1].get("text", "").strip()

                    session.answer_history.append(full_transcript)
                    session.current_window = None

                    import asyncio
                    # Run Grok evaluation in the background without blocking the UI
                    asyncio.create_task(evaluate_question_window(session, win, full_transcript))

                    if session.current_question_n >= MAX_QUESTIONS:
                        await asyncio.sleep(1.0)
                        await ws_manager.send_to_candidate(session.session_id, {"type": "interview_complete"})
                        await ws_manager.send_to_dashboard(session.session_id, {"type": "interview_complete"})
                    else:
                        await asyncio.sleep(1.0)
                        await trigger_next_question(session)

    except WebSocketDisconnect:
        logger.info(f"Candidate disconnected from session {session_id}")
    finally:
        ws_manager.disconnect_candidate(session_id, websocket)
        session.candidate_connected = False
        await ws_manager.send_to_dashboard(session_id, {"type": "candidate_status", "connected": False})


@app.websocket("/ws/dashboard/{session_id}")
async def websocket_dashboard(websocket: WebSocket, session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        session = session_store.create_session(session_id)

    await ws_manager.connect_dashboard(session_id, websocket)
    session.dashboard_connected = True

    # Emit initial status
    await websocket.send_json({"type": "candidate_status", "connected": session.candidate_connected})

    # Catch initial metrics if dashboard connects mid-session
    for result in session.question_results:
        await websocket.send_json({
            "type": "question_result",
            "result": result.model_dump()
        })

    try:
        while True:
            # Dashboard is read-only from socket, keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"Dashboard disconnected from session {session_id}")
    finally:
        ws_manager.disconnect_dashboard(session_id, websocket)
        session.dashboard_connected = False
