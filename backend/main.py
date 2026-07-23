import time
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import BACKEND_PORT
from backend.models import CreateSessionResponse, QuestionAnalysisResult
from backend.session_store import session_store, SessionState, QuestionWindow
from backend.websocket_manager import ws_manager
from backend.question_engine import question_engine
from backend.analyzers.timing_analyzer import TimingAnalyzer
from backend.analyzers.gaze_analyzer import GazeAnalyzer
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

# Configure total questions limit (2 for testing, 5 for final demo)
MAX_QUESTIONS = 3

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


@app.get("/report/{session_id}", response_class=HTMLResponse)
def get_session_report_view(session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        return HTMLResponse(content="<h2>Session not found</h2>", status_code=404)

    rows_html = ""
    for res in session.question_results:
        badge_class = "risk-clean" if res.risk == "clean" else ("risk-suspicious" if res.risk == "suspicious" else "risk-high")
        rows_html += f"""
        <tr class="{badge_class}">
            <td>Q{res.question_n}</td>
            <td><strong>{res.question_text}</strong></td>
            <td>{res.pause_s}s</td>
            <td>{res.gaze_offscreen_pct}%</td>
            <td>{res.blur_count}</td>
            <td>{res.ai_likeness_score}/100</td>
            <td><span class="badge {badge_class}">{res.risk.upper()}</span></td>
            <td><p class="rationale">{res.ai_rationale}</p></td>
        </tr>
        <tr>
            <td colspan="8" class="transcript-cell"><em>Transcript:</em> "{res.transcript_text or 'No speech recorded'}"</td>
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
            .container {{ max-width: 1000px; margin: 0 auto; background: #151c2e; border-radius: 12px; padding: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.08); }}
            h1 {{ color: #38bdf8; margin-top: 0; }}
            .meta {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1rem; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
            th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }}
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
                        <th>Pause</th>
                        <th>Gaze Offscreen</th>
                        <th>Tab Blurs</th>
                        <th>AI Score</th>
                        <th>Risk</th>
                        <th>AI Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html if rows_html else '<tr><td colspan="8">No question analysis data recorded for this session.</td></tr>'}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


async def evaluate_question_window(session: SessionState, win: QuestionWindow):
    # 1. Timing Analysis
    pause_s = TimingAnalyzer.calculate_pause(
        question_start_ts=win.start_ts,
        transcript_chunks=win.transcript_chunks,
        window_end_ts=win.end_ts
    )

    # 2. Gaze Analysis
    gaze_offscreen_pct = gaze_analyzer.calculate_offscreen_percentage(win.gaze_samples)

    # 3. Events Analysis
    blur_count = sum(1 for e in win.event_samples if e.get("name") in ["tab_blur", "visibility_hidden"])

    # 4. Transcript Aggregation & AI Likeness Scoring
    final_transcripts = [c.get("text", "") for c in win.transcript_chunks if c.get("is_final")]
    full_transcript = " ".join(final_transcripts).strip()
    if not full_transcript:
        full_transcript = " ".join([c.get("text", "") for c in win.transcript_chunks]).strip()

    ai_result = await ai_scorer.analyze_transcript(full_transcript, win.question_text)

    # 5. Risk Aggregator
    risk_label, composite_score, breakdown = RiskAggregator.compute_risk(
        pause_s=pause_s,
        gaze_offscreen_pct=gaze_offscreen_pct,
        blur_count=blur_count,
        ai_likeness_score=ai_result.score
    )

    analysis_res = QuestionAnalysisResult(
        question_n=win.question_n,
        question_text=win.question_text,
        pause_s=pause_s,
        gaze_offscreen_pct=gaze_offscreen_pct,
        blur_count=blur_count,
        ai_likeness_score=ai_result.score,
        ai_rationale=ai_result.rationale,
        risk=risk_label,
        transcript_text=full_transcript,
        ts=time.time()
    )

    session.question_results.append(analysis_res)
    session.answer_history.append(full_transcript)

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

    # Get question instantly from hardcoded list
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
                session.add_gaze(x=float(data.get("x", 0.5)), y=float(data.get("y", 0.5)), ts=ts)
            elif msg_type == "transcript":
                session.add_transcript(text=str(data.get("text", "")), is_final=bool(data.get("is_final", False)), ts=ts)
            elif msg_type == "event":
                session.add_event(name=str(data.get("name", "")), ts=ts)
            elif msg_type == "done_answering":
                if session.current_window:
                    win = session.current_window
                    win.end_ts = time.time()
                    session.current_window = None
                    import asyncio
                    # Run Grok evaluation in the background without blocking the UI
                    asyncio.create_task(evaluate_question_window(session, win))
                    # Wait 1.0 second for smooth UI feedback and immediately load next question
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
