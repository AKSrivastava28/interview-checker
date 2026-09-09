from typing import List, Optional
from pydantic import BaseModel, Field

class BaseSignal(BaseModel):
    type: str
    ts: float

class GazeSignal(BaseSignal):
    type: str = "gaze"
    x: float
    y: float

class TranscriptSignal(BaseSignal):
    type: str = "transcript"
    text: str
    is_final: bool = False

class EventSignal(BaseSignal):
    type: str = "event"
    name: str  # tab_blur, tab_focus, window_resize, visibility_hidden, etc.

class AILikenessResult(BaseModel):
    score: int = Field(ge=0, le=100)
    rationale: str
    stall_detected: bool = False
    stall_buffer: str = ""
    substantive_core: str = ""
    effective_latency_offset_s: float = 0.0

class QuestionAnalysisResult(BaseModel):
    question_n: int
    pause_s: float
    effective_latency_s: float = 0.0
    gaze_offscreen_pct: float
    reading_pct: float = 0.0
    speech_reading_score: int = 0
    speech_delivery: str = "spontaneous"  # spontaneous | suspicious_reading | script_reading
    pitch_std: float = 0.0
    vocal_style: str = "expressive"       # expressive | monotone_drone | unmeasured
    prosody_score: int = 0
    blur_count: int
    fullscreen_exit_count: int = 0
    ai_likeness_score: int
    ai_rationale: str
    stall_detected: bool = False
    stall_buffer: str = ""
    confidence_pct: float = 85.0
    risk: str  # clean | suspicious | high_risk
    transcript_text: str
    question_text: str = ""
    ts: float

class CreateSessionResponse(BaseModel):
    session_id: str
    candidate_url: str
    dashboard_url: str
