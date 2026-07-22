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

class QuestionAnalysisResult(BaseModel):
    question_n: int
    pause_s: float
    gaze_offscreen_pct: float
    blur_count: int
    ai_likeness_score: int
    ai_rationale: str
    risk: str  # clean | suspicious | high_risk
    transcript_text: str
    question_text: str = ""
    ts: float

class CreateSessionResponse(BaseModel):
    session_id: str
    candidate_url: str
    dashboard_url: str
