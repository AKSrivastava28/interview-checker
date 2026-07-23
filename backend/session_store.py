import time
from typing import Dict, List, Optional
from backend.models import QuestionAnalysisResult

class QuestionWindow:
    def __init__(self, question_n: int, question_text: str = "", start_ts: Optional[float] = None):
        self.question_n = question_n
        self.question_text = question_text
        self.start_ts = start_ts or time.time()
        self.gaze_samples: List[dict] = []
        self.transcript_chunks: List[dict] = []
        self.event_samples: List[dict] = []
        self.end_ts: Optional[float] = None

class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.candidate_connected = False
        self.dashboard_connected = False
        self.current_question_n = 0
        self.current_window: Optional[QuestionWindow] = None
        self.question_results: List[QuestionAnalysisResult] = []
        
        # History tracks
        self.question_history: List[str] = []
        self.answer_history: List[str] = []

    def start_new_question(self, question_text: str = "") -> QuestionWindow:
        self.current_question_n += 1
        self.current_window = QuestionWindow(
            question_n=self.current_question_n,
            question_text=question_text,
            start_ts=time.time()
        )
        self.question_history.append(question_text)
        return self.current_window

    def add_gaze(self, x: float, y: float, ts: float):
        if self.current_window:
            self.current_window.gaze_samples.append({"x": x, "y": y, "ts": ts})

    def add_transcript(self, text: str, is_final: bool, ts: float):
        if self.current_window:
            # Store the latest cumulative transcript string directly
            self.current_window.transcript_chunks = [{"text": text, "is_final": is_final, "ts": ts}]

    def add_event(self, name: str, ts: float):
        if self.current_window:
            self.current_window.event_samples.append({"name": name, "ts": ts})


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def create_session(self, session_id: str) -> SessionState:
        session = SessionState(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions


# Global singleton instance
session_store = SessionStore()
