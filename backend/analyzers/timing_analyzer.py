from typing import List, Optional

class TimingAnalyzer:
    @staticmethod
    def calculate_pause(question_start_ts: float, transcript_chunks: List[dict], window_end_ts: Optional[float] = None) -> float:
        """
        Calculates pause duration (in seconds) between question_start_ts and candidate's first speech timestamp.
        """
        if not question_start_ts:
            return 0.0

        for chunk in transcript_chunks:
            text = chunk.get("text", "").strip()
            ts = chunk.get("ts")
            if text and ts:
                pause = ts - question_start_ts
                return round(max(0.0, pause), 2)

        # If candidate didn't speak during the window
        if window_end_ts:
            return round(max(0.0, window_end_ts - question_start_ts), 2)
        return 0.0
