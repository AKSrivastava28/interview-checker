import json
import logging
import re
import httpx
from backend.config import GROK_API_KEY, GROK_API_BASE
from backend.models import AILikenessResult

logger = logging.getLogger("ai_likeness_scorer")

SYSTEM_PROMPT = (
    "You are an expert interview integrity evaluator. Your job is to assess transcript chunks "
    "from candidate oral interview answers to detect signs of AI-generated answers read aloud "
    "(e.g., rigid bullet point structure, overly formal textbook phrasing, perfect syntax without fillers) "
    "versus natural spontaneous speech."
)

class AILikenessScorer:
    def __init__(self, api_key: str = GROK_API_KEY, api_base: str = GROK_API_BASE):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    async def analyze_transcript(self, transcript_text: str, question_text: str = "") -> AILikenessResult:
        word_count = len(re.findall(r'\b\w+\b', transcript_text)) if transcript_text else 0
        if word_count < 15:
            return AILikenessResult(
                score=5,
                rationale="Response too short for conclusive AI-likeness evaluation."
            )

        if self.api_key:
            try:
                return await self._call_grok_api(transcript_text, question_text)
            except Exception as e:
                logger.error(f"Grok API call failed: {e}. Falling back to heuristic analysis.")
                return self._fallback_heuristic(transcript_text)
        else:
            return self._fallback_heuristic(transcript_text)

    async def _call_grok_api(self, transcript_text: str, question_text: str) -> AILikenessResult:
        prompt = (
            f"Question asked: \"{question_text}\"\n" if question_text else ""
        ) + (
            f"Candidate Transcript: \"{transcript_text}\"\n\n"
            "Evaluate if this spoken response resembles an AI assistant answer read aloud or natural spontaneous speech.\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "score": <integer 0 to 100>,\n'
            '  "rationale": "<1 sentence explanation>"\n'
            "}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-beta",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            score = int(parsed.get("score", 50))
            rationale = str(parsed.get("rationale", "Evaluated via Grok API."))
            return AILikenessResult(score=max(0, min(100, score)), rationale=rationale)

    def _fallback_heuristic(self, text: str) -> AILikenessResult:
        """
        Heuristic offline scorer when Grok API key is not configured.
        """
        score = 20  # Base natural score
        reasons = []

        lower = text.lower()
        
        # Check formal transitional phrases
        formal_markers = ["firstly", "secondly", "in conclusion", "to summarize", "furthermore", "moreover", "key aspects"]
        matched_markers = [m for m in formal_markers if m in lower]
        if matched_markers:
            score += len(matched_markers) * 15
            reasons.append(f"Contains formal structure markers ({', '.join(matched_markers)})")

        # Check for natural filler words (which reduce AI score)
        fillers = ["um", "uh", "like", "you know", "i mean", "well", "basically"]
        matched_fillers = [f for f in fillers if f in lower]
        if matched_fillers:
            score -= len(matched_fillers) * 10
            reasons.append("Contains natural conversational fillers")

        # Check vocabulary density / average word length
        words = re.findall(r'\b\w+\b', text)
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len > 6.0:
                score += 20
                reasons.append("High academic/formal vocabulary density")

        score = max(5, min(95, score))
        rationale_str = "; ".join(reasons) if reasons else "Spontaneous speech pattern detected."
        return AILikenessResult(
            score=score,
            rationale=f"[Offline Heuristic] {rationale_str}"
        )
