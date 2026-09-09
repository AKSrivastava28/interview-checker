import asyncio
import json
import logging
import re
import httpx
from backend.config import GROK_API_KEY, GROK_API_BASE
from backend.models import AILikenessResult
from backend.analyzers.stall_detector import StallDetector

logger = logging.getLogger("ai_likeness_scorer")

SYSTEM_PROMPT = (
    "You are an expert technical interview integrity evaluator. Your job is to assess candidate oral interview transcripts "
    "to detect signs of AI-generated answers—both rigid textbook recitation AND modern conversational LLM outputs "
    "(e.g., ChatGPT/Claude prompted to 'answer conversationally as a pragmatic senior engineer' with perfectly balanced clauses, "
    "advisory structuring, empirical tuning advice, and zero genuine oral hesitations) versus authentic spontaneous human thought. "
    "Candidates may also use opening verbal stalling to buy time while external AI tools generate answers."
)

class AILikenessScorer:
    def __init__(self, api_key: str = GROK_API_KEY, api_base: str = GROK_API_BASE):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    async def analyze_transcript(self, transcript_text: str, question_text: str = "") -> AILikenessResult:
        word_count = len(re.findall(r'\b\w+\b', transcript_text)) if transcript_text else 0
        if word_count < 10:
            return AILikenessResult(
                score=5,
                rationale="Response too short for conclusive AI-likeness evaluation."
            )

        # Slice transcript to isolate any opening verbal stall from the substantive technical core
        slice_info = StallDetector.slice_transcript(transcript_text, question_text)
        stall_detected = slice_info["stall_detected"]
        stall_buffer = slice_info["stall_buffer"]
        substantive_core = slice_info["substantive_core"]
        latency_offset = slice_info["estimated_stall_duration_s"]

        if self.api_key:
            try:
                return await self._call_grok_api(
                    raw_transcript=transcript_text,
                    question_text=question_text,
                    stall_detected=stall_detected,
                    stall_buffer=stall_buffer,
                    substantive_core=substantive_core,
                    latency_offset=latency_offset
                )
            except Exception as e:
                logger.error(f"Groq API call failed: {e}. Falling back to heuristic analysis.")
                return self._fallback_heuristic(transcript_text, stall_detected, stall_buffer, substantive_core, latency_offset)
        else:
            return self._fallback_heuristic(transcript_text, stall_detected, stall_buffer, substantive_core, latency_offset)

    async def _call_grok_api(
        self,
        raw_transcript: str,
        question_text: str,
        stall_detected: bool,
        stall_buffer: str,
        substantive_core: str,
        latency_offset: float
    ) -> AILikenessResult:
        if stall_detected and stall_buffer:
            prompt = (
                f"Question asked: \"{question_text}\"\n" if question_text else ""
            ) + (
                f"Full Candidate Audio Transcript: \"{raw_transcript}\"\n"
                f"[Detected Opening Camouflage / Stall Buffer: \"{stall_buffer}\"]\n"
                f"[Substantive Technical Core: \"{substantive_core}\"]\n\n"
                "CRITICAL INTEGRITY EVALUATION INSTRUCTION:\n"
                "Candidates frequently cheat using conversational LLM generations (e.g., ChatGPT or Claude prompted to "
                "'answer conversationally like a pragmatic senior engineer').\n"
                "Hallmarks of conversational AI responses:\n"
                "1. Polished rhetorical completeness: perfectly balanced clauses, addressing theoretical baseline then empirical tuning.\n"
                "2. Manufactured first-person pragmatic advice (e.g. 'we start with a reasonable baseline... I wouldn't blindly use these numbers... for legal documents split on headings rather than cutting every 500 tokens').\n"
                "3. Multi-sentence advisory structure delivered without genuine oral hesitations, false starts, or spontaneous self-correction.\n"
                "Notice: Spontaneous oral human speech in live technical interviews is fragmented, hesitant, and struggles for precise phrasing. "
                "Fluently recited multi-sentence advisory paragraphs with clean conceptual coverage are AI-generated scripts.\n"
                "Score high (65-95) if the substantive core exhibits this structured, exhaustive, polished advisory flow (whether formal textbook or conversational AI).\n"
                "Score low (0-35) ONLY if the candidate exhibits authentic spontaneous oral thinking: colloquial false starts, rough fragmented phrasing, or genuine conceptual unpolishedness.\n"
                "Respond ONLY with a JSON object in this format:\n"
                "{\n"
                '  "score": <integer 0 to 100>,\n'
                '  "rationale": "<1 concise sentence stating evaluation, noting if an opening cushion/stall preceded an AI answer>"\n'
                "}"
            )
        else:
            prompt = (
                f"Question asked: \"{question_text}\"\n" if question_text else ""
            ) + (
                f"Candidate Spoken Transcript: \"{raw_transcript}\"\n\n"
                "Evaluate if this spoken response resembles an AI-generated answer (either formal textbook definition OR conversational ChatGPT advice) "
                "versus authentic spontaneous human conversational explanation.\n"
                "Note: Candidates often cheat using conversational AI outputs containing pragmatic phrases ('I wouldn't blindly use these numbers', 'tune it empirically'). "
                "Focus on the underlying syntactic completeness, polished advisory structure, and lack of genuine oral disfluency.\n"
                "Score high (65-100) if it is an AI-generated answer or script recited aloud.\n"
                "Score low (0-35) if it has authentic human oral formulation, fragmented phrasing, or natural spontaneous thought flow.\n"
                "Respond ONLY with a JSON object in this format:\n"
                "{\n"
                '  "score": <integer 0 to 100>,\n'
                '  "rationale": "<1 concise sentence explanation>"\n'
                "}"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 429 and attempt < 4:
                        wait_s = 3.0 * (attempt + 1)
                        logger.warning(f"Groq 429 rate limit reached. Retrying in {wait_s}s...")
                        await asyncio.sleep(wait_s)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    score = int(parsed.get("score", 50))
                    rationale = str(parsed.get("rationale", "Evaluated via Grok API."))
                    return AILikenessResult(
                        score=max(0, min(100, score)),
                        rationale=rationale,
                        stall_detected=stall_detected,
                        stall_buffer=stall_buffer,
                        substantive_core=substantive_core,
                        effective_latency_offset_s=latency_offset
                    )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 4:
                    wait_s = 3.0 * (attempt + 1)
                    await asyncio.sleep(wait_s)
                    continue
                raise

    def _fallback_heuristic(
        self,
        text: str,
        stall_detected: bool = False,
        stall_buffer: str = "",
        substantive_core: str = "",
        latency_offset: float = 0.0
    ) -> AILikenessResult:
        """
        Heuristic offline scorer when Grok API key is not configured.
        """
        eval_text = substantive_core if (stall_detected and substantive_core) else text
        score = 25  # Base natural score
        reasons = []

        lower = eval_text.lower()
        
        # Check formal transitional phrases & textbook markers
        formal_markers = ["firstly", "secondly", "in conclusion", "to summarize", "furthermore", "moreover", "key aspects", "refers to", "architectural pattern", "vector database"]
        matched_markers = [m for m in formal_markers if m in lower]
        if matched_markers:
            score += len(matched_markers) * 15
            reasons.append(f"Contains formal structure markers ({', '.join(matched_markers)})")

        # Check textbook technical terminology
        tech_markers = ["approximate nearest neighbor", "nearest neighbor", "cosine similarity", "retrieval augmented generation", "vector database", "vector index", "hnsw", "faiss", "trade-off", "tradeoff", "token overlap", "chunk size", "minilm", "grounding", "embeddings", "hallucinations"]
        matched_tech = [m for m in tech_markers if m in lower]
        if matched_tech:
            score += min(50, len(matched_tech) * 12)
            reasons.append(f"High technical jargon density ({', '.join(matched_tech[:4])})")

        # Check conversational LLM markers (ChatGPT pragmatic senior engineer persona)
        conversational_ai_markers = [
            "reasonable baseline", "tune it empirically", "blindly use", "in practice",
            "document structure", "legal documents", "split based on", "split on headings",
            "arbitrarily cutting", "arbitrary token count", "retrieval evaluation",
            "latency constraints", "chunk boundary", "chunk overlap", "chunk size"
        ]
        matched_conv_ai = [m for m in conversational_ai_markers if m in lower]
        if matched_conv_ai:
            score += min(50, len(matched_conv_ai) * 15)
            reasons.append(f"Conversational AI advisory phrasing ({', '.join(matched_conv_ai[:4])})")

        # Check for natural filler words in substantive core only (ignored if high technical density)
        fillers = ["um", "uh", "you know", "i mean"]
        matched_fillers = [f for f in fillers if f in lower]
        if matched_fillers and len(matched_tech) < 2:
            score -= len(matched_fillers) * 10
            reasons.append("Contains natural conversational fillers")

        # Check vocabulary density / average word length
        words = re.findall(r'\b\w+\b', eval_text)
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len > 6.0:
                score += 25
                reasons.append("High academic/formal vocabulary density")

        # If stall was detected before a dense response
        if stall_detected and score >= 40:
            score += 20
            reasons.append(f"Opening verbal stall ('{stall_buffer}') preceded dense response")

        score = max(5, min(95, score))
        rationale_str = "; ".join(reasons) if reasons else "Spontaneous speech pattern detected."
        return AILikenessResult(
            score=score,
            rationale=f"[Offline Heuristic] {rationale_str}",
            stall_detected=stall_detected,
            stall_buffer=stall_buffer,
            substantive_core=substantive_core,
            effective_latency_offset_s=latency_offset
        )
