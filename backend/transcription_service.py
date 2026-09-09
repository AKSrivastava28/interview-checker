import logging
import httpx
from backend.config import GROK_API_KEY, GROK_API_BASE

logger = logging.getLogger("transcription_service")

TECHNICAL_VOCABULARY_PROMPT = (
    "Technical engineering interview response. Technical terms: RAG, Retrieval-Augmented Generation, "
    "LLM, Large Language Model, embeddings, vector database, cosine similarity, dot product, "
    "Euclidean distance, approximate nearest neighbor, ANN, Faiss, HNSW, efSearch, precision-latency trade-off, "
    "pre-training, parametric memory, ground truth, hallucinations, chunking, chunk overlap, "
    "multithreading, multiprocessing, CPU bound, I/O bound, concurrency, GIL."
)

class GroqWhisperService:
    def __init__(self, api_key: str = GROK_API_KEY, api_base: str = GROK_API_BASE):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "answer.webm") -> str:
        """
        Transcribes recorded speech using Groq's hosted Whisper-Large-v3-Turbo model.
        Returns high-fidelity transcript with accurate technical vocabulary.
        """
        if not self.api_key or not audio_bytes or len(audio_bytes) < 500:
            logger.warning("Audio bytes empty or Groq API key missing for Whisper transcription.")
            return ""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            files = {
                "file": (filename, audio_bytes, "audio/webm")
            }
            data = {
                "model": "whisper-large-v3-turbo",
                "prompt": TECHNICAL_VOCABULARY_PROMPT,
                "response_format": "json",
                "temperature": "0.0"
            }

            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(
                    f"{self.api_base}/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )
                resp.raise_for_status()
                result = resp.json()
                transcript = result.get("text", "").strip()
                logger.info(f"Groq Whisper transcription success: {transcript[:80]}...")
                return transcript
        except Exception as e:
            logger.error(f"Groq Whisper transcription failed: {e}")
            return ""

whisper_service = GroqWhisperService()
