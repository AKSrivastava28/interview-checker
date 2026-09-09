import logging
import httpx
from typing import List
from backend.config import GROK_API_KEY, GROK_API_BASE

logger = logging.getLogger("question_engine")

HARDCODED_QUESTIONS = [
    "What is RAG (Retrieval-Augmented Generation) and how does it work?",
    "What is multithreading, and how does it differ from multiprocessing?",
    "Explain the difference between synchronous and asynchronous programming, and when you would use each.",
    "Describe a time you had to optimize a slow application. What tools did you use and what was the outcome?",
    "What is your approach to writing clean, maintainable code, and how do you ensure code quality in a team?"
]

class QuestionEngine:
    def __init__(self, api_key: str = GROK_API_KEY, api_base: str = GROK_API_BASE):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    def get_question(self, index: int) -> str:
        """
        Returns the hardcoded question for the given 0-based index.
        """
        if index < len(HARDCODED_QUESTIONS):
            return HARDCODED_QUESTIONS[index]
        return ""

    async def generate_dynamic_question(self, question_history: List[str], answer_history: List[str]) -> str:
        """
        Calls Grok to generate a dynamic next question based on the previous conversation history.
        """
        if not self.api_key:
            # Fallback to hardcoded list if API key is not configured
            next_idx = len(question_history)
            return self.get_question(next_idx)

        try:
            messages = [
                {"role": "system", "content": "You are a senior engineering interviewer. Ask the candidate a concise, single, clear technical or behavioral interview question. Avoid preambles. Just ask the question."}
            ]
            for q, a in zip(question_history, answer_history):
                messages.append({"role": "assistant", "content": q})
                messages.append({"role": "user", "content": a})

            messages.append({"role": "user", "content": "Generate the next technical/behavioral interview question. Make it short and direct."})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "qwen/qwen3.8-27b",
                "messages": messages,
                "temperature": 0.7
            }

            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                question = data["choices"][0]["message"]["content"].strip()
                return question
        except Exception as e:
            logger.error(f"Failed to generate dynamic question with Grok: {e}. Falling back to default list.")
            next_idx = len(question_history)
            return self.get_question(next_idx)

    async def generate_drilldown_question(self, last_question: str, last_answer: str) -> str:
        """
        Calls Grok to generate an immediate conversational drill-down follow-up question
        based specifically on the technical points the candidate just stated.
        """
        if not self.api_key or len(last_answer.split()) < 5:
            return ""

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert technical interviewer in a live senior engineering interview. "
                        "The candidate just provided an answer. Ask an immediate, unscripted follow-up question "
                        "that specifically drills down into a technical detail, trade-off, or architectural choice they just made. "
                        "Keep it to exactly one direct question without preamble or praise. Do not say 'Great answer' or 'Thank you'."
                    )
                },
                {"role": "assistant", "content": last_question},
                {"role": "user", "content": last_answer}
            ]
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "qwen/qwen3.8-27b",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 80
            }

            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                drilldown = data["choices"][0]["message"]["content"].strip()
                logger.info(f"Generated conversational drilldown follow-up: {drilldown}")
                return drilldown
        except Exception as e:
            logger.warning(f"Failed to generate drilldown question: {e}")
            return ""

# Global singleton
question_engine = QuestionEngine()
