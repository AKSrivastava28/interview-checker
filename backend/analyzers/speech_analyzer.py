import math
from typing import List, Dict, Any

class SpeechAnalyzer:
    @staticmethod
    def analyze_speech_delivery(
        question_start_ts: float,
        transcript_chunks: List[Dict[str, Any]],
        window_end_ts: float = 0.0
    ) -> Dict[str, Any]:
        """
        Analyzes the temporal rhythm of speech to detect reading from external AI/prompts
        versus natural spontaneous conversational thinking.
        
        Returns:
            {
                "prompt_latency_s": float,
                "speaking_duration_s": float,
                "total_words": int,
                "cognitive_pauses_count": int,
                "pauses_per_minute": float,
                "cadence_variance": float,
                "reading_cadence_score": int, # 0 (spontaneous) - 100 (script reading)
                "delivery_style": str,        # "spontaneous" | "suspicious_reading" | "script_reading"
                "rationale": str
            }
        """
        if not transcript_chunks or len(transcript_chunks) < 2:
            return {
                "prompt_latency_s": 0.0,
                "speaking_duration_s": 0.0,
                "total_words": 0,
                "cognitive_pauses_count": 0,
                "pauses_per_minute": 0.0,
                "cadence_variance": 0.0,
                "reading_cadence_score": 0,
                "delivery_style": "non_responsive",
                "rationale": "Non-responsive: No audible candidate speech recorded for this question."
            }

        first_speech_ts = transcript_chunks[0].get("ts", question_start_ts)
        prompt_latency_s = round(max(0.0, first_speech_ts - question_start_ts), 2)
        
        last_speech_ts = transcript_chunks[-1].get("ts", first_speech_ts)
        speaking_duration_s = max(1.0, last_speech_ts - first_speech_ts)

        # Extract incremental word gains and timestamp deltas
        deltas = []
        prev_words = 0
        prev_ts = first_speech_ts
        cognitive_pauses = 0

        for chunk in transcript_chunks:
            cur_words = chunk.get("word_count", 0)
            cur_ts = chunk.get("ts", prev_ts)
            dt = cur_ts - prev_ts

            if cur_words > prev_words:
                words_added = cur_words - prev_words
                if dt > 0.1:
                    rate = words_added / dt  # words per second
                    deltas.append((dt, words_added, rate))

                # Cognitive thinking pause: candidate paused for >= 0.8s between words
                if dt >= 0.8:
                    cognitive_pauses += 1

                prev_words = cur_words
                prev_ts = cur_ts

        total_words = prev_words or len(transcript_chunks[-1].get("text", "").split())
        pauses_per_min = round((cognitive_pauses / speaking_duration_s) * 60.0, 1)

        # Calculate cadence variance (variance of speech rates across chunks)
        rates = [d[2] for d in deltas if d[2] > 0]
        if len(rates) >= 3:
            mean_rate = sum(rates) / len(rates)
            variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.8  # default normal variance

        # SCRIPT READING CHARACTERISTICS:
        # 1. Very few cognitive pauses (< 2.5 pauses/min) during a substantial response (> 25 words)
        # 2. Low cadence variance (monotone metronomic reading speed, std_dev < 0.45)
        # 3. Prompt delay (3.0s - 7.0s) followed by continuous, unbroken recitation
        
        reading_score = 0
        reasons = []

        if total_words >= 20:
            # Pacing test:
            if std_dev < 0.45:
                reading_score += 35
                reasons.append(f"Unnaturally uniform speech cadence (std dev {std_dev:.2f} WPS)")
            elif std_dev < 0.65:
                reading_score += 15

            # Cognitive pause test:
            if pauses_per_min < 2.0 and speaking_duration_s > 10.0:
                reading_score += 40
                reasons.append(f"Near-zero cognitive pauses ({cognitive_pauses} pauses in {speaking_duration_s:.1f}s)")
            elif pauses_per_min < 3.5:
                reading_score += 20
                reasons.append(f"Low thinking pauses ({pauses_per_min}/min)")

            # Prompt latency test:
            if 3.0 <= prompt_latency_s <= 8.0 and reading_score >= 30:
                reading_score += 25
                reasons.append(f"Prompt latency gap ({prompt_latency_s}s) followed by immediate recitation")

            # Positive Spontaneous Speech Verification:
            # If cognitive thinking pauses are clearly present (>= 3 pauses or >= 3.5 pauses/min),
            # this is strong positive physical evidence of spontaneous formulation, not script reading.
            if cognitive_pauses >= 3 or pauses_per_min >= 3.5:
                reading_score = max(0, reading_score - 45)

        reading_score = min(100, max(0, reading_score))

        if reading_score >= 60:
            delivery_style = "script_reading"
            rationale = "Scripted Reading Cadence detected: " + "; ".join(reasons)
        elif reading_score >= 35:
            delivery_style = "suspicious_reading"
            rationale = "Suspiciously smooth delivery: " + "; ".join(reasons)
        else:
            delivery_style = "spontaneous"
            rationale = f"Natural spontaneous thinking delivery ({pauses_per_min} pauses/min, expressive cadence variance)."

        return {
            "prompt_latency_s": prompt_latency_s,
            "speaking_duration_s": round(speaking_duration_s, 1),
            "total_words": total_words,
            "cognitive_pauses_count": cognitive_pauses,
            "pauses_per_minute": pauses_per_min,
            "cadence_variance": round(std_dev, 2),
            "reading_cadence_score": reading_score,
            "delivery_style": delivery_style,
            "rationale": rationale
        }
