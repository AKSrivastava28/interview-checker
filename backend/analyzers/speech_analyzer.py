import math
from typing import List, Dict, Any
from backend.analyzers.stall_detector import StallDetector

class SpeechAnalyzer:
    @staticmethod
    def analyze_speech_delivery(
        question_start_ts: float,
        transcript_chunks: List[Dict[str, Any]],
        window_end_ts: float = 0.0,
        full_transcript: str = "",
        question_text: str = ""
    ) -> Dict[str, Any]:
        """
        Analyzes the temporal rhythm of speech to detect reading from external AI/prompts
        versus natural spontaneous conversational thinking.
        Detects verbal stall tactics (repeating phrases) used to evade latency timers.
        
        Returns:
            {
                "prompt_latency_s": float,
                "effective_latency_s": float,
                "speaking_duration_s": float,
                "total_words": int,
                "cognitive_pauses_count": int,
                "stalling_pauses_count": int,
                "pauses_per_minute": float,
                "cadence_variance": float,
                "reading_cadence_score": int, # 0 (spontaneous) - 100 (script reading)
                "delivery_style": str,        # "spontaneous" | "suspicious_reading" | "script_reading"
                "stall_detected": bool,
                "stall_buffer": str,
                "rationale": str
            }
        """
        if not transcript_chunks or len(transcript_chunks) < 2:
            return {
                "prompt_latency_s": 0.0,
                "effective_latency_s": 0.0,
                "speaking_duration_s": 0.0,
                "total_words": 0,
                "cognitive_pauses_count": 0,
                "stalling_pauses_count": 0,
                "pauses_per_minute": 0.0,
                "cadence_variance": 0.0,
                "reading_cadence_score": 0,
                "delivery_style": "non_responsive",
                "stall_detected": False,
                "stall_buffer": "",
                "rationale": "Non-responsive: No audible candidate speech recorded for this question."
            }

        if not full_transcript and transcript_chunks:
            full_transcript = transcript_chunks[-1].get("text", "")

        first_speech_ts = transcript_chunks[0].get("ts", question_start_ts)
        prompt_latency_s = round(max(0.0, first_speech_ts - question_start_ts), 2)
        
        last_speech_ts = transcript_chunks[-1].get("ts", first_speech_ts)
        speaking_duration_s = max(1.0, last_speech_ts - first_speech_ts)

        # Detect verbal stall loops in the transcript
        slice_info = StallDetector.slice_transcript(full_transcript, question_text)
        stall_detected = slice_info["stall_detected"]
        stall_buffer = slice_info["stall_buffer"]
        stall_word_count = slice_info["stall_word_count"]
        estimated_stall_duration_s = slice_info["estimated_stall_duration_s"]

        # Extract incremental word gains and timestamp deltas
        deltas = []
        prev_words = 0
        prev_ts = first_speech_ts
        cognitive_pauses = 0
        stalling_pauses = 0

        for chunk in transcript_chunks:
            cur_words = chunk.get("word_count") or len(chunk.get("text", "").split())
            cur_ts = chunk.get("ts", prev_ts)
            dt = cur_ts - prev_ts

            if cur_words > prev_words:
                words_added = cur_words - prev_words
                if dt > 0.1:
                    rate = words_added / dt  # words per second
                    deltas.append((dt, words_added, rate))

                # Cognitive thinking pause: calculate silent gap between speech chunks
                # Pronunciation time at conversational pace (~2.8 WPS) subtracted from dt
                pronounce_est = words_added / 2.8
                silent_gap = dt - pronounce_est

                # A true pause requires a genuine silent hesitation >= 0.85s.
                # Regular 1.0s browser interim packet arrival intervals are NOT candidate pauses.
                if silent_gap >= 0.85:
                    # Distinguish stalling pauses from genuine cognitive thinking pauses
                    if stall_detected and cur_words <= stall_word_count:
                        stalling_pauses += 1
                    else:
                        cognitive_pauses += 1

                prev_words = cur_words
                prev_ts = cur_ts

        total_words = prev_words or len(full_transcript.split())
        substantive_words = max(0, total_words - stall_word_count)
        pauses_per_min = round((cognitive_pauses / speaking_duration_s) * 60.0, 1)

        # Effective latency: raw latency + time spent stalling
        effective_latency_s = round(prompt_latency_s + estimated_stall_duration_s, 2)

        # Calculate cadence variance (variance of speech rates across chunks)
        rates = [d[2] for d in deltas if d[2] > 0]
        if len(rates) >= 3:
            mean_rate = sum(rates) / len(rates)
            variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.8  # default normal variance

        # SCRIPT READING & STALL DETECTION EVALUATION
        reading_score = 0
        reasons = []

        if total_words >= 18:
            # 1. Pacing test:
            if std_dev < 0.45:
                reading_score += 35
                reasons.append(f"Unnaturally uniform speech cadence (std dev {std_dev:.2f} WPS)")
            elif std_dev < 0.65:
                reading_score += 15

            # 2. Cognitive pause test on substantive explanation:
            if (pauses_per_min < 1.5 and speaking_duration_s > 6.0) or (cognitive_pauses == 0 and total_words >= 25):
                reading_score += 40
                reasons.append(f"Near-zero cognitive pauses ({cognitive_pauses} pauses in {speaking_duration_s:.1f}s)")
            elif pauses_per_min < 3.0:
                reading_score += 20
                reasons.append(f"Low thinking pauses ({pauses_per_min}/min)")

            # 3. Prompt latency & onset test:
            if 3.0 <= prompt_latency_s <= 8.0 and reading_score >= 25:
                reading_score += 25
                reasons.append(f"Prompt latency gap ({prompt_latency_s}s) followed by immediate recitation")
            elif prompt_latency_s < 2.0 and total_words >= 40 and cognitive_pauses == 0:
                reading_score += 25
                reasons.append(f"Immediate onset ({prompt_latency_s}s) with continuous recitation of {total_words} words without conceptualization delay")

            # 4. VERBAL STALL & PHASE SHIFT DETECTION:
            # If candidate stalled/cushioned to evade latency, then launched into substantive recitation
            if stall_detected:
                reasons.append(f"Opening Verbal Camouflage/Stall: '{stall_buffer}' ({estimated_stall_duration_s}s)")
                if substantive_words >= 15:
                    reading_score += 50
                    if cognitive_pauses <= 2:
                        reading_score += 35
                        reasons.append(f"Stall-to-Recitation Phase Shift: Unbroken delivery of {substantive_words} substantive words after opening buffer")

            # 5. Genuine Spontaneous Thinking vs Fluency Verification:
            # Check for conversational discourse markers ("let's suppose", "basically", "so what it does")
            discourse_markers = ["let's suppose", "let's say", "basically", "actually", "what happens is", "so that is why", "you have some", "let me explain"]
            matched_discourse = [m for m in discourse_markers if m in full_transcript.lower()]

            if not stall_detected:
                # Check if candidate delivered an unbroken recitation (>= 25 words with zero or near-zero pauses)
                if cognitive_pauses == 0 and total_words >= 25:
                    reading_score += 45
                    reasons.append(f"Unbroken Recitation: Continuous delivery of {total_words} words with 0 cognitive thinking pauses")
                    if len(matched_discourse) >= 2:
                        # Surface casual fillers in an unbroken script recitation
                        reasons.append(f"Scripted colloquialisms: Casual fillers ({', '.join(matched_discourse[:3])}) read without spontaneous pause hesitations")
                elif cognitive_pauses >= 1 and len(matched_discourse) >= 2 and total_words >= 20:
                    # Genuine conversational fluency: discourse markers accompanied by natural thinking pauses
                    reading_score = min(25, max(5, reading_score - 40))
                    reasons.append(f"Conversational Fluency: Natural discourse markers ({', '.join(matched_discourse[:3])}) with {cognitive_pauses} cognitive thinking pause(s)")
                elif cognitive_pauses >= 3 or (cognitive_pauses >= 2 and pauses_per_min >= 3.0 and speaking_duration_s >= 10.0):
                    reading_score = max(0, reading_score - 40)
                elif cognitive_pauses <= 1 and total_words >= 18:
                    reading_score += 30
                    reasons.append(f"Unbroken recitation with only {cognitive_pauses} pause(s) across {total_words} words")
            else:
                # Stall detected: if followed by unbroken substantive words, lock high reading score
                if substantive_words >= 15 and cognitive_pauses <= 1:
                    reading_score = max(75, reading_score)

        reading_score = min(100, max(0, reading_score))

        if reading_score >= 60:
            delivery_style = "script_reading"
            rationale = "Scripted Reading Cadence detected: " + "; ".join(reasons)
        elif reading_score >= 35:
            delivery_style = "suspicious_reading"
            rationale = "Suspiciously smooth delivery: " + "; ".join(reasons)
        elif len(matched_discourse) >= 2 and (cognitive_pauses >= 1 or std_dev >= 0.50):
            delivery_style = "spontaneous_fluent"
            rationale = "Spontaneous Conversational Fluency: " + "; ".join(reasons)
        else:
            delivery_style = "spontaneous"
            rationale = f"Natural spontaneous thinking delivery ({pauses_per_min} pauses/min, expressive cadence variance)."

        return {
            "prompt_latency_s": prompt_latency_s,
            "effective_latency_s": effective_latency_s,
            "speaking_duration_s": round(speaking_duration_s, 1),
            "total_words": total_words,
            "cognitive_pauses_count": cognitive_pauses,
            "stalling_pauses_count": stalling_pauses,
            "pauses_per_minute": pauses_per_min,
            "cadence_variance": round(std_dev, 2),
            "reading_cadence_score": reading_score,
            "delivery_style": delivery_style,
            "stall_detected": stall_detected,
            "stall_buffer": stall_buffer,
            "rationale": rationale
        }
