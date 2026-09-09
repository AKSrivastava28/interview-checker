from typing import Tuple, Dict

# Multi-signal weight factors (must sum to 1.0)
WEIGHT_SPEECH_CADENCE = 0.30
WEIGHT_AI = 0.30
WEIGHT_BLUR = 0.20
WEIGHT_GAZE = 0.10
WEIGHT_PAUSE = 0.10

# Risk classification thresholds
CLEAN_MAX_SCORE = 38.0
SUSPICIOUS_MAX_SCORE = 68.0

class RiskAggregator:
    @staticmethod
    def compute_risk(
        pause_s: float,
        gaze_offscreen_pct: float,
        blur_count: int,
        ai_likeness_score: int,
        reading_pct: float = 0.0,
        speech_reading_score: int = 0,
        prosody_score: int = 0,
        vocal_style: str = "expressive",
        fullscreen_exit_count: int = 0,
        total_words: int = 0,
        stall_detected: bool = False,
        effective_latency_s: float = 0.0,
        stall_buffer: str = ""
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Combines multi-modal signals into a normalized risk score (0-100) and risk label.
        Returns: (risk_label, composite_score, breakdown_dict)
        """
        # Normalize prompt latency - use effective latency (accounting for verbal stall buffer)
        eval_latency = max(pause_s, effective_latency_s) if effective_latency_s > 0 else pause_s
        pause_norm = min(100.0, max(0.0, (eval_latency - 1.5) / 6.5 * 100.0))

        # Gaze offscreen percentage (physical phone/iPad/2nd monitor turns)
        gaze_norm = min(100.0, max(0.0, gaze_offscreen_pct))

        # Desktop/Window infractions (blur + fullscreen exit)
        total_focus_violations = blur_count + (fullscreen_exit_count * 1.5)
        blur_norm = min(100.0, total_focus_violations * 45.0)

        # AI likeness score (0-100)
        ai_norm = float(min(100, max(0, ai_likeness_score)))

        # Speech cadence & acoustic prosody score (0-100)
        # Speech cadence and acoustic prosody norms (independent observation channels)
        speech_norm = float(min(100.0, max(0.0, speech_reading_score)))
        prosody_norm = float(min(100.0, max(0.0, prosody_score)))

        # Effective speech contribution for composite calculation
        effective_speech = speech_norm
        if vocal_style == "monotone_drone" and prosody_score >= 70:
            effective_speech = max(speech_norm, prosody_norm * 0.85)
        elif vocal_style == "expressive":
            effective_speech = max(0.0, speech_norm - 15.0)

        # If verbal stall was detected, add an explicit integrity weight boost
        stall_boost = 15.0 if stall_detected else 0.0

        composite_score = (
            WEIGHT_SPEECH_CADENCE * effective_speech +
            WEIGHT_AI * ai_norm +
            WEIGHT_BLUR * blur_norm +
            WEIGHT_GAZE * gaze_norm +
            WEIGHT_PAUSE * pause_norm +
            stall_boost
        )

        composite_score = min(100.0, round(composite_score, 1))

        # Cognitive Conceptualization Ratio (CCR): Words delivered per second of conceptualization latency
        ccr = round(total_words / (eval_latency + 0.5), 1) if total_words > 0 else 0.0

        # CONTENT ANCHOR PRINCIPLE:
        # A candidate cannot be guilty of "reading an AI script" if the content being spoken
        # is demonstrably an authentic human explanation (AI Likeness <= 30) with zero desktop infractions
        # AND their speech delivery is authentic/spontaneous (speech_reading_score < 35, zero verbal stalling).
        # A calm voice or fluent delivery with human content is simply personal speaking style.
        is_content_human = (ai_norm <= 30.0 and speech_norm < 35.0 and total_focus_violations == 0 and gaze_norm < 15.0 and not stall_detected)

        # 1. Non-responsive / Empty Answer Evaluation:
        if total_words < 4:
            if gaze_offscreen_pct > 5.0 or total_focus_violations >= 1.0:
                label = "high_risk"
            else:
                label = "suspicious"

        # 2. CONTENT ANCHOR SHIELD (Protects Honest Humans):
        elif is_content_human:
            label = "clean"

        # 3. HIGH-RISK CHEATING PATTERNS (Corroborated Multi-Modal Evidence):
        # A. High AI Likeness corroborated by reading cadence, monotone drone, or stall camouflage
        elif ai_norm >= 45.0 and (speech_norm >= 40.0 or prosody_norm >= 55.0 or stall_detected or vocal_style == "monotone_drone"):
            label = "high_risk"

        # B. High Reading Cadence with elevated AI likeness
        elif speech_norm >= 60.0 and ai_norm >= 35.0:
            label = "high_risk"

        # C. Opening Camouflage/Stall followed by AI/textbook response
        elif stall_detected and (ai_norm >= 35.0 or speech_norm >= 45.0 or prosody_norm >= 60.0):
            label = "high_risk"

        # D. Instant-onset recitation (low latency, long complex response, high reading cadence)
        elif eval_latency < 2.0 and total_words >= 35 and speech_norm >= 65.0 and ai_norm >= 25.0:
            label = "high_risk"

        # E. Desktop & Gaze Infractions
        elif total_focus_violations >= 2.0 or gaze_offscreen_pct > 40.0:
            label = "high_risk"

        # F. Unbroken Reading Cadence with Monotone Drone
        elif speech_norm >= 75.0 and (prosody_norm >= 65.0 or vocal_style == "monotone_drone"):
            label = "high_risk"

        # 4. SUSPICIOUS THRESHOLDS:
        elif (speech_norm >= 45.0 or prosody_norm >= 50.0 or total_focus_violations >= 1.0 or 
              gaze_offscreen_pct > 18.0 or ai_norm >= 40.0 or stall_detected or speech_norm >= 50.0 or
              composite_score >= CLEAN_MAX_SCORE):
            label = "suspicious"

        # 5. CLEAN (Strictly reserved for authentic spontaneous delivery):
        elif speech_norm < 35.0 and ai_norm < 35.0 and composite_score < CLEAN_MAX_SCORE:
            label = "clean"
        else:
            label = "suspicious"

        # CONFIDENCE SCORE ENGINE:
        # Measures statistical concordance across independent multi-modal sensors
        if total_words < 5:
            confidence_pct = 40.0
        elif label == "high_risk":
            agreeing_signals = 0
            if ai_norm >= 45: agreeing_signals += 1
            if prosody_norm >= 60 or vocal_style == "monotone_drone": agreeing_signals += 1
            if speech_norm >= 50: agreeing_signals += 1
            if stall_detected: agreeing_signals += 1
            if total_focus_violations >= 1.0 or gaze_norm >= 20.0: agreeing_signals += 1
            if eval_latency < 2.0 and total_words >= 35 and speech_norm >= 60: agreeing_signals += 1
            confidence_pct = min(98.0, 76.0 + (agreeing_signals * 4.0))
        elif label == "clean":
            clean_factors = 0
            if ai_norm <= 25: clean_factors += 1
            if speech_norm <= 20: clean_factors += 1
            if not stall_detected: clean_factors += 1
            if total_focus_violations == 0: clean_factors += 1
            if gaze_norm < 5.0: clean_factors += 1
            if vocal_style in ["expressive", "calm_steady"]: clean_factors += 1
            confidence_pct = min(98.0, 72.0 + (clean_factors * 4.5))
        else: # suspicious
            confidence_pct = max(55.0, min(85.0, 65.0 + abs(composite_score - 50.0) * 0.4))

        breakdown = {
            "speech_reading_score": round(speech_norm, 1),
            "prosody_score": float(prosody_score),
            "ai_score": round(ai_norm, 1),
            "focus_violation_score": round(blur_norm, 1),
            "gaze_score": round(gaze_norm, 1),
            "pause_score": round(pause_norm, 1),
            "effective_latency_s": round(eval_latency, 1),
            "cognitive_conceptualization_ratio": ccr,
            "verbal_stall_detected": 1.0 if stall_detected else 0.0,
            "confidence_pct": round(confidence_pct, 1),
            "composite": composite_score
        }

        return label, composite_score, breakdown
