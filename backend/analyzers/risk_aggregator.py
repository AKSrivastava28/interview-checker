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
        total_words: int = 0
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Combines multi-modal signals into a normalized risk score (0-100) and risk label.
        Returns: (risk_label, composite_score, breakdown_dict)
        """
        # Normalize prompt latency (e.g. 0-2s = 0, 8s+ = 100)
        pause_norm = min(100.0, max(0.0, (pause_s - 1.5) / 6.5 * 100.0))

        # Gaze offscreen percentage (physical phone/iPad/2nd monitor turns)
        gaze_norm = min(100.0, max(0.0, gaze_offscreen_pct))

        # Desktop/Window infractions (blur + fullscreen exit)
        total_focus_violations = blur_count + (fullscreen_exit_count * 1.5)
        blur_norm = min(100.0, total_focus_violations * 45.0)

        # AI likeness score (0-100)
        ai_norm = float(min(100, max(0, ai_likeness_score)))

        # Speech cadence & acoustic prosody score (0-100)
        # If monotone reading drone is detected, elevate speech score
        effective_speech = float(speech_reading_score)
        if vocal_style == "monotone_drone" or prosody_score >= 50:
            effective_speech = max(effective_speech, float(prosody_score))
        elif vocal_style == "expressive":
            effective_speech = max(0.0, effective_speech - 15.0)

        speech_norm = float(min(100.0, max(0.0, effective_speech)))

        composite_score = (
            WEIGHT_SPEECH_CADENCE * speech_norm +
            WEIGHT_AI * ai_norm +
            WEIGHT_BLUR * blur_norm +
            WEIGHT_GAZE * gaze_norm +
            WEIGHT_PAUSE * pause_norm
        )

        composite_score = round(composite_score, 1)

        # Non-responsive / Empty Answer Evaluation:
        # A candidate who submitted no audible speech or fewer than 4 words did not answer the technical question
        if total_words < 4:
            if gaze_offscreen_pct > 5.0 or total_focus_violations >= 1.0:
                label = "high_risk"
            else:
                label = "suspicious"
        # High-confidence correlated overrides:
        # If candidate reads AI text (speech rhythm/monotone drone and text likeness agree):
        elif (speech_norm >= 55 or prosody_score >= 55) and ai_likeness_score >= 50:
            label = "high_risk"
        elif total_focus_violations >= 2.0 or gaze_offscreen_pct > 40.0:
            label = "high_risk"
        elif speech_norm >= 40 or prosody_score >= 45 or total_focus_violations >= 1.0 or gaze_offscreen_pct > 18.0 or ai_likeness_score >= 60:
            label = "suspicious"
        elif composite_score < CLEAN_MAX_SCORE:
            label = "clean"
        elif composite_score < SUSPICIOUS_MAX_SCORE:
            label = "suspicious"
        else:
            label = "high_risk"

        breakdown = {
            "speech_reading_score": round(speech_norm, 1),
            "prosody_score": float(prosody_score),
            "ai_score": round(ai_norm, 1),
            "focus_violation_score": round(blur_norm, 1),
            "gaze_score": round(gaze_norm, 1),
            "pause_score": round(pause_norm, 1),
            "composite": composite_score
        }

        return label, composite_score, breakdown
