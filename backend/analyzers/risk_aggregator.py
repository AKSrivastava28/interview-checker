from typing import Tuple, Dict

# Easily tunable weight factors (must sum to 1.0)
WEIGHT_GAZE = 0.35
WEIGHT_AI = 0.35
WEIGHT_PAUSE = 0.20
WEIGHT_BLUR = 0.10

# Risk classification thresholds
CLEAN_MAX_SCORE = 40.0
SUSPICIOUS_MAX_SCORE = 70.0

class RiskAggregator:
    @staticmethod
    def compute_risk(
        pause_s: float,
        gaze_offscreen_pct: float,
        blur_count: int,
        ai_likeness_score: int
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Combines multi-modal signals into a normalized risk score (0-100) and risk label.
        Returns: (risk_label, composite_score, breakdown_dict)
        """
        # Normalize pause duration (e.g. 0-2s = 0, 8s+ = 100)
        pause_norm = min(100.0, max(0.0, (pause_s - 1.5) / 6.5 * 100.0))

        # Gaze offscreen percentage is already 0 - 100
        gaze_norm = min(100.0, max(0.0, gaze_offscreen_pct))

        # Tab blur count normalization (0 = 0, 1 = 50, 2+ = 100)
        blur_norm = min(100.0, blur_count * 50.0)

        # AI likeness score is already 0 - 100
        ai_norm = float(min(100, max(0, ai_likeness_score)))

        composite_score = (
            WEIGHT_GAZE * gaze_norm +
            WEIGHT_AI * ai_norm +
            WEIGHT_PAUSE * pause_norm +
            WEIGHT_BLUR * blur_norm
        )

        composite_score = round(composite_score, 1)

        # Security overrides: Elevate risk status directly if a single signal is highly suspicious
        if pause_s > 18.0 or gaze_offscreen_pct > 65.0:
            label = "high_risk"
        elif pause_s > 10.0 or gaze_offscreen_pct > 35.0 or blur_count >= 1:
            label = "suspicious"
        elif composite_score < CLEAN_MAX_SCORE:
            label = "clean"
        elif composite_score < SUSPICIOUS_MAX_SCORE:
            label = "suspicious"
        else:
            label = "high_risk"

        breakdown = {
            "gaze_score": round(gaze_norm, 1),
            "ai_score": round(ai_norm, 1),
            "pause_score": round(pause_norm, 1),
            "blur_score": round(blur_norm, 1),
            "composite": composite_score
        }

        return label, composite_score, breakdown
