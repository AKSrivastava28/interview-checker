import re
from typing import Dict, Any, List

# Cognitive thinking fillers that genuine speakers use when retrieving memory
FILLER_PATTERNS = [
    r"\b(um|uh|erm|ah)\b",
    r"\b(well|like|you know|i mean|sort of|kind of)\b",
    r"\b(basically|essentially|let me see|let's see)\b"
]

# Self-corrections and repair sequences (repetition or self-editing)
REPAIR_PATTERNS = [
    r"\b(wait|actually|or rather|i mean)\b",
    r"\b(\w{3,})\s+\1\b"  # Repeated word e.g. "when when", "threads threads"
]

class AcousticAnalyzer:
    @staticmethod
    def analyze_prosody_and_disfluencies(
        transcript_text: str,
        acoustic_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyzes vocal prosody (pitch variance / F0 contour) and cognitive disfluencies
        (natural thinking fillers and speech repairs) to detect reading aloud vs spontaneous thinking.
        
        Returns:
            {
                "pitch_std": float,
                "mean_pitch": float,
                "disfluency_count": int,
                "self_repair_count": int,
                "vocal_style": str,       # "expressive" | "monotone_drone" | "unmeasured"
                "prosody_score": int,     # 0 (spontaneous) to 100 (script reading drone)
                "rationale": str
            }
        """
        words = transcript_text.lower().split() if transcript_text else []
        word_count = len(words)

        pitch_std = float(acoustic_features.get("pitch_std", 0.0))
        mean_pitch = float(acoustic_features.get("mean_pitch", 0.0))
        samples_count = int(acoustic_features.get("pitch_samples_count", 0))

        # Count cognitive filler disfluencies
        disfluency_count = 0
        if transcript_text:
            text_lower = transcript_text.lower()
            for pat in FILLER_PATTERNS:
                disfluency_count += len(re.findall(pat, text_lower))

        # Count self-corrections / speech repairs
        self_repair_count = 0
        if transcript_text:
            text_lower = transcript_text.lower()
            for pat in REPAIR_PATTERNS:
                self_repair_count += len(re.findall(pat, text_lower))

        # Default if audio samples are unavailable
        if samples_count < 10 or word_count < 5:
            return {
                "pitch_std": pitch_std,
                "mean_pitch": mean_pitch,
                "disfluency_count": disfluency_count,
                "self_repair_count": self_repair_count,
                "vocal_style": "unmeasured",
                "prosody_score": 0,
                "rationale": "Insufficient speech samples to evaluate acoustic prosody."
            }

        prosody_score = 0
        reasons = []

        is_rigid_head = bool(acoustic_features.get("is_rigid_head", False))
        head_motion_std = float(acoustic_features.get("head_motion_std", 0.015))

        # 1. Pitch Variance / Monotone Test:
        # A normal spontaneous voice has pitch inflections (std dev > 22 Hz).
        # Reading from a teleprompter / notes flattens vocal modulation (std dev < 18 Hz).
        if pitch_std < 14.0:
            prosody_score += 45
            reasons.append(f"Monotone reading drone (vocal pitch std dev {pitch_std:.1f} Hz < 14 Hz)")
        elif pitch_std < 19.0:
            prosody_score += 25
            reasons.append(f"Low vocal inflection (pitch std dev {pitch_std:.1f} Hz)")
        elif pitch_std >= 25.0:
            prosody_score = max(0, prosody_score - 25)

        # 2. Cognitive Disfluency Test:
        # Honest human thinking produces natural fillers and self-corrections
        total_cognitive_markers = disfluency_count + (self_repair_count * 2)
        if total_cognitive_markers == 0 and word_count >= 16:
            prosody_score += 35
            reasons.append(f"Zero cognitive fillers or self-corrections across {word_count} words")
        elif total_cognitive_markers >= 2:
            prosody_score = max(0, prosody_score - 30)

        # 3. Teleprompter Stillness Test (Robotic head rigidity while speaking):
        if is_rigid_head or (0 < head_motion_std < 0.0045):
            prosody_score += 30
            reasons.append(f"Rigid Teleprompter Freeze: Unnatural lack of head movement (std dev {head_motion_std:.4f} < 0.0045) while reciting")

        prosody_score = min(100, max(0, prosody_score))

        if prosody_score >= 45 or pitch_std < 16.0 or (is_rigid_head and total_cognitive_markers == 0):
            vocal_style = "monotone_drone"
            rationale = "Monotone Teleprompter Drone: " + "; ".join(reasons)
        else:
            vocal_style = "expressive"
            rationale = f"Expressive Vocal Modulation ({pitch_std:.1f} Hz pitch std dev, {total_cognitive_markers} cognitive markers)."

        return {
            "pitch_std": pitch_std,
            "mean_pitch": mean_pitch,
            "disfluency_count": disfluency_count,
            "self_repair_count": self_repair_count,
            "vocal_style": vocal_style,
            "prosody_score": prosody_score,
            "rationale": rationale
        }
