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

        # Calculate relative pitch variation (Coefficient of Variation)
        # Accounts for natural lower base pitch (male voices ~100-140 Hz naturally have lower Hz spread than female ~200-240 Hz)
        pitch_ratio = (pitch_std / mean_pitch) if mean_pitch > 60.0 else (pitch_std / 140.0)
        is_low_register = 70.0 <= mean_pitch <= 145.0

        # 1. Pitch Variance / Monotone Test (Biological Ground Truth):
        # A true biological flatline drone has pitch std dev < 4.0 Hz or pitch ratio < 3.0%.
        # A calm, deliberate conversational voice (especially in lower register) sits between 5.0 - 10.0 Hz.
        # Expressive human speech modulates with std dev >= 15.0 Hz (or ratio >= 9.0%).
        if pitch_std < 4.0 or pitch_ratio < 0.030:
            prosody_score += 85
            reasons.append(f"Acoustic Flatline Drone: vocal pitch std dev {pitch_std:.1f} Hz (< 4.0 Hz / {pitch_ratio*100:.1f}% ratio)")
        elif pitch_std < 7.5 and not is_low_register:
            prosody_score += 65
            reasons.append(f"Monotone reading drone: vocal pitch std dev {pitch_std:.1f} Hz")
        elif pitch_std < 7.5 and is_low_register:
            prosody_score += 35
            reasons.append(f"Calm low-register pitch: std dev {pitch_std:.1f} Hz ({pitch_ratio*100:.1f}% ratio)")
        elif pitch_std < 14.0:
            prosody_score += 20
            reasons.append(f"Controlled steady inflection: pitch std dev {pitch_std:.1f} Hz")
        elif pitch_std >= 20.0:
            prosody_score = max(0, prosody_score - 25)

        # 2. Cognitive Disfluency & Natural Speech Repair Test:
        total_cognitive_markers = disfluency_count + (self_repair_count * 2)
        if total_cognitive_markers >= 3 and word_count >= 16:
            # Candidate is frequently using thinking fillers ("basically", "actually", "let's suppose")
            # This demonstrates spontaneous cognitive retrieval
            prosody_score = max(10, prosody_score - 30)
            reasons.append(f"Frequent conversational fillers ({total_cognitive_markers} markers) indicate spontaneous formulation")
        elif total_cognitive_markers == 0 and word_count >= 18:
            prosody_score += 25
            reasons.append(f"Zero cognitive fillers across {word_count} words (script recitation pattern)")

        # 3. Teleprompter Stillness Test (Robotic head rigidity while speaking):
        if is_rigid_head or (0 < head_motion_std < 0.0045):
            prosody_score += 20
            reasons.append(f"Rigid Teleprompter Freeze: Lack of head movement (std dev {head_motion_std:.4f} < 0.0045) while reciting")

        prosody_score = min(100, max(0, prosody_score))

        if prosody_score >= 70 or (pitch_std < 4.0 and total_cognitive_markers <= 1):
            vocal_style = "monotone_drone"
            rationale = "Monotone Teleprompter Drone: " + "; ".join(reasons)
        elif prosody_score >= 35 or pitch_std < 12.0:
            vocal_style = "calm_steady"
            rationale = "Calm Steady Voice: " + "; ".join(reasons)
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
