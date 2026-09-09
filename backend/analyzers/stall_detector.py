import re
from typing import Dict, Any, List

class StallDetector:
    """
    Detects latency evasion and verbal stalling tactics where candidates use:
    1. Conversational flattery cushions ("Okay so that is a good question actually...")
    2. Conversational preface fillers ("Okay so like for embeddings...")
    3. Verbal stall loops / phrase repetitions ("so rag is rag is rag is...")
    4. Question parroting / echoing
    to avoid prompt latency detection and trick AI evaluators while waiting for external
    AI tools (ChatGPT, Claude, Perplexity) to stream answers.
    """

    FLATTERY_CUSHION_PATTERNS = [
        re.compile(r"^((o?kay|'kay|k|ok|yeah|well|so)[\s,]+)?(so[\s,]+)?(that('?s|\s+is)|another|this\s+is)?\s*(an?\s+)?(good|great|interesting|fair|nice|cool)\s+question(\s+actually)?(\s*[,.\-—:]*\s*(so|well|basically|essentially))?", re.IGNORECASE),
        re.compile(r"^(well\s+|yeah\s+|yes\s+)?(thanks\s+for\s+asking|i('?m|\s+am)\s+glad\s+you\s+asked|good\s+question|great\s+question)(\s+actually)?(\s*[,.\-—:]*\s*(so|well|basically))?", re.IGNORECASE),
        re.compile(r"^(let\s+me\s+(think|see|break\s+down)|that('?s|\s+is)\s+an\s+interesting\s+one)(\s*[,.\-—:]*\s*(so|well|basically))?", re.IGNORECASE),
    ]

    CONVERSATIONAL_RUNWAY_PATTERNS = [
        # "Okay. So basically what actually happens is...", "So basically what it does is..."
        re.compile(r"^(o?kay|'kay|k|ok|yeah|well|right|so)[\s,]+(so[\s,]+)?(basically[\s,]+)?(what\s+actually\s+happens\s+is|the\s+way\s+it\s+works\s+is|what\s+we\s+do\s+is|what\s+it\s+does\s+is|how\s+it\s+works\s+is)[\s,]*", re.IGNORECASE),
        # "Okay, so RAG actually stands for...", "So RAG basically stands for..."
        re.compile(r"^(o?kay|'kay|k|ok|yeah|well|right)[\s,]+(so[\s,]+)?(\w+[\s,]+)?(actually\s+stands\s+for|basically\s+means|is\s+basically|actually\s+is)[\s,]*", re.IGNORECASE),
        # "So basically, what we start with is...", "Basically what we do is..."
        re.compile(r"^(so[\s,]+)?(basically|actually|essentially)[\s,]+(what\s+we\s+do\s+is|what\s+happens\s+is|the\s+idea\s+is)[\s,]*", re.IGNORECASE)
    ]

    CONVERSATIONAL_PREFACE_PATTERNS = [
        re.compile(r"^(o?kay|'kay|k|ok|yeah|right|well|so)[\s,]+(so[\s,]+)?(like[\s,]+|basically[\s,]+|essentially[\s,]+|actually[\s,]+)+", re.IGNORECASE),
    ]

    @staticmethod
    def detect_phrase_loops(text: str) -> List[Dict[str, Any]]:
        """
        Detects consecutive repeating n-grams (1 to 6 words repeated 2 or more times).
        Returns a list of detected repetition loops.
        """
        if not text or len(text.strip()) == 0:
            return []

        words = text.strip().split()
        if len(words) < 3:
            return []

        clean_words = [re.sub(r'[^\w]', '', w).lower() for w in words]
        loops = []
        n = len(clean_words)

        i = 0
        while i < n:
            best_match = None
            # Check phrase lengths from 6 down to 1
            for phrase_len in range(min(6, (n - i) // 2), 0, -1):
                phrase = clean_words[i:i + phrase_len]
                if not any(phrase):
                    continue

                repeats = 1
                curr_idx = i + phrase_len
                while curr_idx + phrase_len <= n and clean_words[curr_idx:curr_idx + phrase_len] == phrase:
                    repeats += 1
                    curr_idx += phrase_len

                if (phrase_len == 1 and repeats >= 3) or (phrase_len >= 2 and repeats >= 2):
                    original_phrase = " ".join(words[i:i + phrase_len])
                    best_match = {
                        "phrase": original_phrase,
                        "phrase_len": phrase_len,
                        "repeats": repeats,
                        "start_word_idx": i,
                        "end_word_idx": curr_idx,
                        "total_words": phrase_len * repeats
                    }
                    break

            if best_match:
                loops.append(best_match)
                i = best_match["end_word_idx"]
            else:
                i += 1

        return loops

    @classmethod
    def slice_transcript(cls, text: str, question_text: str = "") -> Dict[str, Any]:
        """
        Slices the candidate transcript into:
        1. stall_buffer: Opening repetitive phrases, flattery cushions, or verbal stalling.
        2. substantive_core: The actual technical answer / explanation that follows the stall.

        Returns:
            {
                "stall_detected": bool,
                "stall_buffer": str,
                "substantive_core": str,
                "stall_phrases": List[str],
                "stall_word_count": int,
                "estimated_stall_duration_s": float,
                "camouflage_type": str,
                "raw_text": str
            }
        """
        if not text or not text.strip():
            return {
                "stall_detected": False,
                "stall_buffer": "",
                "substantive_core": "",
                "stall_phrases": [],
                "stall_word_count": 0,
                "estimated_stall_duration_s": 0.0,
                "camouflage_type": "none",
                "raw_text": text
            }

        words = text.strip().split()
        total_words = len(words)
        stall_end_idx = 0
        stall_phrases = []
        camouflage_type = "none"

        # 1. Check for opening flattery cushion ("Okay so that is a good question actually...")
        for pat in cls.FLATTERY_CUSHION_PATTERNS:
            m = pat.match(text.strip())
            if m:
                cushion_str = m.group(0).strip()
                w_count = len(cushion_str.split())
                stall_end_idx = w_count
                stall_phrases.append(f"Conversational flattery cushion: '{cushion_str}'")
                camouflage_type = "flattery_cushion"
                break

        # 2. Check for conversational runway stalls ("Okay. So basically what actually happens is...", "Okay, so RAG actually stands for...")
        if stall_end_idx == 0:
            for pat in cls.CONVERSATIONAL_RUNWAY_PATTERNS:
                m = pat.match(text.strip())
                if m:
                    cushion_str = m.group(0).strip()
                    w_count = len(cushion_str.split())
                    stall_end_idx = w_count
                    stall_phrases.append(f"Conversational runway stall: '{cushion_str}'")
                    camouflage_type = "conversational_runway"
                    break

        # 3. Check for conversational preface buffer if no runway or cushion
        if stall_end_idx == 0:
            for pat in cls.CONVERSATIONAL_PREFACE_PATTERNS:
                m = pat.match(text.strip())
                if m:
                    cushion_str = m.group(0).strip()
                    w_count = len(cushion_str.split())
                    stall_end_idx = w_count
                    stall_phrases.append(f"Conversational preface buffer: '{cushion_str}'")
                    camouflage_type = "conversational_preface"
                    break

        # 3. Check for phrase repetition loops (at start or immediately following cushion)
        loops = cls.detect_phrase_loops(text)
        for loop in loops:
            if loop["start_word_idx"] <= max(4, stall_end_idx + 2):
                stall_end_idx = max(stall_end_idx, loop["end_word_idx"])
                stall_phrases.append(f"Repeated '{loop['phrase']}' x{loop['repeats']}")
                camouflage_type = "verbal_stall_loop" if camouflage_type == "none" else f"{camouflage_type}_plus_loop"
                break

        # 4. Question echoing
        if stall_end_idx == 0 and question_text:
            q_clean = [re.sub(r'[^\w]', '', w).lower() for w in question_text.split()]
            stop_words = {"what", "is", "how", "does", "the", "a", "an", "and", "or", "in", "to", "explain", "describe", "difference", "between", "so", "well", "basically", "like"}
            q_keywords = set(w for w in q_clean if w not in stop_words and len(w) > 2)
            
            if q_keywords:
                first_window = [re.sub(r'[^\w]', '', w).lower() for w in words[:min(12, total_words)]]
                last_echo_idx = 0
                for idx, w in enumerate(first_window):
                    if w in stop_words or w in q_keywords:
                        last_echo_idx = idx + 1
                    else:
                        break

                if last_echo_idx >= 4:
                    stall_end_idx = last_echo_idx
                    stall_phrases.append("Question echoing & opening filler stalling")
                    camouflage_type = "question_echo"

        stall_detected = stall_end_idx > 0 and stall_end_idx < total_words

        if stall_detected:
            stall_buffer = " ".join(words[:stall_end_idx])
            substantive_core = " ".join(words[stall_end_idx:])
            stall_word_count = stall_end_idx
        else:
            stall_buffer = ""
            substantive_core = text.strip()
            stall_word_count = 0

        estimated_stall_duration_s = round(stall_word_count / 2.2, 1) if stall_word_count > 0 else 0.0

        return {
            "stall_detected": stall_detected,
            "stall_buffer": stall_buffer,
            "substantive_core": substantive_core,
            "stall_phrases": stall_phrases,
            "stall_word_count": stall_word_count,
            "estimated_stall_duration_s": estimated_stall_duration_s,
            "camouflage_type": camouflage_type,
            "raw_text": text
        }
