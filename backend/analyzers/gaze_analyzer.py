from typing import List

# Center screen bounding box boundaries (normalized 0.0 - 1.0)
DEFAULT_CENTER_X_MIN = 0.25
DEFAULT_CENTER_X_MAX = 0.75
DEFAULT_CENTER_Y_MIN = 0.20
DEFAULT_CENTER_Y_MAX = 0.80

class GazeAnalyzer:
    def __init__(
        self,
        x_min: float = DEFAULT_CENTER_X_MIN,
        x_max: float = DEFAULT_CENTER_X_MAX,
        y_min: float = DEFAULT_CENTER_Y_MIN,
        y_max: float = DEFAULT_CENTER_Y_MAX,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def calculate_offscreen_percentage(self, gaze_samples: List[dict]) -> float:
        """
        Calculates percentage of gaze samples outside the defined central vision box.
        """
        if not gaze_samples:
            return 0.0

        offscreen_count = 0
        for sample in gaze_samples:
            x = sample.get("x", 0.5)
            y = sample.get("y", 0.5)
            if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
                offscreen_count += 1

        pct = (offscreen_count / len(gaze_samples)) * 100.0
        return round(pct, 1)
