from collections import deque
from typing import Optional
from server.models import DifficultyConfig


class AdaptiveCurriculum:
    def __init__(self):
        self.step = 0
        self.history = deque(maxlen=20)
        self.axes = {
            "lie_subtlety": 0.0,
            "impostor_confidence": 0.0,
            "player_count": 5,
            "impostor_count": 1,
            "red_herrings": 0.0,
        }

    def get_difficulty(self, step: Optional[int] = None) -> DifficultyConfig:
        s = step if step is not None else self.step

        if s < 30:
            return DifficultyConfig(
                lie_subtlety=0.0,
                impostor_confidence=0.2,
                player_count=5,
                impostor_count=1,
                red_herrings=0.0,
                num_rounds=1
            )
        elif s < 80:
            progress = (s - 30) / 50.0
            return DifficultyConfig(
                lie_subtlety=min(0.4, progress * 0.4),
                impostor_confidence=min(0.5, 0.2 + progress * 0.3),
                player_count=5 if progress < 0.5 else 6,
                impostor_count=1,
                red_herrings=min(0.3, progress * 0.3),
                num_rounds=1
            )
        else:
            # Adaptive after step 80
            return DifficultyConfig(
                lie_subtlety=self.axes["lie_subtlety"],
                impostor_confidence=self.axes["impostor_confidence"],
                player_count=int(self.axes["player_count"]),
                impostor_count=int(self.axes["impostor_count"]),
                red_herrings=self.axes["red_herrings"],
                num_rounds=2 if self.axes["lie_subtlety"] > 0.5 else 1
            )

    def update(self, correct: bool):
        self.history.append(1.0 if correct else 0.0)
        self.step += 1

        if len(self.history) < 10:
            return

        accuracy = sum(self.history) / len(self.history)

        if accuracy > 0.75:
            # Increase weakest axis
            weakest = min(
                ["lie_subtlety", "impostor_confidence", "red_herrings"],
                key=lambda k: self.axes[k]
            )
            if weakest in ["lie_subtlety", "impostor_confidence", "red_herrings"]:
                self.axes[weakest] = min(1.0, self.axes[weakest] + 0.1)
            if accuracy > 0.85 and self.axes["player_count"] < 7:
                self.axes["player_count"] = min(7, self.axes["player_count"] + 1)
            if accuracy > 0.90 and self.axes["lie_subtlety"] > 0.6:
                self.axes["impostor_count"] = min(2, self.axes["impostor_count"] + 1)

        elif accuracy < 0.25:
            # Decrease hardest axis
            hardest = max(
                ["lie_subtlety", "impostor_confidence", "red_herrings"],
                key=lambda k: self.axes[k]
            )
            if hardest in ["lie_subtlety", "impostor_confidence", "red_herrings"]:
                self.axes[hardest] = max(0.0, self.axes[hardest] - 0.1)
            if accuracy < 0.1 and self.axes["player_count"] > 5:
                self.axes["player_count"] = max(5, self.axes["player_count"] - 1)
