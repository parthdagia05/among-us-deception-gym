import requests
from typing import Optional, Dict, Any
from server.models import Observation, Action, StepResult, DifficultyConfig


class AmongUsClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, difficulty: Optional[DifficultyConfig] = None) -> Observation:
        payload = difficulty.model_dump() if difficulty else None
        resp = requests.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return Observation.model_validate(resp.json())

    def step(self, action: Action) -> StepResult:
        resp = requests.post(f"{self.base_url}/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()
