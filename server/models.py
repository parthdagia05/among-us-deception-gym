from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Observation(BaseModel):
    game_id: str
    round_number: int
    max_rounds: int
    alive_players: List[str]
    impostor_count: int
    body_found_location: str
    body_found_by: str
    player_statements: Optional[Dict[str, str]] = None
    last_tool_result: Optional[str] = None
    available_tools: List[str]
    investigation_log: List[str] = []
    previous_rounds: Optional[List[Dict]] = None
    turn_number: int
    max_turns: int


class Action(BaseModel):
    action_type: str  # "tool_call" or "vote"
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    vote_target: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class DifficultyConfig(BaseModel):
    lie_subtlety: float = 0.0
    impostor_confidence: float = 0.0
    player_count: int = 5
    impostor_count: int = 1
    red_herrings: float = 0.0
    num_rounds: int = 1
