from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Player:
    name: str
    role: str  # "crew" or "impostor"
    alive: bool = True
    actual_location: str = ""
    actual_task: Optional[str] = None
    claimed_location: str = ""
    claimed_task: str = ""
    saw_players: List[str] = field(default_factory=list)
    personality: str = "confident"
    statement: str = ""
    is_confident_innocent: bool = False
