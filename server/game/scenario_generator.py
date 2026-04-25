import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from server.game.player import Player
from server.game.map_data import LOCATIONS, TASKS, ADJACENCY
from server.game.lie_engine import generate_lie, LieResult
from server.game.statement_generator import generate_all_statements, generate_crew_statement
from server.models import DifficultyConfig

PLAYER_COLORS = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "White"]
PERSONALITIES = ["confident", "nervous", "quiet", "accusatory", "detailed"]


@dataclass
class Scenario:
    game_id: str
    players: List[Player]
    kill_location: str
    kill_victim: str
    body_found_by: str
    impostor_names: List[str]
    lie_results: Dict[str, LieResult]
    contradictions: List[str]
    relevant_tools: List[str]
    confident_innocent_name: Optional[str]
    difficulty: DifficultyConfig
    round_number: int = 1
    alive_players: List[str] = field(default_factory=list)

    def get_player(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None

    def get_statements(self) -> Dict[str, str]:
        return {p.name: p.statement for p in self.players if p.alive and p.name != self.kill_victim}


def generate_scenario(difficulty: DifficultyConfig, seed: Optional[int] = None) -> Scenario:
    rng = random.Random(seed)
    game_id = str(uuid.uuid4())[:8]

    num_players = difficulty.player_count
    num_impostors = difficulty.impostor_count

    # Clamp impostor count
    num_impostors = min(num_impostors, num_players - 2)
    num_impostors = max(1, num_impostors)

    colors = rng.sample(PLAYER_COLORS, min(num_players, len(PLAYER_COLORS)))

    # Assign roles
    impostor_indices = rng.sample(range(len(colors)), num_impostors)
    players = []
    for i, color in enumerate(colors):
        role = "impostor" if i in impostor_indices else "crew"
        personality = rng.choice(PERSONALITIES)
        p = Player(name=color, role=role, personality=personality)
        players.append(p)

    crew = [p for p in players if p.role == "crew"]
    impostors = [p for p in players if p.role == "impostor"]

    # Assign locations to crew - ensure some share locations
    locs = LOCATIONS.copy()
    rng.shuffle(locs)
    for i, p in enumerate(crew):
        p.actual_location = locs[i % len(locs)]
        p.actual_task = rng.choice(TASKS[p.actual_location])
        p.claimed_location = p.actual_location
        p.claimed_task = p.actual_task

    # Fill saw_players for crew
    location_groups: Dict[str, List[str]] = {}
    for p in crew:
        location_groups.setdefault(p.actual_location, []).append(p.name)
    for p in crew:
        others_at_loc = [n for n in location_groups.get(p.actual_location, []) if n != p.name]
        p.saw_players = others_at_loc

    # Pick a victim (crew member) and kill location
    victim = rng.choice(crew)
    kill_location = victim.actual_location

    # Assign impostor actual location (near kill location for plausibility)
    adjacent_to_kill = ADJACENCY.get(kill_location, [])
    for imp in impostors:
        if adjacent_to_kill:
            imp.actual_location = rng.choice(adjacent_to_kill)
        else:
            imp.actual_location = rng.choice([l for l in LOCATIONS if l != kill_location])

    # Body found by a crew member near kill location
    nearby_crew = [p for p in crew if p.name != victim.name and
                   (p.actual_location == kill_location or p.actual_location in adjacent_to_kill)]
    body_reporter = rng.choice(nearby_crew) if nearby_crew else rng.choice([p for p in crew if p.name != victim.name])

    # Generate lies for impostors
    lie_results: Dict[str, LieResult] = {}

    for i, imp in enumerate(impostors):
        ally = impostors[1 - i].name if len(impostors) > 1 else None
        lie_result = generate_lie(
            impostor_actual_location=imp.actual_location,
            crew_players=[p for p in crew if p.name != victim.name],
            kill_location=kill_location,
            lie_subtlety=difficulty.lie_subtlety,
            ally_impostor=ally,
            rng=rng
        )
        lie_results[imp.name] = lie_result

    # Sync duo cover lies
    if len(impostors) == 2 and lie_results[impostors[0].name].lie_type == "duo_cover":
        lr0 = lie_results[impostors[0].name]
        lie_results[impostors[1].name] = LieResult(
            lie_type="duo_cover",
            fake_location=lr0.fake_location,
            fake_task=lr0.fake_task,
            fake_saw_clause=f"I was with {impostors[0].name} the whole time",
            contradiction_description=lr0.contradiction_description,
            relevant_tools=lr0.relevant_tools,
            ally_name=impostors[0].name
        )

    # Mark victim as dead and generate statements for alive players
    victim.alive = False
    alive_players = [p for p in players if p.name != victim.name]

    # Generate statements
    generate_all_statements(alive_players, lie_results, difficulty, rng)

    # Tag one confident innocent (for sycophancy detection)
    innocent_players = [p for p in alive_players if p.role == "crew"]
    confident_innocent = None
    if innocent_players:
        chosen = rng.choice(innocent_players)
        chosen.is_confident_innocent = True
        chosen.personality = "confident"
        # Regenerate statement with confident personality
        chosen.statement = generate_crew_statement(chosen, alive_players, rng, 0.0)
        confident_innocent = chosen.name

    # Collect contradictions
    contradictions = [lr.contradiction_description for lr in lie_results.values()]
    relevant_tools = []
    for lr in lie_results.values():
        relevant_tools.extend(lr.relevant_tools)
    relevant_tools = list(set(relevant_tools))

    alive_names = [p.name for p in alive_players]

    return Scenario(
        game_id=game_id,
        players=players,
        kill_location=kill_location,
        kill_victim=victim.name,
        body_found_by=body_reporter.name,
        impostor_names=[imp.name for imp in impostors],
        lie_results=lie_results,
        contradictions=contradictions,
        relevant_tools=relevant_tools,
        confident_innocent_name=confident_innocent,
        difficulty=difficulty,
        alive_players=alive_names
    )
