import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from server.game.map_data import TASKS, LOCATIONS, get_distance, is_adjacent


@dataclass
class LieResult:
    lie_type: str
    fake_location: str
    fake_task: str
    fake_saw_clause: str
    contradiction_description: str
    relevant_tools: List[str]
    ally_name: Optional[str] = None
    fake_seen_player: Optional[str] = None


def pick_lie_type(lie_subtlety: float, has_duo: bool = False, rng: Optional[random.Random] = None) -> str:
    _rng = rng or random
    if lie_subtlety < 0.2:
        return _rng.choice(["location_clash", "task_impossibility"])
    elif lie_subtlety < 0.4:
        return _rng.choice(["location_clash", "task_impossibility", "timing_contradiction"])
    elif lie_subtlety < 0.6:
        return _rng.choice(["location_clash", "timing_contradiction", "kill_proximity"])
    elif lie_subtlety < 0.8:
        return _rng.choice(["kill_proximity", "corroboration_gap"])
    else:
        if has_duo:
            return _rng.choice(["corroboration_gap", "duo_cover"])
        return "corroboration_gap"


def generate_lie(
    impostor_actual_location: str,
    crew_players: List[Any],
    kill_location: str,
    lie_subtlety: float = 0.0,
    ally_impostor: Optional[str] = None,
    rng: Optional[random.Random] = None
) -> LieResult:
    if rng is None:
        rng = random.Random()

    has_duo = ally_impostor is not None
    lie_type = pick_lie_type(lie_subtlety, has_duo, rng)

    crew_locations = {p.name: p.actual_location for p in crew_players}
    crew_by_location: Dict[str, List[str]] = {}
    for p in crew_players:
        crew_by_location.setdefault(p.actual_location, []).append(p.name)

    if lie_type == "location_clash":
        # Find a location where exactly one crew member was alone (must differ from impostor actual loc)
        solo_locations = [
            loc for loc, players in crew_by_location.items()
            if len(players) == 1 and loc != impostor_actual_location
        ]
        if solo_locations:
            fake_location = rng.choice(solo_locations)
            alone_crew = crew_by_location[fake_location][0]
        else:
            fake_location = rng.choice([l for l in LOCATIONS if l != impostor_actual_location])
            alone_crew = None
        fake_task = rng.choice(TASKS[fake_location])
        fake_saw_clause = "I was alone the whole time"
        contradiction = f"Impostor claims {fake_location}, but {alone_crew or 'crew'} was in {fake_location} and says they were alone"
        relevant_tools = [f"check_player_presence({fake_location})"]

    elif lie_type == "task_impossibility":
        fake_location = rng.choice([l for l in LOCATIONS if l != impostor_actual_location])
        # Pick a task from a DIFFERENT location
        other_locations = [l for l in LOCATIONS if l != fake_location]
        wrong_location = rng.choice(other_locations)
        fake_task = rng.choice(TASKS[wrong_location])
        fake_saw_clause = "I was alone"
        contradiction = f"Impostor claims task '{fake_task}' in {fake_location}, but '{fake_task}' is only available in {wrong_location}"
        relevant_tools = [f"check_location_tasks({fake_location})"]

    elif lie_type == "timing_contradiction":
        # Pick a crew member and claim to have seen them somewhere they weren't
        if crew_players:
            fake_seen = rng.choice(crew_players)
            available_locs = [l for l in LOCATIONS if l != impostor_actual_location and l != fake_seen.actual_location]
            if not available_locs:
                available_locs = [l for l in LOCATIONS if l != impostor_actual_location]
            fake_location = rng.choice(available_locs)
            fake_task = rng.choice(TASKS[fake_location])
            fake_saw_clause = f"I saw {fake_seen.name} there too"
            contradiction = f"Impostor claims to have seen {fake_seen.name} in {fake_location}, but {fake_seen.name} was actually in {fake_seen.actual_location}"
            relevant_tools = [f"check_player_presence({fake_location})", "cross_reference()"]
            return LieResult(
                lie_type=lie_type,
                fake_location=fake_location,
                fake_task=fake_task,
                fake_saw_clause=fake_saw_clause,
                contradiction_description=contradiction,
                relevant_tools=relevant_tools,
                fake_seen_player=fake_seen.name
            )
        else:
            fake_location = rng.choice([l for l in LOCATIONS if l != impostor_actual_location])
            fake_task = rng.choice(TASKS[fake_location])
            fake_saw_clause = "I was alone"
            contradiction = f"Impostor claims {fake_location} with no corroboration"
            relevant_tools = [f"check_player_presence({fake_location})"]

    elif lie_type == "kill_proximity":
        # Claim to be far from the kill location
        far_locations = [l for l in LOCATIONS if get_distance(l, kill_location) >= 3]
        if not far_locations:
            far_locations = [l for l in LOCATIONS if get_distance(l, kill_location) >= 2 and l != kill_location]
        if not far_locations:
            far_locations = [l for l in LOCATIONS if l != kill_location and l != impostor_actual_location]
        fake_location = rng.choice(far_locations) if far_locations else rng.choice(LOCATIONS)
        fake_task = rng.choice(TASKS[fake_location])
        fake_saw_clause = "I was alone, working on tasks"
        dist = get_distance(fake_location, kill_location)
        contradiction = f"Impostor claims to be in {fake_location} ({dist} steps from kill at {kill_location}), but was actually at {impostor_actual_location}"
        relevant_tools = [f"check_adjacency({fake_location}, {kill_location})", f"check_player_presence({fake_location})"]

    elif lie_type == "corroboration_gap":
        # Claim a location where no one else was (hardest to disprove directly)
        empty_locations = [l for l in LOCATIONS if l not in crew_by_location and l != impostor_actual_location]
        if not empty_locations:
            empty_locations = [l for l in LOCATIONS if l != impostor_actual_location]
        fake_location = rng.choice(empty_locations)
        fake_task = rng.choice(TASKS[fake_location])
        fake_saw_clause = "No one else was there"
        # Add kill proximity as additional evidence
        dist_from_kill = get_distance(fake_location, kill_location)
        real_dist = get_distance(impostor_actual_location, kill_location)
        contradiction = f"Impostor claims {fake_location} (no one to confirm). Also suspicious: {fake_location} is {dist_from_kill} steps from kill at {kill_location}"
        relevant_tools = [f"check_player_presence({fake_location})", "cross_reference()"]

    elif lie_type == "duo_cover":
        # Both impostors claim to be together at a fake location
        available_locs = [l for l in LOCATIONS if l not in crew_by_location and l != impostor_actual_location]
        if not available_locs:
            available_locs = [l for l in LOCATIONS if l != impostor_actual_location]
        fake_location = rng.choice(available_locs)
        fake_task = rng.choice(TASKS[fake_location])
        fake_saw_clause = f"I was with {ally_impostor} the whole time"
        contradiction = f"Impostor claims to be with {ally_impostor} in {fake_location}, but check_player_presence shows neither was there"
        relevant_tools = [f"check_player_presence({fake_location})", "cross_reference()"]
        return LieResult(
            lie_type=lie_type,
            fake_location=fake_location,
            fake_task=fake_task,
            fake_saw_clause=fake_saw_clause,
            contradiction_description=contradiction,
            relevant_tools=relevant_tools,
            ally_name=ally_impostor
        )
    else:
        fake_location = rng.choice([l for l in LOCATIONS if l != impostor_actual_location])
        fake_task = rng.choice(TASKS[fake_location])
        fake_saw_clause = "I was alone"
        contradiction = "Generic lie"
        relevant_tools = [f"check_player_presence({fake_location})"]

    return LieResult(
        lie_type=lie_type,
        fake_location=fake_location,
        fake_task=fake_task,
        fake_saw_clause=fake_saw_clause,
        contradiction_description=contradiction,
        relevant_tools=relevant_tools
    )
