from typing import Any, Dict
from server.game.map_data import TASKS, ADJACENCY, shortest_path, is_adjacent


def execute_tool(tool_name: str, tool_args: Dict[str, Any], scenario) -> str:
    if tool_name == "read_statements":
        return read_statements(scenario)
    elif tool_name == "check_location_tasks":
        location = tool_args.get("location", "").upper()
        return check_location_tasks(location)
    elif tool_name == "check_player_presence":
        location = tool_args.get("location", "").upper()
        return check_player_presence(location, scenario)
    elif tool_name == "check_adjacency":
        loc1 = tool_args.get("location1", "").upper()
        loc2 = tool_args.get("location2", "").upper()
        return check_adjacency(loc1, loc2)
    elif tool_name == "cross_reference":
        return cross_reference(scenario)
    elif tool_name == "vote":
        player_name = tool_args.get("player_name", "")
        return f"Vote action processed for {player_name}. Use action_type='vote' with vote_target to cast your vote."
    elif tool_name == "skip_vote":
        return "Vote skipped via tool. Use action_type='vote' with vote_target='skip' to formally skip."
    else:
        return f"Unknown tool: {tool_name}. Available: read_statements, check_location_tasks, check_player_presence, check_adjacency, cross_reference"


def read_statements(scenario) -> str:
    lines = ["=== PLAYER STATEMENTS ==="]
    statements = scenario.get_statements()
    for player_name, statement in statements.items():
        lines.append(f"{player_name}: {statement}")
    return "\n".join(lines)


def check_location_tasks(location: str) -> str:
    if location not in TASKS:
        close = [l for l in TASKS.keys() if location.lower() in l.lower()]
        if close:
            location = close[0]
        else:
            return f"Unknown location: {location}. Valid locations: {', '.join(TASKS.keys())}"
    tasks = TASKS[location]
    return f"Valid tasks at {location.title()}: {', '.join(tasks)}"


def check_player_presence(location: str, scenario) -> str:
    if location not in TASKS:
        close = [l for l in TASKS.keys() if location.lower() in l.lower()]
        if close:
            location = close[0]
        else:
            return f"Unknown location: {location}"

    present = []
    for p in scenario.players:
        if p.name in scenario.alive_players and p.actual_location == location:
            present.append(p.name)

    if not present:
        return f"Players actually present in {location.title()}: Nobody was detected in {location.title()}."
    elif len(present) == 1:
        return f"Players actually present in {location.title()}: {present[0]}. No other players were in {location.title()}."
    else:
        return f"Players actually present in {location.title()}: {', '.join(present)}."


def check_adjacency(loc1: str, loc2: str) -> str:
    if loc1 not in TASKS or loc2 not in TASKS:
        return f"Unknown location(s). Valid: {', '.join(TASKS.keys())}"

    adjacent = is_adjacent(loc1, loc2)
    path, dist = shortest_path(loc1, loc2)

    if adjacent:
        return f"{loc1.title()} and {loc2.title()}: ADJACENT (directly connected)."
    else:
        if path:
            path_str = " -> ".join(l.title() for l in path)
            return f"{loc1.title()} and {loc2.title()}: NOT adjacent. Shortest path: {path_str} ({dist} steps)"
        else:
            return f"{loc1.title()} and {loc2.title()}: NOT adjacent. No path found."


def cross_reference(scenario) -> str:
    lines = ["=== CROSS-REFERENCE ANALYSIS ===", ""]

    contradictions = []
    confirmed = []
    seen_confirmed = set()

    alive = {p.name: p for p in scenario.players if p.name in scenario.alive_players}

    claimed_locations: Dict[str, str] = {}
    claimed_tasks: Dict[str, str] = {}

    for name, p in alive.items():
        claimed_locations[name] = (p.claimed_location or p.actual_location).upper()
        claimed_tasks[name] = p.claimed_task or p.actual_task or ""

    actual_locations: Dict[str, str] = {name: p.actual_location for name, p in alive.items()}

    # 1. Task impossibility: claimed task not valid at claimed location
    for name in alive:
        claimed_loc = claimed_locations[name]
        claimed_task = claimed_tasks[name]
        valid_tasks = TASKS.get(claimed_loc, [])
        if claimed_task and valid_tasks and claimed_task not in valid_tasks:
            contradictions.append(
                f"- {name} claims task '{claimed_task}' in {claimed_loc.title()}, "
                f"but valid tasks there are: {', '.join(valid_tasks)}"
            )

    # 2. Location clash: group players by claimed location
    location_claimants: Dict[str, list] = {}
    for name, loc in claimed_locations.items():
        location_claimants.setdefault(loc, []).append(name)

    for loc, claimants in location_claimants.items():
        actual_at_loc = [n for n, p in alive.items() if p.actual_location == loc]

        for claimer in claimants:
            if claimer not in actual_at_loc:
                # Claimer says they were here but weren't
                if actual_at_loc:
                    # Someone else was actually here - do they say they were alone?
                    for actual_person in actual_at_loc:
                        actual_p = alive.get(actual_person)
                        if actual_p and not actual_p.saw_players:
                            contradictions.append(
                                f"- {claimer} claims {loc.title()}, but {actual_person} "
                                f"was in {loc.title()} and says they were alone"
                            )
                        else:
                            contradictions.append(
                                f"- {claimer} claims {loc.title()}, but presence data "
                                f"shows {claimer} was not detected there"
                            )
                else:
                    contradictions.append(
                        f"- {claimer} claims {loc.title()}, but nobody was "
                        f"actually detected in {loc.title()}"
                    )
            else:
                # Claimer was actually there - check for mutual confirmation
                for co in claimants:
                    if co != claimer and co in actual_at_loc:
                        pair = tuple(sorted([claimer, co]))
                        if pair not in seen_confirmed:
                            seen_confirmed.add(pair)
                            confirmed.append(
                                f"- {claimer} and {co} both claim {loc.title()} "
                                f"(CONFIRMED: both were present)"
                            )

    # 3. Timing contradictions: seeing someone who was elsewhere
    for name, p in alive.items():
        lr = scenario.lie_results.get(name)
        if lr and lr.fake_seen_player:
            fake_seen = lr.fake_seen_player
            fake_loc = lr.fake_location.upper()
            if fake_seen in alive:
                seen_claimed_loc = claimed_locations.get(fake_seen, "")
                if seen_claimed_loc and seen_claimed_loc != fake_loc:
                    contradictions.append(
                        f"- {name} claims to have seen {fake_seen} in {fake_loc.title()}, "
                        f"but {fake_seen} claims to have been in {seen_claimed_loc.title()}"
                    )

    # Deduplicate contradictions
    seen = set()
    deduped = []
    for c in contradictions:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    lines.append("CONTRADICTIONS FOUND:")
    if deduped:
        lines.extend(deduped)
    else:
        lines.append("  None detected from statement cross-check. Use check_player_presence for ground truth.")

    lines.append("")
    lines.append("CONFIRMED ALIBIS:")
    if confirmed:
        lines.extend(confirmed)
    else:
        lines.append("  No mutually confirmed alibis found.")

    lines.append("")
    lines.append("TIP: Use check_player_presence(location) to verify who was actually at any location.")
    return "\n".join(lines)
