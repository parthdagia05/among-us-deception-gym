import random
import json
from pathlib import Path
from typing import Optional
from server.game.player import Player
from server.game.map_data import ADJACENCY

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def load_templates():
    with open(TEMPLATES_DIR / "statement_templates.json") as f:
        stmt = json.load(f)
    with open(TEMPLATES_DIR / "lie_templates.json") as f:
        lie = json.load(f)
    with open(TEMPLATES_DIR / "personality_templates.json") as f:
        pers = json.load(f)
    return stmt, lie, pers


def build_saw_clause(player: Player, all_players: list, rng: random.Random) -> str:
    if player.saw_players:
        seen = rng.choice(player.saw_players)
        return f"I saw {seen} there"
    # Check adjacent locations for flavor
    adjacent = ADJACENCY.get(player.actual_location, [])
    other_players = [p for p in all_players if p.name != player.name and p.actual_location in adjacent]
    if other_players and rng.random() < 0.3:
        seen = rng.choice(other_players)
        adj_loc = seen.actual_location
        return f"I think I saw {seen.name} heading toward {adj_loc}"
    return "I was alone the whole time"


def generate_crew_statement(player: Player, all_players: list, rng: random.Random, red_herring_level: float = 0.0) -> str:
    stmt_templates, _, _ = load_templates()
    personality = player.personality
    templates = stmt_templates.get(personality, stmt_templates["confident"])
    template = rng.choice(templates)

    saw_clause = build_saw_clause(player, all_players, rng)

    # Red herring: accusatory player might point at random innocent
    random_player = None
    impostors = [p.name for p in all_players if p.role == "impostor"]
    innocents = [p.name for p in all_players if p.name != player.name and p.name not in impostors]
    if innocents and red_herring_level > 0:
        random_player = rng.choice(innocents)

    other_names = [p.name for p in all_players if p.name != player.name]
    fallback_player = rng.choice(other_names) if other_names else "someone"

    statement = template.format(
        location=player.actual_location.title(),
        task=player.actual_task or "tasks",
        saw_clause=saw_clause,
        random_player=random_player or fallback_player,
        adjacent_location=rng.choice(ADJACENCY.get(player.actual_location, ["somewhere"])).title()
    )
    return statement


def generate_impostor_statement(player: Player, lie_result, all_players: list, rng: random.Random, impostor_confidence: float = 0.5) -> str:
    _, lie_templates, _ = load_templates()
    lie_type = lie_result.lie_type
    templates = lie_templates.get(lie_type, lie_templates["corroboration_gap"])
    template = rng.choice(templates)

    # High confidence = confident personality
    effective_personality = "confident" if impostor_confidence > 0.6 else player.personality

    # Impostors never say alone if they're claiming to have seen someone
    if lie_result.fake_seen_player:
        fake_saw_clause = f"I saw {lie_result.fake_seen_player} there too"
    elif lie_result.ally_name:
        fake_saw_clause = f"I was with {lie_result.ally_name} the whole time"
    else:
        fake_saw_clause = "I was alone"

    statement = template.format(
        fake_location=lie_result.fake_location.title(),
        fake_task=lie_result.fake_task,
        fake_saw_clause=fake_saw_clause,
        ally_name=lie_result.ally_name or "",
        fake_seen_player=lie_result.fake_seen_player or ""
    )

    # Add confidence boosters at high impostor_confidence
    if impostor_confidence > 0.7:
        boosters = ["I'm 100% sure.", "Anyone can verify.", "Ask around - I was there the whole time.", "Check the records."]
        statement += " " + rng.choice(boosters)

    return statement


def generate_all_statements(players: list, lie_results: dict, difficulty, rng: random.Random) -> None:
    for player in players:
        if player.role == "crew":
            player.statement = generate_crew_statement(player, players, rng, difficulty.red_herrings)
        else:
            lie_result = lie_results.get(player.name)
            if lie_result:
                player.statement = generate_impostor_statement(player, lie_result, players, rng, difficulty.impostor_confidence)
                player.claimed_location = lie_result.fake_location
                player.claimed_task = lie_result.fake_task
            else:
                player.statement = generate_crew_statement(player, players, rng)
