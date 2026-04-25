from typing import Dict, List, Optional, Tuple
from collections import deque

LOCATIONS = [
    "CAFETERIA", "WEAPONS", "NAVIGATION",
    "MEDBAY", "ADMIN", "O2",
    "ELECTRICAL", "STORAGE", "REACTOR"
]

TASKS: Dict[str, List[str]] = {
    "CAFETERIA": ["download_data", "empty_garbage"],
    "WEAPONS": ["clear_asteroids", "calibrate_targeting"],
    "NAVIGATION": ["chart_course", "stabilize_steering"],
    "MEDBAY": ["submit_scan", "inspect_sample"],
    "ADMIN": ["swipe_card", "upload_data"],
    "O2": ["clean_filter", "fill_canisters"],
    "ELECTRICAL": ["fix_wiring", "calibrate_distributor"],
    "STORAGE": ["fuel_engines", "empty_chute"],
    "REACTOR": ["start_reactor", "unlock_manifolds"],
}

ADJACENCY: Dict[str, List[str]] = {
    "CAFETERIA": ["WEAPONS", "MEDBAY"],
    "WEAPONS": ["CAFETERIA", "NAVIGATION", "ADMIN"],
    "NAVIGATION": ["WEAPONS", "O2"],
    "MEDBAY": ["CAFETERIA", "ADMIN", "ELECTRICAL"],
    "ADMIN": ["WEAPONS", "MEDBAY", "O2", "STORAGE"],
    "O2": ["NAVIGATION", "ADMIN", "REACTOR"],
    "ELECTRICAL": ["MEDBAY", "STORAGE"],
    "STORAGE": ["ADMIN", "ELECTRICAL", "REACTOR"],
    "REACTOR": ["O2", "STORAGE"],
}


def shortest_path(loc1: str, loc2: str) -> Tuple[List[str], int]:
    if loc1 == loc2:
        return ([loc1], 0)
    queue = deque([(loc1, [loc1])])
    visited = {loc1}
    while queue:
        current, path = queue.popleft()
        for neighbor in ADJACENCY.get(current, []):
            if neighbor == loc2:
                full_path = path + [neighbor]
                return (full_path, len(full_path) - 1)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return ([], -1)


def is_adjacent(loc1: str, loc2: str) -> bool:
    return loc2 in ADJACENCY.get(loc1, [])


def get_distance(loc1: str, loc2: str) -> int:
    _, dist = shortest_path(loc1, loc2)
    return dist
