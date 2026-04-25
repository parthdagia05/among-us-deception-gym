from typing import Dict, Any, List


def grade_anti_cheat(
    tool_calls: List[Dict],
    vote_actions: int,
    turn_number: int,
    max_turns: int
) -> Dict[str, Any]:
    score = 0.0
    flags = {}

    if len(tool_calls) == 0 and vote_actions > 0:
        score -= 0.2
        flags["no_tools_before_vote"] = True

    if vote_actions > 1:
        score -= 0.1 * (vote_actions - 1)
        flags["multiple_votes"] = vote_actions

    if turn_number >= max_turns and vote_actions == 0:
        score -= 0.3
        flags["timeout_no_vote"] = True

    return {"anti_cheat_score": score, "flags": flags}
