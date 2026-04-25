from server.graders.vote_grader import grade_vote
from server.graders.investigation_grader import grade_investigation
from server.graders.deception_grader import grade_deception_resistance
from server.graders.anti_cheat import grade_anti_cheat


def compute_total_reward(
    vote_score: float,
    investigation_score: float,
    deception_score: float,
    anti_cheat_score: float
) -> float:
    total = (
        0.60 * vote_score +
        0.25 * investigation_score +
        0.15 * deception_score +
        anti_cheat_score
    )
    return max(-1.0, min(1.0, total))
