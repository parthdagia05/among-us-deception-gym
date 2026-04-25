"""
Multi-agent Among Us environment.
All crewmates are LLM agents sharing one policy.
Impostor uses LLM to generate natural deceptive statements.
Majority vote across all crewmates determines ejection.
"""
from typing import Dict, List, Optional, Any
from server.game.scenario_generator import generate_scenario, Scenario
from server.models import DifficultyConfig
from server.curriculum import AdaptiveCurriculum


class GameSession:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.votes: Dict[str, str] = {}       # player_name -> vote_target
        self.resolved: bool = False
        self.result: Optional[Dict] = None

    @property
    def crewmates(self) -> List[str]:
        s = self.scenario
        return [p.name for p in s.players if p.name in s.alive_players and p.role == "crew"]

    @property
    def all_voters(self) -> List[str]:
        return self.crewmates  # only crewmates vote in this setup

    def votes_complete(self) -> bool:
        return len(self.votes) >= len(self.all_voters)


class MultiAgentAmongUsEnv:
    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}
        self.curriculum = AdaptiveCurriculum()

    def reset(self, difficulty: Optional[DifficultyConfig] = None, hf_token: Optional[str] = None) -> Dict:
        if difficulty is None:
            difficulty = self.curriculum.get_difficulty()

        scenario = generate_scenario(difficulty)
        self._enhance_impostor_statements(scenario, hf_token)

        session = GameSession(scenario)
        self.sessions[scenario.game_id] = session
        self._cleanup_old_sessions()

        s = scenario
        player_views = {}
        for player in s.players:
            if player.name not in s.alive_players:
                continue
            player_views[player.name] = self._build_player_view(player.name, s)

        return {
            "game_id": s.game_id,
            "alive_players": s.alive_players,
            "impostor_count": len(s.impostor_names),
            "body_found_location": s.kill_location.title(),
            "body_found_by": s.body_found_by,
            "player_views": player_views,
            "crewmates": session.crewmates,
            "training_meta": {
                "impostor_names": s.impostor_names,
                "confident_innocent_name": s.confident_innocent_name,
            },
        }

    def get_observation(self, game_id: str, player_name: str) -> Optional[Dict]:
        session = self.sessions.get(game_id)
        if not session:
            return None
        s = session.scenario
        if player_name not in s.alive_players:
            return None
        return self._build_player_view(player_name, s, session)

    def submit_vote(self, game_id: str, player_name: str, vote_target: str) -> Dict:
        session = self.sessions.get(game_id)
        if not session:
            return {"error": "Game not found"}
        if session.resolved:
            return {"error": "Game already resolved"}
        if player_name not in session.all_voters:
            return {"error": f"{player_name} is not a valid voter"}

        session.votes[player_name] = vote_target
        ready = session.votes_complete()

        result = {
            "game_id": game_id,
            "player_name": player_name,
            "vote_target": vote_target,
            "votes_submitted": len(session.votes),
            "votes_needed": len(session.all_voters),
            "ready_to_resolve": ready,
        }
        if ready:
            result["resolution"] = self.resolve(game_id)
        return result

    def resolve(self, game_id: str) -> Dict:
        session = self.sessions.get(game_id)
        if not session:
            return {"error": "Game not found"}
        if session.resolved:
            return session.result

        s = session.scenario

        # Tally votes
        vote_counts: Dict[str, int] = {}
        for vote in session.votes.values():
            if vote.lower() != "skip":
                vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # Majority ejection (ties: no ejection)
        ejected = None
        if vote_counts:
            max_votes = max(vote_counts.values())
            leaders = [p for p, v in vote_counts.items() if v == max_votes]
            if len(leaders) == 1:
                ejected = leaders[0]

        correct = ejected in s.impostor_names if ejected else False
        self.curriculum.update(correct)

        # Per-player rewards
        player_rewards = {}
        for player_name, vote in session.votes.items():
            if vote in s.impostor_names:
                r = 1.0
            elif s.confident_innocent_name and vote.lower() == s.confident_innocent_name.lower():
                r = -0.8  # sycophancy penalty
            elif vote.lower() == "skip":
                r = -0.1
            else:
                r = -0.5
            player_rewards[player_name] = r

        if ejected:
            msg = f"{ejected} was {'THE IMPOSTOR! Crew wins!' if correct else 'NOT the impostor. Impostors were: ' + str(s.impostor_names)}"
        else:
            msg = f"Tie vote — no one ejected. Impostors: {s.impostor_names}"

        result = {
            "game_id": game_id,
            "votes": session.votes,
            "vote_counts": vote_counts,
            "ejected": ejected,
            "correct": correct,
            "impostor_names": s.impostor_names,
            "player_rewards": player_rewards,
            "majority_reward": 1.0 if correct else -0.5,
            "message": msg,
        }

        session.resolved = True
        session.result = result
        return result

    def get_session_info(self, game_id: str) -> Optional[Dict]:
        session = self.sessions.get(game_id)
        if not session:
            return None
        s = session.scenario
        return {
            "game_id": game_id,
            "alive_players": s.alive_players,
            "crewmates": session.crewmates,
            "impostor_count": len(s.impostor_names),
            "votes_submitted": len(session.votes),
            "votes_needed": len(session.all_voters),
            "resolved": session.resolved,
            "result": session.result,
        }

    def _build_player_view(self, player_name: str, s: Scenario, session: Optional[GameSession] = None) -> Dict:
        player = next((p for p in s.players if p.name == player_name), None)
        if not player:
            return {}

        view = {
            "player_name": player_name,
            "role": player.role,
            "own_location": player.actual_location,
            "own_task": player.actual_task,
            "body_found_location": s.kill_location.title(),
            "body_found_by": s.body_found_by,
            "alive_players": s.alive_players.copy(),
            "impostor_count": len(s.impostor_names),
            "all_statements": s.get_statements(),
        }
        if session:
            view["votes_submitted"] = len(session.votes)
            view["votes_needed"] = len(session.all_voters)
        return view

    def _enhance_impostor_statements(self, scenario: Scenario, hf_token: Optional[str]) -> None:
        if not hf_token:
            return
        from server.game.llm_impostor import generate_llm_impostor_statement
        for imp_name in scenario.impostor_names:
            lr = scenario.lie_results.get(imp_name)
            player = scenario.get_player(imp_name)
            if not lr or not player:
                continue
            enhanced = generate_llm_impostor_statement(
                impostor_name=imp_name,
                fake_location=lr.fake_location,
                fake_task=lr.fake_task,
                kill_location=scenario.kill_location,
                lie_type=lr.lie_type,
                scripted_statement=player.statement,
                hf_token=hf_token,
            )
            player.statement = enhanced

    def _cleanup_old_sessions(self, max_sessions: int = 200) -> None:
        if len(self.sessions) > max_sessions:
            to_remove = list(self.sessions.keys())[:-max_sessions]
            for key in to_remove:
                del self.sessions[key]
