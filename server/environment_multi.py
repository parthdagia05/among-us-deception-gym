"""
Multi-agent Among Us environment — full game loop.

Round structure (repeats until win condition):
  1. Debate   — every player speaks N rounds (POST /multi/discuss)
  2. Vote     — crewmates vote (POST /multi/vote)
  3. Resolve  — ejection + win check (auto-called when all votes in)
  4. Kill     — impostor kills a crewmate, starts next round (POST /multi/kill)

Win conditions:
  • Crew wins  : all impostors ejected
  • Impostors win : impostor_count >= alive_crew_count (can't be stopped)
"""
import random
from typing import Dict, List, Optional, Set
from server.game.scenario_generator import generate_scenario, Scenario
from server.models import DifficultyConfig
from server.curriculum import AdaptiveCurriculum


class GameSession:
    def __init__(self, scenario: Scenario, max_discussion_rounds: int = 2):
        self.scenario = scenario
        self.max_discussion_rounds = max_discussion_rounds

        # ── current-round kill info (updated each round) ──────────────
        self.kill_victim: str = scenario.kill_victim
        self.kill_location: str = scenario.kill_location
        self.body_found_by: str = scenario.body_found_by

        # ── per-round state (reset after each kill) ───────────────────
        self.votes: Dict[str, str] = {}
        self.discussion_log: List[Dict] = []
        self.discussion_round: int = 0
        self.spoke_this_round: Set[str] = set()
        self.resolved: bool = False
        self.result: Optional[Dict] = None

        # ── game-level state ──────────────────────────────────────────
        self.round_number: int = 1
        self.game_over: bool = False
        self.winner: Optional[str] = None          # "crew" | "impostors"
        self.awaiting_kill: bool = False            # True between resolve and next kill
        self.round_history: List[Dict] = []        # record of every past round

    # ── convenience props ─────────────────────────────────────────────

    @property
    def alive_players(self) -> List[str]:
        return self.scenario.alive_players

    @property
    def crewmates(self) -> List[str]:
        s = self.scenario
        return [p.name for p in s.players if p.name in s.alive_players and p.role == "crew"]

    @property
    def alive_impostors(self) -> List[str]:
        s = self.scenario
        return [n for n in s.alive_players if n in s.impostor_names]

    @property
    def all_voters(self) -> List[str]:
        return self.crewmates

    @property
    def discussion_complete(self) -> bool:
        return self.discussion_round >= self.max_discussion_rounds

    def votes_complete(self) -> bool:
        return len(self.votes) >= len(self.all_voters)

    def can_speak(self, player_name: str) -> bool:
        return (
            not self.discussion_complete
            and player_name in self.alive_players
            and player_name not in self.spoke_this_round
        )

    # ── discussion helpers ────────────────────────────────────────────

    def record_statement(self, player_name: str, statement: str) -> None:
        self.discussion_log.append({
            "player_name": player_name,
            "statement": statement,
            "round": self.discussion_round,
        })
        self.spoke_this_round.add(player_name)
        if self.spoke_this_round >= set(self.alive_players):
            self.discussion_round += 1
            self.spoke_this_round = set()

    # ── round transition ──────────────────────────────────────────────

    def _check_win(self) -> None:
        alive_imp = self.alive_impostors
        alive_crew = self.crewmates
        if len(alive_imp) == 0:
            self.game_over = True
            self.winner = "crew"
        elif len(alive_imp) >= len(alive_crew):
            self.game_over = True
            self.winner = "impostors"
        else:
            self.awaiting_kill = True

    def start_new_round(self, kill_target: str) -> Dict:
        """
        Impostor kills kill_target.  Resets discussion/votes and starts round N+1.
        Returns the new round's opening state or an error dict.
        """
        s = self.scenario
        if self.game_over:
            return {"error": "Game is already over"}
        if not self.awaiting_kill:
            return {"error": "Not waiting for a kill — resolve the vote first"}
        if kill_target not in s.alive_players:
            return {"error": f"{kill_target} is not alive"}
        if kill_target in s.impostor_names:
            return {"error": "Impostors cannot kill each other"}

        # Archive current round before mutation
        self.round_history.append({
            "round": self.round_number,
            "ejected": self.result.get("ejected") if self.result else None,
            "ejection_correct": self.result.get("correct", False) if self.result else False,
            "killed": kill_target,
            "alive_before_kill": s.alive_players.copy(),
        })

        # Kill the target
        victim = s.get_player(kill_target)
        new_kill_location = victim.actual_location if victim else s.kill_location
        if victim:
            victim.alive = False
        s.alive_players = [p for p in s.alive_players if p != kill_target]

        # Pick who finds the body
        alive_crew_players = [p for p in s.players if p.name in s.alive_players and p.role == "crew"]
        finder = random.choice(alive_crew_players).name if alive_crew_players else (
            s.alive_players[0] if s.alive_players else "unknown"
        )

        # Update kill metadata for this round
        self.kill_victim = kill_target
        self.kill_location = new_kill_location
        self.body_found_by = finder

        # Reset per-round state
        self.votes = {}
        self.discussion_log = []
        self.discussion_round = 0
        self.spoke_this_round = set()
        self.resolved = False
        self.result = None
        self.awaiting_kill = False
        self.round_number += 1

        # After the kill, check if impostors now outnumber/match crew
        alive_imp = self.alive_impostors
        alive_crew = self.crewmates
        if len(alive_imp) == 0:
            self.game_over = True
            self.winner = "crew"
        elif len(alive_imp) >= len(alive_crew):
            self.game_over = True
            self.winner = "impostors"

        return {
            "game_id": s.game_id,
            "round": self.round_number,
            "killed": kill_target,
            "kill_location": new_kill_location.title(),
            "body_found_by": finder,
            "alive_players": s.alive_players,
            "alive_impostors_remaining": len(self.alive_impostors),
            "crewmates_remaining": len(self.crewmates),
            "game_over": self.game_over,
            "winner": self.winner,
        }


# ── Main environment class ────────────────────────────────────────────


class MultiAgentAmongUsEnv:
    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}
        self.curriculum = AdaptiveCurriculum()

    def reset(
        self,
        difficulty: Optional[DifficultyConfig] = None,
        hf_token: Optional[str] = None,
        max_discussion_rounds: int = 2,
    ) -> Dict:
        if difficulty is None:
            difficulty = self.curriculum.get_difficulty()

        scenario = generate_scenario(difficulty)
        self._enhance_impostor_statements(scenario, hf_token)

        session = GameSession(scenario, max_discussion_rounds=max_discussion_rounds)
        self.sessions[scenario.game_id] = session
        self._cleanup_old_sessions()

        s = scenario
        player_views = {}
        for player in s.players:
            if player.name not in s.alive_players:
                continue
            player_views[player.name] = self._build_player_view(player.name, session)

        return {
            "game_id": s.game_id,
            "round": 1,
            "alive_players": s.alive_players,
            "impostor_count": len(s.impostor_names),
            "body_found_location": session.kill_location.title(),
            "body_found_by": session.body_found_by,
            "kill_victim": session.kill_victim,
            "player_views": player_views,
            "crewmates": session.crewmates,
            "max_discussion_rounds": max_discussion_rounds,
            "training_meta": {
                "impostor_names": s.impostor_names,
                "confident_innocent_name": s.confident_innocent_name,
            },
        }

    def get_observation(self, game_id: str, player_name: str) -> Optional[Dict]:
        session = self.sessions.get(game_id)
        if not session:
            return None
        if player_name not in session.alive_players:
            return None
        return self._build_player_view(player_name, session)

    def submit_discussion(
        self,
        game_id: str,
        player_name: str,
        statement: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> Dict:
        session = self.sessions.get(game_id)
        if not session:
            return {"error": "Game not found"}
        if session.game_over:
            return {"error": f"Game over — {session.winner} won"}
        if session.awaiting_kill:
            return {"error": "Waiting for impostor kill — call /multi/kill first"}
        if session.resolved:
            return {"error": "Vote already resolved — call /multi/kill to start next round"}
        if player_name not in session.alive_players:
            return {"error": f"{player_name} is not alive"}
        if session.discussion_complete:
            return {"error": "Discussion phase is over — proceed to voting"}
        if not session.can_speak(player_name):
            return {"error": f"{player_name} has already spoken this round"}

        s = session.scenario
        player = s.get_player(player_name)
        if not player:
            return {"error": "Player not found"}

        if not statement:
            from server.game.llm_discussion import generate_discussion_statement
            lr = s.lie_results.get(player_name)
            statement = generate_discussion_statement(
                player_name=player_name,
                role=player.role,
                personality=player.personality,
                all_statements=s.get_statements(),
                discussion_history=session.discussion_log,
                round_num=session.discussion_round,
                kill_location=session.kill_location,
                fake_location=lr.fake_location if lr else None,
                hf_token=hf_token,
            )

        session.record_statement(player_name, statement)

        return {
            "game_id": game_id,
            "player_name": player_name,
            "statement": statement,
            "round": session.discussion_log[-1]["round"],
            "discussion_round": session.discussion_round,
            "discussion_complete": session.discussion_complete,
            "discussion_log": session.discussion_log,
            "spoke_this_round": list(session.spoke_this_round),
            "remaining_speakers": [
                p for p in session.alive_players if p not in session.spoke_this_round
            ] if not session.discussion_complete else [],
        }

    def submit_vote(self, game_id: str, player_name: str, vote_target: str) -> Dict:
        session = self.sessions.get(game_id)
        if not session:
            return {"error": "Game not found"}
        if session.game_over:
            return {"error": f"Game over — {session.winner} won"}
        if session.awaiting_kill:
            return {"error": "Waiting for impostor kill — call /multi/kill first"}
        if session.resolved:
            return {"error": "Round already resolved"}
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

        # Majority ejection (ties → no ejection)
        ejected = None
        if vote_counts:
            max_votes = max(vote_counts.values())
            leaders = [p for p, v in vote_counts.items() if v == max_votes]
            if len(leaders) == 1:
                ejected = leaders[0]

        # Remove ejected player from alive roster
        if ejected and ejected in s.alive_players:
            ejected_player = s.get_player(ejected)
            if ejected_player:
                ejected_player.alive = False
            s.alive_players = [p for p in s.alive_players if p != ejected]

        correct = ejected in s.impostor_names if ejected else False
        self.curriculum.update(correct)

        # Per-player rewards
        player_rewards = {}
        for player_name, vote in session.votes.items():
            if vote in s.impostor_names:
                r = 1.0
            elif s.confident_innocent_name and vote.lower() == s.confident_innocent_name.lower():
                r = -0.8
            elif vote.lower() == "skip":
                r = -0.1
            else:
                r = -0.5
            player_rewards[player_name] = r

        if ejected:
            was = "THE IMPOSTOR! Crew wins this round!" if correct else f"NOT the impostor (was {'crew'})"
            msg = f"{ejected} was {was}."
        else:
            msg = "Tie vote — no one ejected."

        # Check win conditions (mutates session.game_over / winner / awaiting_kill)
        session._check_win()

        if session.game_over:
            final_msg = f"GAME OVER — {session.winner.upper()} WIN! " + msg
        elif session.awaiting_kill:
            alive_imp = session.alive_impostors
            final_msg = msg + f" Impostor(s) {alive_imp} must now kill someone (/multi/kill)."
        else:
            final_msg = msg

        result = {
            "game_id": game_id,
            "round": session.round_number,
            "votes": session.votes,
            "vote_counts": vote_counts,
            "ejected": ejected,
            "ejection_correct": correct,
            "impostor_names": s.impostor_names,
            "player_rewards": player_rewards,
            "majority_reward": 1.0 if correct else -0.5,
            "message": final_msg,
            "game_over": session.game_over,
            "winner": session.winner,
            "alive_players": s.alive_players,
            "awaiting_kill": session.awaiting_kill,
            "discussion_log": session.discussion_log,
        }

        session.resolved = True
        session.result = result
        return result

    def kill(
        self,
        game_id: str,
        impostor_name: str,
        kill_target: Optional[str] = None,
    ) -> Dict:
        """
        Impostor kills a crewmate and starts the next round.
        If kill_target is None, a random crewmate is chosen automatically.
        """
        session = self.sessions.get(game_id)
        if not session:
            return {"error": "Game not found"}
        if session.game_over:
            return {"error": f"Game over — {session.winner} won"}
        if not session.awaiting_kill:
            return {"error": "Not awaiting a kill — resolve the vote first"}
        if impostor_name not in session.scenario.impostor_names:
            return {"error": f"{impostor_name} is not an impostor"}
        if impostor_name not in session.alive_players:
            return {"error": f"{impostor_name} has been ejected"}

        # Auto-pick victim if not specified
        if not kill_target:
            candidates = session.crewmates
            if not candidates:
                return {"error": "No crewmates left to kill"}
            kill_target = random.choice(candidates)

        return session.start_new_round(kill_target)

    def get_session_info(self, game_id: str) -> Optional[Dict]:
        session = self.sessions.get(game_id)
        if not session:
            return None
        s = session.scenario
        return {
            "game_id": game_id,
            "round": session.round_number,
            "alive_players": s.alive_players,
            "crewmates": session.crewmates,
            "impostor_count": len(s.impostor_names),
            "alive_impostors": len(session.alive_impostors),
            "discussion_round": session.discussion_round,
            "max_discussion_rounds": session.max_discussion_rounds,
            "discussion_complete": session.discussion_complete,
            "discussion_log": session.discussion_log,
            "votes_submitted": len(session.votes),
            "votes_needed": len(session.all_voters),
            "resolved": session.resolved,
            "awaiting_kill": session.awaiting_kill,
            "game_over": session.game_over,
            "winner": session.winner,
            "round_history": session.round_history,
            "result": session.result,
        }

    def _build_player_view(self, player_name: str, session: GameSession) -> Dict:
        s = session.scenario
        player = next((p for p in s.players if p.name == player_name), None)
        if not player:
            return {}

        return {
            "player_name": player_name,
            "role": player.role,
            "own_location": player.actual_location,
            "own_task": player.actual_task,
            "game_round": session.round_number,
            "body_found_location": session.kill_location.title(),
            "body_found_by": session.body_found_by,
            "kill_victim": session.kill_victim,
            "alive_players": s.alive_players.copy(),
            "impostor_count": len(s.impostor_names),
            "all_statements": s.get_statements(),
            "discussion_log": session.discussion_log,
            "discussion_round": session.discussion_round,
            "discussion_complete": session.discussion_complete,
            "can_speak": session.can_speak(player_name),
            "votes_submitted": len(session.votes),
            "votes_needed": len(session.all_voters),
            "round_history": session.round_history,
            "game_over": session.game_over,
            "winner": session.winner,
        }

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
