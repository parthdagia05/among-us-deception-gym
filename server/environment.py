import uuid
from typing import Optional, Dict, Any, List
from server.models import Observation, Action, StepResult, DifficultyConfig
from server.game.scenario_generator import generate_scenario, Scenario
from server.tools.investigation import execute_tool
from server.graders import grade_vote, grade_investigation, grade_deception_resistance, grade_anti_cheat, compute_total_reward
from server.curriculum import AdaptiveCurriculum


class AmongUsEnv:
    def __init__(self):
        self.curriculum = AdaptiveCurriculum()
        self.current_scenario: Optional[Scenario] = None
        self.tool_calls: List[Dict] = []
        self.vote_count: int = 0
        self.turn_number: int = 0
        self.max_turns: int = 8
        self.game_id: str = ""
        self.round_number: int = 1
        self.previous_rounds: List[Dict] = []
        self.done: bool = False

    def reset(self, difficulty: Optional[DifficultyConfig] = None) -> Observation:
        if difficulty is None:
            difficulty = self.curriculum.get_difficulty()

        self.current_scenario = generate_scenario(difficulty)
        self.tool_calls = []
        self.investigation_log: List[str] = []
        self.vote_count = 0
        self.turn_number = 0
        self.game_id = self.current_scenario.game_id
        self.round_number = 1
        self.previous_rounds = []
        self.done = False

        return self._make_observation()

    def _make_observation(self) -> Observation:
        s = self.current_scenario
        available_tools = [
            "read_statements", "check_location_tasks",
            "check_player_presence", "check_adjacency", "cross_reference"
        ]
        if s.difficulty.num_rounds > 1:
            available_tools.append("skip_vote")

        return Observation(
            game_id=self.game_id,
            round_number=self.round_number,
            max_rounds=s.difficulty.num_rounds,
            alive_players=s.alive_players.copy(),
            impostor_count=len([n for n in s.impostor_names if n in s.alive_players]),
            body_found_location=s.kill_location.title(),
            body_found_by=s.body_found_by,
            player_statements=s.get_statements(),
            last_tool_result=None,
            available_tools=available_tools,
            investigation_log=[],
            previous_rounds=self.previous_rounds if self.previous_rounds else None,
            turn_number=self.turn_number,
            max_turns=self.max_turns
        )

    def step(self, action: Action) -> StepResult:
        if self.done:
            obs = self._make_observation()
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "Episode already done"})

        s = self.current_scenario
        self.turn_number += 1

        if action.action_type == "tool_call":
            tool_name = action.tool_name or ""
            tool_args = action.tool_args or {}

            result = execute_tool(tool_name, tool_args, s)

            self.tool_calls.append({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result
            })

            log_entry = f"[Turn {self.turn_number}] {tool_name}({tool_args}) -> {result[:120]}"
            self.investigation_log.append(log_entry)

            obs = Observation(
                game_id=self.game_id,
                round_number=self.round_number,
                max_rounds=s.difficulty.num_rounds,
                alive_players=s.alive_players.copy(),
                impostor_count=len([n for n in s.impostor_names if n in s.alive_players]),
                body_found_location=s.kill_location.title(),
                body_found_by=s.body_found_by,
                player_statements=s.get_statements(),
                last_tool_result=result,
                available_tools=["read_statements", "check_location_tasks",
                                  "check_player_presence", "check_adjacency", "cross_reference"] +
                                 (["skip_vote"] if s.difficulty.num_rounds > 1 else []),
                investigation_log=self.investigation_log.copy(),
                previous_rounds=self.previous_rounds if self.previous_rounds else None,
                turn_number=self.turn_number,
                max_turns=self.max_turns
            )

            done = self.turn_number >= self.max_turns
            if done:
                self.done = True
                # Timeout with no vote: apply anti-cheat penalty
                ac_result = grade_anti_cheat(
                    tool_calls=self.tool_calls,
                    vote_actions=self.vote_count,
                    turn_number=self.turn_number,
                    max_turns=self.max_turns
                )
                self.curriculum.update(False)
                timeout_reward = max(-1.0, min(1.0, ac_result["anti_cheat_score"]))
                return StepResult(observation=obs, reward=timeout_reward, done=True,
                                  info={"turn": self.turn_number, "timeout": True,
                                        "anti_cheat_score": ac_result["anti_cheat_score"],
                                        "flags": ac_result["flags"]})

            return StepResult(observation=obs, reward=0.0, done=False, info={"turn": self.turn_number})

        elif action.action_type == "vote":
            self.vote_count += 1
            vote_target = action.vote_target or ""

            vote_result = grade_vote(
                vote_target=vote_target,
                impostor_names=s.impostor_names,
                confident_innocent_name=s.confident_innocent_name,
                investigation_log=[c["tool_name"] for c in self.tool_calls],
                can_skip=s.difficulty.num_rounds > 1
            )

            inv_result = grade_investigation(
                tool_calls=self.tool_calls,
                relevant_tools=s.relevant_tools,
                scenario=s
            )

            dec_result = grade_deception_resistance(
                tool_calls=self.tool_calls,
                vote_target=vote_target,
                confident_innocent_name=s.confident_innocent_name,
                impostor_names=s.impostor_names
            )

            ac_result = grade_anti_cheat(
                tool_calls=self.tool_calls,
                vote_actions=self.vote_count,
                turn_number=self.turn_number,
                max_turns=self.max_turns
            )

            total_reward = compute_total_reward(
                vote_result["vote_score"],
                inv_result["investigation_score"],
                dec_result["deception_score"],
                ac_result["anti_cheat_score"]
            )

            correct = vote_result["correct"]
            self.curriculum.update(correct)
            self.done = True

            # Build result message
            if vote_target in s.impostor_names:
                result_msg = f"You voted for {vote_target}. {vote_target} was THE IMPOSTOR! Crew wins!"
            elif vote_target.lower() == "skip":
                result_msg = "Vote skipped. No one was ejected. Be careful - the impostor is still among you."
            else:
                result_msg = f"You voted for {vote_target}. {vote_target} was NOT the impostor. Impostors were: {', '.join(s.impostor_names)}"

            obs = Observation(
                game_id=self.game_id,
                round_number=self.round_number,
                max_rounds=s.difficulty.num_rounds,
                alive_players=s.alive_players.copy(),
                impostor_count=len(s.impostor_names),
                body_found_location=s.kill_location.title(),
                body_found_by=s.body_found_by,
                player_statements=s.get_statements(),
                last_tool_result=result_msg,
                available_tools=[],
                investigation_log=[c["tool_name"] for c in self.tool_calls],
                previous_rounds=None,
                turn_number=self.turn_number,
                max_turns=self.max_turns
            )

            info = {
                "vote_score": vote_result["vote_score"],
                "investigation_score": inv_result["investigation_score"],
                "deception_score": dec_result["deception_score"],
                "anti_cheat_score": ac_result["anti_cheat_score"],
                "total_reward": total_reward,
                "correct": correct,
                "impostors": s.impostor_names,
                "flags": {**vote_result.get("flags", {}), **inv_result.get("flags", {}),
                          **dec_result.get("flags", {}), **ac_result.get("flags", {})},
                "result_message": result_msg
            }

            return StepResult(observation=obs, reward=total_reward, done=True, info=info)

        else:
            obs = self._make_observation()
            return StepResult(observation=obs, reward=-0.1, done=False, info={"error": f"Unknown action_type: {action.action_type}"})

    def state(self) -> Dict[str, Any]:
        if not self.current_scenario:
            return {"status": "not_started"}

        s = self.current_scenario
        return {
            "game_id": self.game_id,
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "alive_players": s.alive_players,
            "tool_calls_made": len(self.tool_calls),
            "vote_count": self.vote_count,
            "done": self.done,
            "difficulty": s.difficulty.model_dump(),
            "curriculum_step": self.curriculum.step,
        }
