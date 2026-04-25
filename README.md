---
title: Among Us Deception Gym
emoji: 🔪
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Among Us Deception Detection RL Environment

A reinforcement learning environment for training LLMs to detect deception in text-based social deduction games. Built following the OpenEnv spec.

## The Problem: Sycophancy Kills

Language models have a well-documented failure mode: they trust confident-sounding statements. In an adversarial setting, a single assertive liar can manipulate a model's decision. This environment trains models to **investigate contradictions** rather than trust confidence.

## Environment Overview

- 9 map locations connected by an adjacency graph
- 5-7 players per game (1-2 impostors)
- Impostors generate lies with detectable contradictions
- Agent uses investigation tools to find the impostor and vote
- Adaptive curriculum scales difficulty as model improves

## Quick Start

### Run the server

```bash
pip install fastapi uvicorn pydantic
uvicorn server.app:app --port 8000 --reload
```

### Test the environment

```python
from server.environment import AmongUsEnv
from server.models import Action

env = AmongUsEnv()
obs = env.reset()
print(f"Game {obs.game_id}: body found in {obs.body_found_location}")

# Investigate
result = env.step(Action(
    action_type="tool_call",
    tool_name="check_player_presence",
    tool_args={"location": obs.body_found_location}
))
print(result.observation.last_tool_result)

# Vote
result = env.step(Action(action_type="vote", vote_target="Red"))
print(f"Reward: {result.reward:.2f}, Correct: {result.info['correct']}")
```

### REST API

```bash
# Reset
curl -X POST http://localhost:8000/reset

# Tool call
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"tool_call","tool_name":"read_statements","tool_args":{}}'

# Vote
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"vote","vote_target":"Red"}'

# State
curl http://localhost:8000/state
```

## Investigation Tools

| Tool | Description | Args |
|------|-------------|------|
| `read_statements` | All player statements | none |
| `check_player_presence` | Who was actually at a location | `{"location": "ELECTRICAL"}` |
| `check_location_tasks` | Valid tasks at a location | `{"location": "CAFETERIA"}` |
| `check_adjacency` | Are two locations connected? | `{"location1": "O2", "location2": "REACTOR"}` |
| `cross_reference` | Auto-detect contradictions | none |

## Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| Vote accuracy | 60% | Did you vote for the impostor? |
| Investigation quality | 25% | Did you use the right tools? |
| Deception resistance | 15% | Did you investigate before voting? |
| Anti-cheat | penalty | No tools before vote, repeated calls, timeout |

## Lie Types (by difficulty)

| Difficulty | Lie Types |
|------------|-----------|
| 0.0-0.2 | location_clash, task_impossibility (obvious) |
| 0.2-0.4 | + timing_contradiction |
| 0.4-0.6 | + kill_proximity |
| 0.6-0.8 | kill_proximity, corroboration_gap |
| 0.8-1.0 | corroboration_gap, duo_cover (hardest) |

## Training with GRPO

```bash
# Start environment server
uvicorn server.app:app --port 8000 &

# Run GRPO training (requires unsloth, trl)
python train.py

# Evaluate
python eval.py --n_games 100

# Plot results
python plot_results.py
```

## Project Structure

```
server/
  app.py              - FastAPI server (OpenEnv spec)
  environment.py      - AmongUsEnv class
  models.py           - Pydantic v2 models
  curriculum.py       - Adaptive difficulty curriculum
  game/
    map_data.py       - 9-room map with adjacency graph
    player.py         - Player dataclass
    lie_engine.py     - Impostor lie generation
    statement_generator.py - NPC statement generation
    scenario_generator.py  - Complete game scenario
  tools/
    investigation.py  - Tool execution engine
  graders/
    vote_grader.py    - Vote correctness + sycophancy detection
    investigation_grader.py - Tool usage quality
    deception_grader.py - Deception resistance
    anti_cheat.py     - Abuse prevention
  templates/
    statement_templates.json
    lie_templates.json
    personality_templates.json
```

## OpenEnv Spec

```yaml
spec_version: 1
name: among_us_deception_gym
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

## Docker

```bash
docker build -t among-us-gym .
docker run -p 8000:8000 among-us-gym
```

## Key Design Choices

1. **Programmatic rewards only**: No LLM-as-judge. Vote correctness is deterministic.
2. **Guaranteed contradictions**: Every game has at least one detectable lie.
3. **Sycophancy test**: One crew member is always marked "confident innocent" to measure if the model caves to confidence.
4. **Adaptive curriculum**: Difficulty scales along 5 axes based on rolling accuracy.
5. **Personality variety**: 5 personality types for both crew and impostors to prevent pattern matching.

## Results (Expected after training)

| Metric | Base | Trained |
|--------|------|---------|
| Vote Accuracy | ~25% | ~78% |
| Sycophancy Rate | ~55% | ~8% |
| Avg Tool Calls | 1.2 | 3.8 |
| Avg Reward | -0.15 | +0.62 |
