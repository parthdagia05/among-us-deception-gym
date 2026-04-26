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

A multi-agent reinforcement learning environment for training LLMs to detect deception in social-deduction games. Built to the **OpenEnv spec**.

## 🔗 Links

| | |
|---|---|
| Live game environment (HF Space) | https://huggingface.co/spaces/parthdagia/among-us-deception-gym |
| Trained model (HF Hub) | https://huggingface.co/parthdagia/among-us-multiagent-detective |
| Source code (GitHub) | https://github.com/parthdagia05/among-us-deception-gym |

## 🧠 The Problem: Sycophancy Kills

Language models trust confident-sounding statements. In an adversarial setting, one assertive liar can manipulate the model's decision. This environment trains models to **investigate contradictions** rather than trust confidence.

## 📊 Headline Result

After 1500 GRPO iterations on a single A10G, the trained model goes from **32.7% → 96.7% impostor-detection accuracy** and reduces sycophancy by **17×**.

| Metric | **Trained** | Base (Qwen 2.5 1.5B) | Δ |
|---|---|---|---|
| Vote accuracy | **96.7 %** | 32.7 % | **+64.0 %** |
| Sycophancy rate | **1.3 %** | 22.0 % | **−20.7 %** |
| Crew win rate | **96.0 %** | 30.0 % | **+66.0 %** |
| Malformed votes | 0.0 % | 0.0 % | — |

*Eval: 50 unseen games × 3 crewmate votes = 150 votes per model.*

![Trained vs Base comparison](https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_comparison.png)

### Learning curve (real training metrics)

The model converges in ~500 iterations and saturates at perfect reward by iteration 600.

![GRPO reward over 1500 iterations](https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_reward.png)

## 🎮 Game Mechanics

A real Among Us-style loop, in pure text:

1. **Spawn**: 5 colored players with locations, tasks, personalities. 1 victim dies, 1 is the impostor (hard-locked to single impostor).
2. **Initial statements**: Every alive player gives a 1-2 sentence alibi. Impostor's lie has at least one detectable contradiction.
3. **Debate** (`/multi/discuss`): Each alive player speaks for **N rounds** — accusing, defending, deflecting. Impostor actively defends itself when accused.
4. **Vote** (`/multi/vote`): Crewmates cast a vote based on initial statements + full debate transcript.
5. **Resolve**: Majority ejects someone. Crew wins if impostor ejected.
6. **Kill** (`/multi/kill`): If wrong ejection, impostor murders a crewmate. New body, new round.
7. Repeat until **crew wins** (impostor ejected) or **impostors win** (parity reached: alive impostors ≥ alive crew).

The trained model sees the full debate transcript and votes based on contradictions, not confidence.

## 🌐 REST API (multi-agent endpoints)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/multi/reset` | Start a new game |
| `GET`  | `/multi/observation/{game_id}/{player}` | Per-player view (no leak of impostor) |
| `POST` | `/multi/discuss` | Submit a debate statement |
| `POST` | `/multi/vote` | Crewmate votes |
| `POST` | `/multi/kill` | Impostor kills (between rounds) |
| `GET`  | `/multi/resolve/{game_id}` | Tally votes / win check |
| `GET`  | `/multi/session/{game_id}` | Full game state + round history |

```bash
# Live demo against the deployed Space
curl -X POST https://parthdagia-among-us-deception-gym.hf.space/multi/reset \
  -H "Content-Type: application/json" \
  -d '{"max_discussion_rounds": 2}'
```

## 🏋️ Training

GRPO training runs on **HuggingFace Jobs** with no local GPU needed. The script in [`jobs/train_job.py`](jobs/train_job.py) is a self-contained UV script.

```bash
# Submit training (a10g-large, ~4 hours, ~$5.65 of HF credits)
hf jobs uv run \
  --flavor a10g-large \
  --secrets HF_TOKEN \
  --env N_GAMES=500 \
  --env N_STEPS=500 \
  --env OUTPUT_HF=youruser/your-model \
  jobs/train_job.py
```

The script:
1. Clones this repo into the container for in-process env imports
2. Loads `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (r=16)
3. Builds dataset: 500 games × debate phase × ~3 crewmates = **1500 prompts**
4. Runs 1500 GRPO iterations (8 rollouts per prompt)
5. Merges LoRA into base, pushes merged model to HF Hub

## 🧪 Evaluation

```bash
# Compare trained model against base on 50 fresh games
hf jobs uv run \
  --flavor t4-medium \
  --secrets HF_TOKEN \
  --env MODEL=youruser/your-model \
  --env COMPARE=1 \
  --env N_GAMES=50 \
  jobs/eval_job.py
```

Reports vote accuracy, sycophancy rate, malformed-output rate, and crew win rate for both trained and base.

## 🎯 Reward Structure

Per-vote reward, computed deterministically from the scenario's ground truth:

| Outcome | Reward |
|---|---|
| Voted the impostor | **+1.0** |
| Voted the "confident innocent" (sycophancy trap) | **−0.8** |
| Voted any other crewmate | **−0.5** |
| Skipped or malformed | **−0.3 to −0.1** |
| Vote includes substantive `REASONING:` (≥30 chars) | **+0.1 bonus** |

No LLM-as-judge — every reward is mechanical.

## 🎭 Lie Types (curriculum-scaled)

| Difficulty | Lie Types |
|---|---|
| 0.0-0.2 | location_clash, task_impossibility |
| 0.2-0.4 | + timing_contradiction |
| 0.4-0.6 | + kill_proximity |
| 0.6-0.8 | kill_proximity, corroboration_gap |
| 0.8-1.0 | corroboration_gap, duo_cover |

## 🏗️ Key Design Choices

1. **Programmatic rewards only** — no LLM-as-judge. Vote correctness is ground truth.
2. **Guaranteed contradictions** — every impostor lie has at least one provably wrong claim.
3. **Sycophancy test** — one crewmate is always tagged "confident innocent" with an assertive personality. Voting for them = the model caved to confidence. We measure this rate explicitly.
4. **Multi-round game loop** — impostor kills a crewmate after each wrong ejection. Game ends when impostor ejected (crew wins) or parity reached (impostors win).
5. **Shared crewmate policy** — all 3 crewmates share one LoRA, just told "you are Red / Green / Yellow" in the prompt. One model, three roles.
6. **LLM impostor** — uses HF Inference API (Qwen 2.5 1.5B) to generate natural-sounding lies + active debate defense. Falls back to scripted statements if no token.
7. **Single-impostor lock** — `scenario_generator.py` forces `num_impostors=1` regardless of difficulty config, for consistent training signal.

## 🧱 Project Structure

```
.
├── server/
│   ├── app.py                # FastAPI server (OpenEnv spec, all endpoints)
│   ├── environment_multi.py  # Multi-agent env: debate + vote + kill loop
│   ├── environment.py        # Single-agent env (legacy)
│   ├── curriculum.py         # Adaptive difficulty
│   ├── models.py             # Pydantic models
│   ├── game/
│   │   ├── scenario_generator.py  # Game scenarios + lie injection
│   │   ├── lie_engine.py          # 6 lie archetypes
│   │   ├── statement_generator.py # NPC alibi statements
│   │   ├── llm_impostor.py        # LLM-driven natural lies
│   │   ├── llm_discussion.py      # LLM-driven debate (crew + impostor)
│   │   ├── map_data.py            # 9-room adjacency graph
│   │   └── player.py              # Player dataclass
│   ├── graders/                   # Vote / investigation / deception graders
│   └── tools/investigation.py     # 5 investigation tools
├── jobs/
│   ├── train_job.py          # GRPO training (HF Jobs UV script)
│   ├── eval_job.py           # Trained vs base eval (HF Jobs UV script)
│   └── README.md             # HF Jobs cost / submission docs
├── train_multiagent.ipynb    # Equivalent training pipeline for Colab
├── Dockerfile                # HF Spaces container
├── openenv.yaml              # OpenEnv spec
└── requirements.txt
```

## 📦 OpenEnv Spec

```yaml
spec_version: 1
name: among_us_deception_gym
type: space
runtime: fastapi
app: server.app:app
port: 7860
```

## 🐳 Run Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --port 8000
# or
docker build -t among-us-gym . && docker run -p 7860:7860 among-us-gym
```

## 🔐 Environment Variables

Copy [`.env.example`](.env.example) to `.env` and fill in:

```bash
HF_TOKEN=hf_xxx   # write access + Manage Jobs scope
```

## 📜 License

MIT.
