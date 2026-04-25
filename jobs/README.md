# HF Jobs — Among Us GRPO Training

Submit `train_job.py` to HuggingFace Jobs to run GRPO on cloud GPU, billed to your HF credit balance.

## One-time setup

```bash
pip install -U "huggingface_hub[cli]>=0.26.0"
huggingface-cli login        # paste your HF write token
huggingface-cli auth whoami  # confirm you're logged in
```

Add your token as a job secret (one-time):

```bash
huggingface-cli auth switch                 # ensure correct user
# Token is automatically picked up from your login session by --secrets HF_TOKEN
```

## Submit the job

A10G-large is the right GPU for a 1.5B model with LoRA — plenty of VRAM, ~$1.30/hr.

```bash
cd /home/parth-dagia/amongus-rl-env/jobs

hf jobs uv run \
  --flavor a10g-large \
  --secrets HF_TOKEN \
  --env N_GAMES=500 \
  --env N_STEPS=500 \
  --env N_ROUNDS=2 \
  --env OUTPUT_HF=parthdagia/among-us-multiagent-detective \
  train_job.py
```

The CLI prints a job URL. Open it to watch live logs.

## Monitor

```bash
hf jobs ps              # list running jobs
hf jobs logs <job-id>   # tail logs
hf jobs inspect <job-id> # full job details
```

## Cost estimate

| Hardware | $/hr | Est. time | Est. cost |
|----------|------|-----------|-----------|
| `t4-medium`   | $0.60 | ~3.5 hr | ~$2.10 |
| `a10g-large`  | $1.50 | ~1.8 hr | ~$2.70 |
| `a100-large`  | $2.50 | ~1.0 hr | ~$2.50 |

Recommended: **a100-large** — finishes fastest, similar total cost (24 GB unused but burns no extra).

## What it does

1. Clones the Space repo into `/tmp/gym` so it can `import server.environment_multi`
2. Loads Qwen2.5-1.5B in bf16 with a LoRA head
3. Builds dataset locally (no HTTP) — 500 games × debate phase × ~5 crewmates ≈ 2500 prompts
4. Runs 500 GRPO steps
5. Merges LoRA into base, pushes to `parthdagia/among-us-multiagent-detective`

## Knobs (override via `--env`)

| Var | Default | What |
|-----|---------|------|
| `N_GAMES`   | 500 | Games to generate |
| `N_ROUNDS`  | 2   | Debate rounds per game |
| `N_STEPS`   | 500 | GRPO training steps |
| `MODEL_ID`  | `Qwen/Qwen2.5-1.5B-Instruct` | Base model |
| `OUTPUT_HF` | `parthdagia/among-us-multiagent-detective` | Push target |

## Smoke test before full run

To burn ~$0.50 to verify the path end-to-end:

```bash
hf jobs uv run \
  --flavor a10g-large \
  --secrets HF_TOKEN \
  --env N_GAMES=20 \
  --env N_STEPS=20 \
  --env OUTPUT_HF=parthdagia/among-us-multiagent-smoke \
  train_job.py
```

Should finish in ~15 min. If it succeeds → submit the full 500-game run.
