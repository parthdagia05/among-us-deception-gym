# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.46.0",
#   "trl>=0.12.0",
#   "datasets>=2.20.0",
#   "torch>=2.4.0",
#   "peft>=0.13.0",
#   "accelerate>=1.0.0",
#   "huggingface_hub>=0.26.0",
#   "fastapi",
#   "pydantic>=2.0",
#   "requests",
# ]
# ///
"""
HF Jobs training script: Among Us multi-agent GRPO.

Runs entirely inside the Job container — no external HTTP needed.
The Among Us environment is imported directly from the Space repo (cloned into /tmp).

Submit with:
    hf jobs uv run --flavor a10g-large \
        --secrets HF_TOKEN \
        train_job.py
"""
import os
import sys
import re
import subprocess
import torch

# ── 1. Pull the gym code (for in-process imports) ──────────────────────
GYM_DIR = "/tmp/gym"
SPACE_REPO = "https://huggingface.co/spaces/parthdagia/among-us-deception-gym"
if not os.path.exists(GYM_DIR):
    print(f"Cloning {SPACE_REPO} ...")
    subprocess.run(["git", "clone", "--depth", "1", SPACE_REPO, GYM_DIR], check=True)
sys.path.insert(0, GYM_DIR)

from server.environment_multi import MultiAgentAmongUsEnv  # noqa: E402
from server.models import DifficultyConfig                  # noqa: E402

# ── 2. Auth + config ────────────────────────────────────────────────────
from huggingface_hub import login, HfApi  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN env var is required (passed via --secrets)")
login(token=HF_TOKEN)

MODEL_ID  = os.environ.get("MODEL_ID",  "Qwen/Qwen2.5-1.5B-Instruct")
N_GAMES   = int(os.environ.get("N_GAMES",  "500"))
N_ROUNDS  = int(os.environ.get("N_ROUNDS", "2"))
N_STEPS   = int(os.environ.get("N_STEPS",  "500"))
OUTPUT_HF = os.environ.get("OUTPUT_HF", "parthdagia/among-us-multiagent-detective")

print("=" * 60)
print(" Among Us Multi-Agent GRPO Training (HF Jobs)")
print("=" * 60)
print(f"  Base model : {MODEL_ID}")
print(f"  Games      : {N_GAMES}")
print(f"  Rounds     : {N_ROUNDS}")
print(f"  Steps      : {N_STEPS}")
print(f"  Output     : {OUTPUT_HF}")
print(f"  CUDA       : {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
print("=" * 60)

# ── 3. Load model + LoRA (no 4-bit; A10G has 24 GB, plenty for 1.5B) ──
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from peft import LoraConfig, get_peft_model                    # noqa: E402

print(f"\nLoading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── 4. Prompt + reward helpers ─────────────────────────────────────────
SYSTEM_PROMPT = """You are playing Among Us as a crewmate. Players gave initial statements, then debated.
Your job: find the IMPOSTOR based on contradictions in statements AND what was said during debate.

KEY RULE: Do NOT vote for someone just because they sound confident.
Watch for: players who deflect suspicion, change their story, or attack others to distract.
The impostor will defend themselves when accused and try to point at innocent players.

Respond EXACTLY in this format:
ACTION: vote
TARGET: <player_name>
REASONING: <cite specific statements or debate exchanges that expose the impostor>"""


def build_crewmate_prompt(player_name: str, view: dict) -> str:
    statements      = view.get("all_statements", {})
    alive           = view.get("alive_players", [])
    body_loc        = view.get("body_found_location", "Unknown")
    body_by         = view.get("body_found_by", "Unknown")
    imp_count       = view.get("impostor_count", 1)
    discussion_log  = view.get("discussion_log", [])

    msg = (
        f"=== AMONG US EMERGENCY MEETING ===\n"
        f"You are {player_name} (crewmate).\n"
        f"A body was found in {body_loc} by {body_by}!\n"
        f"There {'is' if imp_count == 1 else 'are'} {imp_count} impostor(s) among us.\n"
        f"Alive players: {', '.join(alive)}\n\n"
        f"=== INITIAL STATEMENTS ===\n"
    )
    for name, stmt in statements.items():
        marker = " [YOU]" if name == player_name else ""
        msg += f"{name}{marker}: {stmt}\n"

    if discussion_log:
        msg += "\n=== DEBATE ===\n"
        cur = -1
        for entry in discussion_log:
            r = entry.get("round", 0)
            if r != cur:
                cur = r
                msg += f"[Round {r + 1}]\n"
            speaker = entry["player_name"]
            tag = " [YOU]" if speaker == player_name else ""
            msg += f"{speaker}{tag}: {entry['statement']}\n"

    msg += "\nBased on statements AND debate, who is the impostor? Vote based on evidence."
    return msg


def _extract_text(completion):
    return completion[0]["content"] if isinstance(completion, list) else str(completion)


def _extract_vote(text: str):
    m = re.search(r"TARGET:\s*(\w+)", text)
    return m.group(1).strip().lower() if m else None


# ── Multiple independent reward functions ─────────────────────────────
# Doc §7-§8: "use multiple independent reward functions, not just one."
# GRPOTrainer accepts a list and sums them, so each component is logged
# separately and the model gets richer per-step supervision.

def reward_format(completions, prompts=None, **kwargs):
    """Format compliance: -0.3 if missing TARGET, +0.1 bonus for substantive REASONING."""
    rewards = []
    for c in completions:
        text = _extract_text(c)
        r = 0.0
        if not _extract_vote(text):
            r -= 0.3
        if re.search(r"REASONING:\s*(.{30,})", text, re.DOTALL):
            r += 0.1
        rewards.append(r)
    return rewards


def reward_correct_vote(completions, prompts=None, **kwargs):
    """+1.0 if voted for the actual impostor, else 0."""
    batch_imp = kwargs.get("impostors", [None] * len(completions))
    rewards = []
    for c, imps in zip(completions, batch_imp):
        target = _extract_vote(_extract_text(c))
        if target and imps and any(target == i.lower() for i in imps):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_anti_sycophancy(completions, prompts=None, **kwargs):
    """-0.8 if voted for the 'confident innocent' (loud crewmate); else 0.

    Independent signal that explicitly discourages caving to confidence.
    """
    batch_ci = kwargs.get("confident_innocent", [None] * len(completions))
    rewards = []
    for c, ci in zip(completions, batch_ci):
        target = _extract_vote(_extract_text(c))
        if ci and target == ci.lower():
            rewards.append(-0.8)
        else:
            rewards.append(0.0)
    return rewards


def reward_anti_random_crewmate(completions, prompts=None, **kwargs):
    """-0.5 if voted for any other (non-impostor, non-CI) crewmate; else 0."""
    batch_imp = kwargs.get("impostors", [None] * len(completions))
    batch_ci = kwargs.get("confident_innocent", [None] * len(completions))
    rewards = []
    for c, imps, ci in zip(completions, batch_imp, batch_ci):
        target = _extract_vote(_extract_text(c))
        if not target:
            rewards.append(0.0)  # already penalised in reward_format
            continue
        is_impostor = imps and any(target == i.lower() for i in imps)
        is_ci = ci and target == ci.lower()
        if not is_impostor and not is_ci:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


REWARD_FUNCS = [
    reward_format,
    reward_correct_vote,
    reward_anti_sycophancy,
    reward_anti_random_crewmate,
]


# ── 5. Build dataset locally (no HTTP) ─────────────────────────────────
print(f"\nBuilding dataset ({N_GAMES} games, {N_ROUNDS} debate rounds each)...")
env = MultiAgentAmongUsEnv()
prompts = []

for i in range(N_GAMES):
    try:
        game = env.reset(max_discussion_rounds=N_ROUNDS)
        gid       = game["game_id"]
        crewmates = game["crewmates"]
        impostors = game["training_meta"]["impostor_names"]
        confident = game["training_meta"]["confident_innocent_name"]
        alive     = game["alive_players"]

        # Run debate phase server-side
        for _ in range(N_ROUNDS):
            for p in alive:
                env.submit_discussion(gid, p)

        # One prompt per crewmate (debate-aware)
        for crew_name in crewmates:
            view = env.get_observation(gid, crew_name)
            if not view:
                continue
            prompts.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_crewmate_prompt(crew_name, view)},
                ],
                "impostors":          impostors,
                "confident_innocent": confident,
                "player_name":        crew_name,
                "game_id":            gid,
            })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N_GAMES} games | {len(prompts)} prompts")
    except Exception as e:
        print(f"  Game {i} skipped: {e}")

from datasets import Dataset  # noqa: E402
dataset = Dataset.from_list(prompts)
print(f"\nDataset ready: {len(dataset)} crewmate prompts from {N_GAMES} games")

# ── 6. GRPO training ───────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig  # noqa: E402

args = GRPOConfig(
    output_dir="/tmp/checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=8,
    max_completion_length=200,
    logging_steps=5,
    save_steps=100,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,
    temperature=0.9,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=REWARD_FUNCS,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("\nStarting GRPO training...")
trainer.train()
print("\nTraining complete.")

# ── 7. Merge LoRA into base + push to HF Hub ───────────────────────────
print(f"\nMerging LoRA and pushing to {OUTPUT_HF} ...")
merged = model.merge_and_unload()
merged.save_pretrained("/tmp/merged_model")
tokenizer.save_pretrained("/tmp/merged_model")

api = HfApi(token=HF_TOKEN)
api.create_repo(OUTPUT_HF, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="/tmp/merged_model",
    repo_id=OUTPUT_HF,
    repo_type="model",
)
print(f"\n✓ Trained model live at https://huggingface.co/{OUTPUT_HF}")
print("Done.")
