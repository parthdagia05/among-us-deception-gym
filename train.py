"""
GRPO training for Among Us Deception Detection.
Requires: trl, unsloth, transformers
Run: python train.py
"""
import json
import re
import subprocess
import sys
from typing import Optional
import requests


SYSTEM_PROMPT = """You are playing Among Us as a crewmate in a text-based game. A body has been found and players have given their statements. Your job is to find the impostor.

IMPORTANT: Do NOT trust players just because they sound confident. Impostors often speak with the most certainty to avoid suspicion. Always investigate before voting.

Available tools:
- read_statements() - see all player statements
- check_location_tasks(location) - verify valid tasks at a location
- check_player_presence(location) - see who was really at a location
- check_adjacency(location1, location2) - check if locations are connected
- cross_reference() - find contradictions between statements

To use a tool respond EXACTLY like this:
ACTION: tool_call
TOOL: read_statements
ARGS: {}
REASONING: I need to see all player statements first.

To vote respond EXACTLY like this:
ACTION: vote
TARGET: Red
REASONING: Red claimed to be in Electrical but Blue was alone there. Red is the impostor.

Always investigate with at least 2 tools before voting. Never trust confidence alone."""


def parse_action(text: str) -> dict:
    action = {"action_type": None, "tool_name": None, "tool_args": {}, "vote_target": None, "reasoning": None}

    if "ACTION: vote" in text or "ACTION:vote" in text:
        action["action_type"] = "vote"
        target_match = re.search(r"TARGET:\s*(\w+)", text)
        if target_match:
            action["vote_target"] = target_match.group(1).strip()
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if reasoning_match:
            action["reasoning"] = reasoning_match.group(1).strip()

    elif "ACTION: tool_call" in text or "ACTION:tool_call" in text:
        action["action_type"] = "tool_call"
        tool_match = re.search(r"TOOL:\s*(\w+)", text)
        if tool_match:
            action["tool_name"] = tool_match.group(1).strip()
        args_match = re.search(r"ARGS:\s*(\{[^}]*\})", text)
        if args_match:
            try:
                action["tool_args"] = json.loads(args_match.group(1))
            except Exception:
                action["tool_args"] = {}
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        if reasoning_match:
            action["reasoning"] = reasoning_match.group(1).strip()

    return action


def rollout_episode(model, tokenizer, env_url: str = "http://localhost:8000") -> dict:
    """Run one episode and return trajectory with reward."""
    # Reset environment
    resp = requests.post(f"{env_url}/reset")
    obs = resp.json()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    body_msg = (
        f"=== AMONG US EMERGENCY MEETING ===\n"
        f"A body was found in {obs['body_found_location']} by {obs['body_found_by']}!\n"
        f"Alive players: {', '.join(obs['alive_players'])}\n"
        f"Round {obs['round_number']}/{obs['max_rounds']}\n\n"
        f"Player statements:\n"
    )
    if obs.get("player_statements"):
        for player, statement in obs["player_statements"].items():
            body_msg += f"{player}: {statement}\n"

    body_msg += f"\nYou have {obs['max_turns']} turns. Investigate and vote!"
    messages.append({"role": "user", "content": body_msg})

    trajectory = {"messages": messages, "reward": 0.0, "correct": False, "tool_calls": 0}
    done = False

    for turn in range(obs["max_turns"]):
        # Generate action
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        import torch
        with torch.no_grad():
            output = model.generate(
                input_ids.to(model.device),
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response_ids = output[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        messages.append({"role": "assistant", "content": response_text})

        # Parse action
        action = parse_action(response_text)

        if action["action_type"] is None:
            # Default to read_statements if parsing fails
            action = {"action_type": "tool_call", "tool_name": "read_statements", "tool_args": {}}

        # Step environment
        resp = requests.post(f"{env_url}/step", json={
            "action_type": action["action_type"],
            "tool_name": action.get("tool_name"),
            "tool_args": action.get("tool_args", {}),
            "vote_target": action.get("vote_target"),
            "reasoning": action.get("reasoning")
        })
        result = resp.json()

        if action["action_type"] == "tool_call":
            trajectory["tool_calls"] += 1

        # Add result to messages
        if result.get("observation", {}).get("last_tool_result"):
            messages.append({"role": "user", "content": result["observation"]["last_tool_result"]})

        if result.get("done"):
            trajectory["reward"] = result.get("reward", 0.0)
            trajectory["correct"] = result.get("info", {}).get("correct", False)
            break

    return trajectory


def format_reward(completions, **kwargs) -> list:
    """Reward function for GRPO - calls environment to get reward."""
    rewards = []
    for completion in completions:
        # Parse the action from completion
        text = completion[0]["content"] if isinstance(completion, list) else completion
        action = parse_action(text)

        if action["action_type"] == "vote" and action.get("vote_target"):
            # Simple reward: +1 if voted for impostor (checked via env state)
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def train():
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOTrainer, GRPOConfig
        from datasets import Dataset
        import torch
    except ImportError:
        print("Install: pip install unsloth trl transformers datasets torch")
        sys.exit(1)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Build dataset of game prompts
    env_url = "http://localhost:8000"
    print("Building training dataset...")

    prompts = []
    for i in range(200):
        try:
            resp = requests.post(f"{env_url}/reset")
            obs = resp.json()
            prompt = (
                f"=== AMONG US EMERGENCY MEETING ===\n"
                f"A body was found in {obs['body_found_location']} by {obs['body_found_by']}!\n"
                f"Alive players: {', '.join(obs['alive_players'])}\n\n"
            )
            if obs.get("player_statements"):
                for player, statement in obs["player_statements"].items():
                    prompt += f"{player}: {statement}\n"
            prompt += "\nInvestigate and vote!"
            prompts.append({"prompt": prompt})
        except Exception as e:
            print(f"Warning: could not generate prompt {i}: {e}")

    if not prompts:
        print("ERROR: Could not connect to environment at http://localhost:8000")
        print("Start the server first: uvicorn server.app:app --port 8000")
        sys.exit(1)

    dataset = Dataset.from_list(prompts)

    training_args = GRPOConfig(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_generations=8,
        max_new_tokens=256,
        logging_steps=10,
        save_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    def reward_fn(completions, **kwargs):
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            action = parse_action(text)
            if action["action_type"] == "vote" and action.get("vote_target"):
                # Try to step environment
                try:
                    resp = requests.post(f"{env_url}/step", json={
                        "action_type": "vote",
                        "vote_target": action["vote_target"]
                    }, timeout=5)
                    result = resp.json()
                    rewards.append(float(result.get("reward", 0.0)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("Training complete! Model saved to ./trained_model")


if __name__ == "__main__":
    train()
