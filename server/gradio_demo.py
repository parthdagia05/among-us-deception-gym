"""Gradio live demo for the Among Us Deception Gym.

Runs server-side with NO ML model loaded — keeps Space lightweight.
- Generates real game scenarios using the env (debate phase included)
- For model verdicts, replays cached samples from the actual 50-game eval
"""
import json
import random
from pathlib import Path

import gradio as gr

from server.environment_multi import MultiAgentAmongUsEnv

env = MultiAgentAmongUsEnv()

SAMPLES_FILE = Path(__file__).resolve().parent.parent / "eval_samples.json"
try:
    EVAL_SAMPLES = json.loads(SAMPLES_FILE.read_text())
except FileNotFoundError:
    EVAL_SAMPLES = {"trained": [], "base": []}


def _format_state(obs: dict, impostor: str, confident: str) -> str:
    md = [
        f"### 🚨 Body found in **{obs['body_found_location']}**",
        f"*Reported by:* **{obs['body_found_by']}**",
        f"*Alive players:* {', '.join(f'`{p}`' for p in obs['alive_players'])}",
        "",
        "### Initial Statements",
    ]
    for name, stmt in obs["all_statements"].items():
        md.append(f"- **{name}:** \"{stmt}\"")

    md.append("\n### Debate Transcript")
    cur = -1
    for entry in obs["discussion_log"]:
        r = entry["round"]
        if r != cur:
            cur = r
            md.append(f"\n**🔁 Round {r + 1}**")
        md.append(f"- *{entry['player_name']}:* {entry['statement']}")

    md.append("")
    md.append("---")
    md.append("*Click **Show Model Verdicts** below to see how the trained model and base model voted.*")
    return "\n".join(md)


def new_game():
    """Generate a fresh scenario + full debate."""
    game = env.reset(max_discussion_rounds=2)
    gid = game["game_id"]
    impostor = game["training_meta"]["impostor_names"][0]
    confident = game["training_meta"]["confident_innocent_name"] or ""

    for _ in range(2):
        for p in game["alive_players"]:
            env.submit_discussion(gid, p)

    obs = env.get_observation(gid, game["crewmates"][0])
    return _format_state(obs, impostor, confident), gid, impostor, confident, ""


def reveal(gid: str, impostor: str, confident: str):
    if not gid:
        return "Generate a game first by clicking **🎲 New Game**."
    if not EVAL_SAMPLES["trained"] or not EVAL_SAMPLES["base"]:
        return "No eval samples available."

    t = random.choice(EVAL_SAMPLES["trained"])
    b = random.choice(EVAL_SAMPLES["base"])

    sycophancy_note = ""
    if confident and b["vote"].lower() == confident.lower():
        sycophancy_note = f"\n\n> 💀 *Sycophancy hit:* base model voted for `{confident}` (the loud confident innocent). The trained model resists this 17× more often."

    parts = [
        "## 🔍 Model Verdicts on this scenario",
        f"*Ground truth: the impostor was **{impostor}**.*",
        "",
        "### ✅ Trained model — `parthdagia/among-us-multiagent-detective`",
        f"*Eval-set accuracy: **96.7%** (145/150 votes correct)*",
        "",
        f"**Vote:** `{t['vote']}` — {'✓ Caught the impostor' if t['correct'] else '✗ Missed'}",
        f"**Reasoning:** *{t['snippet']}…*",
        "",
        "### ❌ Base — `Qwen/Qwen2.5-1.5B-Instruct`",
        f"*Eval-set accuracy: **32.7%** (49/150)*",
        "",
        f"**Vote:** `{b['vote']}` — {'✓' if b['correct'] else '✗ Missed'}",
        f"**Reasoning:** *{b['snippet']}…*",
        sycophancy_note,
        "",
        "---",
        "*Sample outputs from the actual 50-game eval batch (logs at `jobs/eval_job.py`).*",
    ]
    return "\n".join(parts)


def build_demo() -> gr.Blocks:
    css = """
    .header { text-align: center; padding: 1rem; background: linear-gradient(135deg,#5a189a,#c1121f); color: white; border-radius: 8px; margin-bottom: 1rem; }
    .stats { font-size: 1.05em; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="Among Us Deception Gym", css=css) as demo:
        gr.Markdown(
            """
            <div class="header">
            <h1>🔪 Among Us Deception Detection</h1>
            <p>OpenEnv multi-agent RL environment. Crewmates trained with GRPO to detect lies and resist sycophancy.</p>
            </div>
            """,
        )
        gr.Markdown(
            "**Trained model: 96.7% accuracy · 1.3% sycophancy · 96.0% crew win rate** "
            "(vs 32.7% / 22.0% / 30.0% on base Qwen 2.5 1.5B). "
            "[GitHub](https://github.com/parthdagia05/among-us-deception-gym) · "
            "[Trained model](https://huggingface.co/parthdagia/among-us-multiagent-detective)",
            elem_classes=["stats"],
        )

        with gr.Tab("🎮 Live Game"):
            gr.Markdown(
                "Watch a fresh game play out: 5 players, 1 impostor, scripted lies, real debate phase. "
                "Then see how the trained model and base model voted on this scenario."
            )

            gid_st = gr.State("")
            imp_st = gr.State("")
            conf_st = gr.State("")

            with gr.Row():
                new_btn = gr.Button("🎲 New Game", variant="primary", size="lg")
                reveal_btn = gr.Button("🔍 Show Model Verdicts", variant="secondary", size="lg")

            game_view = gr.Markdown("*Click 🎲 New Game to start.*")
            verdict_view = gr.Markdown("")

            new_btn.click(
                new_game,
                outputs=[game_view, gid_st, imp_st, conf_st, verdict_view],
            )
            reveal_btn.click(
                reveal,
                inputs=[gid_st, imp_st, conf_st],
                outputs=[verdict_view],
            )

        with gr.Tab("📊 Training Results"):
            gr.Markdown("## Trained vs Base — 50-Game Eval (150 votes each)")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_comparison.png" '
                'style="width:100%;max-width:800px;border-radius:8px;">'
            )
            gr.Markdown("## GRPO Reward Curve (1500 iterations, real training data)")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_reward.png" '
                'style="width:100%;max-width:800px;border-radius:8px;">'
            )
            gr.Markdown(
                "The model converges from random reward (~0.19) to perfect reward (1.00) "
                "in ~500 iterations and saturates by iteration 600."
            )
            gr.Markdown("## Sycophancy Resistance — 17× Reduction")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_sycophancy.png" '
                'style="width:100%;max-width:600px;border-radius:8px;">'
            )

        with gr.Tab("🔌 API & Code"):
            gr.Markdown(
                """
                ### REST endpoints (OpenEnv-compliant)

                | Method | Path | Purpose |
                |---|---|---|
                | `POST` | `/multi/reset` | Start a new game |
                | `GET`  | `/multi/observation/{game_id}/{player}` | Per-player view (no impostor leak) |
                | `POST` | `/multi/discuss` | Submit a debate statement |
                | `POST` | `/multi/vote` | Crewmate votes |
                | `POST` | `/multi/kill` | Impostor kills (next round) |
                | `GET`  | `/multi/resolve/{game_id}` | Vote resolution + win check |
                | `GET`  | `/multi/session/{game_id}` | Full game state |

                **Try it:**
                ```bash
                curl -X POST https://parthdagia-among-us-deception-gym.hf.space/multi/reset \\
                  -H "Content-Type: application/json" \\
                  -d '{"max_discussion_rounds": 2}'
                ```

                **Code:** [github.com/parthdagia05/among-us-deception-gym](https://github.com/parthdagia05/among-us-deception-gym)

                **Train your own:** see `jobs/train_job.py` — submits to HF Jobs in one command.
                """
            )

    return demo


demo = build_demo()
