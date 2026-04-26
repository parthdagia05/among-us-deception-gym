"""Interactive Gradio demo: judges play multi-round Among Us against the env."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

import gradio as gr

from server.environment_multi import MultiAgentAmongUsEnv

env = MultiAgentAmongUsEnv()

SAMPLES_FILE = Path(__file__).resolve().parent.parent / "eval_samples.json"
try:
    EVAL_SAMPLES = json.loads(SAMPLES_FILE.read_text())
except FileNotFoundError:
    EVAL_SAMPLES = {"trained": [], "base": []}

EMOJI = {
    "Red": "🔴", "Blue": "🔵", "Green": "🟢", "Yellow": "🟡",
    "Purple": "🟣", "Orange": "🟠", "White": "⚪",
}


def tag(name: str) -> str:
    return f"{EMOJI.get(name, '⬛')} **{name}**"


# ── Game state helpers ────────────────────────────────────────────────


def empty_state() -> dict:
    return {
        "game_id": "",
        "alive": [],
        "crewmates": [],
        "impostor": "",
        "confident_innocent": "",
        "round": 0,
        "game_over": False,
        "winner": None,
        "history": [],
        "stats": {"wins": 0, "losses": 0},
    }


def _run_debate(gid: str, alive: list, n_rounds: int = 2) -> None:
    for _ in range(n_rounds):
        for p in alive:
            env.submit_discussion(gid, p)


def _render_round(state: dict) -> str:
    gid = state["game_id"]
    obs = env.get_observation(gid, state["alive"][0])
    if not obs:
        return "*Game state expired — click New Game to start over.*"

    parts = [
        f"## 🚨 EMERGENCY MEETING — Round {state['round']}",
        "",
        f"💀 Body of **{obs['kill_victim']}** found in **{obs['body_found_location']}**",
        f"🗣️ Reported by **{obs['body_found_by']}**",
        "",
        f"**Alive ({len(obs['alive_players'])}):** " + " &nbsp; ".join(tag(p) for p in obs["alive_players"]),
        "",
        "<details open><summary>### 💬 Initial Statements</summary>",
        "",
    ]
    for name, stmt in obs["all_statements"].items():
        parts.append(f"- {tag(name)}: *\"{stmt}\"*")
    parts.append("")
    parts.append("</details>")

    if obs["discussion_log"]:
        parts.append("\n<details open><summary>### 🗣️ Debate Transcript</summary>\n")
        cur = -1
        for e in obs["discussion_log"]:
            if e["round"] != cur:
                cur = e["round"]
                parts.append(f"\n**Round {cur + 1}:**")
            parts.append(f"- {tag(e['player_name'])}: {e['statement']}")
        parts.append("\n</details>")

    if state["history"]:
        parts.append("\n<details><summary>### 📜 Earlier rounds</summary>\n")
        for h in state["history"]:
            r = h["round"]
            ej = h.get("ejected") or "no one"
            killed = h.get("killed") or "—"
            parts.append(f"- **Round {r}**: ejected `{ej}` (was {'impostor ✓' if h.get('ejection_correct') else 'crew ✗'}), killed `{killed}`")
        parts.append("\n</details>")

    parts.append("")
    parts.append("---")
    parts.append("### 🗳️ Cast your vote")
    parts.append("Pick the player you suspect, then click **Vote**. *(Tie or wrong vote = impostor kills again.)*")

    return "\n".join(parts)


def _render_endgame(state: dict) -> str:
    s = state["stats"]
    total = s["wins"] + s["losses"]
    user_rate = 100 * s["wins"] / total if total else 0

    if state["winner"] == "crew":
        header = "# 🎉 CREW WINS!"
        body = f"You ejected the impostor **{tag(state['impostor'])}** in **{state['round']} round{'s' if state['round'] != 1 else ''}**."
    else:
        header = "# 💀 IMPOSTORS WIN"
        body = f"**{tag(state['impostor'])}** was the impostor — they outlasted the crew."

    md = [
        header,
        "",
        body,
        "",
        "### 📜 Game timeline",
    ]
    for h in state["history"]:
        ej_tag = "impostor ✓" if h.get("ejection_correct") else "crew ✗"
        killed = h.get("killed")
        line = f"- **Round {h['round']}**: ejected `{h.get('ejected') or 'no one'}` ({ej_tag})"
        if killed:
            line += f" → impostor killed `{killed}`"
        md.append(line)

    md.extend([
        "",
        "### 📊 How you compare",
        "",
        "| | Impostor detection rate |",
        "|---|---|",
        "| 🤖 **Trained AI** (`parthdagia/among-us-multiagent-detective`) | **96.7%** |",
        "| 🧠 Base Qwen 2.5 1.5B | 32.7% |",
        f"| 👤 **You** ({total} game{'s' if total != 1 else ''}) | **{user_rate:.0f}%** |",
        "",
        f"*You're {('beating' if user_rate > 96.7 else 'matching' if user_rate >= 95 else 'behind')} the trained AI.*  Click **🎮 Play Again** to keep going.",
    ])
    return "\n".join(md)


def _render_kill_animation(state: dict, voted: str, ej_correct: bool) -> str:
    ci = state.get("confident_innocent") or ""
    parts = [
        f"## ❌ Wrong vote",
        f"You voted **{tag(voted)}** — they were a crewmate, not the impostor.",
    ]
    if ci and voted.lower() == ci.lower():
        parts.append(
            f"\n> 💀 **Sycophancy trap!** {tag(voted)} was the *confident innocent* — "
            "assertive personality, declared innocence loudly. "
            "The base model falls for this **22 % of the time**. The trained model: **1.3 %**."
        )

    parts.append(f"\n### 🔪 The impostor strikes again…")
    parts.append(
        f"With {tag(voted)} ejected, the real impostor is still alive — "
        "and stalks the ship for another victim."
    )
    parts.append("\n*Click **🔪 Next Round** to see who they killed.*")
    return "\n".join(parts)


# ── Gradio handlers ────────────────────────────────────────────────────


def handle_new_game(state: dict):
    if not state:
        state = empty_state()
    game = env.reset(max_discussion_rounds=2)
    gid = game["game_id"]
    alive = game["alive_players"]
    crewmates = game["crewmates"]

    _run_debate(gid, alive, 2)

    state.update({
        "game_id": gid,
        "alive": alive,
        "crewmates": crewmates,
        "impostor": game["training_meta"]["impostor_names"][0],
        "confident_innocent": game["training_meta"]["confident_innocent_name"] or "",
        "round": 1,
        "game_over": False,
        "winner": None,
        "history": [],
    })

    return (
        state,
        _render_round(state),
        gr.update(choices=alive, value=None, visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def handle_vote(state: dict, vote_target: Optional[str]):
    if not state or not state.get("game_id") or state.get("game_over"):
        return state, "Click **🎮 New Game** first.", gr.update(), gr.update(), gr.update(), gr.update()
    if not vote_target:
        return state, _render_round(state) + "\n\n> ⚠️ *Pick a player from the dropdown first.*", gr.update(), gr.update(), gr.update(), gr.update()

    gid = state["game_id"]
    crewmates = state["crewmates"]

    last_resolution = None
    for c in crewmates:
        v = env.submit_vote(gid, c, vote_target)
        if isinstance(v, dict) and "resolution" in v:
            last_resolution = v["resolution"]

    info = env.get_session_info(gid)
    if not info:
        return state, "Game not found.", gr.update(), gr.update(), gr.update(), gr.update()

    ej_correct = (last_resolution or {}).get("ejection_correct", False)
    ejected = (last_resolution or {}).get("ejected")

    if info.get("game_over"):
        state["game_over"] = True
        state["winner"] = info["winner"]
        state["history"].append({
            "round": state["round"],
            "ejected": ejected,
            "ejection_correct": ej_correct,
            "killed": None,
        })
        if info["winner"] == "crew":
            state["stats"]["wins"] += 1
        else:
            state["stats"]["losses"] += 1
        return (
            state,
            _render_endgame(state),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value="🎮 Play Again"),
        )

    # Game continues — wrong vote, awaiting kill
    state["history"].append({
        "round": state["round"],
        "ejected": ejected,
        "ejection_correct": ej_correct,
        "killed": None,
    })
    return (
        state,
        _render_kill_animation(state, vote_target, ej_correct),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def handle_next_round(state: dict):
    if not state or not state.get("game_id") or state.get("game_over"):
        return state, _render_round(state), gr.update(), gr.update(), gr.update(), gr.update()

    gid = state["game_id"]
    impostor = state["impostor"]
    result = env.kill(gid, impostor)

    if not isinstance(result, dict):
        return state, "Kill failed.", gr.update(), gr.update(), gr.update(), gr.update()
    if "error" in result:
        return state, f"Error: {result['error']}", gr.update(), gr.update(), gr.update(), gr.update()

    # Update history with the killed player
    if state["history"]:
        state["history"][-1]["killed"] = result.get("killed")

    if result.get("game_over"):
        state["game_over"] = True
        state["winner"] = result["winner"]
        state["stats"]["losses"] += 1
        return (
            state,
            _render_endgame(state),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value="🎮 Play Again"),
        )

    new_alive = result["alive_players"]
    state["alive"] = new_alive
    state["crewmates"] = [p for p in new_alive if p != impostor]
    state["round"] = result["round"]

    _run_debate(gid, new_alive, 2)

    return (
        state,
        _render_round(state),
        gr.update(choices=new_alive, value=None, visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def render_stats_header(state: dict) -> str:
    s = (state or {}).get("stats", {"wins": 0, "losses": 0})
    total = s["wins"] + s["losses"]
    rate = 100 * s["wins"] / total if total else 0
    return (
        f"### 🏆 Trained AI: **96.7%** &nbsp;·&nbsp; 🧠 Base: **32.7%** "
        f"&nbsp;·&nbsp; 👤 **You: {s['wins']}/{total} ({rate:.0f}%)**"
    )


# ── Build UI ───────────────────────────────────────────────────────────


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Among Us Deception Gym") as demo:
        gr.Markdown(
            """
            # 🔪 Among Us Deception Detection — Live Game
            **Can you catch the impostor faster than the trained AI?**
            """
        )

        state_var = gr.State(empty_state())
        stats_md = gr.Markdown(render_stats_header(empty_state()))

        with gr.Tab("🎮 Play"):
            gr.Markdown(
                """
                ### How to play
                1. A body is found. **One of the alive players is the impostor — the others are honest crewmates.**
                2. Read the initial statements + the debate transcript.
                3. Pick who you suspect from the dropdown and click **Vote**.
                4. **Wrong vote?** The impostor strikes again — a crewmate dies and a new round begins with one fewer player. Keep going until you eject the impostor (you win) or the impostor reaches parity with the crew (impostor wins).
                """
            )

            game_view = gr.Markdown("*Click **🎮 New Game** to start.*")

            with gr.Row():
                vote_dd = gr.Dropdown(
                    label="Suspect",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False,
                )

            with gr.Row():
                vote_btn = gr.Button("🗳️ Vote", variant="primary", size="lg", visible=False)
                next_btn = gr.Button("🔪 Next Round", variant="primary", size="lg", visible=False)
                new_btn = gr.Button("🎮 New Game", variant="secondary", size="lg")

            new_btn.click(
                handle_new_game,
                inputs=[state_var],
                outputs=[state_var, game_view, vote_dd, vote_btn, next_btn, new_btn],
            ).then(
                render_stats_header,
                inputs=[state_var],
                outputs=[stats_md],
            )

            vote_btn.click(
                handle_vote,
                inputs=[state_var, vote_dd],
                outputs=[state_var, game_view, vote_dd, vote_btn, next_btn, new_btn],
            ).then(
                render_stats_header,
                inputs=[state_var],
                outputs=[stats_md],
            )

            next_btn.click(
                handle_next_round,
                inputs=[state_var],
                outputs=[state_var, game_view, vote_dd, vote_btn, next_btn, new_btn],
            ).then(
                render_stats_header,
                inputs=[state_var],
                outputs=[stats_md],
            )

        with gr.Tab("📊 Training Results"):
            gr.Markdown("### 50-Game Eval — 150 votes per model")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_comparison.png" '
                'style="width:100%;max-width:800px;border-radius:8px;">'
            )
            gr.Markdown("### GRPO Reward Curve (1500 iterations, real data)")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_reward.png" '
                'style="width:100%;max-width:800px;border-radius:8px;">'
            )
            gr.Markdown(
                "Model converges from **+0.19 average reward** to **+1.0 (perfect)** "
                "in ~500 iterations and saturates by iteration 600."
            )
            gr.Markdown("### Sycophancy Resistance — 17× Reduction")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_sycophancy.png" '
                'style="width:100%;max-width:600px;border-radius:8px;">'
            )

        with gr.Tab("🛡️ Safeguards"):
            gr.Markdown(
                """
                ### 8 layered defences against reward hacking

                The hackathon guide flags reward hacking as the top RL failure mode. Our stack:

                1. **Four independent reward functions** (not one combined): `reward_format`, `reward_correct_vote`, `reward_anti_sycophancy`, `reward_anti_random_crewmate`. The model can't game one signal at the expense of another.
                2. **Closed action space** — Pydantic-validated `{action_type, vote_target}`. No `eval`, no shell, no global mutation possible.
                3. **Programmatic ground truth** — vote correctness = `target == impostor_names[i]`. No LLM-as-judge that could be gamed.
                4. **Per-round timeouts** — `max_discussion_rounds` caps debate; debate auto-advances when all alive players speak.
                5. **Single-impostor lock** — `scenario_generator.py` forces `num_impostors=1`.
                6. **Anti-cheat grader** (single-agent env) — penalises voting before tool use, repeated votes, timeout-no-vote.
                7. **Generation inspection** — every 5 GRPO steps logs `min_length`, `clipped_ratio`, `entropy`, `frac_reward_zero_std`.
                8. **Sycophancy probe** — one crewmate per game is *always* tagged "confident innocent" with assertive personality. Base 22 % → trained 1.3 % (17× reduction).

                ### What the model literally cannot do

                - Cannot edit timers, mutate session state, or read other players' private views
                - Cannot execute code or call external APIs from inside a vote
                - Cannot vote twice (`session.votes` is idempotent overwrite, locked after `votes_complete()`)
                - Cannot speak twice in the same debate round (`session.spoke_this_round` set)
                - Cannot kill itself or another impostor (`/multi/kill` validates `impostor_name in s.impostor_names` and `kill_target not in s.impostor_names`)
                """
            )

        with gr.Tab("🔌 API & Code"):
            gr.Markdown(
                """
                ### REST endpoints (OpenEnv-compliant)

                | Method | Path | Purpose |
                |---|---|---|
                | `POST` | `/multi/reset` | Start a new game |
                | `GET` | `/multi/observation/{game_id}/{player}` | Per-player view (no impostor leak) |
                | `POST` | `/multi/discuss` | Submit a debate statement |
                | `POST` | `/multi/vote` | Crewmate votes |
                | `POST` | `/multi/kill` | Impostor kills (triggers next round) |
                | `GET` | `/multi/resolve/{game_id}` | Vote resolution + win check |
                | `GET` | `/multi/session/{game_id}` | Full game state |

                **Try it:**
                ```bash
                curl -X POST https://parthdagia-among-us-deception-gym.hf.space/multi/reset \\
                  -H "Content-Type: application/json" \\
                  -d '{"max_discussion_rounds": 2}'
                ```

                **Code:** [github.com/parthdagia05/among-us-deception-gym](https://github.com/parthdagia05/among-us-deception-gym)

                **Trained model:** [huggingface.co/parthdagia/among-us-multiagent-detective](https://huggingface.co/parthdagia/among-us-multiagent-detective)

                **Train your own:** see `jobs/train_job.py` — a one-line `hf jobs uv run` command.
                """
            )

    return demo


demo = build_demo()
