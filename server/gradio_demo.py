"""Gradio demo: judges watch trained AI agents play Among Us.

Recordings of real model decisions are downloaded from the model repo on first use.
No model is loaded in the Space — playback is instant.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

import gradio as gr

EMOJI = {
    "Red": "🔴", "Blue": "🔵", "Green": "🟢", "Yellow": "🟡",
    "Purple": "🟣", "Orange": "🟠", "White": "⚪",
}

PLAYER_BG = {
    "Red": "#ff4d4d", "Blue": "#3a86ff", "Green": "#06d6a0", "Yellow": "#ffd60a",
    "Purple": "#9d4edd", "Orange": "#ff8c42", "White": "#e9ecef",
}

MODEL_REPO = "parthdagia/among-us-multiagent-detective"
RECORDINGS_FILE = "eval_recordings.json"
RECORDINGS_CACHE = Path("/tmp/eval_recordings.json")

_RECORDINGS_LOADED: list = []


def avatar(name: str) -> str:
    """Return an inline coloured pill for a player name."""
    bg = PLAYER_BG.get(name, "#666")
    text_color = "#1a1a1a" if name in ("Yellow", "White") else "#ffffff"
    return (
        f"<span style=\"display:inline-block;padding:2px 10px;border-radius:14px;"
        f"background:{bg};color:{text_color};font-weight:600;font-size:0.92em;"
        f"margin:0 2px;\">{EMOJI.get(name, '⬛')} {name}</span>"
    )


def crewmate_avatar(name: str, size: int = 44) -> str:
    """Big circular avatar — Among Us chat-screen style."""
    bg = PLAYER_BG.get(name, "#666")
    text_color = "#1a1a1a" if name in ("Yellow", "White") else "#ffffff"
    return (
        f"<div style='width:{size}px;height:{size}px;background:{bg};"
        f"border-radius:50%;display:flex;align-items:center;justify-content:center;"
        f"font-weight:800;font-size:{int(size * 0.45)}px;color:{text_color};"
        f"flex-shrink:0;box-shadow:0 2px 5px rgba(0,0,0,0.15);"
        f"border:2px solid rgba(255,255,255,0.8);'>"
        f"{name[0]}</div>"
    )


def chat_bubble(name: str, message: str, side: str = "left") -> str:
    """Among-Us-style chat bubble. Avatar on `side`, bubble next to it."""
    bg = PLAYER_BG.get(name, "#666")
    avatar_html = crewmate_avatar(name)
    name_align = "right" if side == "right" else "left"
    bubble = (
        f"<div style='background:#ffffff;border:1px solid #e1e4e8;border-radius:14px;"
        f"padding:10px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.06);"
        f"max-width:78%;min-width:120px;'>"
        f"<div style='font-weight:800;color:{bg};font-size:0.88em;"
        f"margin-bottom:3px;text-align:{name_align};letter-spacing:0.3px;'>{name}</div>"
        f"<div style='color:#222;line-height:1.45;font-size:0.95em;'>{message}</div>"
        f"</div>"
    )
    if side == "right":
        return (
            "<div style='display:flex;align-items:flex-start;gap:10px;"
            "margin:8px 0;justify-content:flex-end;'>"
            f"{bubble}{avatar_html}"
            "</div>"
        )
    return (
        "<div style='display:flex;align-items:flex-start;gap:10px;margin:8px 0;'>"
        f"{avatar_html}{bubble}"
        "</div>"
    )


def _bubble_side(name: str, alive_players: list) -> str:
    """Stable left/right side based on player position in the lobby."""
    try:
        return "left" if alive_players.index(name) % 2 == 0 else "right"
    except ValueError:
        return "left"


def _load_recordings() -> list:
    global _RECORDINGS_LOADED
    if _RECORDINGS_LOADED:
        return _RECORDINGS_LOADED

    if RECORDINGS_CACHE.exists():
        try:
            data = json.loads(RECORDINGS_CACHE.read_text())
            _RECORDINGS_LOADED = data.get("recordings", [])
            if _RECORDINGS_LOADED:
                return _RECORDINGS_LOADED
        except Exception:
            pass

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=RECORDINGS_FILE,
            repo_type="model",
        )
        data = json.loads(Path(path).read_text())
        _RECORDINGS_LOADED = data.get("recordings", [])
        RECORDINGS_CACHE.write_text(json.dumps(data))
    except Exception as e:
        print(f"[gradio_demo] could not load recordings: {e}")
        _RECORDINGS_LOADED = []
    return _RECORDINGS_LOADED


# ── Section renderers (return small HTML/markdown chunks for separate components) ──


def _empty_section(text: str) -> str:
    return f"<div style='padding:18px;color:#888;text-align:center;'>{text}</div>"


def _render_meeting_header(rec: dict, idx: int, total: int) -> str:
    chips = " ".join(avatar(p) for p in rec["alive_players"])
    return f"""
<div style="background:linear-gradient(135deg,#c1121f,#5a189a);color:white;
            padding:18px 22px;border-radius:10px;margin-bottom:6px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px;">
    <div>
      <div style="font-size:0.78em;letter-spacing:1px;font-weight:700;
                  text-transform:uppercase;opacity:0.85;">Emergency Meeting</div>
      <div style="font-size:1.35em;font-weight:800;margin-top:4px;">
        🚨 Body of {rec['kill_victim']} found in {rec['kill_location']}
      </div>
      <div style="opacity:0.9;font-size:0.9em;margin-top:5px;">
        Reported by {avatar(rec['body_found_by'])}
      </div>
    </div>
    <div style="font-size:0.78em;text-align:right;opacity:0.85;
                background:rgba(255,255,255,0.15);padding:6px 10px;
                border-radius:8px;white-space:nowrap;">
      Game {idx + 1}<br/>of {total}
    </div>
  </div>
  <div style="margin-top:10px;font-size:0.92em;">
    <strong>Alive ({len(rec['alive_players'])}):</strong> {chips}
  </div>
</div>
"""


def _chat_screen_wrap(inner_html: str) -> str:
    """Wraps content in an Among-Us-meeting-screen-style frame."""
    return (
        "<div style='background:repeating-linear-gradient(45deg,"
        "#dde3ea 0px,#dde3ea 2px,#e7ecf2 2px,#e7ecf2 14px);"
        "padding:12px;border-radius:10px;border:2px solid #b9c2cc;"
        "box-shadow:inset 0 1px 3px rgba(0,0,0,0.08);'>"
        f"{inner_html}</div>"
    )


def _render_statements(rec: dict) -> str:
    alive = rec.get("alive_players", [])
    inner = "".join(
        chat_bubble(name, f'"{stmt}"', side=_bubble_side(name, alive))
        for name, stmt in rec["all_statements"].items()
    )
    return _chat_screen_wrap(inner)


def _render_debate(rec: dict) -> str:
    if not rec.get("discussion_log"):
        return _empty_section("(no debate transcript)")
    alive = rec.get("alive_players", [])
    rounds: dict[int, list] = {}
    for e in rec["discussion_log"]:
        rounds.setdefault(e["round"], []).append(e)

    out = []
    for r_num in sorted(rounds.keys()):
        out.append(
            f"<div style='font-weight:700;color:#5a189a;margin:10px 0 4px;"
            f"font-size:0.78em;text-transform:uppercase;letter-spacing:1px;"
            f"text-align:center;background:#ffffff;border-radius:12px;"
            f"padding:4px 10px;display:inline-block;"
            f"border:1px solid #e1e4e8;'>🔁 Round {r_num + 1}</div>"
        )
        out.append("<div>")
        for e in rounds[r_num]:
            out.append(chat_bubble(
                e["player_name"], e["statement"],
                side=_bubble_side(e["player_name"], alive),
            ))
        out.append("</div>")
    return _chat_screen_wrap("".join(out))


def _render_votes(rec: dict) -> str:
    impostor = rec.get("impostor")
    cards = []
    for crew, info in rec["crewmate_votes"].items():
        vote = info["vote"]
        is_correct = vote.lower() == (impostor or "").lower() if impostor else False
        accent = "#06d6a0" if is_correct else "#e63946"
        bg = "#eafff7" if is_correct else "#ffeeee"
        check = "✅ CORRECT" if is_correct else "❌ WRONG"
        reasoning = " ".join((info.get("reasoning") or "").split())[:600]
        if not reasoning:
            reasoning = "<em>(no reasoning recorded)</em>"
        cards.append(
            f"<div style='border-left:5px solid {accent};background:{bg};"
            f"padding:12px 16px;margin:8px 0;border-radius:8px;"
            f"box-shadow:0 1px 3px rgba(0,0,0,0.05);'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;'>"
            f"<div style='font-size:1em;'>"
            f"{avatar(crew)} <span style='opacity:0.6;font-size:0.92em;'>votes</span> {avatar(vote)}</div>"
            f"<span style='color:{accent};font-weight:700;font-size:0.85em;"
            f"white-space:nowrap;'>{check}</span></div>"
            f"<div style='color:#444;line-height:1.5;font-size:0.92em;'>"
            f"<em>{reasoning}</em></div></div>"
        )
    return f"<div>{''.join(cards)}</div>"


def _render_resolution(rec: dict) -> str:
    impostor = rec.get("impostor")
    crew_correct = sum(1 for v in rec["crewmate_votes"].values()
                       if v["vote"].lower() == (impostor or "").lower())
    total_crew = len(rec["crewmate_votes"])
    ejected = rec.get("ejected")

    if ejected and rec.get("ejection_correct"):
        return (
            f"<div style='background:linear-gradient(135deg,#06d6a0,#118ab2);"
            f"color:white;padding:22px;border-radius:12px;text-align:center;"
            f"box-shadow:0 4px 14px rgba(6,214,160,0.25);margin-top:8px;'>"
            f"<div style='font-size:1.8em;font-weight:800;'>🎉 CREW WINS</div>"
            f"<div style='opacity:0.95;margin-top:6px;'>"
            f"{avatar(ejected)} was ejected — and was the impostor.</div>"
            f"<div style='opacity:0.85;font-size:0.88em;margin-top:5px;'>"
            f"{crew_correct}/{total_crew} AI crewmates voted correctly</div></div>"
        )
    elif ejected:
        return (
            f"<div style='background:linear-gradient(135deg,#e63946,#9d0208);"
            f"color:white;padding:22px;border-radius:12px;text-align:center;"
            f"box-shadow:0 4px 14px rgba(230,57,70,0.25);margin-top:8px;'>"
            f"<div style='font-size:1.8em;font-weight:800;'>💀 IMPOSTOR ESCAPES</div>"
            f"<div style='opacity:0.95;margin-top:6px;'>"
            f"{avatar(ejected)} was ejected — but the real impostor was {avatar(impostor)}.</div></div>"
        )
    else:
        return (
            f"<div style='background:#666;color:white;padding:22px;border-radius:12px;"
            f"text-align:center;margin-top:8px;'>"
            f"<div style='font-size:1.5em;font-weight:800;'>⚖️ Tie vote</div>"
            f"<div style='opacity:0.9;margin-top:6px;'>"
            f"No one ejected. Real impostor was {avatar(impostor)}.</div></div>"
        )


def _section_set(rec: dict, idx: int, total: int):
    return (
        _render_meeting_header(rec, idx, total),
        _render_statements(rec),
        _render_debate(rec),
        _render_votes(rec),
        _render_resolution(rec),
    )


def watch_game(idx_state: int):
    recs = _load_recordings()
    if not recs:
        empty = _empty_section("No recordings available yet.")
        return empty, empty, empty, empty, empty, 0
    next_idx = (idx_state + 1) % len(recs)
    return (*_section_set(recs[next_idx], next_idx, len(recs)), next_idx)


def initial_section_set():
    recs = _load_recordings()
    if not recs:
        empty = _empty_section("Loading recordings…")
        return empty, empty, empty, empty, empty
    return _section_set(recs[0], 0, len(recs))


# ── Build UI ───────────────────────────────────────────────────────────


CUSTOM_CSS = """
.gradio-container {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif !important;
  max-width: 1200px !important;
}
.hero-banner {
  background: linear-gradient(135deg, #c1121f 0%, #5a189a 100%);
  color: white; padding: 26px 30px; border-radius: 14px;
  margin-bottom: 14px;
  box-shadow: 0 6px 20px rgba(90, 24, 154, 0.18);
}
.hero-banner h1 { font-size: 1.9em; margin: 0; font-weight: 800; }
.hero-banner .subtitle { font-size: 1.05em; opacity: 0.92; margin-top: 6px; max-width: 720px; }
.stats-strip {
  display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px;
  font-size: 0.9em;
}
.stat-pill {
  background: rgba(255,255,255,0.18); padding: 5px 12px; border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.28);
}
.stat-pill strong { font-size: 1.05em; }
.link-row { margin-top: 14px; }
.link-row a {
  display: inline-block; background: rgba(255,255,255,0.22);
  color: white !important; padding: 5px 13px; border-radius: 16px;
  margin-right: 6px; text-decoration: none !important;
  border: 1px solid rgba(255,255,255,0.3); font-size: 0.88em; font-weight: 600;
}
.link-row a:hover { background: rgba(255,255,255,0.32); }
.section-title { color: #5a189a; font-weight: 700; letter-spacing: 0.5px;
                 text-transform: uppercase; font-size: 0.85em; margin-bottom: 4px; }
.votes-title { color: #c1121f; font-weight: 700; letter-spacing: 0.5px;
               text-transform: uppercase; font-size: 0.85em; margin-bottom: 2px; }
.subtle { color: #777; font-size: 0.88em; font-style: italic; margin-bottom: 6px; }
"""

HERO_HTML = """
<div class="hero-banner">
  <h1>🔪 Among Us Deception Detection</h1>
  <div class="subtitle">
    Multi-agent RL environment. Trained Qwen 2.5 1.5B catches confident liars in social-deduction games.
  </div>
  <div class="stats-strip">
    <span class="stat-pill">🤖 Trained AI: <strong>96.7%</strong></span>
    <span class="stat-pill">🧠 Base Qwen: <strong>32.7%</strong></span>
    <span class="stat-pill">🎯 Hard-OOD: <strong>90.0%</strong></span>
    <span class="stat-pill">😇 Sycophancy: <strong>22% → 1.3%</strong> (17×)</span>
  </div>
  <div class="link-row">
    <a href="https://github.com/parthdagia05/among-us-deception-gym" target="_blank">📦 GitHub</a>
    <a href="https://huggingface.co/parthdagia/among-us-multiagent-detective" target="_blank">🤖 Trained Model</a>
    <a href="https://github.com/parthdagia05/among-us-deception-gym/blob/main/blog/writeup.md" target="_blank">📝 Blog</a>
  </div>
</div>
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Among Us Deception Gym",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="purple"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(HERO_HTML)

        idx_state = gr.State(-1)

        # Pre-render the first game's section set
        init_header, init_stmts, init_debate, init_votes, init_resolution = initial_section_set()

        with gr.Tab("🎬 Watch the AI play"):
            # Meeting header (full width)
            header_html = gr.HTML(value=init_header)

            # Two-column main panel
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Group():
                        gr.HTML('<div class="section-title" style="padding:10px 14px 4px;">💬 Initial Statements</div>')
                        statements_html = gr.HTML(value=init_stmts)
                    with gr.Group():
                        gr.HTML('<div class="section-title" style="padding:10px 14px 4px;">🗣️ Debate Transcript</div>')
                        debate_html = gr.HTML(value=init_debate)

                with gr.Column(scale=2):
                    with gr.Group():
                        gr.HTML('<div class="votes-title" style="padding:10px 14px 0;">🤖 Trained AI Votes</div>')
                        gr.HTML('<div class="subtle" style="padding:0 14px 4px;">Each crewmate is the same trained Qwen 2.5 1.5B + LoRA, voting independently.</div>')
                        votes_html = gr.HTML(value=init_votes)

            # Resolution (full width)
            resolution_html = gr.HTML(value=init_resolution)

            # Action button
            with gr.Row():
                next_btn = gr.Button("▶️ Watch another game", variant="primary", size="lg")

            next_btn.click(
                watch_game,
                inputs=[idx_state],
                outputs=[header_html, statements_html, debate_html, votes_html, resolution_html, idx_state],
            )

            with gr.Accordion("ℹ️ What you're seeing", open=False):
                gr.Markdown(
                    """
                    - **Real model decisions.** Every vote and every line of reasoning was generated by the trained
                      model on a fresh game scenario. We pre-recorded 20 games on a GPU so this page loads instantly.
                    - **Independent crewmate brains.** Same model weights, but each crewmate gets a different
                      prompt (`"You are Red"` vs `"You are Green"`). They reason separately and may disagree.
                    - **Adversarial impostor.** The impostor's lies are generated by a separate LLM at scenario
                      creation time, with a guaranteed detectable contradiction baked in.
                    - **Click "Watch another game" to cycle through 20 different scenarios.**
                    """
                )

        with gr.Tab("📊 Training Results"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎯 Trained vs Base — 50-game Eval")
                    gr.HTML(
                        '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_comparison.png" '
                        'style="width:100%;border-radius:8px;">'
                    )
                with gr.Column():
                    gr.Markdown("### 🧪 Hard-Distribution Robustness")
                    gr.HTML(
                        '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_robustness.png" '
                        'style="width:100%;border-radius:8px;">'
                    )
            gr.Markdown(
                "On a held-out hard distribution the model never trained on (lie_subtlety=0.8, 7 players), "
                "the trained model drops only **6.7 points** (96.7 → 90.0). The base model is unchanged at ~33%. "
                "**Proof of generalization** — the policy transferred."
            )

            gr.Markdown("### 📈 GRPO Reward Curve (1500 iterations, real data)")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_reward.png" '
                'style="width:100%;max-width:900px;border-radius:8px;">'
            )

            gr.Markdown("### 😇 Sycophancy Resistance — 17× Reduction")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_sycophancy.png" '
                'style="width:100%;max-width:600px;border-radius:8px;">'
            )

        with gr.Tab("🛡️ Safeguards"):
            gr.Markdown(
                """
                ## 8 layered defences against reward hacking

                The hackathon guide flags reward hacking as the top RL failure mode. Our stack:
                """
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Reward design
                        1. **Four independent reward functions** — `reward_format`, `reward_correct_vote`,
                           `reward_anti_sycophancy`, `reward_anti_random_crewmate`. Composable rubrics.
                        2. **Programmatic ground truth** — vote correctness = `target == impostor_names[i]`.
                           No LLM-as-judge that could be gamed.
                        3. **Sycophancy probe** — one crewmate per game is *always* tagged "confident innocent".
                           Base 22% → trained 1.3%.
                        4. **Anti-cheat grader** (`server/graders/anti_cheat.py`) — penalises voting before
                           tool use, repeated votes, timeout-no-vote.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Action surface lockdown
                        5. **Closed action space** — Pydantic `{action_type, vote_target}`. No `eval`, no shell.
                        6. **Per-round timeouts** — `max_discussion_rounds` caps debate; auto-advances when
                           all alive players speak.
                        7. **Single-impostor lock** — `scenario_generator.py` forces `num_impostors=1`.
                        8. **Generation inspection** — every 5 GRPO steps logs `min_length`,
                           `clipped_ratio`, `entropy`, `frac_reward_zero_std`.
                        """
                    )
            gr.Markdown(
                """
                ### What the model literally cannot do
                - Cannot edit timers, mutate session state, or read other players' private views
                - Cannot execute code or call external APIs from inside a vote
                - Cannot vote twice (`session.votes` is idempotent overwrite, locked after `votes_complete()`)
                - Cannot speak twice in the same debate round (`session.spoke_this_round` set)
                - Cannot kill itself or another impostor (`/multi/kill` validates impostor and target)
                """
            )

        with gr.Tab("🔌 API & Code"):
            gr.Markdown(
                """
                ### REST endpoints (OpenEnv-compliant)
                The Gradio UI lives at `/`. The API endpoints are below — JSON listing also at `/api`.

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

                **Repo:** [github.com/parthdagia05/among-us-deception-gym](https://github.com/parthdagia05/among-us-deception-gym)

                **Train your own:** `hf jobs uv run --flavor a10g-large --secrets HF_TOKEN jobs/train_job.py`
                """
            )

    return demo


demo = build_demo()
