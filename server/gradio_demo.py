"""Gradio demo: judges watch trained AI agents play Among Us.

Single tablet UI with two screens (chat ↔ voting), Among-Us-style.
Recordings of real model decisions are downloaded from the model repo on first use.
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
PLAYER_DARK = {
    "Red": "#a01313", "Blue": "#1d4ed8", "Green": "#057a5b", "Yellow": "#a37f00",
    "Purple": "#5d2599", "Orange": "#a14a1c", "White": "#888",
}

MODEL_REPO = "parthdagia/among-us-multiagent-detective"
RECORDINGS_FILE = "eval_recordings.json"
RECORDINGS_CACHE = Path("/tmp/eval_recordings.json")

_RECORDINGS_LOADED: list = []


# ── Crewmate SVG (bean shape + visor + backpack, Among-Us-inspired) ───


def crewmate_svg(name: str, size: int = 44, dead: bool = False) -> str:
    """Inline SVG of a stylized crewmate — bean body + glass visor + backpack."""
    body = PLAYER_BG.get(name, "#666")
    dark = PLAYER_DARK.get(name, "#444")
    if dead:
        body = "#7a3a3a"
        dark = "#4a1a1a"
    visor = "#a5d8ff"
    visor_hi = "#e0f2ff"
    return (
        f"<svg viewBox='0 0 60 64' width='{size}' height='{int(size * 1.07)}' "
        f"style='flex-shrink:0;display:block;'>"
        f"<ellipse cx='12' cy='42' rx='6' ry='13' fill='{dark}'/>"      # backpack
        f"<ellipse cx='32' cy='38' rx='20' ry='24' fill='{body}'/>"      # body
        f"<path d='M22,52 Q22,60 26,60 L30,60 L30,55 L22,55 Z' fill='{dark}'/>"  # left leg
        f"<path d='M34,55 L34,60 L38,60 Q42,60 42,52 L42,55 Z' fill='{dark}'/>"  # right leg
        f"<ellipse cx='35' cy='28' rx='14' ry='9' fill='{visor}'/>"      # visor
        f"<ellipse cx='30' cy='26' rx='4' ry='3' fill='{visor_hi}' opacity='0.85'/>"  # highlight
        f"<ellipse cx='44' cy='30' rx='2' ry='1.5' fill='{visor_hi}' opacity='0.6'/>"  # tiny gleam
        + (
            f"<line x1='8' y1='10' x2='52' y2='58' stroke='#c00' stroke-width='5' "
            f"stroke-linecap='round'/>"
            f"<line x1='52' y1='10' x2='8' y2='58' stroke='#c00' stroke-width='5' "
            f"stroke-linecap='round'/>"
            if dead else ""
        )
        + "</svg>"
    )


def chat_bubble(name: str, message: str, side: str = "left") -> str:
    """Among-Us-style chat bubble. Avatar on `side`, bubble next to it."""
    color = PLAYER_BG.get(name, "#666")
    avatar_html = crewmate_svg(name, size=44)
    name_align = "right" if side == "right" else "left"
    bubble = (
        f"<div style='background:#ffffff;border:1px solid #e1e4e8;border-radius:14px;"
        f"padding:9px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.08);"
        f"max-width:78%;min-width:140px;'>"
        f"<div style='font-weight:800;color:{color};font-size:0.9em;"
        f"margin-bottom:3px;text-align:{name_align};letter-spacing:0.4px;'>{name}</div>"
        f"<div style='color:#222;line-height:1.45;font-size:0.93em;'>{message}</div>"
        f"</div>"
    )
    if side == "right":
        return (
            "<div style='display:flex;align-items:flex-end;gap:8px;"
            "margin:6px 0;justify-content:flex-end;'>"
            f"{bubble}{avatar_html}"
            "</div>"
        )
    return (
        "<div style='display:flex;align-items:flex-end;gap:8px;margin:6px 0;'>"
        f"{avatar_html}{bubble}"
        "</div>"
    )


def _bubble_side(name: str, alive_players: list) -> str:
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
            repo_id=MODEL_REPO, filename=RECORDINGS_FILE, repo_type="model",
        )
        data = json.loads(Path(path).read_text())
        _RECORDINGS_LOADED = data.get("recordings", [])
        RECORDINGS_CACHE.write_text(json.dumps(data))
    except Exception as e:
        print(f"[gradio_demo] could not load recordings: {e}")
        _RECORDINGS_LOADED = []
    return _RECORDINGS_LOADED


# ── Chat screen content (scenario + statements + round 1 + round 2) ────


def _round_banner(text: str, color: str = "#5a189a") -> str:
    return (
        f"<div style='display:flex;justify-content:center;margin:14px 0 8px;'>"
        f"<div style='background:#ffffff;color:{color};font-weight:800;"
        f"letter-spacing:1px;text-transform:uppercase;font-size:0.78em;"
        f"border:2px solid {color};padding:5px 16px;border-radius:18px;"
        f"box-shadow:0 2px 6px rgba(0,0,0,0.1);'>{text}</div></div>"
    )


def _render_chat_screen(rec: dict) -> str:
    alive = rec.get("alive_players", [])

    parts = [
        # Body found banner
        f"<div style='background:linear-gradient(135deg,#c1121f,#5a189a);"
        f"color:white;padding:12px 16px;border-radius:10px;margin-bottom:12px;'>"
        f"<div style='font-size:0.78em;letter-spacing:1px;font-weight:700;"
        f"text-transform:uppercase;opacity:0.85;'>🚨 Body Reported</div>"
        f"<div style='font-size:1.08em;font-weight:700;margin-top:3px;'>"
        f"{rec['kill_victim']} found in {rec['kill_location']}</div>"
        f"<div style='font-size:0.86em;opacity:0.92;margin-top:2px;'>"
        f"Reported by <strong>{rec['body_found_by']}</strong></div>"
        f"</div>",
        _round_banner("📜 Initial Statements", "#1d4ed8"),
    ]
    for name, stmt in rec["all_statements"].items():
        parts.append(chat_bubble(name, f'"{stmt}"', side=_bubble_side(name, alive)))

    if rec.get("discussion_log"):
        rounds: dict[int, list] = {}
        for e in rec["discussion_log"]:
            rounds.setdefault(e["round"], []).append(e)
        for r_num in sorted(rounds.keys()):
            parts.append(_round_banner(f"🔁 Debate — Round {r_num + 1}"))
            for e in rounds[r_num]:
                parts.append(chat_bubble(
                    e["player_name"], e["statement"],
                    side=_bubble_side(e["player_name"], alive),
                ))

    parts.append(
        "<div style='display:flex;justify-content:center;margin-top:16px;'>"
        "<div style='background:#fff3cd;border:2px solid #ffb703;color:#7a4f00;"
        "font-weight:700;padding:10px 18px;border-radius:14px;font-size:0.92em;'>"
        "🗳️ Discussion ended — tap the vote icon (top right) to see the verdict</div></div>"
    )
    return "".join(parts)


# ── Voting screen content (player blocks + tiny voter avatars) ─────────


def _render_voting_screen(rec: dict) -> str:
    alive = rec.get("alive_players", [])
    impostor = rec.get("impostor")
    ejected = rec.get("ejected")
    ej_correct = rec.get("ejection_correct", False)

    # tally votes per target
    voters_per_target: dict[str, list[str]] = {}
    for crew, info in rec["crewmate_votes"].items():
        voters_per_target.setdefault(info["vote"], []).append(crew)

    blocks = []
    for player in alive:
        is_ejected = (player == ejected)
        bg_color = "#a8b3bd" if is_ejected else "#ffffff"
        text_opacity = 0.55 if is_ejected else 1.0
        voters = voters_per_target.get(player, [])

        voter_chips = "".join(crewmate_svg(v, size=24) for v in voters)
        voter_html = (
            f"<div style='display:flex;gap:2px;align-items:center;'>{voter_chips}</div>"
            if voters else
            "<div style='color:#888;font-size:0.75em;font-style:italic;'>no votes</div>"
        )

        blocks.append(
            f"<div style='background:{bg_color};border-radius:10px;padding:10px 14px;"
            f"display:flex;align-items:center;gap:14px;"
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);min-height:64px;'>"
            f"<div style='opacity:{text_opacity};'>{crewmate_svg(player, size=44, dead=is_ejected)}</div>"
            f"<div style='flex:1;font-weight:800;letter-spacing:1.5px;"
            f"font-size:1.15em;color:{PLAYER_BG.get(player, '#666')};"
            f"opacity:{text_opacity};text-transform:uppercase;'>{player}</div>"
            f"{voter_html}"
            f"</div>"
        )

    # 2-column grid
    grid_html = (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));"
        "gap:10px;margin-bottom:14px;'>"
        + "".join(blocks)
        + "</div>"
    )

    # Resolution banner
    if ejected and ej_correct:
        resolution = (
            f"<div style='background:linear-gradient(135deg,#06d6a0,#118ab2);"
            f"color:white;padding:18px;border-radius:12px;text-align:center;"
            f"box-shadow:0 4px 14px rgba(6,214,160,0.25);'>"
            f"<div style='font-size:1.4em;font-weight:800;'>🎉 CREW WINS</div>"
            f"<div style='opacity:0.95;margin-top:4px;font-size:0.92em;'>"
            f"<strong>{ejected}</strong> was ejected — and was the impostor.</div></div>"
        )
    elif ejected:
        resolution = (
            f"<div style='background:linear-gradient(135deg,#e63946,#9d0208);"
            f"color:white;padding:18px;border-radius:12px;text-align:center;"
            f"box-shadow:0 4px 14px rgba(230,57,70,0.25);'>"
            f"<div style='font-size:1.4em;font-weight:800;'>💀 IMPOSTOR ESCAPES</div>"
            f"<div style='opacity:0.95;margin-top:4px;font-size:0.92em;'>"
            f"<strong>{ejected}</strong> was ejected — but the real impostor was "
            f"<strong>{impostor}</strong>.</div></div>"
        )
    else:
        resolution = (
            f"<div style='background:#666;color:white;padding:18px;border-radius:12px;"
            f"text-align:center;'>"
            f"<div style='font-size:1.3em;font-weight:800;'>⚖️ Tie vote</div>"
            f"<div style='opacity:0.9;margin-top:4px;font-size:0.92em;'>"
            f"No one ejected. Real impostor was <strong>{impostor}</strong>.</div></div>"
        )

    return grid_html + resolution


# ── Tablet wrapper ─────────────────────────────────────────────────────


def _render_tablet(rec: dict, idx: int, total: int, mode: str) -> str:
    """Wraps either chat or voting screen in the tablet frame."""
    if mode == "voting":
        body_html = _render_voting_screen(rec)
        toggle_label = "💬"
        toggle_title = "Switch to chat"
    else:
        body_html = _render_chat_screen(rec)
        toggle_label = "🗳️"
        toggle_title = "Switch to vote"

    return f"""
<div style="max-width:880px;margin:0 auto;padding:0 4px;font-family:-apple-system,system-ui,sans-serif;">
  <!-- Tablet outer frame -->
  <div style="background:linear-gradient(135deg,#9aa5b1,#6c757d);
              padding:16px;border-radius:32px;
              box-shadow:0 12px 32px rgba(0,0,0,0.25),inset 0 2px 4px rgba(255,255,255,0.4);
              position:relative;">

    <!-- Tablet inner screen -->
    <div style="background:repeating-linear-gradient(135deg,
                #d8e4f0 0px,#d8e4f0 4px,#e7eef5 4px,#e7eef5 18px);
                border-radius:18px;padding:0;
                border:2px solid #5a6a7a;
                box-shadow:inset 0 2px 6px rgba(0,0,0,0.15);
                overflow:hidden;">

      <!-- Tablet header -->
      <div style="background:rgba(0,0,0,0.04);padding:14px 22px;
                  display:flex;justify-content:space-between;align-items:center;
                  border-bottom:1px solid rgba(0,0,0,0.08);">
        <div style="font-size:1.5em;font-weight:900;color:#fff;
                    text-shadow:2px 2px 0 #1a1a2e,-1px -1px 0 #1a1a2e,
                                1px -1px 0 #1a1a2e,-1px 1px 0 #1a1a2e;
                    letter-spacing:1.5px;">
          Who Is The Impostor?
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <div style="background:rgba(255,255,255,0.7);font-size:0.78em;
                      padding:4px 10px;border-radius:10px;color:#444;font-weight:700;">
            Game {idx + 1}/{total}
          </div>
          <div title="{toggle_title}"
               style="width:42px;height:42px;background:#ffffff;border-radius:10px;
                      display:flex;align-items:center;justify-content:center;
                      font-size:1.3em;border:2px solid #5a6a7a;
                      box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            {toggle_label}
          </div>
        </div>
      </div>

      <!-- Tablet body — scrollable -->
      <div style="padding:18px 22px;max-height:640px;overflow-y:auto;">
        {body_html}
      </div>
    </div>

    <!-- Tablet home button (decorative) -->
    <div style="position:absolute;right:6px;top:50%;transform:translateY(-50%);
                width:14px;height:14px;border-radius:50%;
                background:#fff;box-shadow:inset 0 1px 2px rgba(0,0,0,0.3);"></div>
  </div>
</div>
"""


# ── Gradio handlers ────────────────────────────────────────────────────


def watch_next_game(idx_state: int, mode: str):
    recs = _load_recordings()
    if not recs:
        return _render_no_recordings(), 0, mode
    next_idx = (idx_state + 1) % len(recs)
    return _render_tablet(recs[next_idx], next_idx, len(recs), mode), next_idx, mode


def toggle_view(idx_state: int, mode: str):
    recs = _load_recordings()
    if not recs:
        return _render_no_recordings(), idx_state, mode
    new_mode = "voting" if mode == "chat" else "chat"
    cur_idx = idx_state if idx_state >= 0 else 0
    return _render_tablet(recs[cur_idx], cur_idx, len(recs), new_mode), cur_idx, new_mode


def _render_no_recordings() -> str:
    return (
        "<div style='text-align:center;padding:40px;color:#666;'>"
        "<div style='font-size:1.4em;'>🎬 Demo recordings loading…</div>"
        "<div style='margin-top:8px;'>If this persists, recordings will be downloaded "
        f"from <a href='https://huggingface.co/{MODEL_REPO}' target='_blank'>{MODEL_REPO}</a>.</div></div>"
    )


def initial_view():
    recs = _load_recordings()
    if not recs:
        return _render_no_recordings()
    return _render_tablet(recs[0], 0, len(recs), "chat")


# ── Build UI ───────────────────────────────────────────────────────────


CUSTOM_CSS = """
.gradio-container {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif !important;
  max-width: 1100px !important;
}
.hero-banner {
  background: linear-gradient(135deg, #c1121f 0%, #5a189a 100%);
  color: white; padding: 22px 28px; border-radius: 14px;
  margin-bottom: 14px;
  box-shadow: 0 6px 20px rgba(90, 24, 154, 0.18);
}
.hero-banner h1 { font-size: 1.85em; margin: 0; font-weight: 800; }
.hero-banner .subtitle { font-size: 1.02em; opacity: 0.92; margin-top: 4px; max-width: 720px; }
.stats-strip {
  display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
  font-size: 0.86em;
}
.stat-pill {
  background: rgba(255,255,255,0.18); padding: 4px 11px; border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.28);
}
.stat-pill strong { font-size: 1.05em; }
.link-row { margin-top: 12px; }
.link-row a {
  display: inline-block; background: rgba(255,255,255,0.22);
  color: white !important; padding: 4px 12px; border-radius: 14px;
  margin-right: 5px; text-decoration: none !important;
  border: 1px solid rgba(255,255,255,0.3); font-size: 0.86em; font-weight: 600;
}
.link-row a:hover { background: rgba(255,255,255,0.32); }
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

        idx_state = gr.State(0)
        mode_state = gr.State("chat")

        with gr.Tab("🎬 Watch the AI play"):
            tablet_view = gr.HTML(initial_view())

            with gr.Row():
                toggle_btn = gr.Button("🔄 Toggle Chat / Vote view", variant="secondary", size="lg")
                next_btn = gr.Button("▶️ Watch another game", variant="primary", size="lg")

            toggle_btn.click(
                toggle_view,
                inputs=[idx_state, mode_state],
                outputs=[tablet_view, idx_state, mode_state],
            )
            next_btn.click(
                watch_next_game,
                inputs=[idx_state, mode_state],
                outputs=[tablet_view, idx_state, mode_state],
            )

            with gr.Accordion("ℹ️ What you're seeing", open=False):
                gr.Markdown(
                    """
                    The tablet has two screens, just like in real Among Us:

                    - **💬 Chat screen** — Body found, every player's initial alibi, two rounds
                      of debate (every player speaks each round). Bubbles alternate left/right
                      to mimic the in-game chat layout.
                    - **🗳️ Voting screen** — One block per alive crewmate. Tiny crewmate icons
                      on the right show *which AI crewmates voted for that player*. The ejected
                      player gets a red ✖ over their avatar.

                    Click **🔄 Toggle Chat / Vote** to switch between screens. Click
                    **▶️ Watch another game** to load the next of 20 recorded games.

                    Every word of reasoning behind each vote was generated by the trained
                    Qwen 2.5 1.5B + LoRA model on a fresh game. We pre-recorded these so the
                    page loads instantly (no GPU in this Space).
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
                """
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Reward design
                        1. **Four independent reward functions** — `reward_format`,
                           `reward_correct_vote`, `reward_anti_sycophancy`,
                           `reward_anti_random_crewmate`. Composable rubrics.
                        2. **Programmatic ground truth** — vote correctness =
                           `target == impostor_names[i]`. No LLM-as-judge.
                        3. **Sycophancy probe** — one crewmate per game is *always* tagged
                           "confident innocent". Base 22% → trained 1.3%.
                        4. **Anti-cheat grader** — penalises voting before tool use,
                           repeated votes, timeout-no-vote.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        ### Action surface lockdown
                        5. **Closed action space** — Pydantic `{action_type, vote_target}`.
                           No `eval`, no shell.
                        6. **Per-round timeouts** — `max_discussion_rounds` caps debate.
                        7. **Single-impostor lock** — `scenario_generator.py` forces
                           `num_impostors=1`.
                        8. **Generation inspection** — every 5 GRPO steps logs
                           `min_length`, `clipped_ratio`, `entropy`,
                           `frac_reward_zero_std`.
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

                **Repo:** [github.com/parthdagia05/among-us-deception-gym](https://github.com/parthdagia05/among-us-deception-gym)
                """
            )

    return demo


demo = build_demo()
