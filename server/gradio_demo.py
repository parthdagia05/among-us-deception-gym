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


def _render_game_html(rec: dict, idx: int, total: int) -> str:
    impostor = rec.get("impostor")
    crew_correct = sum(1 for v in rec["crewmate_votes"].values()
                       if v["vote"].lower() == (impostor or "").lower())
    total_crew = len(rec["crewmate_votes"])

    crew_chips = " ".join(avatar(p) for p in rec["alive_players"])

    stmts_html = "".join(
        f"<div style=\"padding:6px 10px;margin:4px 0;background:#f8f9fa;border-radius:6px;\">"
        f"{avatar(name)} <span style=\"color:#444;font-style:italic;\">\"{stmt}\"</span></div>"
        for name, stmt in rec["all_statements"].items()
    )

    debate_html = ""
    if rec.get("discussion_log"):
        rounds: dict[int, list] = {}
        for e in rec["discussion_log"]:
            rounds.setdefault(e["round"], []).append(e)
        for r_num in sorted(rounds.keys()):
            debate_html += (
                f"<div style=\"font-weight:700;color:#5a189a;margin-top:10px;"
                f"font-size:0.9em;text-transform:uppercase;letter-spacing:0.5px;\">"
                f"🔁 Round {r_num + 1}</div>"
            )
            for e in rounds[r_num]:
                debate_html += (
                    f"<div style=\"padding:6px 10px;margin:3px 0;background:#fffaf0;"
                    f"border-left:3px solid #ffb703;border-radius:4px;\">"
                    f"{avatar(e['player_name'])} {e['statement']}</div>"
                )

    vote_cards = ""
    for crew, info in rec["crewmate_votes"].items():
        vote = info["vote"]
        is_correct = vote.lower() == (impostor or "").lower() if impostor else False
        accent = "#06d6a0" if is_correct else "#e63946"
        bg = "#eafff7" if is_correct else "#ffeeee"
        check = "✅ CORRECT" if is_correct else "❌ WRONG"
        reasoning = " ".join((info.get("reasoning") or "").split())[:600]
        if not reasoning:
            reasoning = "<em>(no reasoning recorded)</em>"
        vote_cards += (
            f"<div style=\"border-left:5px solid {accent};background:{bg};"
            f"padding:14px 18px;margin:10px 0;border-radius:8px;"
            f"box-shadow:0 1px 3px rgba(0,0,0,0.06);\">"
            f"<div style=\"display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:8px;\">"
            f"<div style=\"font-size:1.05em;\">"
            f"{avatar(crew)} <span style=\"opacity:0.6;\">votes</span> {avatar(vote)}</div>"
            f"<span style=\"color:{accent};font-weight:700;font-size:0.9em;\">{check}</span>"
            f"</div>"
            f"<div style=\"color:#444;line-height:1.5;font-size:0.95em;\">"
            f"<em>{reasoning}</em></div>"
            f"</div>"
        )

    ejected = rec.get("ejected")
    if ejected:
        if rec.get("ejection_correct"):
            res_html = (
                f"<div style=\"background:linear-gradient(135deg,#06d6a0,#118ab2);"
                f"color:white;padding:24px;border-radius:12px;text-align:center;"
                f"box-shadow:0 4px 14px rgba(6,214,160,0.3);margin-top:18px;\">"
                f"<div style=\"font-size:2em;font-weight:800;margin-bottom:6px;\">"
                f"🎉 CREW WINS!</div>"
                f"<div style=\"opacity:0.95;\">{avatar(ejected)} was ejected — and was the impostor.</div>"
                f"<div style=\"opacity:0.85;font-size:0.92em;margin-top:6px;\">"
                f"{crew_correct}/{total_crew} AI crewmates voted correctly</div>"
                f"</div>"
            )
        else:
            res_html = (
                f"<div style=\"background:linear-gradient(135deg,#e63946,#9d0208);"
                f"color:white;padding:24px;border-radius:12px;text-align:center;"
                f"box-shadow:0 4px 14px rgba(230,57,70,0.3);margin-top:18px;\">"
                f"<div style=\"font-size:2em;font-weight:800;margin-bottom:6px;\">"
                f"💀 IMPOSTOR ESCAPES</div>"
                f"<div style=\"opacity:0.95;\">{avatar(ejected)} was ejected — but the real impostor was {avatar(impostor)}.</div>"
                f"</div>"
            )
    else:
        res_html = (
            f"<div style=\"background:#666;color:white;padding:24px;border-radius:12px;"
            f"text-align:center;margin-top:18px;\">"
            f"<div style=\"font-size:1.6em;font-weight:800;\">⚖️ Tie vote</div>"
            f"<div style=\"opacity:0.9;\">No one ejected. Real impostor was {avatar(impostor)}.</div>"
            f"</div>"
        )

    return f"""
<div style="max-width:920px;margin:0 auto;font-family:-apple-system,system-ui,sans-serif;">

  <!-- Game header -->
  <div style="background:linear-gradient(135deg,#c1121f 0%,#5a189a 100%);
              color:white;padding:20px 24px;border-radius:12px 12px 0 0;
              box-shadow:0 4px 14px rgba(193,18,31,0.25);">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <div style="font-size:0.82em;opacity:0.85;letter-spacing:1px;
                    text-transform:uppercase;font-weight:700;">Emergency Meeting</div>
        <div style="font-size:1.5em;font-weight:800;margin-top:4px;">
          🚨 Body of {rec['kill_victim']} found in {rec['kill_location']}
        </div>
        <div style="opacity:0.9;font-size:0.95em;margin-top:6px;">
          Reported by {avatar(rec['body_found_by'])}
        </div>
      </div>
      <div style="opacity:0.9;font-size:0.85em;text-align:right;">
        Game {idx + 1}<br/>of {total}
      </div>
    </div>
    <div style="margin-top:12px;font-size:0.92em;opacity:0.95;">
      <strong>Alive ({len(rec['alive_players'])}):</strong> {crew_chips}
    </div>
  </div>

  <!-- Body card -->
  <div style="background:#ffffff;padding:22px 26px;border-radius:0 0 12px 12px;
              border:1px solid #e9ecef;border-top:none;
              box-shadow:0 4px 14px rgba(0,0,0,0.04);">

    <div style="font-weight:700;color:#5a189a;letter-spacing:0.5px;
                text-transform:uppercase;font-size:0.85em;margin-bottom:8px;">
      💬 Initial Statements
    </div>
    {stmts_html}

    <div style="font-weight:700;color:#5a189a;letter-spacing:0.5px;
                text-transform:uppercase;font-size:0.85em;margin:16px 0 8px;">
      🗣️ Debate Transcript
    </div>
    {debate_html}

    <div style="font-weight:700;color:#c1121f;letter-spacing:0.5px;
                text-transform:uppercase;font-size:0.85em;margin:20px 0 4px;">
      🤖 The Trained AI Crewmates Vote
    </div>
    <div style="color:#777;font-size:0.88em;font-style:italic;margin-bottom:6px;">
      Each crewmate is the same trained Qwen 2.5 1.5B + LoRA, voting independently.
    </div>
    {vote_cards}

    {res_html}

  </div>

</div>
"""


def _render_no_recordings() -> str:
    return (
        "<div style='text-align:center;padding:40px 20px;color:#666;'>"
        "<div style='font-size:1.4em;'>🎬 Demo recordings not available yet</div>"
        "<div style='margin-top:8px;'>Recordings are generated by <code>jobs/eval_job.py</code> "
        "and uploaded to "
        f"<a href='https://huggingface.co/{MODEL_REPO}' target='_blank'>{MODEL_REPO}</a> "
        "as <code>eval_recordings.json</code>.</div>"
        "</div>"
    )


def watch_game(idx_state: int):
    recs = _load_recordings()
    if not recs:
        return _render_no_recordings(), 0
    next_idx = (idx_state + 1) % len(recs)
    return _render_game_html(recs[next_idx], next_idx, len(recs)), next_idx


def initial_view():
    recs = _load_recordings()
    if not recs:
        return _render_no_recordings()
    return _render_game_html(recs[0], 0, len(recs))


# ── Build UI ───────────────────────────────────────────────────────────


CUSTOM_CSS = """
.gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif !important; }
.hero-banner {
  background: linear-gradient(135deg, #c1121f 0%, #5a189a 100%);
  color: white; padding: 28px 32px; border-radius: 14px;
  margin-bottom: 14px;
  box-shadow: 0 6px 20px rgba(90, 24, 154, 0.18);
}
.hero-banner h1 { font-size: 2em; margin: 0; font-weight: 800; }
.hero-banner .subtitle { font-size: 1.1em; opacity: 0.92; margin-top: 6px; }
.stats-strip {
  display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px;
  font-size: 0.92em;
}
.stat-pill {
  background: rgba(255,255,255,0.18); padding: 6px 12px; border-radius: 20px;
  border: 1px solid rgba(255,255,255,0.28);
}
.stat-pill strong { font-size: 1.1em; }
.link-row { margin-top: 14px; }
.link-row a {
  display: inline-block; background: rgba(255,255,255,0.22);
  color: white !important; padding: 6px 14px; border-radius: 18px;
  margin-right: 6px; text-decoration: none !important;
  border: 1px solid rgba(255,255,255,0.3); font-size: 0.9em; font-weight: 600;
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

        idx_state = gr.State(-1)

        with gr.Tab("🎬 Watch the AI play"):
            view = gr.HTML(initial_view())

            with gr.Row():
                next_btn = gr.Button("▶️ Watch another game", variant="primary", size="lg")

            next_btn.click(
                watch_game,
                inputs=[idx_state],
                outputs=[view, idx_state],
            )

            gr.Markdown(
                """
                ### What you're seeing

                - **Real model decisions.** Every vote and every line of reasoning was generated by the trained
                  model on a fresh game scenario. We pre-recorded 20 games so the page loads instantly.
                - **Independent crewmate brains.** Same model weights, but each crewmate gets a different
                  prompt (`"You are Red"` vs `"You are Green"`). They reason separately and may disagree.
                - **Adversarial impostor.** The impostor's lies are generated by a separate LLM at scenario
                  creation time, with a guaranteed detectable contradiction baked in.
                """
            )

        with gr.Tab("📊 Training Results"):
            gr.Markdown("### 50-Game Eval — 150 votes per model")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_comparison.png" '
                'style="width:100%;max-width:800px;border-radius:8px;">'
            )
            gr.Markdown("### Robustness — Hard-distribution OOD eval (lie_subtlety=0.8, 7 players)")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_robustness.png" '
                'style="width:100%;max-width:800px;border-radius:8px;">'
            )
            gr.Markdown(
                "On a held-out hard distribution the model never trained on, the trained model "
                "drops only **6.7 points** (96.7 → 90.0). The base model is unchanged at ~33%. "
                "**Proof of generalization** — the policy transferred."
            )
            gr.Markdown("### GRPO Reward Curve (1500 iterations, real data)")
            gr.HTML(
                '<img src="https://raw.githubusercontent.com/parthdagia05/among-us-deception-gym/main/plot_reward.png" '
                'style="width:100%;max-width:800px;border-radius:8px;">'
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

                1. **Four independent reward functions** (not one combined): `reward_format`,
                   `reward_correct_vote`, `reward_anti_sycophancy`, `reward_anti_random_crewmate`.
                   The model can't game one signal at the expense of another.
                2. **Closed action space** — Pydantic-validated `{action_type, vote_target}`.
                   No `eval`, no shell, no global mutation.
                3. **Programmatic ground truth** — vote correctness = `target == impostor_names[i]`.
                   No LLM-as-judge that could be gamed.
                4. **Per-round timeouts** — `max_discussion_rounds` caps debate; auto-advances when
                   all alive players speak.
                5. **Single-impostor lock** — `scenario_generator.py` forces `num_impostors=1`.
                6. **Anti-cheat grader** (single-agent env) — penalises voting before tool use,
                   repeated votes, timeout-no-vote.
                7. **Generation inspection** — every 5 GRPO steps logs `min_length`, `clipped_ratio`,
                   `entropy`, `frac_reward_zero_std`.
                8. **Sycophancy probe** — one crewmate per game is *always* tagged "confident innocent"
                   with assertive personality. Base 22% → trained 1.3% (17× reduction).

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
