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
        # Body found banner — solid color, no gradient
        f"<div style='background:#c1121f;"
        f"color:white;padding:14px 18px;border-radius:10px;margin-bottom:12px;"
        f"border:1px solid #8b0e17;box-shadow:0 2px 6px rgba(193,18,31,0.18);'>"
        f"<div style='font-size:0.78em;letter-spacing:1px;font-weight:700;"
        f"text-transform:uppercase;opacity:0.9;'>🚨 Body Reported</div>"
        f"<div style='font-size:1.08em;font-weight:700;margin-top:3px;'>"
        f"{rec['kill_victim']} found in {rec['kill_location']}</div>"
        f"<div style='font-size:0.86em;opacity:0.95;margin-top:2px;'>"
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
    else:
        body_html = _render_chat_screen(rec)

    # Tab bar inside the tablet — clear "Chat | Vote" toggle.
    # Clicking the inactive tab fires the hidden Gradio button to swap mode.
    chat_active = mode == "chat"
    vote_active = mode == "voting"
    chat_click = "" if chat_active else "document.querySelector('#tablet-toggle-trigger button').click()"
    vote_click = "" if vote_active else "document.querySelector('#tablet-toggle-trigger button').click()"

    def tab(label, icon, is_active, onclick):
        active_bg = "linear-gradient(135deg,#c1121f,#5a189a)" if is_active else "transparent"
        active_color = "#ffffff" if is_active else "#3a3a4a"
        active_shadow = (
            "box-shadow:0 2px 10px rgba(193,18,31,0.35),"
            "inset 0 1px 0 rgba(255,255,255,0.18);"
            if is_active else "box-shadow:none;"
        )
        cursor = "default" if is_active else "pointer"
        weight = "800" if is_active else "700"
        return (
            f"<div onclick=\"{onclick}\" "
            f"style='flex:1;padding:11px 18px;text-align:center;border-radius:10px;"
            f"background:{active_bg};color:{active_color};{active_shadow}"
            f"font-weight:{weight};font-size:1em;letter-spacing:0.4px;cursor:{cursor};"
            f"transition:all 0.18s ease;user-select:none;display:flex;align-items:center;"
            f"justify-content:center;gap:8px;'>{icon} {label}</div>"
        )

    tab_bar = (
        "<div style='display:flex;gap:6px;background:rgba(0,0,0,0.06);"
        "padding:4px;border-radius:13px;margin:0 22px 14px;"
        "border:1px solid rgba(0,0,0,0.08);'>"
        + tab("CHAT", "💬", chat_active, chat_click)
        + tab("VOTE", "🗳", vote_active, vote_click)
        + "</div>"
    )

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

      <!-- Tablet header (centered title) -->
      <div style="background:rgba(0,0,0,0.04);padding:14px 22px 6px;
                  text-align:center;
                  border-bottom:1px solid rgba(0,0,0,0.06);">
        <div style="font-size:1.45em;font-weight:900;color:#1a2e44;
                    letter-spacing:1.2px;font-family:'Trebuchet MS',sans-serif;">
          Who Is The Impostor?
        </div>
      </div>

      <!-- Tab bar inside the iPad — Chat / Vote toggle -->
      <div style="padding:14px 0 0;">{tab_bar}</div>

      <!-- Tablet body — scrollable -->
      <div style="padding:6px 22px 22px;max-height:600px;overflow-y:auto;">
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

/* ── Hero ────────────────────────────────────────────────── */
.hero-banner {
  background: #14141f;
  color: #f5f5f7;
  padding: 30px 34px;
  border-radius: 16px;
  margin-bottom: 20px;
  border: 1px solid #2a2a3d;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
}
.hero-banner .hero-title {
  font-size: 2.15em;
  margin: 0;
  font-weight: 900;
  letter-spacing: -0.5px;
  color: #ffffff;
  display: flex;
  align-items: center;
  gap: 12px;
  line-height: 1.15;
}
.hero-banner .subtitle {
  font-size: 1.05em;
  color: #c8cdd6;
  margin-top: 10px;
  max-width: 740px;
  line-height: 1.55;
}
.stats-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}
.stat-pill {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 10px 14px;
  border-radius: 12px;
  display: inline-flex;
  flex-direction: column;
  gap: 2px;
  min-width: 130px;
}
.stat-pill .stat-label {
  font-size: 0.72em;
  color: #9aa0ac;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  font-weight: 600;
}
.stat-pill .stat-value {
  font-size: 1.18em;
  font-weight: 800;
  color: #ffffff;
}
.stat-pill.trained .stat-value { color: #06d6a0; }
.stat-pill.base    .stat-value { color: #ff6b6b; }
.stat-pill.hard    .stat-value { color: #ffd166; }
.stat-pill.syco    .stat-value { color: #8ab4ff; }
.link-row {
  margin-top: 18px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.link-row a {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(255, 255, 255, 0.08);
  color: #ffffff !important;
  padding: 9px 16px;
  border-radius: 10px;
  text-decoration: none !important;
  border: 1px solid rgba(255, 255, 255, 0.16);
  font-size: 0.95em;
  font-weight: 600;
  transition: background 0.15s, border-color 0.15s;
}
.link-row a:hover {
  background: rgba(255, 255, 255, 0.16);
  border-color: rgba(255, 255, 255, 0.3);
}

/* ── Nav buttons (replacing tabs) ────────────────────────── */
.nav-row { gap: 8px !important; margin-bottom: 14px; }
.nav-btn button {
  background: #1f1f2e !important;
  color: #c8cdd6 !important;
  border: 1px solid #2e2e44 !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: 0.96em !important;
  padding: 10px 14px !important;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
  box-shadow: none !important;
}
.nav-btn button:hover {
  background: #29293d !important;
  color: #ffffff !important;
}
.nav-btn-active button {
  background: #c1121f !important;
  color: #ffffff !important;
  border-color: #c1121f !important;
}
.nav-btn-active button:hover {
  background: #d32030 !important;
  color: #ffffff !important;
}

/* ── App header (top bar above hero) ────────────────────── */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 22px;
  background: linear-gradient(90deg, #0d1421 0%, #1a1530 50%, #0d1421 100%);
  border-bottom: 1px solid rgba(103, 232, 249, 0.15);
  margin: -16px -16px 16px;
  border-radius: 0;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
}
.app-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.app-logo { font-size: 1.4em; }
.app-brand {
  font-weight: 800;
  font-size: 1.1em;
  color: #ffffff;
  letter-spacing: -0.2px;
}
.app-tag {
  font-size: 0.68em;
  letter-spacing: 1.5px;
  font-weight: 700;
  color: #67e8f9;
  background: rgba(103, 232, 249, 0.12);
  padding: 3px 9px;
  border-radius: 12px;
  border: 1px solid rgba(103, 232, 249, 0.25);
}
.app-event {
  font-size: 0.82em;
  color: #b8c2d8;
  letter-spacing: 0.3px;
}

/* ── App footer (bottom credits) ────────────────────────── */
.app-footer {
  margin: 32px -16px -16px;
  padding: 24px 22px 18px;
  background: linear-gradient(180deg, #0a0e1a 0%, #060914 100%);
  border-top: 1px solid rgba(103, 232, 249, 0.18);
  color: #b8c2d8;
}
.app-footer-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
}
.app-footer-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #ffffff;
  font-size: 1.05em;
}
.app-footer-links { display: flex; gap: 14px; flex-wrap: wrap; }
.app-footer-links a {
  color: #67e8f9 !important;
  text-decoration: none !important;
  font-weight: 600;
  font-size: 0.92em;
  transition: color 0.15s ease;
}
.app-footer-links a:hover {
  color: #ffffff !important;
}
.app-footer-stack {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 14px 0;
  justify-content: center;
}
.footer-pill {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 14px;
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0.4px;
  border: 1px solid;
}
.footer-pill.cyan   { color: #67e8f9; background: rgba(103, 232, 249, 0.10); border-color: rgba(103, 232, 249, 0.3); }
.footer-pill.purple { color: #c084fc; background: rgba(192, 132, 252, 0.10); border-color: rgba(192, 132, 252, 0.3); }
.footer-pill.red    { color: #ff8a90; background: rgba(255, 138, 144, 0.10); border-color: rgba(255, 138, 144, 0.3); }
.footer-pill.green  { color: #6ee7b7; background: rgba(110, 231, 183, 0.10); border-color: rgba(110, 231, 183, 0.3); }
.footer-pill.amber  { color: #fcd34d; background: rgba(252, 211, 77, 0.10); border-color: rgba(252, 211, 77, 0.3); }
.app-footer-credit {
  text-align: center;
  font-size: 0.86em;
  color: #6b7686;
  margin-top: 12px;
  padding-top: 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}
.app-footer-credit a { color: #a8b3c8 !important; text-decoration: none !important; font-weight: 600; }
.app-footer-credit a:hover { color: #67e8f9 !important; }

/* ── Section accent colors (per-tab tint) ───────────────── */
.gradio-container > div[id^="component-"]:has(.gr-group) { transition: background 0.4s ease; }

/* Hide Gradio's default footer (Use via API · Built with Gradio · Settings)
   so judges only see our custom footer. */
.gradio-container > footer,
.gradio-container .footer,
.gradio-container .footer-toolbar,
footer:has(button[aria-label="Settings"]),
footer:has(a[href*="gradio.app"]) {
  display: none !important;
}

/* hidden trigger for tablet-header toggle icon */
.hidden-trigger {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
}

/* ── Polish pass: animations, transitions, hover states ──── */

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-cyan {
  0%, 100% { box-shadow: 0 0 25px rgba(103, 232, 249, 0.18),
                         0 4px 14px rgba(0, 0, 0, 0.35),
                         inset 0 1px 0 rgba(255, 255, 255, 0.07); }
  50%      { box-shadow: 0 0 40px rgba(103, 232, 249, 0.35),
                         0 4px 14px rgba(0, 0, 0, 0.4),
                         inset 0 1px 0 rgba(255, 255, 255, 0.1); }
}

/* Hero entrance animation */
.hero-banner {
  animation: fadeIn 0.6s ease-out;
}

/* Stat pills — staggered entrance + hover lift */
.stat-pill {
  animation: fadeIn 0.5s ease-out backwards;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
  cursor: default;
}
.stat-pill:nth-child(1) { animation-delay: 0.05s; }
.stat-pill:nth-child(2) { animation-delay: 0.12s; }
.stat-pill:nth-child(3) { animation-delay: 0.19s; }
.stat-pill:nth-child(4) { animation-delay: 0.26s; }
.stat-pill:hover {
  transform: translateY(-2px) scale(1.02);
  background: rgba(103, 232, 249, 0.06) !important;
  border-color: rgba(103, 232, 249, 0.3) !important;
}

/* Hero link buttons — lift on hover */
.link-row a {
  transition: transform 0.15s, background 0.15s, border-color 0.15s !important;
}
.link-row a:hover { transform: translateY(-1px); }

/* Nav buttons — animated cyan underline on hover */
.nav-btn button {
  position: relative;
  overflow: hidden;
}
.nav-btn button::after {
  content: "";
  position: absolute;
  bottom: 0;
  left: 50%;
  width: 0;
  height: 2px;
  background: #67e8f9;
  transition: width 0.3s ease, left 0.3s ease;
}
.nav-btn button:hover::after { width: 80%; left: 10%; }
.nav-btn-active button::after { display: none; }

/* Primary button — gentle breathing glow when idle, instant on hover */
.gradio-container .primary button.lg,
.gradio-container button.lg.primary {
  animation: pulse-cyan 3.5s ease-in-out infinite;
}
.gradio-container .primary button.lg:hover,
.gradio-container button.lg.primary:hover { animation: none !important; }

/* Section group entrance */
.gradio-container .gr-group { animation: fadeIn 0.45s ease-out backwards; }

/* Plot images — frame and lift on dark theme */
.gradio-container .prose img,
.gradio-container img[src*="raw.githubusercontent"] {
  border: 1px solid rgba(103, 232, 249, 0.12);
  background: #0e1424;
  padding: 4px;
  border-radius: 12px !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4),
              0 0 0 1px rgba(103, 232, 249, 0.08);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.gradio-container .prose img:hover,
.gradio-container img[src*="raw.githubusercontent"]:hover {
  transform: translateY(-4px);
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.5),
              0 0 0 1px rgba(103, 232, 249, 0.18),
              0 0 24px rgba(103, 232, 249, 0.12);
}

/* Numbered ordered lists (safeguards) — turn each into a card */
.gradio-container .prose ol {
  counter-reset: sg;
  list-style: none;
  padding-left: 0;
}
.gradio-container .prose ol > li {
  counter-increment: sg;
  position: relative;
  padding: 14px 16px 14px 56px;
  margin: 8px 0;
  background: linear-gradient(135deg, rgba(20, 20, 31, 0.6), rgba(26, 26, 46, 0.45));
  border: 1px solid rgba(103, 232, 249, 0.14);
  border-radius: 10px;
  transition: border-color 0.2s ease, transform 0.2s ease;
  list-style: none;
}
.gradio-container .prose ol > li:hover {
  border-color: rgba(103, 232, 249, 0.4);
  transform: translateX(3px);
}
.gradio-container .prose ol > li::before {
  content: counter(sg);
  position: absolute;
  left: 14px;
  top: 12px;
  width: 30px;
  height: 30px;
  background: linear-gradient(135deg, #c1121f, #5a189a);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-weight: 800;
  font-size: 0.88em;
  box-shadow: 0 2px 8px rgba(193, 18, 31, 0.4);
}

/* Cleaner cyan scrollbar */
.gradio-container *::-webkit-scrollbar { width: 8px; height: 8px; }
.gradio-container *::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 4px; }
.gradio-container *::-webkit-scrollbar-thumb { background: rgba(103, 232, 249, 0.25); border-radius: 4px; }
.gradio-container *::-webkit-scrollbar-thumb:hover { background: rgba(103, 232, 249, 0.45); }

/* Section heading spacing */
.gradio-container h3 {
  margin-top: 18px !important;
  margin-bottom: 10px !important;
  letter-spacing: -0.2px;
}

/* Mobile fallback */
@media (max-width: 720px) {
  .stat-pill { min-width: calc(50% - 5px); }
  .gradio-container button.lg {
    padding: 14px 18px !important;
    min-height: 56px !important;
    font-size: 0.92em !important;
  }
  .hero-banner { padding: 22px 18px !important; }
  .hero-banner .hero-title { font-size: 1.6em !important; }
}
"""

HEADER_HTML = """
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:14px 24px;
            background:linear-gradient(90deg,#0d1421 0%,#1a1530 50%,#0d1421 100%);
            border-bottom:1px solid rgba(103,232,249,0.18);
            margin:-16px -16px 18px;
            box-shadow:0 4px 18px rgba(0,0,0,0.35);">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <span style="font-size:1.5em;">🔪</span>
    <span style="font-weight:800;font-size:1.1em;color:#ffffff;letter-spacing:-0.2px;">
      Among Us Deception Gym
    </span>
    <span style="font-size:0.66em;letter-spacing:1.5px;font-weight:700;color:#67e8f9;
                 background:rgba(103,232,249,0.12);padding:3px 10px;border-radius:12px;
                 border:1px solid rgba(103,232,249,0.3);">RL ENV · OPENENV</span>
  </div>
  <span style="font-size:0.82em;color:#b8c2d8;letter-spacing:0.3px;">
    Meta OpenEnv Hackathon · April 2026
  </span>
</div>
"""


HERO_HTML = """
<div style="background:linear-gradient(135deg,#14141f 0%,#1a1530 50%,#1f0d1a 100%);
            color:#f5f5f7;padding:30px 34px;border-radius:16px;margin-bottom:18px;
            border:1px solid rgba(103,232,249,0.18);
            box-shadow:0 12px 32px rgba(0,0,0,0.35),
                       0 0 60px rgba(103,232,249,0.06) inset;">
  <h1 style="font-size:2em;margin:0 0 6px;font-weight:900;letter-spacing:-0.5px;
             color:#ffffff;display:flex;align-items:center;gap:12px;line-height:1.15;">
    🔪 Among Us Deception Detection
  </h1>
  <div style="font-size:1.05em;color:#c8cdd6;margin-top:6px;max-width:740px;line-height:1.55;">
    Multi-agent RL environment. Trained Qwen 2.5 1.5B catches confident liars
    in social-deduction games.
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:18px;">
    <div style="background:rgba(6,214,160,0.12);border:1px solid rgba(6,214,160,0.4);
                padding:10px 16px;border-radius:12px;display:flex;flex-direction:column;
                gap:2px;min-width:130px;">
      <span style="font-size:0.7em;color:#9aa0ac;text-transform:uppercase;
                   letter-spacing:0.6px;font-weight:600;">🤖 Trained AI</span>
      <span style="font-size:1.2em;font-weight:800;color:#06d6a0;">96.7%</span>
    </div>
    <div style="background:rgba(255,107,107,0.12);border:1px solid rgba(255,107,107,0.4);
                padding:10px 16px;border-radius:12px;display:flex;flex-direction:column;
                gap:2px;min-width:130px;">
      <span style="font-size:0.7em;color:#9aa0ac;text-transform:uppercase;
                   letter-spacing:0.6px;font-weight:600;">🧠 Base Qwen</span>
      <span style="font-size:1.2em;font-weight:800;color:#ff6b6b;">32.7%</span>
    </div>
    <div style="background:rgba(255,209,102,0.12);border:1px solid rgba(255,209,102,0.4);
                padding:10px 16px;border-radius:12px;display:flex;flex-direction:column;
                gap:2px;min-width:130px;">
      <span style="font-size:0.7em;color:#9aa0ac;text-transform:uppercase;
                   letter-spacing:0.6px;font-weight:600;">🎯 Hard-OOD</span>
      <span style="font-size:1.2em;font-weight:800;color:#ffd166;">90.0%</span>
    </div>
    <div style="background:rgba(138,180,255,0.12);border:1px solid rgba(138,180,255,0.4);
                padding:10px 16px;border-radius:12px;display:flex;flex-direction:column;
                gap:2px;min-width:130px;">
      <span style="font-size:0.7em;color:#9aa0ac;text-transform:uppercase;
                   letter-spacing:0.6px;font-weight:600;">😇 Sycophancy</span>
      <span style="font-size:1.2em;font-weight:800;color:#8ab4ff;">
        22% → 1.3%<span style="font-size:0.7em;opacity:0.75;font-weight:600;"> (17×)</span>
      </span>
    </div>
  </div>

  <div style="margin-top:18px;display:flex;gap:10px;flex-wrap:wrap;">
    <a href="https://github.com/parthdagia05/among-us-deception-gym" target="_blank"
       style="display:inline-flex;align-items:center;gap:6px;
              background:rgba(255,255,255,0.08);color:#ffffff;padding:9px 16px;
              border-radius:10px;text-decoration:none;border:1px solid rgba(255,255,255,0.16);
              font-size:0.92em;font-weight:600;transition:background 0.15s,border-color 0.15s;"
       onmouseover="this.style.background='rgba(255,255,255,0.16)';this.style.borderColor='rgba(255,255,255,0.3)';"
       onmouseout="this.style.background='rgba(255,255,255,0.08)';this.style.borderColor='rgba(255,255,255,0.16)';">
      📦 GitHub
    </a>
    <a href="https://huggingface.co/parthdagia/among-us-multiagent-detective" target="_blank"
       style="display:inline-flex;align-items:center;gap:6px;
              background:rgba(255,255,255,0.08);color:#ffffff;padding:9px 16px;
              border-radius:10px;text-decoration:none;border:1px solid rgba(255,255,255,0.16);
              font-size:0.92em;font-weight:600;transition:background 0.15s,border-color 0.15s;"
       onmouseover="this.style.background='rgba(255,255,255,0.16)';this.style.borderColor='rgba(255,255,255,0.3)';"
       onmouseout="this.style.background='rgba(255,255,255,0.08)';this.style.borderColor='rgba(255,255,255,0.16)';">
      🤖 Trained Model
    </a>
    <a href="https://github.com/parthdagia05/among-us-deception-gym/blob/main/blog/writeup.md" target="_blank"
       style="display:inline-flex;align-items:center;gap:6px;
              background:rgba(255,255,255,0.08);color:#ffffff;padding:9px 16px;
              border-radius:10px;text-decoration:none;border:1px solid rgba(255,255,255,0.16);
              font-size:0.92em;font-weight:600;transition:background 0.15s,border-color 0.15s;"
       onmouseover="this.style.background='rgba(255,255,255,0.16)';this.style.borderColor='rgba(255,255,255,0.3)';"
       onmouseout="this.style.background='rgba(255,255,255,0.08)';this.style.borderColor='rgba(255,255,255,0.16)';">
      📝 Blog
    </a>
  </div>
</div>
"""


def _foot_pill(text: str, color: str, glow: str) -> str:
    return (
        f"<span style='display:inline-block;padding:5px 13px;border-radius:14px;"
        f"font-size:0.78em;font-weight:700;letter-spacing:0.4px;color:{color};"
        f"background:{glow};border:1px solid {color}55;'>{text}</span>"
    )


FOOTER_HTML = """
<div style="margin:32px -16px -16px;padding:24px 24px 18px;
            background:linear-gradient(180deg,#0a0e1a 0%,#060914 100%);
            border-top:1px solid rgba(103,232,249,0.18);color:#b8c2d8;">

  <div style="display:flex;justify-content:space-between;align-items:center;
              flex-wrap:wrap;gap:12px;margin-bottom:14px;">
    <div style="display:flex;align-items:center;gap:8px;color:#ffffff;font-size:1em;font-weight:700;">
      🔪 Among Us Deception Gym
    </div>
    <div style="display:flex;gap:18px;flex-wrap:wrap;">
      <a href="https://github.com/parthdagia05/among-us-deception-gym" target="_blank"
         style="color:#67e8f9;text-decoration:none;font-weight:600;font-size:0.92em;">GitHub</a>
      <a href="https://huggingface.co/spaces/parthdagia/among-us-deception-gym" target="_blank"
         style="color:#67e8f9;text-decoration:none;font-weight:600;font-size:0.92em;">HF Space</a>
      <a href="https://huggingface.co/parthdagia/among-us-multiagent-detective" target="_blank"
         style="color:#67e8f9;text-decoration:none;font-weight:600;font-size:0.92em;">Model</a>
      <a href="https://github.com/parthdagia05/among-us-deception-gym/blob/main/blog/writeup.md" target="_blank"
         style="color:#67e8f9;text-decoration:none;font-weight:600;font-size:0.92em;">Blog</a>
    </div>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:8px;margin:14px 0;justify-content:center;">
""" + _foot_pill("Qwen 2.5 1.5B + LoRA", "#67e8f9", "rgba(103,232,249,0.10)") + """
""" + _foot_pill("GRPO via TRL",         "#c084fc", "rgba(192,132,252,0.10)") + """
""" + _foot_pill("HF Jobs · A10G",       "#ff8a90", "rgba(255,138,144,0.10)") + """
""" + _foot_pill("FastAPI · OpenEnv",    "#6ee7b7", "rgba(110,231,183,0.10)") + """
""" + _foot_pill("Gradio",               "#fcd34d", "rgba(252,211,77,0.10)") + """
  </div>

  <div style="text-align:center;font-size:0.86em;color:#6b7686;margin-top:12px;
              padding-top:14px;border-top:1px solid rgba(255,255,255,0.05);">
    Built by
    <a href="https://github.com/parthdagia05" target="_blank"
       style="color:#a8b3c8;text-decoration:none;font-weight:600;">parthdagia05</a>
    for the Meta OpenEnv Hackathon ·
    96.7% accuracy · 1.3% sycophancy · 17× resistance to confident liars
  </div>
</div>
"""


def _nav_classes(is_active: bool) -> list[str]:
    return ["nav-btn", "nav-btn-active"] if is_active else ["nav-btn"]


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Among Us Deception Gym",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="purple"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(HEADER_HTML)
        gr.HTML(HERO_HTML)

        idx_state = gr.State(0)
        mode_state = gr.State("chat")

        # ── Nav buttons (replacing tabs) ───────────────────────────────
        with gr.Row(elem_classes=["nav-row"]):
            nav_watch = gr.Button(
                "🎬 Watch the AI play", elem_classes=_nav_classes(True),
            )
            nav_results = gr.Button(
                "📊 Training Results", elem_classes=_nav_classes(False),
            )
            nav_safeguards = gr.Button(
                "🛡️ Safeguards", elem_classes=_nav_classes(False),
            )
            nav_api = gr.Button(
                "🔌 API & Code", elem_classes=_nav_classes(False),
            )

        # ── Section: Watch the AI play ─────────────────────────────────
        with gr.Group(visible=True) as section_watch:
            tablet_view = gr.HTML(initial_view())

            # Hidden trigger fired by the chat/vote icon inside the tablet header.
            toggle_btn = gr.Button(
                "toggle", elem_id="tablet-toggle-trigger",
                elem_classes=["hidden-trigger"],
            )
            next_btn = gr.Button(
                "▶️ Watch another game", variant="primary", size="lg",
            )

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

                    Tap the **chat / vote icon in the tablet's top-right** to switch screens.
                    Click **▶️ Watch another game** to load the next of 20 recorded games.

                    Every word of reasoning behind each vote was generated by the trained
                    Qwen 2.5 1.5B + LoRA model on a fresh game. We pre-recorded these so the
                    page loads instantly (no GPU in this Space).
                    """
                )

        # ── Section: Training Results ──────────────────────────────────
        with gr.Group(visible=False) as section_results:
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

        # ── Section: Safeguards ────────────────────────────────────────
        with gr.Group(visible=False) as section_safeguards:
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

        # ── Section: API & Code ────────────────────────────────────────
        with gr.Group(visible=False) as section_api:
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

        # ── Nav routing ────────────────────────────────────────────────
        nav_outputs = [
            section_watch, section_results, section_safeguards, section_api,
            nav_watch, nav_results, nav_safeguards, nav_api,
        ]

        def _show(active: str):
            return (
                gr.update(visible=active == "watch"),
                gr.update(visible=active == "results"),
                gr.update(visible=active == "safeguards"),
                gr.update(visible=active == "api"),
                gr.update(elem_classes=_nav_classes(active == "watch")),
                gr.update(elem_classes=_nav_classes(active == "results")),
                gr.update(elem_classes=_nav_classes(active == "safeguards")),
                gr.update(elem_classes=_nav_classes(active == "api")),
            )

        nav_watch.click(lambda: _show("watch"), outputs=nav_outputs)
        nav_results.click(lambda: _show("results"), outputs=nav_outputs)
        nav_safeguards.click(lambda: _show("safeguards"), outputs=nav_outputs)
        nav_api.click(lambda: _show("api"), outputs=nav_outputs)

        gr.HTML(FOOTER_HTML)

    return demo


demo = build_demo()
