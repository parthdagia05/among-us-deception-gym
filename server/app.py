from fastapi import FastAPI, WebSocket, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import os

from server.models import Action, DifficultyConfig, Observation, StepResult
from server.environment import AmongUsEnv
from server.environment_multi import MultiAgentAmongUsEnv



app = FastAPI(title="Among Us Deception Gym", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = AmongUsEnv()
multi_env = MultiAgentAmongUsEnv()


@app.get("/api")
def api_info():
    """API endpoint listing. The Gradio demo lives at /."""
    return {
        "name": "Among Us Deception Gym",
        "version": "1.0.0",
        "status": "ok",
        "live_demo": "/",
        "endpoints": [
            "/ (Gradio UI — live demo)",
            "/api", "/health",
            "/reset", "/step", "/state", "/ws",
            "/multi/reset", "/multi/discuss", "/multi/vote", "/multi/kill",
            "/multi/observation/{game_id}/{player_name}",
            "/multi/resolve/{game_id}", "/multi/session/{game_id}",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "environment": "among_us_deception_gym"}


@app.post("/reset")
def reset(difficulty: Optional[DifficultyConfig] = None) -> Observation:
    return env.reset(difficulty)


@app.post("/step")
def step(action: Action) -> StepResult:
    return env.step(action)


@app.get("/state")
def state():
    return env.state()


@app.get("/training/game_info")
def training_game_info():
    """Returns ground truth for reward computation during training. Not for agents."""
    if not env.current_scenario:
        return {"error": "No active game. Call /reset first."}
    s = env.current_scenario
    return {
        "game_id": env.game_id,
        "impostor_names": s.impostor_names,
        "confident_innocent_name": s.confident_innocent_name,
        "alive_players": s.alive_players,
    }


@app.post("/multi/reset")
def multi_reset(
    difficulty: Optional[DifficultyConfig] = None,
    x_hf_token: Optional[str] = Header(default=None),
    max_discussion_rounds: int = 2,
):
    token = x_hf_token or os.environ.get("HF_TOKEN")
    return multi_env.reset(
        difficulty=difficulty, hf_token=token, max_discussion_rounds=max_discussion_rounds
    )


@app.get("/multi/observation/{game_id}/{player_name}")
def multi_observation(game_id: str, player_name: str):
    obs = multi_env.get_observation(game_id, player_name)
    if obs is None:
        return {"error": f"Game {game_id} or player {player_name} not found"}
    return obs


@app.post("/multi/discuss")
def multi_discuss(
    payload: dict,
    x_hf_token: Optional[str] = Header(default=None),
):
    """
    Submit a discussion statement for a player.
    If 'statement' is omitted, the server generates one via LLM (requires HF token).

    Payload: {game_id, player_name, statement (optional)}
    """
    game_id = payload.get("game_id", "")
    player_name = payload.get("player_name", "")
    statement = payload.get("statement", None)
    token = x_hf_token or os.environ.get("HF_TOKEN")
    return multi_env.submit_discussion(game_id, player_name, statement=statement, hf_token=token)


@app.post("/multi/vote")
def multi_vote(payload: dict):
    game_id = payload.get("game_id", "")
    player_name = payload.get("player_name", "")
    vote_target = payload.get("vote_target", "")
    return multi_env.submit_vote(game_id, player_name, vote_target)


@app.post("/multi/kill")
def multi_kill(payload: dict):
    """
    Impostor kills a crewmate to start the next round.
    Call this after /multi/resolve returns awaiting_kill=true.

    Payload: {game_id, impostor_name, kill_target (optional — auto-picked if omitted)}
    """
    game_id = payload.get("game_id", "")
    impostor_name = payload.get("impostor_name", "")
    kill_target = payload.get("kill_target", None)
    return multi_env.kill(game_id, impostor_name, kill_target=kill_target)


@app.get("/multi/resolve/{game_id}")
def multi_resolve(game_id: str):
    return multi_env.resolve(game_id)


@app.get("/multi/session/{game_id}")
def multi_session(game_id: str):
    info = multi_env.get_session_info(game_id)
    if info is None:
        return {"error": f"Game {game_id} not found"}
    return info


# Mount Gradio live demo at / (root) so judges land on the demo immediately.
# JSON API listing lives at /api. Lazy import so failures don't break the API.
try:
    import gradio as gr
    from server.gradio_demo import demo as gradio_demo

    app = gr.mount_gradio_app(app, gradio_demo, path="/")
except Exception as exc:  # noqa: BLE001
    print(f"[gradio] not mounted: {exc}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    obs = env.reset()
    await websocket.send_text(obs.model_dump_json())

    try:
        while True:
            data = await websocket.receive_text()
            action = Action.model_validate_json(data)
            result = env.step(action)
            await websocket.send_text(result.model_dump_json())
            if result.done:
                break
    except Exception as e:
        await websocket.close()
