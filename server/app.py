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


@app.get("/")
def root():
    return {
        "name": "Among Us Deception Gym",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": ["/health", "/reset", "/step", "/state", "/ws"]
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
def multi_reset(difficulty: Optional[DifficultyConfig] = None,
                x_hf_token: Optional[str] = Header(default=None)):
    token = x_hf_token or os.environ.get("HF_TOKEN")
    return multi_env.reset(difficulty=difficulty, hf_token=token)


@app.get("/multi/observation/{game_id}/{player_name}")
def multi_observation(game_id: str, player_name: str):
    obs = multi_env.get_observation(game_id, player_name)
    if obs is None:
        return {"error": f"Game {game_id} or player {player_name} not found"}
    return obs


@app.post("/multi/vote")
def multi_vote(payload: dict):
    game_id = payload.get("game_id", "")
    player_name = payload.get("player_name", "")
    vote_target = payload.get("vote_target", "")
    return multi_env.submit_vote(game_id, player_name, vote_target)


@app.get("/multi/resolve/{game_id}")
def multi_resolve(game_id: str):
    return multi_env.resolve(game_id)


@app.get("/multi/session/{game_id}")
def multi_session(game_id: str):
    info = multi_env.get_session_info(game_id)
    if info is None:
        return {"error": f"Game {game_id} not found"}
    return info


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
