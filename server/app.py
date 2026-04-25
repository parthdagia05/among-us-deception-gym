from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json

from server.models import Action, DifficultyConfig, Observation, StepResult
from server.environment import AmongUsEnv

app = FastAPI(title="Among Us Deception Gym", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = AmongUsEnv()


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
