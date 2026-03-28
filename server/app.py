import json
import uuid

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from environment import SupplyChainEnvironment
from models import SupplyChainAction

app = FastAPI(title="SupplyChainEnv", version="1.0.0")
sessions: dict[str, SupplyChainEnvironment] = {}


@app.get("/")
def health():
    return {"status": "ok", "env": "supply-chain-env", "version": "1.0.0"}


@app.get("/health")
def health_alt():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task_id: str = "reorder_point", seed: int = 42, session_id: str | None = None):
    sid = session_id or str(uuid.uuid4())
    env = SupplyChainEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    sessions[sid] = env
    return {"session_id": sid, "observation": obs.model_dump()}


@app.post("/step")
def step(action: SupplyChainAction, session_id: str = "default"):
    env = sessions.get(session_id)
    if not env:
        env = SupplyChainEnvironment()
        env.reset("reorder_point")
        sessions[session_id] = env
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state(session_id: str = "default"):
    env = sessions.get(session_id)
    if not env:
        return {"error": "no session found"}
    return env.state()


@app.get("/grade")
def grade(session_id: str = "default"):
    env = sessions.get(session_id)
    if not env:
        return {"error": "no session found"}
    from graders import run_all_graders

    result = run_all_graders(env.state()["history"], env.task_id)
    return result.model_dump()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "reorder_point", "difficulty": "easy", "max_steps": 10},
            {"id": "vendor_selection", "difficulty": "medium", "max_steps": 20},
            {"id": "disruption_recovery", "difficulty": "hard", "max_steps": 30},
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket session — used by EnvClient."""
    await websocket.accept()
    env = SupplyChainEnvironment()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            cmd = msg.get("command")
            if cmd == "reset":
                obs = env.reset(
                    task_id=msg.get("task_id", "reorder_point"),
                    seed=msg.get("seed", 42),
                )
                await websocket.send_text(
                    json.dumps({"type": "reset", "observation": obs.model_dump()})
                )
            elif cmd == "step":
                action = SupplyChainAction(**msg.get("action", {}))
                result = env.step(action)
                await websocket.send_text(
                    json.dumps({"type": "step", **result.model_dump()})
                )
            elif cmd == "state":
                await websocket.send_text(
                    json.dumps({"type": "state", "state": env.state()})
                )
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()
