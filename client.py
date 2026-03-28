from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets

from models import SupplyChainAction, SupplyChainObservation, StepResult


class EnvClient:
    """WebSocket client for SupplyChainEnv (`/ws` protocol)."""

    def __init__(self, uri: str = "ws://127.0.0.1:7860/ws") -> None:
        self.uri = uri
        self._ws: Any = None

    async def connect(self) -> EnvClient:
        self._ws = await websockets.connect(self.uri)
        return self

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def reset(
        self, task_id: str = "reorder_point", seed: int = 42
    ) -> tuple[SupplyChainObservation, dict[str, Any]]:
        assert self._ws is not None
        await self._ws.send(
            json.dumps({"command": "reset", "task_id": task_id, "seed": seed})
        )
        raw = json.loads(await self._ws.recv())
        if raw.get("type") != "reset":
            raise RuntimeError(raw)
        obs = SupplyChainObservation(**raw["observation"])
        return obs, raw

    async def step(self, action: SupplyChainAction) -> StepResult:
        assert self._ws is not None
        await self._ws.send(
            json.dumps({"command": "step", "action": action.model_dump()})
        )
        raw = json.loads(await self._ws.recv())
        if raw.get("type") != "step":
            raise RuntimeError(raw)
        raw.pop("type", None)
        return StepResult.model_validate(raw)

    async def state(self) -> dict[str, Any]:
        assert self._ws is not None
        await self._ws.send(json.dumps({"command": "state"}))
        raw = json.loads(await self._ws.recv())
        if raw.get("type") != "state":
            raise RuntimeError(raw)
        return raw["state"]


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)
