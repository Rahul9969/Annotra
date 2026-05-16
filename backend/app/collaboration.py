"""Real-time project broadcast for multi-client annotation sync."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket

_lock = asyncio.Lock()
_subscribers: dict[int, list[WebSocket]] = {}


async def register_ws(project_id: int, ws: WebSocket) -> None:
    async with _lock:
        _subscribers.setdefault(project_id, []).append(ws)


async def unregister_ws(project_id: int, ws: WebSocket) -> None:
    async with _lock:
        lst = _subscribers.get(project_id)
        if not lst:
            return
        if ws in lst:
            lst.remove(ws)


async def broadcast_project(project_id: int, payload: dict[str, Any]) -> None:
    async with _lock:
        conns = list(_subscribers.get(project_id, []))
    dead: list[WebSocket] = []
    for ws in conns:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    if not dead:
        return
    async with _lock:
        lst = _subscribers.get(project_id)
        if not lst:
            return
        for ws in dead:
            if ws in lst:
                lst.remove(ws)
