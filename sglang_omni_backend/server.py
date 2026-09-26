"""HTTP/WebSocket surface of the backend protocol (same endpoints as py_backend/server.py).

    GET  /health                      readiness (also checks the upstream /health)
    WS   /backend                     session.init -> session.created, input.append -> events
    POST /sessions/{session_id}/close close (completion semantics), 404 for unknown ids

Run: ``python -m sglang_omni_backend --port 22500 --upstream-url ws://127.0.0.1:18260/v1/realtime``
and point the worker at it: ``python worker.py --backend-server-url http://127.0.0.1:22500``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from aiohttp import web

from .session import BACKEND_NAME, BackendConfig, BackendSession, ProtocolViolation

log = logging.getLogger("sglang_omni_backend.server")

WS_MAX_MESSAGE_BYTES = 128 * 1024 * 1024  # same as py_backend (uvicorn ws_max_size)


class Registry:
    """Open sessions. The reference backend allows exactly one; here ``max_sessions`` (default 1)."""

    def __init__(self, max_sessions: int) -> None:
        self.max_sessions = max_sessions
        self.sessions: dict[str, BackendSession] = {}

    def register(self, session: BackendSession) -> None:
        if len(self.sessions) >= self.max_sessions:
            raise ProtocolViolation(f"backend already has {len(self.sessions)} active session(s) (max {self.max_sessions})")
        self.sessions[session.session_id] = session

    def forget(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def _upstream_health_url(ws_url: str) -> str:
    parts = urlsplit(ws_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "/health", "", ""))


async def health(request: web.Request) -> web.Response:
    config: BackendConfig = request.app["config"]
    registry: Registry = request.app["registry"]
    upstream_ok = False
    try:
        async with request.app["http"].get(_upstream_health_url(config.upstream_url), timeout=aiohttp.ClientTimeout(total=2)) as response:
            upstream_ok = response.status == 200
    except Exception:
        upstream_ok = False
    body = {
        "status": "ready" if upstream_ok else "upstream_unavailable",
        "backend": BACKEND_NAME,
        "upstream_url": config.upstream_url,
        "active_session_id": next(iter(registry.sessions), None),
        "active_sessions": len(registry.sessions),
        "max_sessions": registry.max_sessions,
    }
    return web.json_response(body, status=200 if upstream_ok else 503)


async def backend_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=WS_MAX_MESSAGE_BYTES, heartbeat=None)
    await ws.prepare(request)
    config: BackendConfig = request.app["config"]
    registry: Registry = request.app["registry"]
    session: Optional[BackendSession] = None
    try:
        first = await ws.receive()
        if first.type != aiohttp.WSMsgType.TEXT:
            raise ProtocolViolation("first message must be session.init")
        message = json.loads(first.data)
        if message.get("type") != "session.init":
            raise ProtocolViolation("first message must be session.init")
        params = message.get("payload")
        if not isinstance(params, dict):
            raise ProtocolViolation("message must carry an object `payload`")
        # Session identity is assigned here, never taken from the client (schema §3.1).
        session = BackendSession(session_id=f"sess_{uuid.uuid4().hex[:12]}", ws=ws, config=config, registry=registry)
        registry.register(session)
        await session.init(params)
        async for msg in ws:
            if session.closed:
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(msg.data)
                if event.get("type") != "input.append":
                    # close only travels over HTTP (network §3.2); anything else is illegal.
                    raise ProtocolViolation(f"unsupported message type: {event.get('type')}")
                await session.push(event)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                raise ProtocolViolation("binary frames are not part of the protocol")
            else:
                break
        if not session.closed:
            await session.close(reason="client_disconnected", emit_event=False)
    except Exception as exc:
        if session is not None and session.session_id in registry.sessions:
            await session.fatal("backend_error", message=str(exc))  # no-op if already closed
        else:
            # Rejected before the session was registered (bad first message or capacity).
            log.warning("backend websocket rejected: %s", exc)
            try:
                await ws.send_str(json.dumps({"type": "session.closed", "reason": "backend_error", "diagnostic": {"message": str(exc)}}))
            except Exception:
                pass
        if not ws.closed:
            await ws.close(code=1011, message=b"backend_error")
    return ws


async def close_session(request: web.Request) -> web.Response:
    registry: Registry = request.app["registry"]
    session_id = request.match_info["session_id"]
    session = registry.sessions.get(session_id)
    if session is None:
        return web.json_response({"detail": "session not found"}, status=404)
    reason = "client_closed"
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("reason"):
            reason = str(body["reason"])
    except Exception:
        pass
    await session.close(reason=reason)
    return web.json_response({"ok": True, "session_id": session_id, "closed": True})


def create_app(config: BackendConfig, *, max_sessions: int = 1) -> web.Application:
    app = web.Application(client_max_size=WS_MAX_MESSAGE_BYTES)
    app["config"] = config
    app["registry"] = Registry(max_sessions)

    async def on_startup(app: web.Application) -> None:
        app["http"] = aiohttp.ClientSession()

    async def on_shutdown(app: web.Application) -> None:
        for session in list(app["registry"].sessions.values()):
            await session.close(reason="server_shutdown")
        await app["http"].close()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/health", health)
    app.router.add_get("/backend", backend_ws)
    app.router.add_post("/sessions/{session_id}/close", close_session)
    return app
