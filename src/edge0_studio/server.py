"""FastAPI app: serves the chat UI and the streaming agent endpoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .agent import AgentLoop
from .config import Config
from .mcp_manager import McpManager

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

config = Config.load()
mcp = McpManager(config.mcp_servers)
agent = AgentLoop(config.base_url, config.model, config.api_key, mcp)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await mcp.start()
    try:
        yield
    finally:
        await mcp.stop()


app = FastAPI(title="edge0-studio", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> JSONResponse:
    upstream: dict[str, Any] = {"base_url": config.base_url, "model": config.model,
                                "reachable": False, "models": []}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{config.base_url}/models")
            if resp.status_code == 200:
                upstream["reachable"] = True
                upstream["models"] = [
                    m.get("id") for m in resp.json().get("data", [])
                ]
    except httpx.HTTPError:
        pass
    return JSONResponse(
        {
            "upstream": upstream,
            "mcp": mcp.describe(),
            "tool_count": len(mcp.openai_tools),
            "config_path": str(config.path),
        }
    )


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    messages = body.get("messages") or []
    temperature = body.get("temperature")
    use_tools = bool(body.get("use_tools", True))
    return StreamingResponse(
        agent.run(messages, temperature=temperature, use_tools=use_tools),
        media_type="text/event-stream",
    )
