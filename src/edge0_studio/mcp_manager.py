"""MCP client manager: connects to configured MCP servers over stdio and
exposes their tools in OpenAI function-calling format."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import McpServerConfig

log = logging.getLogger(__name__)


class McpManager:
    """Owns sessions to every configured MCP server for the app's lifetime."""

    def __init__(self, servers: list[McpServerConfig]):
        self._configs = servers
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        # tool name (exposed to the model) -> (server name, original tool name)
        self._tool_index: dict[str, tuple[str, str]] = {}
        self._openai_tools: list[dict[str, Any]] = []

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        for cfg in self._configs:
            try:
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=cfg.env or None
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[cfg.name] = session
                await self._register_tools(cfg.name, session)
                log.info("MCP server '%s' connected", cfg.name)
            except Exception:
                log.exception("MCP server '%s' failed to start; skipping", cfg.name)

    async def stop(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack = None
        self._sessions.clear()
        self._tool_index.clear()
        self._openai_tools.clear()

    async def _register_tools(self, server_name: str, session: ClientSession) -> None:
        result = await session.list_tools()
        for tool in result.tools:
            schema = (
                getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
                or {"type": "object", "properties": {}}
            )
            exposed = f"{server_name}__{tool.name}"
            self._tool_index[exposed] = (server_name, tool.name)
            self._openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                }
            )

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return list(self._openai_tools)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_index)

    def describe(self) -> list[dict[str, Any]]:
        by_server: dict[str, list[str]] = {}
        for exposed, (server, original) in self._tool_index.items():
            by_server.setdefault(server, []).append(original)
        return [
            {"server": name, "connected": name in self._sessions,
             "tools": sorted(by_server.get(name, []))}
            for name in {c.name for c in self._configs}
        ]

    async def call_tool(self, exposed_name: str, arguments: dict[str, Any]) -> str:
        if exposed_name not in self._tool_index:
            return json.dumps({"error": f"unknown tool: {exposed_name}"})
        server_name, tool_name = self._tool_index[exposed_name]
        session = self._sessions.get(server_name)
        if session is None:
            return json.dumps({"error": f"MCP server not connected: {server_name}"})
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments), timeout=120
            )
        except asyncio.TimeoutError:
            return json.dumps({"error": f"tool call timed out: {exposed_name}"})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)})
        parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(json.dumps(item.model_dump(), default=str))
        return "\n".join(parts) if parts else "(empty result)"
