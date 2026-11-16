"""Configuration for edge0-studio: upstream server and MCP servers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("EDGE0_STUDIO_CONFIG", "~/.config/edge0-studio/config.yaml")
).expanduser()

DEFAULT_CONFIG = {
    "upstream": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "edge0_35b",
        "api_key": "",
    },
    "mcp_servers": {
        # Example:
        # "filesystem": {
        #     "command": "npx",
        #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        #     "env": {},
        # },
    },
}


@dataclass
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    base_url: str
    model: str
    api_key: str
    mcp_servers: list[McpServerConfig]
    path: Path

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or DEFAULT_CONFIG_PATH
        if path.exists():
            raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        else:
            raw = {}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False))
        upstream = {**DEFAULT_CONFIG["upstream"], **(raw.get("upstream") or {})}
        servers = []
        for name, spec in (raw.get("mcp_servers") or {}).items():
            servers.append(
                McpServerConfig(
                    name=name,
                    command=spec["command"],
                    args=list(spec.get("args") or []),
                    env=dict(spec.get("env") or {}),
                )
            )
        return cls(
            base_url=upstream["base_url"].rstrip("/"),
            model=upstream["model"],
            api_key=upstream.get("api_key") or "",
            mcp_servers=servers,
            path=path,
        )
