"""Agent loop: streams chat from the upstream OpenAI-compatible server,
executes MCP tool calls, and feeds results back until the model answers."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from .mcp_manager import McpManager

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


class AgentLoop:
    def __init__(self, base_url: str, model: str, api_key: str, mcp: McpManager):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.mcp = mcp

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def run(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        use_tools: bool = True,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Yields SSE lines: token / thinking / tool_call / tool_result / done / error."""
        convo = list(messages)
        tools = self.mcp.openai_tools if use_tools else []
        model_name = model or self.model

        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
            started = time.monotonic()
            first_token_at: float | None = None
            total_tokens = 0
            for _round in range(MAX_TOOL_ROUNDS + 1):
                body: dict[str, Any] = {
                    "model": model_name,
                    "messages": convo,
                    "stream": True,
                }
                if temperature is not None:
                    body["temperature"] = temperature
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"

                content_parts: list[str] = []
                tool_calls: dict[int, dict[str, Any]] = {}

                try:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        json=body,
                        headers=self._headers(),
                    ) as resp:
                        if resp.status_code != 200:
                            detail = (await resp.aread()).decode(errors="replace")[:500]
                            yield _sse({"type": "error",
                                        "message": f"upstream {resp.status_code}: {detail}"})
                            return
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            choice = (chunk.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                            if reasoning:
                                yield _sse({"type": "thinking", "text": reasoning})
                            text = delta.get("content")
                            if text:
                                if first_token_at is None:
                                    first_token_at = time.monotonic()
                                total_tokens += 1
                                content_parts.append(text)
                                yield _sse({"type": "token", "text": text})
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index", 0)
                                slot = tool_calls.setdefault(
                                    idx,
                                    {"id": "", "type": "function",
                                     "function": {"name": "", "arguments": ""}},
                                )
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["function"]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    slot["function"]["arguments"] += fn["arguments"]
                except httpx.HTTPError as exc:
                    yield _sse({"type": "error", "message": f"upstream connection failed: {exc}"})
                    return

                if not tool_calls:
                    elapsed = time.monotonic() - started
                    decode_s = (
                        time.monotonic() - first_token_at if first_token_at else 0.0
                    )
                    yield _sse({
                        "type": "done",
                        "stats": {
                            "total_seconds": round(elapsed, 2),
                            "first_token_seconds": round(
                                first_token_at - started, 2
                            ) if first_token_at else None,
                            "tokens": total_tokens,
                            "tokens_per_second": round(total_tokens / decode_s, 1)
                            if decode_s > 0.5 and total_tokens else None,
                        },
                    })
                    return

                calls = [tool_calls[i] for i in sorted(tool_calls)]
                convo.append(
                    {
                        "role": "assistant",
                        "content": "".join(content_parts) or None,
                        "tool_calls": calls,
                    }
                )
                for call in calls:
                    name = call["function"]["name"]
                    raw_args = call["function"]["arguments"] or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                    yield _sse({"type": "tool_call", "name": name, "arguments": raw_args})
                    result = await self.mcp.call_tool(name, args)
                    yield _sse({"type": "tool_result", "name": name,
                                "result": result[:4000]})
                    convo.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"] or name,
                            "content": result,
                        }
                    )

            yield _sse({"type": "error",
                        "message": f"stopped after {MAX_TOOL_ROUNDS} tool rounds"})
