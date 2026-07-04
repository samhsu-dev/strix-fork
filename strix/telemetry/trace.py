"""JSONL tracing of every agent LLM turn and tool call within one scan run."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from itertools import count
from typing import TYPE_CHECKING, Any

from agents.lifecycle import RunHooks


if TYPE_CHECKING:
    from pathlib import Path

    from agents import RunContextWrapper
    from agents.agent import Agent
    from agents.items import ModelResponse, TResponseInputItem
    from agents.tool import Tool


logger = logging.getLogger(__name__)

TRACE_FILENAME = "trace.jsonl"
_TRACE_ENV = "STRIX_TRACE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def tracing_enabled() -> bool:
    """True when the STRIX_TRACE environment variable selects a truthy value."""
    return os.environ.get(_TRACE_ENV, "").strip().lower() in _TRUTHY


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _to_jsonable(dump(mode="json"))
        except Exception:
            return str(value)
    return str(value)


def _parse_arguments(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _agent_name(agent: Any) -> str | None:
    name = getattr(agent, "name", None)
    return name if isinstance(name, str) else None


def _agent_id(context: Any, agent: Any) -> str:
    ctx = getattr(context, "context", None)
    agent_id = ctx.get("agent_id") if isinstance(ctx, dict) else None
    if isinstance(agent_id, str) and agent_id:
        return agent_id
    return _agent_name(agent) or "unknown"


def _parent_id(context: Any) -> str | None:
    ctx = getattr(context, "context", None)
    parent_id = ctx.get("parent_id") if isinstance(ctx, dict) else None
    return parent_id if isinstance(parent_id, str) else None


def _tool_fields(context: Any, tool: Any) -> dict[str, Any]:
    return {
        "tool_name": getattr(context, "tool_name", None) or getattr(tool, "name", None),
        "tool_call_id": getattr(context, "tool_call_id", None),
        "arguments": _parse_arguments(getattr(context, "tool_arguments", None)),
    }


class TraceWriter:
    """Append-only JSONL sink for one scan's trace events."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = count(1)

    def emit(self, record: dict[str, Any]) -> None:
        """Serialize one event to a single JSONL line, stamped with sequence and time."""
        with self._lock:
            stamped = {"seq": next(self._seq), "ts": time.time(), **record}
            self._file.write(json.dumps(_to_jsonable(stamped), ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


class TracingHooks(RunHooks[dict[str, Any]]):
    """Record every LLM turn, tool call, and agent transition to a TraceWriter."""

    def __init__(self, writer: TraceWriter) -> None:
        self._writer = writer

    def _emit(self, record: dict[str, Any], context: Any, agent: Any) -> None:
        record["agent_id"] = _agent_id(context, agent)
        record["agent_name"] = _agent_name(agent)
        record["parent_id"] = _parent_id(context)
        try:
            self._writer.emit(record)
        except Exception:
            logger.exception("failed to write trace event %s", record.get("event"))

    async def on_llm_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        record = {"event": "llm_start", "system_prompt": system_prompt, "input": input_items}
        self._emit(record, context, agent)

    async def on_llm_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        response: ModelResponse,
    ) -> None:
        record = {
            "event": "llm_end",
            "output": getattr(response, "output", None),
            "usage": getattr(response, "usage", None),
            "response_id": getattr(response, "response_id", None),
        }
        self._emit(record, context, agent)

    async def on_tool_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        tool: Tool,
    ) -> None:
        record = {"event": "tool_start", **_tool_fields(context, tool)}
        self._emit(record, context, agent)

    async def on_tool_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        tool: Tool,
        result: str,
    ) -> None:
        record = {"event": "tool_end", "result": result, **_tool_fields(context, tool)}
        self._emit(record, context, agent)

    async def on_agent_start(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
    ) -> None:
        self._emit({"event": "agent_start"}, context, agent)

    async def on_agent_end(
        self,
        context: RunContextWrapper[dict[str, Any]],
        agent: Agent[dict[str, Any]],
        output: Any,
    ) -> None:
        self._emit({"event": "agent_end", "output": output}, context, agent)

    async def on_handoff(
        self,
        context: RunContextWrapper[dict[str, Any]],
        from_agent: Agent[dict[str, Any]],
        to_agent: Agent[dict[str, Any]],
    ) -> None:
        record = {"event": "handoff", "to_agent": _agent_name(to_agent)}
        self._emit(record, context, from_agent)
