"""Durable scan event logging in the unified normalized contract.

Every line of ``events.jsonl`` is one Normalized Event carrying the full
envelope shared with Shannon (schema, seq, identity, kind, redacted data).
The wire contract is docs/security-event-logging.md in the harness repo.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from pathlib import Path


SCHEMA_VERSION = "1.0"
TOOL_NAME = "strix"

# Contract bound: values over this UTF-8 byte length are externalized (spec).
INLINE_PAYLOAD_LIMIT = 8192
PAYLOAD_FIELDS = ("content", "arguments", "output")

# Object keys whose value is masked regardless of content (contract Redaction).
REDACTION_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "password",
        "passwd",
        "secret",
        "auth_token",
    }
)

# Environment variables holding a live run secret masked by value everywhere.
SECRET_ENV_VARS = ("LLM_API_KEY",)


class IdentitySource(Protocol):
    """Agent identity lookup the writer reads at emit time."""

    names: dict[str, str]
    parent_of: dict[str, str | None]


class ScanEventWriter:
    """Write one unified-envelope JSON event per line for every stream event."""

    def __init__(self, path: Path, *, run_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._payload_dir = path.parent / "payloads"
        self._lock = Lock()
        self._run_id = run_id
        self._coordinator: IdentitySource | None = None
        self._secrets = [v for v in (os.environ.get(k) for k in SECRET_ENV_VARS) if v]
        self._seq = 0
        self._seen: set[str] = set()
        self._turn: dict[str, int] = {}
        self._payload_names: dict[str, str] = {}

    def set_coordinator(self, coordinator: IdentitySource) -> None:
        """Bind the identity source read for agent_name and parent_agent_id."""
        self._coordinator = coordinator

    def run_started(self, *, model: str, target: str, params: dict[str, Any]) -> None:
        """Emit the run_started marker at the head of the stream."""
        with self._lock:
            data = {"model": model, "target": target, "params": params}
            self._write_locked(self._run_id, "run_started", data)

    def run_finished(self, *, status: str, duration_s: float) -> None:
        """Emit the run_finished marker with terminal status and duration."""
        with self._lock:
            data = {"status": status, "duration_s": duration_s}
            self._write_locked(self._run_id, "run_finished", data)

    def __call__(self, agent_id: str, event: Any) -> None:
        classified = _classify_event(event)
        if classified is None:
            return
        kind, item = classified
        with self._lock:
            if agent_id not in self._seen:
                self._seen.add(agent_id)
                self._write_locked(agent_id, "agent_started", {})
            data = self._build_data(agent_id, kind, item)
            self._write_locked(agent_id, kind, data)

    def close(self) -> None:
        with self._lock:
            self._file.flush()
            self._file.close()

    def _build_data(self, agent_id: str, kind: str, item: Any) -> dict[str, Any]:
        if kind == "agent_input":
            self._turn[agent_id] = self._turn.get(agent_id, 0) + 1
            prefix_len = len(item) if isinstance(item, list) else 0
            content = self._jsonable(item)
            return {"turn_id": self._turn[agent_id], "prefix_len": prefix_len, "content": content}
        if kind == "agent_output":
            return self._agent_output_data(agent_id, item)
        if kind == "tool_call_command":
            call_id, name, args = _tool_call_fields(item)
            return {
                "turn_id": self._turn.get(agent_id, 0),
                "tool_name": name,
                "tool_call_id": call_id,
                "arguments": args,
            }
        call_id, name, output = _tool_output_fields(item)
        return {"tool_name": name, "tool_call_id": call_id, "output": self._jsonable(output)}

    def _agent_output_data(self, agent_id: str, item: Any) -> dict[str, Any]:
        turn_id = self._turn.get(agent_id, 0)
        if isinstance(item, dict) and item.get("type") == "agent_error":
            return {"turn_id": turn_id, "content": str(item.get("data")), "is_error": True}
        return {"turn_id": turn_id, "content": _message_text(item)}

    def _write_locked(self, agent_id: str, event: str, data: dict[str, Any]) -> None:
        self._seq += 1
        redacted = _redact(data, self._secrets)
        payload = self._externalize(redacted, self._seq)
        name, parent = self._identity(agent_id)
        record = {
            "schema_version": SCHEMA_VERSION,
            "seq": self._seq,
            "timestamp": _now_iso(),
            "run_id": self._run_id,
            "tool": TOOL_NAME,
            "agent_id": agent_id,
            "agent_name": name,
            "attempt": 1,
            "parent_agent_id": parent,
            "event": event,
            "data": payload,
        }
        self._file.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        self._file.flush()

    def _identity(self, agent_id: str) -> tuple[str, str | None]:
        if self._coordinator is None:
            return agent_id, None
        return self._coordinator.names.get(agent_id, agent_id), self._coordinator.parent_of.get(
            agent_id
        )

    def _externalize(self, data: dict[str, Any], seq: int) -> dict[str, Any]:
        for field in PAYLOAD_FIELDS:
            if field not in data:
                continue
            value = data[field]
            serialized = value if isinstance(value, str) else json.dumps(value, default=str)
            if len(serialized.encode("utf-8")) <= INLINE_PAYLOAD_LIMIT:
                continue
            data[f"{field}_ref"] = f"payloads/{self._store_payload(serialized, seq, field)}"
            del data[field]
        return data

    def _store_payload(self, serialized: str, seq: int, field: str) -> str:
        digest = hashlib.sha1(serialized.encode("utf-8"), usedforsecurity=False).hexdigest()
        name = self._payload_names.get(digest)
        if name is not None:
            return name
        self._payload_dir.mkdir(parents=True, exist_ok=True)
        name = f"{seq}-{field}.txt"
        (self._payload_dir / name).write_text(serialized, encoding="utf-8")
        self._payload_names[digest] = name
        return name

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): ScanEventWriter._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ScanEventWriter._jsonable(v) for v in value]
        for method in ("model_dump", "to_dict"):
            converter = getattr(value, method, None)
            if not callable(converter):
                continue
            with contextlib.suppress(Exception):
                return ScanEventWriter._jsonable(converter())
        return str(value)


def _classify_event(event: Any) -> tuple[str, Any] | None:
    if isinstance(event, dict) and event.get("type") == "agent_input":
        return "agent_input", event.get("data")
    if isinstance(event, dict) and event.get("type") == "agent_error":
        return "agent_output", event
    item = getattr(event, "item", None)
    item_type = str(getattr(item, "type", "")) if item is not None else ""
    if item_type == "message_output_item":
        return "agent_output", item
    if item_type == "tool_call_item":
        return "tool_call_command", item
    if item_type == "tool_call_output_item":
        return "tool_call_command_output", item
    return None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_field(key, sub, secrets) for key, sub in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        masked = value
        for secret in secrets:
            masked = masked.replace(secret, "[REDACTED:credential]")
        return masked
    return value


def _redact_field(key: str, value: Any, secrets: list[str]) -> Any:
    if key.lower() in REDACTION_KEYS:
        return f"[REDACTED:{key.lower()}]"
    return _redact(value, secrets)


def _raw_field(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _message_text(item: Any) -> str:
    content = _raw_field(getattr(item, "raw_item", None), "content", [])
    parts: list[str] = []
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, str):
            parts.append(part)
            continue
        text = _raw_field(part, "text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _tool_call_fields(item: Any) -> tuple[str, str, Any]:
    raw = getattr(item, "raw_item", None)
    call_id = str(_raw_field(raw, "call_id") or _raw_field(raw, "id") or id(item))
    name = str(_raw_field(raw, "name") or _raw_field(raw, "type") or "tool")
    return call_id, name, _parse_arguments(_raw_field(raw, "arguments"))


def _tool_output_fields(item: Any) -> tuple[str, str, Any]:
    raw = getattr(item, "raw_item", None)
    call_id = str(_raw_field(raw, "call_id") or _raw_field(raw, "id") or id(item))
    name = str(_raw_field(raw, "name") or _raw_field(raw, "type") or "tool")
    return call_id, name, getattr(item, "output", _raw_field(raw, "output"))


def _parse_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value
