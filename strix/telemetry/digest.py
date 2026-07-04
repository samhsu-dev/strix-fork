"""Transform one scan's trace.jsonl into split payload files plus a summary manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

DIGEST_DIRNAME = "digest"
PAYLOADS_DIRNAME = "payloads"
SUMMARY_FILENAME = "summary.json"


def _read_events(trace_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


class _PayloadStore:
    """Write one payload to its own file and return the manifest-relative path."""

    def __init__(self, payloads_dir: Path) -> None:
        self._dir = payloads_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, stem: str, suffix: str, content: Any) -> str:
        name = f"{stem}.{suffix}"
        path = self._dir / name
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        path.write_text(text, encoding="utf-8")
        return f"{PAYLOADS_DIRNAME}/{name}"


def _slug(value: Any) -> str:
    text = str(value) if value is not None else "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text)[:40]


def _agent_record(start: dict[str, Any], end: dict[str, Any] | None, store: _PayloadStore) -> dict[str, Any]:
    seq = start.get("seq")
    stem = f"{seq:05d}-agent-{_slug(start.get('agent_id'))}"
    prompt_path = store.write(f"{stem}-input", "json", start.get("input"))
    output_path = store.write(f"{stem}-output", "json", end.get("output")) if end else None
    return {
        "type": "agent",
        "agent_id": start.get("agent_id"),
        "agent_name": start.get("agent_name"),
        "invoker": start.get("parent_id"),
        "prompt": prompt_path,
        "output": output_path,
        "system_prompt": start.get("system_prompt"),
        "usage": end.get("usage") if end else None,
        "response_id": end.get("response_id") if end else None,
        "timestamp": start.get("ts"),
    }


def _tool_record(start: dict[str, Any], end: dict[str, Any] | None, store: _PayloadStore) -> dict[str, Any]:
    seq = start.get("seq")
    stem = f"{seq:05d}-tool-{_slug(start.get('tool_name'))}-{_slug(start.get('tool_call_id'))}"
    result = end.get("result") if end else None
    output_path = store.write(f"{stem}-output", "txt", result) if result is not None else None
    return {
        "type": "tool_call",
        "agent_id": start.get("agent_id"),
        "invoker": start.get("agent_id"),
        "tool": start.get("tool_name"),
        "tool_call_id": start.get("tool_call_id"),
        "command": start.get("arguments"),
        "output": output_path,
        "timestamp": start.get("ts"),
    }


def _pair_events(events: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tool_ends: dict[str, dict[str, Any]] = {}
    llm_ends: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        kind = event.get("event")
        if kind == "tool_end":
            call_id = event.get("tool_call_id")
            if call_id is not None:
                tool_ends[call_id] = event
        elif kind == "llm_end":
            llm_ends.setdefault(event.get("agent_id"), []).append(event)
    return tool_ends, llm_ends


def build_digest(run_dir: Path) -> Path | None:
    """Read run_dir/trace.jsonl and write run_dir/digest/ with split payloads and summary.json."""
    from strix.telemetry.trace import TRACE_FILENAME

    trace_path = run_dir / TRACE_FILENAME
    if not trace_path.exists():
        logger.warning("no trace file to digest at %s", trace_path)
        return None

    events = _read_events(trace_path)
    digest_dir = run_dir / DIGEST_DIRNAME
    store = _PayloadStore(digest_dir / PAYLOADS_DIRNAME)
    tool_ends, llm_ends = _pair_events(events)

    records: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("event")
        if kind == "llm_start":
            pending = llm_ends.get(event.get("agent_id"))
            end = pending.pop(0) if pending else None
            records.append(_agent_record(event, end, store))
        elif kind == "tool_start":
            end = tool_ends.get(event.get("tool_call_id"))
            records.append(_tool_record(event, end, store))

    summary_path = digest_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote scan digest: %s (%d records)", summary_path, len(records))
    return summary_path


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a structured digest from a scan's trace.jsonl")
    parser.add_argument("run_dir", type=Path, help="Scan run directory containing trace.jsonl")
    args = parser.parse_args()
    result = build_digest(args.run_dir)
    if result is None:
        raise SystemExit(1)
    print(result)


if __name__ == "__main__":
    _main()
