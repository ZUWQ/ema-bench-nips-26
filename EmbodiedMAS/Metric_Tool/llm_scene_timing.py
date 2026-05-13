"""Scene pause around LLM calls; memory and timing logs as separate files in the same folder.

Privacy / repository hygiene: Memory and timing text logs may contain prompts or host-specific paths under
``EMBODIED_BENCHMARK_LOG_DIR`` / ``EMBODIED_BENCHMARK_DATA_ROOT``. Do not commit secrets; if sensitive material
was ever committed, rewrite Git history with ``git filter-repo`` or BFG before a public push (see
``EmbodiedMAS/ExperimentRunning/Automation_runner.py``).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

import tongsim as ts

T = TypeVar("T")


def ensure_session_id(session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_benchmark_log_dir() -> Path:
    """
    Directory for LLM memory / timing text logs.

    Precedence: ``EMBODIED_BENCHMARK_LOG_DIR`` → ``EMBODIED_BENCHMARK_DATA_ROOT/memory_logs`` → ./memory_logs
    (alongside ``llm_tokens`` under the same task workspace ``DATA_ROOT`` in tongsim_automation).
    """
    explicit = os.environ.get("EMBODIED_BENCHMARK_LOG_DIR")
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        root = os.environ.get("EMBODIED_BENCHMARK_DATA_ROOT")
        if root:
            p = Path(root).expanduser().resolve() / "memory_logs"
        else:
            p = Path("memory_logs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_agent_id(agent_id: Optional[str]) -> str:
    aid = agent_id or "agent"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in aid)


def session_memory_path(
    log_dir: Path,
    agent_id: Optional[str],
    session_id: str,
) -> Path:
    """Append-only memory snapshot file (same naming idea as before)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_aid = _safe_agent_id(agent_id)
    return log_dir / f"memory_{safe_aid}_{session_id}.txt"


def session_timing_path(
    log_dir: Path,
    agent_id: Optional[str],
    session_id: str,
) -> Path:
    """Append-only timing events file (JSON lines per step block)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_aid = _safe_agent_id(agent_id)
    return log_dir / f"timing_{safe_aid}_{session_id}.txt"


def session_log_path(
    log_dir: Path,
    agent_id: Optional[str],
    session_id: str,
) -> Path:
    """Alias for memory log path (backward compatible name)."""
    return session_memory_path(log_dir, agent_id, session_id)


def append_memory_block(
    path: Path,
    *,
    log_round: int,
    memory_lines: list[str],
    session_header: Optional[str] = None,
) -> None:
    recorded_at = datetime.now().isoformat()
    parts: list[str] = [
        f"===== step={log_round} recorded_at={recorded_at} =====",
    ]
    if session_header:
        parts.append(session_header)
    parts.append("## MEMORY")
    parts.extend(memory_lines if memory_lines else ["(empty)"])
    parts.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(parts))


def append_timing_block(
    path: Path,
    *,
    log_round: int,
    timing_events: list[dict[str, Any]],
    session_header: Optional[str] = None,
) -> None:
    recorded_at = datetime.now().isoformat()
    parts: list[str] = [
        f"===== step={log_round} recorded_at={recorded_at} =====",
    ]
    if session_header:
        parts.append(session_header)
    if timing_events:
        for ev in timing_events:
            parts.append(json.dumps(ev, ensure_ascii=False))
    else:
        parts.append("(no timing events this step)")
    parts.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(parts))


async def run_with_scene_paused(
    conn: Any,
    coro_factory: Callable[[], Awaitable[T]],
    *,
    replan: bool = False,
) -> tuple[T, dict[str, Any]]:
    """
    pause_scene(True) -> await coro_factory() -> pause_scene(False) in finally.
    Returns (result, timing dict) with name llm_thinking.
    """
    await ts.UnaryAPI.pause_scene(conn, True)
    start_wall = datetime.now().isoformat()
    t0 = time.monotonic()
    try:
        result = await coro_factory()
    finally:
        await ts.UnaryAPI.pause_scene(conn, False)
    end_wall = datetime.now().isoformat()
    duration_sec = round(time.monotonic() - t0, 6)
    ev: dict[str, Any] = {
        "name": "llm_thinking",
        "start": start_wall,
        "end": end_wall,
        "duration_sec": duration_sec,
    }
    if replan:
        ev["replan"] = True
    return result, ev
