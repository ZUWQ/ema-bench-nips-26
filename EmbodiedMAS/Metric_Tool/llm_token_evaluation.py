"""
Monkey-patch ``openai`` ``chat.completions.create`` to record token usage (thread-safe); aggregates are written by
``export_summary_json`` / ``flush_token_summary`` to ``data_save/llm_tokens/summary_*.json``.

Call ``install()`` once at process entry; ``atexit`` writes ``summary_*.json`` on exit.
Timestamps and the default ``run_id`` use Asia/Shanghai when available, otherwise UTC+8.

Environment variables: ``LLM_TOKEN_LOG_DIR``, ``LLM_TOKEN_RUN_ID``, ``LLM_TOKEN_JSONL`` (default 1),
``LLM_TOKEN_DISABLE_ATEXIT`` (non-empty disables automatic flush on exit).
If ``LLM_TOKEN_LOG_DIR`` is unset and ``EMBODIED_BENCHMARK_DATA_ROOT`` is set, logs go to ``<DATA_ROOT>/llm_tokens``;
for the same tongsim_automation run, LLM memory/timing logs use ``EMBODIED_BENCHMARK_LOG_DIR`` or ``<DATA_ROOT>/memory_logs``.

Does not import ``evaluation.py``, avoiding a tongsim/matplotlib dependency.

Privacy / repository hygiene: Token logs and summaries must not include API keys (only usage counts and metadata).
Store keys in environment variables or an untracked ``llm_profiles.json``; rotate leaked keys and rewrite Git history
with ``git filter-repo`` or BFG if secrets were ever committed — old clones can still expose them (see
``EmbodiedMAS/ExperimentRunning/Automation_runner.py``).
"""

from __future__ import annotations

import atexit
import inspect
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo

    _TZ_BEIJING = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover — fallback to fixed UTC+8 when tzdata is missing
    _TZ_BEIJING = timezone(timedelta(hours=8))


def _now_beijing() -> datetime:
    return datetime.now(_TZ_BEIJING)


_lock = threading.Lock()
_installed = False
_original_create: Optional[Callable[..., Any]] = None
_patched_completions_class: Optional[type] = None
_cached_run_id: Optional[str] = None
_atexit_registered = False

# Aggregates: key = (model, caller_module) -> {prompt_tokens, completion_tokens, total_tokens, calls}
_aggregates: Dict[Tuple[str, str], Dict[str, int]] = {}

_ENV_LOG_DIR = "LLM_TOKEN_LOG_DIR"
_ENV_RUN_ID = "LLM_TOKEN_RUN_ID"
_ENV_JSONL = "LLM_TOKEN_JSONL"
_ENV_DISABLE_ATEXIT = "LLM_TOKEN_DISABLE_ATEXIT"


def _log_base_dir() -> Path:
    explicit = os.getenv(_ENV_LOG_DIR)
    if explicit:
        return Path(explicit)
    base = os.environ.get("EMBODIED_BENCHMARK_DATA_ROOT")
    if base:
        return Path(base) / "llm_tokens"
    return Path("data_save/llm_tokens")


def _run_id() -> str:
    global _cached_run_id
    if _cached_run_id is None:
        _cached_run_id = os.getenv(_ENV_RUN_ID) or _now_beijing().strftime(
            "%Y%m%d_%H%M%S"
        )
    return _cached_run_id


def get_log_run_id() -> str:
    """This process's token log run_id (matches events_/summary_ filenames)."""
    return _run_id()


def _jsonl_enabled() -> bool:
    return os.getenv(_ENV_JSONL, "1").strip().lower() not in ("0", "false", "no")


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump(mode="python")
        except Exception:
            pass
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _guess_caller_module() -> str:
    """Skip openai and this module's stack frames; return the first caller module name."""
    skip_substrings = (
        "openai",
        "Metric_Tool/llm_token_evaluation",
        "llm_token_evaluation.py",
        "asyncio",
        "concurrent/futures",
    )
    try:
        for frame_info in inspect.stack()[3:]:
            mod = inspect.getmodule(frame_info.frame)
            if mod is None:
                continue
            name = getattr(mod, "__name__", "") or ""
            file = getattr(mod, "__file__", "") or ""
            if not name or name == "__main__":
                if file and not any(s in file for s in skip_substrings):
                    return Path(file).stem
                continue
            if any(s in file for s in skip_substrings):
                continue
            if name.startswith("openai"):
                continue
            if "Metric_Tool" in name and "llm_token_evaluation" in name:
                continue
            return name
    except Exception:
        pass
    return "unknown"


def _append_jsonl(record: Dict[str, Any]) -> None:
    if not _jsonl_enabled():
        return
    base = _log_base_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"events_{_run_id()}.jsonl"
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def record_chat_completion(completion: Any, model_kw: Optional[str] = None) -> None:
    """Public API: manually record one completion when ``install()`` is not used."""
    _record_completion(completion, model_kw)


def _record_completion(
    completion: Any,
    model_kw: Optional[str],
) -> None:
    usage = getattr(completion, "usage", None)
    udict = _usage_to_dict(usage)
    if not udict or all(v is None for v in udict.values()):
        return

    model = model_kw or getattr(completion, "model", None) or "unknown"
    caller = _guess_caller_module()

    pt = int(udict.get("prompt_tokens") or 0)
    ct = int(udict.get("completion_tokens") or 0)
    tt = int(udict.get("total_tokens") or (pt + ct))

    ts_iso = _now_beijing().isoformat()
    event = {
        "timestamp": ts_iso,
        "model": model,
        "caller_module": caller,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "usage_raw": udict,
    }
    _append_jsonl(event)

    key = (str(model), caller)
    with _lock:
        bucket = _aggregates.setdefault(
            key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            },
        )
        bucket["prompt_tokens"] += pt
        bucket["completion_tokens"] += ct
        bucket["total_tokens"] += tt
        bucket["calls"] += 1


def _make_wrapped_create(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        completion = original(self, *args, **kwargs)
        try:
            _record_completion(completion, kwargs.get("model"))
        except Exception:
            pass
        return completion

    return wrapped


def _resolve_completions_create() -> Tuple[Any, str]:
    """Return ``(Completions class, qualified_name)`` for patching; supports common openai>=1.0 layouts."""
    import importlib

    for modname, clsname in (
        ("openai.resources.chat.completions", "Completions"),
        ("openai.resources.chat.completions.completions", "Completions"),
    ):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        cls = getattr(mod, clsname, None)
        if cls is not None and callable(getattr(cls, "create", None)):
            return cls, f"{modname}.{clsname}"
    raise ImportError(
        "Could not locate openai Chat Completions class (requires openai>=1.0)."
    )


def is_installed() -> bool:
    return _installed


def ensure_atexit_flush_registered() -> None:
    """Register ``atexit`` flush when using ``create_openai_client`` without ``install()``."""
    global _atexit_registered
    if _atexit_registered or os.getenv(_ENV_DISABLE_ATEXIT, "").strip():
        return
    atexit.register(flush_token_summary)
    _atexit_registered = True


def install() -> None:
    """Replace ``Chat Completions.create`` with a recording wrapper (idempotent)."""
    global _installed, _original_create, _patched_completions_class
    if _installed:
        return

    CompletionsCls, _ = _resolve_completions_create()
    _patched_completions_class = CompletionsCls
    _original_create = CompletionsCls.create
    CompletionsCls.create = _make_wrapped_create(_original_create)  # type: ignore[method-assign]
    _installed = True

    ensure_atexit_flush_registered()


def uninstall() -> None:
    """For tests: restore the original ``create``."""
    global _installed, _original_create, _patched_completions_class
    if not _installed or _original_create is None or _patched_completions_class is None:
        return
    _patched_completions_class.create = _original_create  # type: ignore[method-assign]
    _installed = False
    _original_create = None
    _patched_completions_class = None


def get_totals() -> Dict[str, Any]:
    """Return a shallow copy of aggregate totals."""
    with _lock:
        per_key: Dict[str, Any] = {}
        grand = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        for (model, caller), v in _aggregates.items():
            key = f"{model}::{caller}"
            per_key[key] = dict(v)
            grand["prompt_tokens"] += v["prompt_tokens"]
            grand["completion_tokens"] += v["completion_tokens"]
            grand["total_tokens"] += v["total_tokens"]
            grand["calls"] += v["calls"]
        return {"by_model_caller": per_key, "grand_total": grand}


def export_summary_json(path: Optional[Path] = None) -> Path:
    """Write summary JSON; default path ``data_save/llm_tokens/summary_<run_id>.json``."""
    base = _log_base_dir()
    base.mkdir(parents=True, exist_ok=True)
    out = path or (base / f"summary_{_run_id()}.json")
    payload = {
        "run_id": _run_id(),
        "written_at": _now_beijing().isoformat(),
        "totals": get_totals(),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def flush_token_summary() -> Path:
    """Write summary file (``atexit`` or manual); ``grand_total`` is zero when there were no calls."""
    return export_summary_json()


def _written_at_beijing_iso() -> str:
    return _now_beijing().isoformat()


@dataclass
class LLMTokenEvaluation:
    task_id: str
    task_type: str = "llm_token_usage"
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def build_payload(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "run_id": get_log_run_id(),
            "written_at": _written_at_beijing_iso(),
            "is_installed": is_installed(),
            "totals": get_totals(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.build_payload()
