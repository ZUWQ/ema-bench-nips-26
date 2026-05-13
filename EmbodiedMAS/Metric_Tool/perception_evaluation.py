"""
Call ``install_perception_evaluation()`` once at process entry to monkey-patch each loaded
``ActionAPI.get_perception_object_list``: after each successful call, append one JSON line to that agent's JSONL file.

Before ``BaseAgent.spawn_agent`` returns, code typically awaits ``UnaryAPI.query_info`` and
``record_query_info_snapshot`` appends a line with every actor id in the scene to the same JSONL; that line
matches the later perception rows' ``result.actor_info`` shape (id list only).

The initial filename uses the simulation actor id (stable); each JSON line's ``agent_label`` prefers a human-readable
``name`` when available. On process exit (``atexit``), files are renamed to
``{sanitized(name)}_{original timestamp or RUN_ID suffix}.jsonl`` using ``agent_label`` from the file (non-UUID preferred).

Each record stores ``actor.id`` for every item in ``result.actor_info``. If any object has ``burning_state`` true
(read from ``actor`` then outer ``actor_info``), the line's top level sets ``burning_state: true`` once; that agent
file (``id_stem``) is not checked for burning again afterward.
Timestamps use Asia/Shanghai when available, otherwise UTC+8.

Environment variables:
  perception_evaluation_DIR                 default data_save/explore_perception
  perception_evaluation_RUN_ID              if set, filename is {id_stem}_{run_id}.jsonl
  perception_evaluation_USE_SESSION_SUFFIX  set to 0/false/no to drop the time suffix (only {id_stem}.jsonl)
  perception_evaluation_SKIP_FINAL_RENAME   set to 1/true/yes/on to skip rename-by-agent_label on exit
  perception_evaluation_ENABLED             set to 0/false/no to disable writes

If ``agent_id`` is None or no id can be resolved, the original method still runs but no dump is written (avoids junk filenames).

Privacy / repository hygiene: Perception JSONL may list actor ids, labels, and paths under your data root. Do not commit
secrets, API keys, or ``llm_profiles.json``; prefer environment variables. If sensitive logs or paths were ever committed,
rotate credentials and rewrite Git history with ``git filter-repo`` or BFG before a public push — old clones can still
expose that material (see ``EmbodiedMAS/ExperimentRunning/Automation_runner.py``).
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys
import threading
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from zoneinfo import ZoneInfo

    _TZ_BEIJING = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    _TZ_BEIJING = timezone(timedelta(hours=8))


def _now_beijing() -> datetime:
    return datetime.now(_TZ_BEIJING)


_lock = threading.Lock()
_originals: Dict[type, Callable[..., Any]] = {}

_session_suffix: Optional[str] = None
_written_dump_paths: Set[Path] = set()
_log_path_by_id_stem: Dict[str, Path] = {}
_perception_burning_logged_stems: Set[str] = set()
_atexit_rename_registered = False

_ENV_DIR = "perception_evaluation_DIR"
_ENV_ENABLED = "perception_evaluation_ENABLED"
_ENV_RUN_ID = "perception_evaluation_RUN_ID"
_ENV_SESSION_SUFFIX = "perception_evaluation_USE_SESSION_SUFFIX"
_ENV_SKIP_RENAME = "perception_evaluation_SKIP_FINAL_RENAME"

_PATCH_FILE_RELATIVE: Tuple[str, ...] = (
    "Only_Language-base_Agent/action_ol.py",
    "VLM_Agent/action_vlm.py",
    "VLM_Agent/Single_agent/action_ol.py",
    "Vision_Language-base_Agent/action_vl.py",
    "Vision_VLM_Agent/action_vlm.py",
    "Human_Agent/action_h.py",
)


def _embodiedmas_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dump_enabled() -> bool:
    return os.getenv(_ENV_ENABLED, "1").strip().lower() not in ("0", "false", "no", "off")


def _dump_base_dir() -> Path:
    return Path(os.getenv(_ENV_DIR, "data_save/explore_perception"))


def _use_session_suffix() -> bool:
    return os.getenv(_ENV_SESSION_SUFFIX, "1").strip().lower() not in ("0", "false", "no", "off")


def _skip_final_rename() -> bool:
    return os.getenv(_ENV_SKIP_RENAME, "").strip().lower() in ("1", "true", "yes", "on")


def _ensure_session_suffix() -> str:
    global _session_suffix
    with _lock:
        if _session_suffix is None:
            _session_suffix = _now_beijing().strftime("%Y%m%d_%H%M%S")
        return _session_suffix


def _to_jsonable(obj: Any, depth: int = 0) -> Any:
    if depth > 40:
        return "<max_depth>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        h = obj.hex()
        return {"__bytes_hex__": h[:128] + ("..." if len(h) > 128 else "")}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x, depth + 1) for x in obj]
    if hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z"):
        try:
            return {
                "x": float(obj.x),
                "y": float(obj.y),
                "z": float(obj.z),
            }
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump(mode="python"), depth + 1)
        except Exception:
            pass
    return str(obj)


def _sanitize_filename_part(s: str, max_len: int = 64) -> str:
    s = re.sub(r'[/\\:*?"<>|\s]+', "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "agent"
    return s[:max_len]


def _is_uuid_like_label(s: str) -> bool:
    t = s.strip()
    if len(t) < 32:
        return False
    return bool(
        re.fullmatch(
            r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
            t,
        )
    )


def _agent_id_stem_for_filename(agent_id: Any) -> str:
    """Filename stem from id/guid or str/bytes only; never from name."""
    if isinstance(agent_id, dict):
        aid = agent_id.get("id") or agent_id.get("guid")
        if aid is not None:
            return _sanitize_filename_part(str(aid)[:64])
        return "agent_dict_no_id"
    if agent_id is None:
        return "unknown"
    if isinstance(agent_id, bytes):
        return _sanitize_filename_part(agent_id.hex()[:48])
    return _sanitize_filename_part(str(agent_id)[:64])


def _agent_display_label(agent_id: Any) -> str:
    """JSON ``agent_label``: prefer ``name``, else id."""
    if isinstance(agent_id, dict):
        name = agent_id.get("name") or agent_id.get("Name")
        if name:
            t = str(name).strip()
            if t:
                return t
        return _agent_id_stem_for_filename(agent_id)
    if agent_id is None:
        return "unknown"
    if isinstance(agent_id, bytes):
        return _sanitize_filename_part(agent_id.hex()[:48])
    t = str(agent_id).strip()
    return t if t else "unknown"


def _agent_id_summary(agent_id: Any) -> Any:
    if isinstance(agent_id, dict):
        return {
            k: _to_jsonable(v)
            for k, v in agent_id.items()
            if k in ("name", "Name", "id", "guid", "tags")
        }
    return _to_jsonable(agent_id)


def _should_record_for_agent_id(agent_id: Any) -> bool:
    if agent_id is None:
        return False
    if isinstance(agent_id, dict):
        return bool(agent_id.get("id") or agent_id.get("guid"))
    return True


def _result_dict_any_burning_true(result: Any) -> bool:
    """True if any ``actor_info`` entry has ``burning_state`` true on the same path as ``_simplify_result_actor_info_only``."""
    if not isinstance(result, dict):
        return False
    for actor_info in result.get("actor_info", []):
        if not isinstance(actor_info, dict):
            continue
        actor = actor_info.get("actor")
        if not isinstance(actor, dict):
            actor = {}
        b = actor.get("burning_state")
        if b is None:
            b = actor_info.get("burning_state")
        if b is True:
            return True
    return False


def _simplify_result_actor_info_only(result: Any) -> Dict[str, Any]:
    """Keep only each ``result.actor_info`` item's actor id (top-level ``burning_state`` handled in ``record_after_get_perception``)."""
    if not isinstance(result, dict):
        return {"actor_info": []}
    out_items: List[Dict[str, Any]] = []
    for actor_info in result.get("actor_info", []):
        if not isinstance(actor_info, dict):
            continue
        actor = actor_info.get("actor")
        if not isinstance(actor, dict):
            actor = {}
        out_items.append(_to_jsonable(actor.get("id")))
        # burning = actor.get("burning_state")
        # if burning is None:
        #     burning = actor_info.get("burning_state")
        # out_items.append(
        #     {
        #         "actor": {
        #             "id": _to_jsonable(actor.get("id")),
        #             "name": _to_jsonable(actor.get("name")),
        #             "location": _to_jsonable(actor.get("location")),
        #             "burning_state": _to_jsonable(burning),
        #         }
        #     }
        # )
    return {"actor_info": out_items}


def _append_log_path_for_id_stem(id_stem: str) -> Path:
    base = _dump_base_dir()
    base.mkdir(parents=True, exist_ok=True)
    rid = os.getenv(_ENV_RUN_ID)
    if rid:
        rid = rid.strip()
        return base / f"{id_stem}_{rid}.jsonl"
    if _use_session_suffix():
        suf = _ensure_session_suffix()
        return base / f"{id_stem}_{suf}.jsonl"
    return base / f"{id_stem}.jsonl"


def _log_path_for_id_stem(id_stem: str) -> Path:
    """
    Each agent's JSONL path is resolved once per process via ``_append_log_path_for_id_stem`` and cached.
    ``record_query_info_snapshot`` and ``record_after_get_perception`` share that path so logs are not split.

    Do not call ``_append_log_path_for_id_stem`` while holding ``_lock`` (it may call ``_ensure_session_suffix``,
    which also takes ``_lock``); ``threading.Lock`` is not re-entrant and will deadlock on the same thread.
    """
    with _lock:
        cached = _log_path_by_id_stem.get(id_stem)
        if cached is not None:
            return cached
    path = _append_log_path_for_id_stem(id_stem)
    resolved = path.resolve()
    with _lock:
        existing = _log_path_by_id_stem.get(id_stem)
        if existing is not None:
            return existing
        _log_path_by_id_stem[id_stem] = resolved
        return resolved


def _best_agent_label_from_jsonl(path: Path) -> Optional[str]:
    """Last non-UUID-shaped ``agent_label`` in the file."""
    best: Optional[str] = None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        lab = obj.get("agent_label")
        if not isinstance(lab, str) or not lab.strip():
            continue
        lab = lab.strip()
        if not _is_uuid_like_label(lab):
            best = lab
    return best


def finalize_perception_evaluation_renames() -> List[Tuple[Path, Path]]:
    """
    Rename written ``{id_stem}_{suffix}.jsonl`` files to ``{agent_label}_{suffix}.jsonl``.
    ``agent_label`` is taken from JSON lines in the file (non-UUID preferred); skip if no better name.
    Returns a list of ``(old_path, new_path)``.
    """
    if _skip_final_rename() or not _dump_enabled():
        return []
    done: List[Tuple[Path, Path]] = []
    with _lock:
        paths = sorted(_written_dump_paths, key=lambda p: str(p))
    run_id = os.getenv(_ENV_RUN_ID)
    run_id = run_id.strip() if run_id else ""

    for path in paths:
        try:
            rp = path.resolve()
        except OSError:
            continue
        if not rp.is_file():
            continue

        stem = rp.stem
        if run_id and stem.endswith(f"_{run_id}"):
            id_part = stem[: -(len(run_id) + 1)]
            suffix_part = run_id
        else:
            m = re.match(r"^(.+)_(\d{8}_\d{6})$", stem)
            if m:
                id_part, suffix_part = m.group(1), m.group(2)
            else:
                id_part, suffix_part = stem, ""

        display = _best_agent_label_from_jsonl(rp)
        if not display:
            continue
        new_stem_san = _sanitize_filename_part(display)
        id_san = _sanitize_filename_part(id_part)
        if new_stem_san == id_san or _is_uuid_like_label(display):
            continue

        if suffix_part:
            new_path = rp.parent / f"{new_stem_san}_{suffix_part}.jsonl"
        else:
            new_path = rp.parent / f"{new_stem_san}.jsonl"
        if new_path == rp:
            continue
        n = 2
        while new_path.exists():
            if suffix_part:
                new_path = rp.parent / f"{new_stem_san}_{suffix_part}_{n}.jsonl"
            else:
                new_path = rp.parent / f"{new_stem_san}_{n}.jsonl"
            n += 1
        try:
            rp.rename(new_path)
            done.append((rp, new_path))
        except OSError:
            pass
    return done


def _register_atexit_rename_once() -> None:
    global _atexit_rename_registered
    with _lock:
        if _atexit_rename_registered:
            return
        _atexit_rename_registered = True
    atexit.register(finalize_perception_evaluation_renames)


def _ids_from_query_info_list(qlist: Any) -> List[Any]:
    """Extract each item's id from ``UnaryAPI.query_info`` list[dict] (``_to_jsonable`` normalized)."""
    if not isinstance(qlist, list):
        return []
    out: List[Any] = []
    for item in qlist:
        if not isinstance(item, dict):
            continue
        aid = item.get("id")
        if aid is None:
            continue
        out.append(_to_jsonable(aid))
    return out


def record_query_info_snapshot(agent_id: Any, query_info_list: Any) -> None:
    """Append one line with full-scene actor ids from ``query_info`` to this agent's .jsonl (shared with get_perception dumps)."""
    if not _dump_enabled():
        return
    if not _should_record_for_agent_id(agent_id):
        return
    id_stem = _agent_id_stem_for_filename(agent_id)
    display_label = _agent_display_label(agent_id)
    path = _log_path_for_id_stem(id_stem)
    payload = {
        "timestamp": _now_beijing().replace(tzinfo=None).isoformat(timespec='milliseconds'),
        "agent_label": display_label,
        "result": {"actor_info": _ids_from_query_info_list(query_info_list)},
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        _written_dump_paths.add(path.resolve())
    _register_atexit_rename_once()


def record_after_get_perception(
    agent_id: Any,
    result_dict: Any,
    _perception_object_list_snapshot: Any,
    source_module: str,
) -> None:
    """Append one JSON line for this call to the agent's .jsonl (thread-safe)."""
    if not _dump_enabled():
        return
    if not _should_record_for_agent_id(agent_id):
        return
    id_stem = _agent_id_stem_for_filename(agent_id)
    display_label = _agent_display_label(agent_id)
    path = _log_path_for_id_stem(id_stem)
    payload = {
        "timestamp": _now_beijing().replace(tzinfo=None).isoformat(timespec='milliseconds'),
        # "source_action_module": source_module,
        "agent_label": display_label,
        # "dump_id_stem": id_stem,
        # "agent_id_summary": _agent_id_summary(agent_id),
        "result": _simplify_result_actor_info_only(result_dict),
    }
    with _lock:
        if id_stem not in _perception_burning_logged_stems:
            if _result_dict_any_burning_true(result_dict):
                payload["burning_state"] = True
                _perception_burning_logged_stems.add(id_stem)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        _written_dump_paths.add(path.resolve())
    _register_atexit_rename_once()


def _find_loaded_module(abs_path: Path) -> Optional[types.ModuleType]:
    try:
        target = abs_path.resolve()
    except Exception:
        target = abs_path
    for m in list(sys.modules.values()):
        if m is None:
            continue
        fp = getattr(m, "__file__", None)
        if not fp or not isinstance(fp, str):
            continue
        try:
            if Path(fp).resolve() == target:
                return m
        except Exception:
            continue
    return None


def _make_wrapper(
    orig: Callable[..., Any],
    source_tag: str,
) -> Callable[..., Any]:
    async def wrapped(self: Any, agent_id: Any = None, timeout: float = 5.0) -> Any:
        out = await orig(self, agent_id, timeout)
        try:
            snap = getattr(self, "_perception_object_list", None)
            record_after_get_perception(agent_id, out, snap, source_tag)
        except Exception:
            pass
        return out

    return wrapped


def _patch_action_api_class(cls: type, source_tag: str) -> bool:
    if cls in _originals:
        return False
    orig = getattr(cls, "get_perception_object_list", None)
    if not callable(orig):
        return False
    _originals[cls] = orig
    cls.get_perception_object_list = _make_wrapper(orig, source_tag)  # type: ignore[method-assign]
    return True


def install_perception_evaluation() -> None:
    """
    Scan ``sys.modules`` for already-loaded action modules (matched by path under EmbodiedMAS) and patch their ``ActionAPI``.
    Safe to call multiple times: modules imported later are patched on a subsequent call.
    Call once after importing application code and before ``main()`` in entry scripts.
    """
    root = _embodiedmas_root()
    for rel in _PATCH_FILE_RELATIVE:
        abs_path = root / rel
        if not abs_path.is_file():
            continue
        mod = _find_loaded_module(abs_path)
        if mod is None:
            continue
        api = getattr(mod, "ActionAPI", None)
        if api is None or not isinstance(api, type):
            continue
        tag = f"{mod.__name__}@{rel}"
        _patch_action_api_class(api, tag)


def uninstall_perception_evaluation() -> None:
    """Restore original ``get_perception_object_list`` on patched ``ActionAPI`` classes (for tests)."""
    for cls, orig in list(_originals.items()):
        cls.get_perception_object_list = orig  # type: ignore[method-assign]
    _originals.clear()
    with _lock:
        _written_dump_paths.clear()
        _log_path_by_id_stem.clear()
        _perception_burning_logged_stems.clear()


def is_perception_evaluation_installed() -> bool:
    return bool(_originals)
