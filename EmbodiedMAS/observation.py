"""Observation camera and map building utilities.

Privacy / repository hygiene: Saved RGB/depth paths and logs may reveal host layout or scene content. Do not commit
API keys or ``llm_profiles.json``; prefer environment variables. If secrets or sensitive paths were ever committed,
rotate keys and rewrite Git history with ``git filter-repo`` or BFG before a public push — old clones can still expose
that material (see ``EmbodiedMAS/ExperimentRunning/Automation_runner.py``).
"""
from __future__ import annotations

import binascii
import csv
import os
import re
import struct
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tongsim as ts
from tongsim.core.world_context import WorldContext

_OBSERVATION_FILE = Path(__file__).resolve()


def _infer_log_dir_from_calling_agent() -> Path:
    """
    Default save directory: walk the stack, take the deepest frame inside EmbodiedMAS that is not this module
    (usually the script that constructs the Agent, e.g. Single_agent/llm_agent_vl.py), and use that file's ``logs/`` folder.
    If none matches, fall back to ``logs/`` under the parent of EmbodiedMAS (legacy behavior).
    """
    import inspect

    chosen: Path | None = None
    frame = inspect.currentframe()
    try:
        f = frame.f_back if frame else None
        while f is not None:
            f = f.f_back
            if f is None:
                break
            try:
                co_file = Path(f.f_code.co_filename).resolve()
            except (OSError, ValueError):
                continue
            if co_file == _OBSERVATION_FILE:
                continue
            if co_file.name == "observation.py" and "EmbodiedMAS" in str(co_file):
                continue
            co_str = str(co_file)
            if "EmbodiedMAS" not in co_str:
                continue
            if "site-packages" in co_str:
                continue
            chosen = co_file.parent / "logs"
    finally:
        del frame
    return chosen if chosen is not None else _OBSERVATION_FILE.parents[1] / "logs"


class ObservationCamera:
    """First-person RGB and depth capture; see capture_demo.py."""
    _GLOBAL_AGENT_CAMERAS: dict[str, bytes | str] = {}

    def __init__(self, context: WorldContext, log_dir: Optional[Path] = None):
        self._context = context
        self._conn = context.conn
        self._log_dir = log_dir if log_dir is not None else _infer_log_dir_from_calling_agent()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # Instance cache: agent_id -> camera_id
        self._agent_cameras: dict[str, bytes | str] = {}
        # Instance cache: agent_id -> save directory
        self._agent_capture_dirs: dict[str, Path] = {}

    @staticmethod
    def _normalize_agent_id(agent_id: bytes | str | None) -> str:
        """Normalize ``agent_id`` to a string key."""
        if agent_id is None:
            return ""
        if isinstance(agent_id, (bytes, bytearray)):
            return binascii.hexlify(agent_id).decode("ascii")
        return str(agent_id)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a label to alphanumeric plus ``_`` and ``-``."""
        if not name:
            return "agent"
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)

    def register_agent_camera(self, agent_id: bytes | str | None, camera_id: bytes | str | None) -> None:
        """Register or update the camera bound to an agent (instance + global cache)."""
        if camera_id is None:
            return
        agent_key = self._normalize_agent_id(agent_id)
        if not agent_key:
            return
        self._agent_cameras[agent_key] = camera_id
        ObservationCamera._GLOBAL_AGENT_CAMERAS[agent_key] = camera_id

    def unregister_agent_camera(self, agent_id: bytes | str | None) -> None:
        """Drop cached camera mapping for an agent (instance + global cache)."""
        agent_key = self._normalize_agent_id(agent_id)
        if not agent_key:
            return
        self._agent_cameras.pop(agent_key, None)
        ObservationCamera._GLOBAL_AGENT_CAMERAS.pop(agent_key, None)
        self._agent_capture_dirs.pop(agent_key, None)

    def _ensure_capture_dir(self, agent_cache_key: str, agent_label: str) -> Path:
        """Ensure the capture directory exists, creating it if needed."""
        capture_dir = self._agent_capture_dirs.get(agent_cache_key)
        if capture_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{agent_label}_{timestamp}"
            capture_dir = self._log_dir / folder_name
            capture_dir.mkdir(parents=True, exist_ok=True)
            self._agent_capture_dirs[agent_cache_key] = capture_dir
        return capture_dir

    def _camera_tag(self, camera_id: Any) -> str:
        """String tag for a ``camera_id`` value."""
        if isinstance(camera_id, (bytes, bytearray)):
            return camera_id.hex()
        if isinstance(camera_id, str):
            return camera_id
        return str(camera_id)

    def _save_color_image(self, frame: dict[str, Any], path: Path, fmt: str = "png") -> Path | None:
        """Save a color frame; see capture_demo.py."""
        rgba = frame.get("rgba8")
        if not rgba:
            return None
        width = int(frame.get("width", 0))
        height = int(frame.get("height", 0))
        if width <= 0 or height <= 0:
            return None

        # Prefer PIL when available
        try:
            from PIL import Image  # type: ignore
            img = Image.frombytes("RGBA", (width, height), bytes(rgba), "raw", "BGRA")
            ext = fmt.lower()
            if ext in ("jpg", "jpeg"):
                img = img.convert("RGB")
                out = path.with_suffix(".jpg")
                img.save(out, quality=95)
                return out
            else:
                out = path.with_suffix(".png")
                img.save(out)
                return out
        except Exception:
            pass

        # Fallback: manual PNG encoding
        def _crc(chunk_type: bytes, data: bytes) -> bytes:
            c = binascii.crc32(chunk_type)
            c = binascii.crc32(data, c)
            return struct.pack(">I", (c & 0xFFFFFFFF))

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + chunk_type + data + _crc(chunk_type, data)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        raw = bytearray()
        mv = memoryview(rgba)
        row_bytes = width * 4
        for y in range(height):
            raw.append(0)
            start = y * row_bytes
            row = mv[start : start + row_bytes]
            # BGRA -> RGBA
            for x in range(width):
                b = row[x * 4 + 0]
                g = row[x * 4 + 1]
                r = row[x * 4 + 2]
                a = row[x * 4 + 3] if len(row) >= (x * 4 + 4) else 255
                raw += bytes((r, g, b, a))
        idat = zlib.compress(bytes(raw))
        png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
        out = path.with_suffix(".png")
        out.write_bytes(png)
        return out

    def _save_depth_image(self, frame: dict[str, Any], path: Path) -> Path | None:
        """Save depth payload as EXR; see capture_demo.py."""
        depth_data = frame.get("depth") or frame.get("depth_r32")
        if not depth_data:
            return None
        
        out = path.with_suffix(".exr")
        out.write_bytes(depth_data)
        return out

    async def _find_best_camera(self, agent_id: bytes | str | None, agent_name: str | None, prefer_depth: bool = False) -> Optional[bytes | str]:
        """Pick the best camera for this agent using cache and name heuristics."""
        agent_key = self._normalize_agent_id(agent_id)
        
        # Check cache
        if agent_key in self._agent_cameras:
            return self._agent_cameras[agent_key]
        
        # List cameras and match by name
        all_cameras = await ts.CaptureAPI.list_cameras(self._conn)
        candidates = []
        
        for cam_desc in all_cameras:
            cam_info = cam_desc.get("camera", {})
            cam_name = cam_info.get("name", "")
            cam_id = cam_info.get("id")
            if not cam_id:
                continue
            
            # Match camera name to agent
            if (str(agent_id) in cam_name 
                or (agent_name and agent_name in cam_name)
                or (agent_name and f"AgentCamera_{agent_name}" in cam_name)
                or f"AgentCamera_{agent_id}" in cam_name):
                candidates.append((cam_name, cam_id))

        if candidates:
            if prefer_depth:
                for name, cid in candidates:
                    if "depth" in name.lower():
                        return cid
            return candidates[0][1]

        # Fallback: last camera in the list (often the most recently created)
        if all_cameras:
            last_cam = all_cameras[-1].get("camera", {})
            fallback_id = last_cam.get("id")
            if fallback_id:
                self._agent_cameras[agent_key] = fallback_id
                return fallback_id

        return None

    async def resolve_camera_id(
        self,
        agent_id: bytes | str | dict,
        *,
        prefer_depth: bool = True,
    ) -> Optional[bytes | str]:
        """
        Resolve the Capture camera id for an agent, same contract as :meth:`capture_rgb_and_depth`,
        reusing ``_find_best_camera`` cache and naming rules.
        """
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
            dict_cam_id = agent_id.get("camera_id")
            if dict_cam_id:
                agent_key = self._normalize_agent_id(agent_id_str)
                self._agent_cameras[agent_key] = dict_cam_id
        else:
            agent_id_str = agent_id
            agent_name = None

        return await self._find_best_camera(agent_id_str, agent_name, prefer_depth=prefer_depth)

    async def capture_rgb_image(
        self, 
        agent_id: bytes | str | dict, 
        image_prefix: Optional[str] = None, 
        image_format: str = "png"
    ) -> Optional[str]:
        """Capture and save an RGB image."""
        return await self._capture_impl(
            agent_id, 
            image_prefix, 
            image_format, 
            include_color=True, 
            include_depth=False, 
            return_type="color"
        )

    async def capture_depth_image(
        self, 
        agent_id: bytes | str | dict, 
        image_prefix: Optional[str] = None
    ) -> Optional[str]:
        """Capture and save depth; returns the ``.exr`` path."""
        return await self._capture_impl(
            agent_id, 
            image_prefix, 
            "exr", 
            include_color=True, 
            include_depth=True, 
            return_type="depth"
        )

    async def capture_rgb_and_depth(
        self,
        agent_id: bytes | str | dict,
        image_prefix: Optional[str] = None,
        image_format: str = "png",
    ) -> dict[str, Optional[str]]:
        """
        Single snapshot saves RGB and depth together, avoiding two ``capture_snapshot`` calls from
        separate ``capture_rgb_image`` / ``capture_depth_image`` invocations.
        """
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
            dict_cam_id = agent_id.get("camera_id")
            if dict_cam_id:
                agent_key = self._normalize_agent_id(agent_id_str)
                self._agent_cameras[agent_key] = dict_cam_id
        else:
            agent_id_str = agent_id
            agent_name = None

        agent_key = self._normalize_agent_id(agent_id_str)
        agent_label = self._sanitize_name(agent_name or agent_key or str(agent_id_str))

        camera_id = await self._find_best_camera(agent_id_str, agent_name, prefer_depth=True)
        if camera_id is None:
            return {"rgb": None, "depth": None}

        snapshot = await ts.CaptureAPI.capture_snapshot(
            self._conn,
            camera_id,
            include_color=True,
            include_depth=True,
            timeout_seconds=3.0,
        )
        if not snapshot:
            return {"rgb": None, "depth": None}

        capture_dir = self._ensure_capture_dir(agent_key or agent_label, agent_label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        camera_tag = self._camera_tag(camera_id)
        prefix = image_prefix or f"agent_{str(agent_id_str)[:8]}_{camera_tag[:8]}"
        base_name = f"{prefix}_{timestamp}"

        rgb_path: Optional[str] = None
        depth_path: Optional[str] = None
        if snapshot.get("rgba8"):
            c_path = capture_dir / base_name
            saved_color = self._save_color_image(snapshot, c_path, fmt=image_format)
            if saved_color:
                rgb_path = str(saved_color)
        d_path = capture_dir / base_name
        saved_depth = self._save_depth_image(snapshot, d_path)
        if saved_depth:
            depth_path = str(saved_depth)

        return {"rgb": rgb_path, "depth": depth_path}

    async def _capture_impl(
        self, 
        agent_id: bytes | str | dict, 
        image_prefix: str | None, 
        image_format: str, 
        include_color: bool, 
        include_depth: bool,
        return_type: str
    ) -> Optional[str]:
        """Shared capture/save implementation."""
        # Normalize agent_id
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
            dict_cam_id = agent_id.get("camera_id")
            if dict_cam_id:
                agent_key = self._normalize_agent_id(agent_id_str)
                self._agent_cameras[agent_key] = dict_cam_id
        else:
            agent_id_str = agent_id
            agent_name = None

        agent_key = self._normalize_agent_id(agent_id_str)
        agent_label = self._sanitize_name(agent_name or agent_key or str(agent_id_str))

        # Resolve camera
        camera_id = await self._find_best_camera(agent_id_str, agent_name, prefer_depth=(return_type == "depth"))
        if camera_id is None:
            return None

        # Capture snapshot
        snapshot = await ts.CaptureAPI.capture_snapshot(
            self._conn,
            camera_id,
            include_color=include_color,
            include_depth=include_depth,
            timeout_seconds=3.0,
        )

        if not snapshot:
            return None

        # Write files
        capture_dir = self._ensure_capture_dir(agent_key or agent_label, agent_label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        camera_tag = self._camera_tag(camera_id)
        prefix = image_prefix or f"agent_{str(agent_id_str)[:8]}_{camera_tag[:8]}"
        base_name = f"{prefix}_{timestamp}"

        saved_path = None
        if include_color and snapshot.get('rgba8'):
            c_path = capture_dir / base_name
            saved_path_color = self._save_color_image(snapshot, c_path, fmt=image_format)
            if return_type == "color":
                saved_path = saved_path_color

        if include_depth:
            d_path = capture_dir / base_name
            saved_path_depth = self._save_depth_image(snapshot, d_path)
            if return_type == "depth":
                saved_path = saved_path_depth

        return str(saved_path) if saved_path else None

    async def get_latest_image_path(self, agent_id: bytes | str | dict) -> Optional[str]:
        """Return the path of the most recently saved image under this agent's capture dir."""
        # Normalize agent_id
        if isinstance(agent_id, dict):
            agent_id_str = agent_id.get("id", agent_id)
            agent_name = agent_id.get("name")
        else:
            agent_id_str = agent_id
            agent_name = None

        agent_key = self._normalize_agent_id(agent_id_str)
        agent_label = self._sanitize_name(agent_name or agent_key or str(agent_id_str))

        # Resolve capture directory
        capture_dir = self._agent_capture_dirs.get(agent_key or agent_label)
        if capture_dir is None or not capture_dir.exists():
            return None

        # Collect image files
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.exr"]:
            image_files.extend(capture_dir.glob(ext))

        if not image_files:
            return None

        return str(max(image_files, key=lambda p: p.stat().st_mtime))


_EMBODIED_MAS_DIR = Path(__file__).resolve().parent
_OBJECT_NAME_MAP: dict[str, str] | None = None
_OBJECT_DELETE_NAMES: frozenset[str] | None = None
_OBJECT_YOLO_DELETE_NAMES: frozenset[str] | None = None


def _load_object_name_map() -> dict[str, str]:
    global _OBJECT_NAME_MAP
    if _OBJECT_NAME_MAP is not None:
        return _OBJECT_NAME_MAP
    path = _EMBODIED_MAS_DIR / "ObjectDictionary.csv"
    mapping: dict[str, str] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                old = (row.get("object_name") or "").strip()
                new = (row.get("object_name_new") or "").strip()
                if old and new:
                    mapping[old] = new
    _OBJECT_NAME_MAP = mapping
    return mapping


def _load_object_delete_names() -> frozenset[str]:
    global _OBJECT_DELETE_NAMES
    if _OBJECT_DELETE_NAMES is not None:
        return _OBJECT_DELETE_NAMES
    path = _EMBODIED_MAS_DIR / "ObjectDelete.csv"
    names: set[str] = set()
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                n = (row.get("object_name_new") or "").strip()
                if n:
                    names.add(n)
    _OBJECT_DELETE_NAMES = frozenset(names)
    return _OBJECT_DELETE_NAMES


def _load_object_yolo_delete_names() -> frozenset[str]:
    """YOLO class-name denylist (one name per line in ObjectYoloDelete.csv); case-insensitive match."""
    global _OBJECT_YOLO_DELETE_NAMES
    if _OBJECT_YOLO_DELETE_NAMES is not None:
        return _OBJECT_YOLO_DELETE_NAMES
    path = _EMBODIED_MAS_DIR / "ObjectYoloDelete.csv"
    names: set[str] = set()
    if path.is_file():
        text = path.read_text(encoding="utf-8-sig")
        for line in text.splitlines():
            n = line.strip()
            if n and not n.startswith("#"):
                names.add(n.casefold())
    _OBJECT_YOLO_DELETE_NAMES = frozenset(names)
    return _OBJECT_YOLO_DELETE_NAMES


_INSTANCE_SUFFIX_RE = re.compile(r"_C_\d+$")


def _lookup_name_in_map(nm: str, name_map: dict[str, str]) -> str | None:
    """Map by exact name; if missing, strip trailing ``_C_<digits>`` and retry (e.g. BP_xxx_C_1 -> BP_xxx)."""
    if nm in name_map:
        return name_map[nm]
    base = _INSTANCE_SUFFIX_RE.sub("", nm)
    if base != nm and base in name_map:
        return name_map[base]
    return None


def _postprocess_embodied_perception(result: dict) -> dict:
    """Rename via ObjectDictionary, then drop entries listed in ObjectDelete."""
    name_map = _load_object_name_map()
    delete_names = _load_object_delete_names()

    raw_actors = result.get("actor_info") or []
    kept_actors: list[Any] = []
    for entry in raw_actors:
        actor = entry.get("actor")
        if isinstance(actor, dict):
            nm = actor.get("name")
            if isinstance(nm, str):
                mapped = _lookup_name_in_map(nm, name_map)
                if mapped is not None:
                    actor["name"] = mapped
            final = actor.get("name")
            if isinstance(final, str) and final in delete_names:
                continue
        kept_actors.append(entry)
    result["actor_info"] = kept_actors

    raw_npcs = result.get("npc_info") or []
    kept_npcs: list[Any] = []
    for entry in raw_npcs:
        if not isinstance(entry, dict):
            kept_npcs.append(entry)
            continue
        oi = entry.get("object_info")
        if isinstance(oi, dict):
            nm = oi.get("name")
            if isinstance(nm, str):
                mapped = _lookup_name_in_map(nm, name_map)
                if mapped is not None:
                    oi["name"] = mapped
            final = oi.get("name")
            if isinstance(final, str) and final in delete_names:
                continue
        kept_npcs.append(entry)
    result["npc_info"] = kept_npcs

    return result


def _destroyed_names_in_perception_space(destroyed_raw: list[Any]) -> frozenset[str]:
    """
    Map engine-reported destroyed object names into the same perception namespace as ``_postprocess_embodied_perception``:
    keep the stripped raw name and add the ObjectDictionary-mapped name when present.
    """
    name_map = _load_object_name_map()
    out: set[str] = set()
    for d in destroyed_raw or []:
        rs = str(d).strip()
        if not rs:
            continue
        out.add(rs)
        mapped = _lookup_name_in_map(rs, name_map)
        if mapped is not None:
            out.add(mapped)
    return frozenset(out)


def _filter_destroyed_from_embodied_perception(result: dict, destroyed_perception_names: frozenset[str]) -> None:
    """Remove entries whose names appear in ``destroyed_perception_names`` from ``actor_info`` / ``npc_info`` (mutates ``result``)."""
    if not destroyed_perception_names:
        return

    def _in_destroyed(nm: object) -> bool:
        if not isinstance(nm, str):
            return False
        s = nm.strip()
        return bool(s) and s in destroyed_perception_names

    raw_actors = result.get("actor_info") or []
    kept_actors: list[Any] = []
    for entry in raw_actors:
        actor = entry.get("actor") if isinstance(entry, dict) else None
        if isinstance(actor, dict) and _in_destroyed(actor.get("name")):
            continue
        kept_actors.append(entry)
    result["actor_info"] = kept_actors

    raw_npcs = result.get("npc_info") or []
    kept_npcs: list[Any] = []
    for entry in raw_npcs:
        if not isinstance(entry, dict):
            kept_npcs.append(entry)
            continue
        oi = entry.get("object_info")
        if isinstance(oi, dict) and _in_destroyed(oi.get("name")):
            continue
        kept_npcs.append(entry)
    result["npc_info"] = kept_npcs


class PerceptionInfo:
    """Fetch embodied perception (actors/NPCs) around an agent."""
    
    def __init__(self, context: WorldContext):
        self._conn = context.conn
        self._last_destroyed_perception_names: frozenset[str] = frozenset()

    @property
    def last_destroyed_perception_names(self) -> frozenset[str]:
        """Name set from the latest ``get_perception`` when ``filter_destroyed=True``; used to filter simplified caches."""
        return self._last_destroyed_perception_names

    def filter_simplified_object_list_cache(self, cache: dict[str, Any] | None) -> None:
        """Drop destroyed objects from a simplified perception cache (``actor_info`` / ``npc_info`` lists with ``name`` keys)."""
        if not cache or not self._last_destroyed_perception_names:
            return
        dset = self._last_destroyed_perception_names
        for key in ("actor_info", "npc_info"):
            lst = cache.get(key)
            if not isinstance(lst, list):
                continue
            cache[key] = [
                it
                for it in lst
                if not (
                    isinstance(it, dict)
                    and isinstance(it.get("name"), str)
                    and it.get("name", "").strip() in dset
                )
            ]
    
    async def get_perception(
        self,
        agent_id: bytes | str | dict,
        timeout: float = 5.0,
        *,
        filter_destroyed: bool = True,
    ) -> dict:
        """Return embodied perception around the given agent."""
        self._last_destroyed_perception_names = frozenset()
        result = await ts.UnaryAPI.get_embodied_perception(
            self._conn,
            agent_id,
            timeout=timeout
        )
        # print(f"[PerceptionInfo] result: {result}")
        _postprocess_embodied_perception(result)
        if filter_destroyed:
            destroyed_raw = await ts.UnaryAPI.get_destroyed_objects(self._conn) or []
            dset = _destroyed_names_in_perception_space(destroyed_raw)
            self._last_destroyed_perception_names = dset
            _filter_destroyed_from_embodied_perception(result, dset)
        return result
    
    async def receive_sos(self, agent_ids: list[bytes | str | dict]):
        """Async generator of SOS events for the given agent ids."""
        async for sos in ts.UnaryAPI.receive_npc_sos(self._conn, agent_ids):
            yield sos
