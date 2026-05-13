from __future__ import annotations

import asyncio
import json
import os
import multiprocessing as mp
import threading
import tempfile
import time
import uuid
import warnings
from dataclasses import asdict
from urllib.parse import unquote, urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class HumanDecisionRequest:
    """Payload presented to a human participant."""

    step: int
    observation_text: str
    agent_position_text: str
    memory_text: str
    other_agent_info: Optional[str]
    perception_image_paths: list[str]
    observation_space_text: str
    action_space_text: str
    system_prompt_text: Optional[str] = None
    scenario_mode: Optional[str] = None
    scenario_constraints: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class HumanDecisionResult:
    """Human-selected action compatible with VLM action schema."""

    summary: str
    high_level_plan: str
    action: str
    action_parameter: Optional[dict[str, Any]]


class WebUIManager:
    """Thread-safe bridge between simulation asyncio tasks and web request handlers."""

    def __init__(self):
        self._lock = threading.RLock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._path_to_token: dict[str, str] = {}
        self._token_to_path: dict[str, str] = {}
        self._static_dir = Path(tempfile.gettempdir()) / "embodiedmas_webui_assets"
        self._static_dir.mkdir(parents=True, exist_ok=True)

    @property
    def static_dir(self) -> Path:
        return self._static_dir

    def register_request(
        self,
        agent_id: str,
        request: HumanDecisionRequest,
        *,
        loop: Any,
        event: Any,
    ) -> str:
        request_id = uuid.uuid4().hex
        request_dict = asdict(request)
        with self._lock:
            self._pending[agent_id] = {
                "request_id": request_id,
                "request": request,
                "request_dict": request_dict,
                "event": event,
                "loop": loop,
                "result": None,
                "created_at": time.time(),
            }
        return request_id

    def get_agent_snapshot(self, agent_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            item = self._pending.get(agent_id)
            if item is None:
                return None
            return {
                "request_id": item["request_id"],
                "request": item["request"],
                "request_dict": dict(item["request_dict"]),
                "created_at": item["created_at"],
                "has_result": item["result"] is not None,
            }

    def submit_result(
        self,
        agent_id: str,
        result: HumanDecisionResult,
        *,
        request_id: Optional[str] = None,
    ) -> bool:
        with self._lock:
            item = self._pending.get(agent_id)
            if item is None:
                return False
            if request_id is not None and item["request_id"] != request_id:
                return False
            item["result"] = result
            loop = item["loop"]
            event = item["event"]

        # Wake simulation coroutine from web thread safely.
        loop.call_soon_threadsafe(event.set)
        return True

    def pop_result(self, agent_id: str) -> Optional[HumanDecisionResult]:
        with self._lock:
            item = self._pending.pop(agent_id, None)
            if item is None:
                return None
            result = item.get("result")
            return result

    def register_image_path(self, path: str) -> Optional[str]:
        src = Path(path)
        if not src.is_file():
            return None

        src_key = str(src.resolve())
        with self._lock:
            if src_key in self._path_to_token:
                token = self._path_to_token[src_key]
                ext = Path(src_key).suffix or ".img"
                return f"/human-ui-assets/{token}{ext}"

            token = uuid.uuid4().hex
            ext = src.suffix or ".img"
            dst = self._static_dir / f"{token}{ext}"
            try:
                dst.write_bytes(src.read_bytes())
            except Exception:
                return None

            self._path_to_token[src_key] = token
            self._token_to_path[token] = str(dst)
            return f"/human-ui-assets/{token}{ext}"

    def clear_all(self) -> None:
        with self._lock:
            self._pending.clear()


WEB_UI_MANAGER = WebUIManager()


class WebHumanDecisionProvider:
    """Async web provider that bridges simulation loop and NiceGUI frontend."""

    def __init__(
        self,
        agent_id: str,
        *,
        auto_start_server: bool = True,
        host: str = "0.0.0.0",
        port: int = 8080,
        timeout_seconds: Optional[float] = None,
    ):
        self._agent_id = agent_id
        self._auto_start_server = auto_start_server
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds

    async def async_decide(self, request: HumanDecisionRequest) -> HumanDecisionResult:
        if self._auto_start_server:
            try:
                # Prefer the colocated server module so UI changes in this package are always used.
                try:
                    from .web_server_ui import ensure_web_ui_server_started  # type: ignore
                except Exception:
                    try:
                        from Human_Agent.Single_agent.web_server_ui import ensure_web_ui_server_started
                    except Exception:
                        # Legacy fallback for older deployments.
                        from Vision_VLM_Agent.Single_agent.web_server_ui import ensure_web_ui_server_started

                ensure_web_ui_server_started(
                    host=self._host,
                    port=self._port,
                )
            except Exception as e:
                print(f"[WebHumanDecisionProvider] Failed to auto-start web server: {e}")

        agent_id = request.agent_id or self._agent_id
        request.agent_id = agent_id

        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        WEB_UI_MANAGER.register_request(agent_id, request, loop=loop, event=event)

        try:
            if self._timeout_seconds is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=self._timeout_seconds)
        except asyncio.TimeoutError:
            _ = WEB_UI_MANAGER.pop_result(agent_id)
            return HumanDecisionResult(
                summary="Decision timeout; default wait.",
                high_level_plan="Hold current position until next step.",
                action="wait",
                action_parameter=None,
            )

        result = WEB_UI_MANAGER.pop_result(agent_id)
        if result is None:
            return HumanDecisionResult(
                summary="Missing web decision result; default wait.",
                high_level_plan="Hold current position until next step.",
                action="wait",
                action_parameter=None,
            )
        return result

    def decide(self, request: HumanDecisionRequest) -> HumanDecisionResult:
        # Backward-compatible sync API for code paths that still expect decide().
        return asyncio.run(self.async_decide(request))


_ALLOWED_ACTIONS = {
    "move_to",
    "move_by",
    "extinguish_fire",
    "explore",
    "send_follow",
    "send_stop_follow",
    "wait",
}


class ScriptedDecisionProvider:
    """Deterministic provider for tests; no GUI required."""

    def __init__(self, scripted_results: list[HumanDecisionResult]):
        self._results = list(scripted_results)
        self._idx = 0

    def decide(self, _request: HumanDecisionRequest) -> HumanDecisionResult:
        if self._idx >= len(self._results):
            return HumanDecisionResult(
                summary="No scripted action available; waiting.",
                high_level_plan="Hold position.",
                action="wait",
                action_parameter=None,
            )
        item = self._results[self._idx]
        self._idx += 1
        return item


class TkHumanDecisionProvider:
    """Simple blocking Tk UI for collecting a single human decision per step."""

    def __init__(self, title: str = "Human Baseline Controller"):
        warnings.warn(
            "TkHumanDecisionProvider is deprecated; use WebHumanDecisionProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._title = title

    def decide(self, request: HumanDecisionRequest) -> HumanDecisionResult:
        try:
            import tkinter as tk
            from tkinter import messagebox, scrolledtext
        except Exception:
            return _console_decide(request)

        # If no display is available (e.g., headless CI), fallback to console.
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            return _console_decide(request)

        result_holder: dict[str, Any] = {}
        done = threading.Event()

        root = tk.Tk()
        root.title(f"{self._title} - Step {request.step}")
        # Layout tuning knobs for different displays.
        root.geometry("1480x900")
        right_panel_width = 400
        image_max_w = 420
        image_max_h = 260
        context_text_height = 20
        image_box_sizes: dict[str, tuple[int, int]] = {
            "RGB": (460, 310),
            "Seg/Overlay": (460, 310),
            "Depth": (460, 310),
        }

        # Keep references to PhotoImage objects to avoid garbage collection.
        image_refs: list[Any] = []
        ppm_cache: dict[str, Path] = {}

        def _normalize_image_path(raw_path: str) -> Path:
            ptxt = (raw_path or "").strip().strip('"').strip("'")
            if ptxt.lower().startswith("file://"):
                parsed = urlparse(ptxt)
                ptxt = unquote(parsed.path or "")
                # Handle Windows file URI forms like /D:/...
                if os.name == "nt" and len(ptxt) >= 3 and ptxt[0] == "/" and ptxt[2] == ":":
                    ptxt = ptxt[1:]
            return Path(ptxt)

        def _maybe_convert_p3_ppm_to_p6(path: Path) -> Optional[Path]:
            """Convert ASCII P3 PPM to binary P6 PPM for Tk compatibility."""
            key = str(path.resolve())
            if key in ppm_cache:
                return ppm_cache[key]

            try:
                with path.open("rb") as f:
                    raw = f.read()
                if not raw.startswith(b"P3"):
                    return None

                tokens: list[str] = []
                for line in raw.decode("ascii", errors="strict").splitlines():
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue
                    tokens.extend(line.split())

                if len(tokens) < 4 or tokens[0] != "P3":
                    return None

                width = int(tokens[1])
                height = int(tokens[2])
                maxval = int(tokens[3])
                if maxval <= 0 or maxval > 255:
                    return None

                expected = width * height * 3
                vals = [int(v) for v in tokens[4:4 + expected]]
                if len(vals) != expected:
                    return None

                pix = bytes(max(0, min(255, v)) for v in vals)
                header = f"P6\n{width} {height}\n255\n".encode("ascii")

                tmp = Path(tempfile.gettempdir()) / f"tk_preview_{path.stem}_p6.ppm"
                tmp.write_bytes(header + pix)
                ppm_cache[key] = tmp
                return tmp
            except Exception:
                return None

        def _load_preview_image(path: str, max_w: int, max_h: int) -> Optional[Any]:
            p = _normalize_image_path(path)
            if not p.is_file():
                return None

            # Tk/Tcl interprets backslashes as escapes on Windows, so use POSIX form.
            tk_path = p.resolve().as_posix()
            max_w = min(max_w, image_max_w)
            max_h = min(max_h, image_max_h)

            # Prefer PIL for broader format support and decoding consistency.
            try:
                from PIL import Image, ImageTk  # type: ignore

                pil_img = Image.open(p)
                pil_img.load()
                pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(pil_img)
                return img
            except Exception:
                # Fall back to native Tk decoder.
                try:
                    img = tk.PhotoImage(file=tk_path)
                except Exception:
                    # Some Tk builds reject ASCII P3 PPM; convert to P6 and retry.
                    converted = _maybe_convert_p3_ppm_to_p6(p)
                    if converted is None:
                        return None
                    try:
                        img = tk.PhotoImage(file=converted.resolve().as_posix())
                    except Exception:
                        return None

            # Fit image into a small preview panel while preserving coarse aspect.
            w = max(1, img.width())
            h = max(1, img.height())
            scale = max((w + max_w - 1) // max_w, (h + max_h - 1) // max_h, 1)
            if scale > 1 and hasattr(img, "subsample"):
                img = img.subsample(scale, scale)
            return img

        # Horizontal split with fixed-width right panel for comfortable text entry.
        frame_right = tk.Frame(root, width=right_panel_width)
        frame_right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        frame_right.pack_propagate(False)

        # Left pane: step context + image previews
        frame_left = tk.Frame(root)
        frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        obs_label = tk.Label(frame_left, text="Current Step Context", font=("TkDefaultFont", 11, "bold"))
        obs_label.pack(anchor="w")

        obs_text = scrolledtext.ScrolledText(frame_left, wrap=tk.WORD, height=context_text_height, width=92)
        composed = []
        composed.append("=== CURRENT STEP INPUT ===\n")
        composed.append(request.agent_position_text + "\n\n")
        composed.append(request.observation_text + "\n\n")
        if request.other_agent_info:
            composed.append("INFORMATION FROM OTHER AGENTS:\n")
            composed.append(request.other_agent_info + "\n\n")
        composed.append("MEMORY:\n")
        composed.append(request.memory_text + "\n\n")
        if request.perception_image_paths:
            composed.append("IMAGE PATHS (RGB/Depth/Overlay):\n")
            composed.extend([f"- {p}\n" for p in request.perception_image_paths])
        obs_text.insert(tk.END, "".join(composed))
        obs_text.configure(state=tk.DISABLED)
        # Keep this area compact so image previews retain vertical space.
        obs_text.pack(fill=tk.X, expand=False)

        def _open_text_window(window_title: str, body: str, width: int = 980, height: int = 780) -> None:
            win = tk.Toplevel(root)
            win.title(window_title)
            win.geometry(f"{width}x{height}")
            area = scrolledtext.ScrolledText(win, wrap=tk.WORD)
            area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            area.insert(tk.END, body)
            area.configure(state=tk.DISABLED)

        prompt_window_text = (request.system_prompt_text or "System prompt not available.").strip()

        modal_bar = tk.Frame(frame_left)
        modal_bar.pack(anchor="w", fill=tk.X, pady=(6, 2))
        tk.Button(
            modal_bar,
            text="Open Full System Prompt",
            command=lambda: _open_text_window("System Prompt (Full)", prompt_window_text),
        ).pack(side=tk.LEFT, padx=(0, 8))

        # Open full system prompt window immediately so it is always available.
        _open_text_window("System Prompt (Full)", prompt_window_text)

        preview_frame = tk.Frame(frame_left)
        # Let this section absorb remaining left-pane height.
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        tk.Label(preview_frame, text="Perception Image Previews (Larger)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        image_grid = tk.Frame(preview_frame)
        image_grid.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        image_grid.grid_columnconfigure(0, weight=1)
        image_grid.grid_columnconfigure(1, weight=1)
        image_grid.grid_rowconfigure(0, weight=1)
        image_grid.grid_rowconfigure(1, weight=1)

        display_items = [
            ("RGB", request.perception_image_paths[0] if len(request.perception_image_paths) > 0 else None, 0, 0),
            ("Seg/Overlay", request.perception_image_paths[1] if len(request.perception_image_paths) > 1 else None, 0, 1),
            ("Depth", request.perception_image_paths[2] if len(request.perception_image_paths) > 2 else None, 1, 0),
        ]

        for title, path, row_idx, col_idx in display_items:
            box_w, box_h = image_box_sizes.get(title, (460, 310))
            card = tk.LabelFrame(image_grid, text=title, padx=4, pady=4, width=box_w, height=box_h)
            card.grid(row=row_idx, column=col_idx, sticky="nsew", padx=4, pady=4)
            card.grid_propagate(False)

            if path is None:
                tk.Label(card, text="No image path", fg="gray40").pack(anchor="center", pady=20)
                continue

            img = _load_preview_image(path, box_w - 20, box_h - 56)
            if img is None:
                tk.Label(card, text=f"Cannot load image\n{path}", fg="gray40", justify=tk.LEFT, wraplength=420).pack(
                    anchor="w", pady=8
                )
                continue

            image_refs.append(img)
            img_label = tk.Label(card, image=img)
            img_label.pack(anchor="center")
            tk.Label(card, text=Path(path).name, fg="gray35").pack(anchor="w", pady=(4, 0))

        # Empty bottom-right cell keeps Depth directly under RGB in a 2x2 grid.
        spacer = tk.Frame(image_grid)
        spacer.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        mode_frame = tk.LabelFrame(frame_right, text="Scenario Mode", padx=8, pady=8)
        mode_frame.pack(anchor="w", fill=tk.X, pady=(0, 10))
        mode_text = request.scenario_mode or "Unknown"
        constraint_text = request.scenario_constraints or "No additional constraints."
        tk.Label(mode_frame, text=f"Mode: {mode_text}", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        tk.Label(mode_frame, text=f"Constraints: {constraint_text}", wraplength=360, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        tk.Label(frame_right, text="Action", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        action_var = tk.StringVar(value="wait")
        actions = [
            "move_to",
            "move_by",
            "extinguish_fire",
            "explore",
            "send_follow",
            "send_stop_follow",
            "wait",
        ]
        action_menu = tk.OptionMenu(frame_right, action_var, *actions)
        action_menu.config(width=24)
        action_menu.pack(anchor="w", pady=(4, 12))

        tk.Label(
            frame_right,
            text="Mosaic pixel px (required for move_to; optional pair for extinguish_fire)",
        ).pack(anchor="w")
        pixel_x_entry = tk.Entry(frame_right, width=36)
        pixel_x_entry.pack(anchor="w", pady=(2, 4))
        tk.Label(frame_right, text="Mosaic pixel py").pack(anchor="w")
        pixel_y_entry = tk.Entry(frame_right, width=36)
        pixel_y_entry.pack(anchor="w", pady=(2, 10))

        tk.Label(frame_right, text="Move-by distance (required for move_by)").pack(anchor="w")
        distance_entry = tk.Entry(frame_right, width=36)
        distance_entry.insert(0, "200")
        distance_entry.pack(anchor="w", pady=(2, 10))

        tk.Label(frame_right, text="Move-by angle in degrees (required for move_by)").pack(anchor="w")
        angle_entry = tk.Entry(frame_right, width=36)
        angle_entry.insert(0, "0")
        angle_entry.pack(anchor="w", pady=(2, 10))

        tk.Label(frame_right, text="High-level plan").pack(anchor="w")
        plan_text = scrolledtext.ScrolledText(frame_right, wrap=tk.WORD, height=10, width=54)
        plan_text.pack(anchor="w", pady=(2, 10))

        tk.Label(frame_right, text="Summary").pack(anchor="w")
        summary_text = scrolledtext.ScrolledText(frame_right, wrap=tk.WORD, height=10, width=54)
        summary_text.pack(anchor="w", pady=(2, 10))

        def on_submit() -> None:
            action = action_var.get().strip()
            px_text = pixel_x_entry.get().strip()
            py_text = pixel_y_entry.get().strip()
            distance_text = distance_entry.get().strip()
            angle_text = angle_entry.get().strip()
            high_level_plan = plan_text.get("1.0", tk.END).strip()
            summary = summary_text.get("1.0", tk.END).strip()

            if action not in _ALLOWED_ACTIONS:
                messagebox.showerror("Invalid Action", f"Unsupported action: {action}")
                return

            action_parameter: Optional[dict[str, Any]] = None
            if action == "move_to":
                if not px_text or not py_text:
                    messagebox.showerror("Missing pixel", "move_to requires mosaic pixel px and py.")
                    return
                try:
                    action_parameter = {"x": float(px_text), "y": float(py_text)}
                except ValueError:
                    messagebox.showerror("Invalid pixel", "px and py must be numbers.")
                    return
            elif action == "extinguish_fire":
                if px_text or py_text:
                    if not px_text or not py_text:
                        messagebox.showerror(
                            "Missing pixel",
                            "For extinguish_fire pixel input, provide both px and py.",
                        )
                        return
                    try:
                        action_parameter = {"x": float(px_text), "y": float(py_text)}
                    except ValueError:
                        messagebox.showerror("Invalid pixel", "px and py must be numbers.")
                        return
            elif action == "move_by":
                try:
                    distance = float(distance_text)
                    angle = float(angle_text)
                except ValueError:
                    messagebox.showerror("Invalid move_by", "move_by requires numeric distance and angle.")
                    return
                action_parameter = {"distance": distance, "angle": angle}

            result_holder["result"] = HumanDecisionResult(
                summary=summary or f"Selected action: {action}",
                high_level_plan=high_level_plan or "",
                action=action,
                action_parameter=action_parameter,
            )
            done.set()
            root.destroy()

        def on_cancel() -> None:
            result_holder["result"] = HumanDecisionResult(
                summary="User canceled UI input; default wait.",
                high_level_plan="Pause until next step.",
                action="wait",
                action_parameter=None,
            )
            done.set()
            root.destroy()

        tk.Button(frame_right, text="Submit", width=16, command=on_submit).pack(anchor="w", pady=(6, 4))
        tk.Button(frame_right, text="Cancel (wait)", width=16, command=on_cancel).pack(anchor="w")

        root.protocol("WM_DELETE_WINDOW", on_cancel)
        root.mainloop()
        done.wait()

        return result_holder["result"]


def _tk_decide_worker(
    request: HumanDecisionRequest,
    title: str,
    result_queue: Any,
) -> None:
    """Run one Tk decision dialog inside a dedicated process."""
    try:
        result = TkHumanDecisionProvider(title=title).decide(request)
        result_queue.put(
            {
                "summary": result.summary,
                "high_level_plan": result.high_level_plan,
                "action": result.action,
                "action_parameter": result.action_parameter,
            }
        )
    except Exception as e:
        result_queue.put(
            {
                "error": str(e),
            }
        )


class ParallelTkHumanDecisionProvider:
    """Concurrent decision provider: each request uses an isolated Tk process/window.

    This allows multiple agent decision windows to remain open simultaneously.
    """

    def __init__(self, title: str = "Human Baseline Controller"):
        warnings.warn(
            "ParallelTkHumanDecisionProvider is deprecated; use WebHumanDecisionProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._title = title

    def decide(self, request: HumanDecisionRequest) -> HumanDecisionResult:
        # If no display is available (e.g., headless CI), fallback to console.
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            return _console_decide(request)

        result_queue = mp.Queue()
        proc = mp.Process(
            target=_tk_decide_worker,
            args=(request, f"{self._title}", result_queue),
            daemon=False,
        )
        proc.start()

        payload: Optional[dict[str, Any]] = None
        try:
            payload = result_queue.get()
        finally:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

        if not payload:
            return HumanDecisionResult(
                summary="No UI result received; default wait.",
                high_level_plan="Pause until next step.",
                action="wait",
                action_parameter=None,
            )

        if payload.get("error"):
            print(f"[ParallelTkHumanDecisionProvider] Child UI error: {payload['error']}")
            return HumanDecisionResult(
                summary="UI error; default wait.",
                high_level_plan="Pause until next step.",
                action="wait",
                action_parameter=None,
            )

        return HumanDecisionResult(
            summary=str(payload.get("summary") or "Selected action: wait"),
            high_level_plan=str(payload.get("high_level_plan") or ""),
            action=str(payload.get("action") or "wait"),
            action_parameter=payload.get("action_parameter"),
        )


def _console_decide(request: HumanDecisionRequest) -> HumanDecisionResult:
    print("\n" + "=" * 80)
    print(f"[Human UI Console] Step {request.step}")
    print("=" * 80)
    print("OBSERVATION SPACE/ACTION SPACE are omitted in console step view because they are embedded in system prompt.")
    print("\nSCENARIO MODE:\n" + (request.scenario_mode or "Unknown"))
    print("SCENARIO CONSTRAINTS:\n" + (request.scenario_constraints or "No additional constraints."))
    if request.system_prompt_text:
        print("\nSYSTEM PROMPT (VLM2):\n" + request.system_prompt_text)
    print("\nNEW OBSERVATION MESSAGE (CURRENT STEP):\n" + request.agent_position_text)
    print(request.observation_text)
    if request.other_agent_info:
        print("\nINFORMATION FROM OTHER AGENTS:\n" + request.other_agent_info)
    print("\nMEMORY:\n" + request.memory_text)
    if request.perception_image_paths:
        print("\nIMAGE PATHS:")
        for p in request.perception_image_paths:
            print(f"- {p}")

    action = input(
        "Choose action [move_to/move_by/extinguish_fire/explore/send_follow/send_stop_follow/wait]: "
    ).strip()
    if action not in _ALLOWED_ACTIONS:
        print(f"Invalid action '{action}', fallback to wait.")
        action = "wait"

    move_by_distance: Optional[float] = None
    move_by_angle: Optional[float] = None
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None
    if action == "move_to":
        try:
            pixel_x = float(input("Mosaic pixel px: ").strip())
            pixel_y = float(input("Mosaic pixel py: ").strip())
        except ValueError:
            print("Invalid pixel, fallback to wait.")
            action = "wait"
    elif action == "extinguish_fire":
        raw = input("Mosaic pixel px,py (optional, blank for forward extinguish): ").strip()
        if raw:
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
            if len(parts) >= 2:
                try:
                    pixel_x = float(parts[0])
                    pixel_y = float(parts[1])
                except ValueError:
                    print("Invalid pixel, fallback to wait.")
                    action = "wait"
            else:
                print("Need two numbers, fallback to wait.")
                action = "wait"
    elif action == "move_by":
        try:
            move_by_distance = float(input("Distance: ").strip())
            move_by_angle = float(input("Angle (degrees): ").strip())
        except ValueError:
            print("Invalid distance/angle, fallback to wait.")
            action = "wait"

    high_level_plan = input("High-level plan (optional): ").strip()
    summary = input("Summary (optional): ").strip()

    action_parameter = None
    if action == "move_to" and pixel_x is not None and pixel_y is not None:
        action_parameter = {"x": pixel_x, "y": pixel_y}
    elif action == "extinguish_fire" and pixel_x is not None and pixel_y is not None:
        action_parameter = {"x": pixel_x, "y": pixel_y}
    elif action == "move_by" and move_by_distance is not None and move_by_angle is not None:
        action_parameter = {
            "distance": move_by_distance,
            "angle": move_by_angle,
        }

    return HumanDecisionResult(
        summary=summary or f"Selected action: {action}",
        high_level_plan=high_level_plan,
        action=action,
        action_parameter=action_parameter,
    )


def load_prompt_text(prompt_file: str) -> str:
    p = Path(prompt_file)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return f"Prompt file not found: {prompt_file}"


def serialize_human_result(result: HumanDecisionResult) -> str:
    return json.dumps(
        {
            "summary": result.summary,
            "high_level_plan": result.high_level_plan,
            "action": result.action,
            "action_parameter": result.action_parameter,
        },
        ensure_ascii=False,
    )
