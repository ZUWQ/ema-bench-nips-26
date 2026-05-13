from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from Human_Agent.Single_agent.human_ui_vlm import (
    HumanDecisionResult,
    WEB_UI_MANAGER,
)


_SERVER_LOCK = threading.RLock()
_SERVER_THREAD: Optional[threading.Thread] = None
_SERVER_STARTED = False
_SERVER_HOST = "0.0.0.0"
_SERVER_PORT = 8080

_ACTION_OPTIONS = [
    "move_to",
    "move_by",
    "extinguish_fire",
    "explore",
    "send_follow",
    "send_stop_follow",
]

# RGB viewer display controls (UI-only; source image pixels are unchanged).
# Width is fixed to keep runtime and mock image display consistent.
RGB_DISPLAY_WIDTH = 900
RGB_DISPLAY_FALLBACK_HEIGHT = 576


def _import_nicegui() -> tuple[Any, Any]:
    from nicegui import app, ui  # type: ignore

    return ui, app


def _extract_observation_object_names(observation_text: str) -> list[str]:
    names: list[str] = []
    in_object_names_block = False

    for raw_line in (observation_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("OBJECT NAMES"):
            in_object_names_block = True
            continue

        if not in_object_names_block:
            continue

        if upper.startswith("DISTRESS SIGNALS"):
            break

        if line.startswith("Count:"):
            continue

        if line.startswith("- "):
            name = line[2:].strip()
            if name and name not in names:
                names.append(name)

    return names


def _render_text_card(ui: Any, title: str, content: str, *, mono: bool = False) -> None:
    with ui.card().classes("w-full"):
        ui.label(title).classes("text-subtitle1")
        style = "white-space: pre-wrap;"
        if mono:
            style += " font-family: 'Consolas', 'Courier New', monospace;"
        ui.label(content or "").classes("w-full").style(style)


def _get_image_size(path: str) -> Optional[tuple[int, int]]:
    try:
        import cv2  # type: ignore

        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        h, w = arr.shape[:2]
        if w <= 0 or h <= 0:
            return None
        return int(w), int(h)
    except Exception:
        return None


def _mount_static_assets(app: Any) -> None:
    if getattr(app.state, "_human_ui_assets_mounted", False):
        return
    app.add_static_files("/human-ui-assets", str(WEB_UI_MANAGER.static_dir))
    app.state._human_ui_assets_mounted = True


def _register_pages(ui: Any) -> None:
    if getattr(ui, "_human_ui_routes_registered", False):
        return

    @ui.page("/")
    def index_page() -> None:
        ui.label("Human-in-the-loop Control Panel").classes("text-h5")
        ui.label("Open /agent/{agent_id} for each participant.")

    @ui.page("/agent/{agent_id}")
    def agent_page(agent_id: str) -> None:
        ui.label(f"Agent Console: {agent_id}").classes("text-h5")
        status_label = ui.label("Waiting for next step...")
        root = ui.column().classes("w-full")

        state: dict[str, Any] = {
            "request_id": None,
        }

        def render_waiting() -> None:
            root.clear()
            with root:
                with ui.card().classes("w-full"):
                    ui.label("Waiting for next step...").classes("text-subtitle1")

        def render_request(snapshot: dict[str, Any]) -> None:
            root.clear()
            request_id = snapshot["request_id"]
            req = snapshot["request"]

            rgb_path = req.perception_image_paths[0] if req.perception_image_paths else None
            rgb_src = WEB_UI_MANAGER.register_image_path(rgb_path) if rgb_path else None
            rgb_size = _get_image_size(rgb_path) if rgb_path else None
            display_w = RGB_DISPLAY_WIDTH
            display_h = (
                int(display_w * (rgb_size[1] / rgb_size[0]))
                if rgb_size and rgb_size[0] > 0
                else RGB_DISPLAY_FALLBACK_HEIGHT
            )

            with root:
                with ui.row().classes("w-full items-start no-wrap").style("gap: 16px;"):
                    with ui.column().classes("min-w-0").style("flex: 1 1 0;"):
                        with ui.card().classes("w-full"):
                            ui.label("RGB Image (Click to Select Pixel)").classes("text-subtitle1")
                            if rgb_src is None:
                                ui.label(f"Missing RGB image: {rgb_path}")
                            else:
                                picked_label = ui.label("Selected pixel: none")

                                pixel_state: dict[str, Optional[int]] = {"x": None, "y": None}

                                def _set_pixel(x: int, y: int) -> None:
                                    if rgb_size:
                                        x = max(0, min(rgb_size[0] - 1, int(x)))
                                        y = max(0, min(rgb_size[1] - 1, int(y)))
                                    else:
                                        x = max(0, int(x))
                                        y = max(0, int(y))
                                    pixel_state["x"] = x
                                    pixel_state["y"] = y
                                    pixel_x.value = str(x)
                                    pixel_y.value = str(y)
                                    picked_label.text = f"Selected pixel: ({x}, {y})"

                                def _on_rgb_mouse(e: Any) -> None:
                                    x = int(round(float(getattr(e, "image_x", 0.0))))
                                    y = int(round(float(getattr(e, "image_y", 0.0))))
                                    _set_pixel(x, y)

                                ui.interactive_image(
                                    rgb_src,
                                    on_mouse=_on_rgb_mouse,
                                    cross="orange",
                                ).style(
                                    f"width: {display_w}px; height: {display_h}px; object-fit: contain;"
                                ).classes("border rounded")
                                ui.label(Path(rgb_path).name).classes("text-caption")

                        with ui.row().classes("w-full items-start no-wrap").style("gap: 10px;"):
                            with ui.column().classes("min-w-0").style("flex: 1 1 0;"):
                                _render_text_card(ui, "Current Step Input", req.agent_position_text or "")

                            with ui.column().classes("min-w-0").style("flex: 1 1 0;"):
                                _render_text_card(ui, "Observation", req.observation_text or "")

                            with ui.column().classes("min-w-0").style("flex: 1 1 0;"):
                                _render_text_card(ui, "Memory", req.memory_text or "", mono=True)
                                if req.other_agent_info:
                                    _render_text_card(ui, "Information From Other Agents", req.other_agent_info)

                    with ui.column().classes("min-w-0").style("flex: 0 0 420px; max-width: 420px;"):
                        with ui.card().classes("w-full"):
                            ui.label("Action Input").classes("text-subtitle1")
                            ui.label("Click one action:").classes("text-caption")
                            action = ui.radio(_ACTION_OPTIONS, value="explore").classes("w-full").style(
                                "display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 10px;"
                            )

                            pixel_container = ui.column().classes("w-full")
                            with pixel_container:
                                pixel_x = ui.input("arg1: pixel_x (required for move_to/extinguish_fire)").classes("w-full")
                                pixel_y = ui.input("arg2: pixel_y (required for move_to/extinguish_fire)").classes("w-full")

                            move_by_container = ui.column().classes("w-full")
                            with move_by_container:
                                move_by_distance = ui.input("Move-by distance (required for move_by)", value="200").classes("w-full")
                                move_by_angle = ui.input("Move-by angle in degrees (required for move_by)", value="0").classes("w-full")

                            high_level_plan = ui.input("High-level plan (optional)").classes("w-full")
                            summary = ui.input("Summary (optional)").classes("w-full")

                            def _sync_action_inputs() -> None:
                                selected_action = str(action.value or "explore").strip()
                                pixel_container.set_visibility(
                                    selected_action in {"move_to", "extinguish_fire"}
                                )
                                move_by_container.set_visibility(selected_action == "move_by")

                            action.on_value_change(lambda _e: _sync_action_inputs())
                            _sync_action_inputs()

                            def on_submit() -> None:
                                selected_action = str(action.value or "explore").strip()

                                move_by_parameter = None
                                if selected_action == "move_by":
                                    try:
                                        move_by_parameter = {
                                            "distance": float(str(move_by_distance.value or "").strip()),
                                            "angle": float(str(move_by_angle.value or "").strip()),
                                        }
                                    except ValueError:
                                        ui.notify("move_by requires numeric distance and angle.", color="negative")
                                        return

                                pixel_parameter = None
                                if selected_action in {"move_to", "extinguish_fire"}:
                                    try:
                                        px = int(round(float(str(pixel_x.value or "").strip())))
                                        py = int(round(float(str(pixel_y.value or "").strip())))
                                        pixel_parameter = {
                                            "pixel_x": px,
                                            "pixel_y": py,
                                        }
                                    except ValueError:
                                        ui.notify(
                                            "move_to/extinguish_fire require numeric pixel_x and pixel_y (click the RGB image).",
                                            color="negative",
                                        )
                                        return

                                action_parameter = (
                                    pixel_parameter
                                    if selected_action in {"move_to", "extinguish_fire"}
                                    else move_by_parameter
                                )
                                result = HumanDecisionResult(
                                    summary=str(summary.value or f"Selected action: {selected_action}"),
                                    high_level_plan=str(high_level_plan.value or ""),
                                    action=selected_action,
                                    action_parameter=action_parameter,
                                )
                                ok = WEB_UI_MANAGER.submit_result(
                                    agent_id,
                                    result,
                                    request_id=request_id,
                                )
                                if ok:
                                    ui.notify("Decision submitted.", color="positive")
                                else:
                                    ui.notify("Request expired. Please wait for the next step.", color="warning")

                            ui.button("Submit", on_click=on_submit).classes("w-full")

        def refresh() -> None:
            snapshot = WEB_UI_MANAGER.get_agent_snapshot(agent_id)
            new_request_id = snapshot["request_id"] if snapshot is not None else None
            if new_request_id == state["request_id"]:
                return

            state["request_id"] = new_request_id
            if snapshot is None:
                status_label.text = "Waiting for next step..."
                render_waiting()
            else:
                req = snapshot["request"]
                status_label.text = f"Pending decision for step {req.step}."
                render_request(snapshot)

        refresh()
        ui.timer(0.4, refresh)

    ui._human_ui_routes_registered = True


def _run_server(host: str, port: int) -> None:
    ui, app = _import_nicegui()
    _mount_static_assets(app)
    _register_pages(ui)
    ui.run(
        host=host,
        port=port,
        title="EmbodiedMAS Human Control",
        reload=False,
        show=False,
    )


def ensure_web_ui_server_started(*, host: str = "0.0.0.0", port: int = 8080) -> None:
    global _SERVER_THREAD
    global _SERVER_STARTED
    global _SERVER_HOST
    global _SERVER_PORT

    with _SERVER_LOCK:
        if _SERVER_STARTED:
            return

        _SERVER_HOST = host
        _SERVER_PORT = port

        thread = threading.Thread(
            target=_run_server,
            args=(host, port),
            name="embodiedmas-nicegui-server",
            daemon=True,
        )
        thread.start()
        _SERVER_THREAD = thread
        _SERVER_STARTED = True


def web_ui_agent_url(agent_id: str, *, host: str = "127.0.0.1", port: int = 8080) -> str:
    return f"http://{host}:{port}/agent/{agent_id}"
