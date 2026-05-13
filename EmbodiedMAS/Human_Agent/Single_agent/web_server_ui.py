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

_ACTION_LABELS = {
    "move_to": "Move to a selected location",
    "move_by": "Move by direction and distance",
    "send_follow": "Send follow command to casualty",
    "send_stop_follow": "Send stop-follow command to casualty",
    "wait": "Refresh observations",
    "extinguish_fire": "Extinguish fire",
}
_ACTION_OPTIONS = list(_ACTION_LABELS.keys())
_ACTION_LABEL_OPTIONS = [_ACTION_LABELS[action_key] for action_key in _ACTION_OPTIONS]
_ACTION_LABEL_TO_KEY = {label: key for key, label in _ACTION_LABELS.items()}

# RGB viewer display controls (UI-only; source image pixels are unchanged).
# Width is fixed to keep runtime and mock image display consistent.
RGB_DISPLAY_WIDTH = 1200
RGB_DISPLAY_FALLBACK_HEIGHT = 1152

_IMAGE_ORIENTATION_OVERLAY_HTML = """
<div style="
    position: absolute;
    top: 10px;
    right: 10px;
    left: auto;
    z-index: 5;
    pointer-events: none;
    user-select: none;
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #bdbdbd;
    border-radius: 8px;
    padding: 8px 10px;
    box-shadow: 0 1px 6px rgba(0, 0, 0, 0.12);
    font-size: 13px;
    font-weight: 600;
    color: #212121;
">
    <div style="
        display: grid;
        grid-template-columns: repeat(3, minmax(36px, 1fr));
        grid-template-rows: repeat(3, auto);
        gap: 2px 4px;
        text-align: center;
        align-items: center;
        justify-items: center;
    ">
        <span></span><span>Fwd</span><span></span>
        <span>Left</span><span></span><span>Right</span>
        <span></span><span>Back</span><span></span>
    </div>
</div>
"""


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
        ui.label("Firefighting drill — human decision UI").classes("text-h5")
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
            memory_text = req.memory_text or ""
            memory_lines = memory_text.splitlines()
            memory_last_line = memory_lines[-1] if memory_lines else ""

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
                            ui.label("RGB view (click to pick pixels)").classes("text-subtitle1")
                            if rgb_src is None:
                                ui.label(f"RGB image missing: {rgb_path}")
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

                                with ui.element("div").style(
                                    f"position: relative; width: {display_w}px; height: {display_h}px;"
                                ):
                                    ui.interactive_image(
                                        rgb_src,
                                        on_mouse=_on_rgb_mouse,
                                        cross="orange",
                                    ).style(
                                        f"width: {display_w}px; height: {display_h}px; object-fit: contain; display: block;"
                                    )
                                    ui.html(_IMAGE_ORIENTATION_OVERLAY_HTML)
                                    ui.html(
                                        """
                                        <div style="
                                            position: absolute;
                                            top: 50%;
                                            left: 50%;
                                            transform: translate(-50%, -50%);
                                            width: 96px;
                                            height: 96px;
                                            pointer-events: none;
                                            user-select: none;
                                        ">
                                            <svg
                                                viewBox="0 0 100 100"
                                                width="96"
                                                height="96"
                                                xmlns="http://www.w3.org/2000/svg"
                                                style="overflow: visible; filter: drop-shadow(0 0 2px #fff) drop-shadow(0 0 5px #fff);"
                                                aria-hidden="true"
                                            >
                                                <defs>
                                                    <mask id="humanNavArrowMask">
                                                        <!-- Upper triangle: white in mask = visible -->
                                                        <polygon points="50,6 94,96 6,96" fill="white" />
                                                        <!-- Lower triangle: black in mask = cut tail for stemless arrow -->
                                                        <polygon points="50,54 78,97 22,97" fill="black" />
                                                    </mask>
                                                </defs>
                                                <polygon
                                                    points="50,6 94,96 6,96"
                                                    fill="rgba(0, 0, 0, 0.88)"
                                                    mask="url(#humanNavArrowMask)"
                                                />
                                            </svg>
                                        </div>
                                        """
                                    )

                        with ui.row().classes("w-full items-start no-wrap").style("gap: 10px;"):
                            with ui.column().classes("min-w-0").style("flex: 1 1 0;"):
                                _render_text_card(ui, "Current state", req.agent_position_text or "")
                                
                                if req.other_agent_info:
                                    _render_text_card(ui, "Other agents", req.other_agent_info)

                            # with ui.column().classes("min-w-0").style("flex: 1 1 0;"):
                            #     _render_text_card(ui, "Observation", req.observation_text or "")

                    with ui.column().classes("min-w-0").style("flex: 0 0 420px; max-width: 420px;"):
                        with ui.card().classes("w-full"):
                            # ui.label("Action Input").classes("text-subtitle1")
                            # ui.label("Click one action:").classes("text-caption")
                            ui.label("Decision input").classes("text-subtitle1")
                            ui.label("Choose one action:").classes("text-caption")
                            # ui.label("Click one action:").classes("text-caption")
                            action = ui.radio(
                                _ACTION_LABEL_OPTIONS,
                                value=_ACTION_LABELS["move_to"],
                            ).classes("w-full").style(
                                "display: grid; grid-template-columns: 1fr; gap: 4px 0;"
                            )

                            pixel_container = ui.column().classes("w-full")
                            with pixel_container:
                                # pixel_x = ui.input("arg1: pixel_x (required for move_to/extinguish_fire)").classes("w-full")
                                # pixel_y = ui.input("arg2: pixel_y (required for move_to/extinguish_fire)").classes("w-full")
                                pixel_x = ui.input("Arg 1: pixel x").classes("w-full")
                                pixel_y = ui.input("Arg 2: pixel y").classes("w-full")

                            move_by_container = ui.column().classes("w-full")
                            with move_by_container:
                                # move_by_angle = ui.input("Move-by angle in degrees (required for move_by)", value="0").classes("w-full")
                                # move_by_distance = ui.input("Move-by distance (required for move_by)", value="200").classes("w-full")
                                move_by_angle = ui.input("Move-by angle (degrees)", value="0").classes("w-full")
                                move_by_distance = ui.input("Move-by distance (cm)", value="200").classes("w-full")

                            def _selected_action_key() -> str:
                                selected_label = str(action.value or _ACTION_LABELS["move_to"]).strip()
                                return _ACTION_LABEL_TO_KEY.get(selected_label, "move_to")

                            def _sync_action_inputs() -> None:
                                selected_action = _selected_action_key()
                                pixel_container.set_visibility(
                                    selected_action in {"move_to", "extinguish_fire"}
                                )
                                move_by_container.set_visibility(selected_action == "move_by")

                            action.on_value_change(lambda _e: _sync_action_inputs())
                            _sync_action_inputs()

                            def on_submit() -> None:
                                selected_action = _selected_action_key()

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
                                    summary=f"Selected action: {selected_action}",
                                    high_level_plan="",
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

                            # ui.button("Submit", on_click=on_submit).classes("w-full")
                            ui.button("Submit", on_click=on_submit).classes("w-full")

                        _render_text_card(ui, "Observation history", memory_last_line, mono=True)

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
                # status_label.text = f"Pending decision for step {req.step}."
                status_label.text = f"Your id: {agent_id}, current step: {req.step}."
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
