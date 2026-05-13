from __future__ import annotations

import asyncio
import importlib.util
import math
from typing import Optional, List, Callable, Awaitable, Dict, Any
from pathlib import Path
import numpy as np
import cv2

import tongsim as ts
from tongsim.core.world_context import WorldContext
from tongsim.type.rl_demo import RLDemoOrientationMode

# Ensure EmbodiedMAS imports work when this file is run as a script
if __package__ is None or __package__ == "":
    import os
    import sys
    # Repo root (PythonClient): resolves EmbodiedMAS.* and tongsim
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

# Do not use `from EmbodiedMAS...` here: it runs EmbodiedMAS/__init__.py, which may exec
# base_agent_vlm → import action_vlm and trigger a partially-initialized import cycle.
_embodied_root = Path(__file__).resolve().parent.parent

_obs_spec = importlib.util.spec_from_file_location(
    "embodiedmas_observation_vlm", str(_embodied_root / "observation.py")
)
if not _obs_spec or not _obs_spec.loader:
    raise ImportError(f"cannot load observation from {_embodied_root / 'observation.py'}")
_obs_module = importlib.util.module_from_spec(_obs_spec)
_obs_spec.loader.exec_module(_obs_module)
PerceptionInfo = _obs_module.PerceptionInfo
ObservationCamera = _obs_module.ObservationCamera

_l4h_spec = importlib.util.spec_from_file_location(
    "embodiedmas_location4human_for_action_h",
    str(_embodied_root / "perception_yolo" / "location4human.py"),
)
if not _l4h_spec or not _l4h_spec.loader:
    raise ImportError(
        f"cannot load location4human from {_embodied_root / 'perception_yolo' / 'location4human.py'}"
    )
_l4h_module = importlib.util.module_from_spec(_l4h_spec)
_l4h_spec.loader.exec_module(_l4h_module)
explore_poses_to_pose_by_direction = _l4h_module.explore_poses_to_pose_by_direction
pixel_to_world_from_paths = _l4h_module.pixel_to_world_from_paths
_read_rgb_hw_l4h = _l4h_module._read_rgb_hw
_explore_pose_to_cam_arrays_l4h = _l4h_module._explore_pose_to_cam_arrays
_intrinsic_matrix_from_wh_fov_l4h = _l4h_module._intrinsic_matrix_from_wh_fov
depth_path_inferred_from_rgb_path_l4h = _l4h_module.depth_path_inferred_from_rgb_path

# Match base_agent_h.AGENT_CAMERA_PARAMS horizontal FOV (back-projection)
EXPLORE_CAMERA_HFOV_DEG = 90.0

# Human explore: fixed per-tile RGB size (no rescaling; mismatch raises ValueError)
EXPLORE_TILE_WIDTH = 640
EXPLORE_TILE_HEIGHT = 480


def _assert_explore_tile_bgr_shape(label: str, image: np.ndarray) -> None:
    """Each BGR tile must be ``EXPLORE_TILE_HEIGHT × EXPLORE_TILE_WIDTH``."""
    if image.ndim < 2:
        raise ValueError(f"explore RGB '{label}': invalid array shape {image.shape!r}")
    h, w = int(image.shape[0]), int(image.shape[1])
    if (h, w) != (EXPLORE_TILE_HEIGHT, EXPLORE_TILE_WIDTH):
        raise ValueError(
            f"explore RGB '{label}' must be {EXPLORE_TILE_WIDTH}×{EXPLORE_TILE_HEIGHT}, got {w}×{h}"
        )


def _assert_explore_tile_rgb_path_hw(label: str, rgb_path: str) -> tuple[int, int]:
    """Read RGB dimensions from disk and validate; returns ``(height, width)``."""
    h, w = _read_rgb_hw_l4h(str(rgb_path))
    if (h, w) != (EXPLORE_TILE_HEIGHT, EXPLORE_TILE_WIDTH):
        raise ValueError(
            f"explore RGB '{label}' must be {EXPLORE_TILE_WIDTH}×{EXPLORE_TILE_HEIGHT}, got {w}×{h} ({rgb_path})"
        )
    return h, w


def _human_explore_mosaic_size_hw(h0: int, w0: int) -> tuple[int, int]:
    """
    Human ``explore`` mosaic size: width ``3*w0`` (default 1920), height ``2*h0 + h0//2`` (1200 when h0=480).
    Layout: three columns — left column centers ``left``; middle column top ``front``, bottom ``back``,
    with ``h0//2`` white gap between; right column centers ``right``; remaining pixels are white.
    """
    return int(2 * h0 + (h0 // 2)), int(3 * w0)


def _human_mosaic_pixel_to_direction_local(
    px: float, py: float, h0: int, w0: int
) -> tuple[str, float, float]:
    """Map mosaic pixel → (direction, local_px, local_py); blanks map to ``_blank_`` (no depth / back-proj)."""
    mosaic_h, mosaic_w = _human_explore_mosaic_size_hw(h0, w0)
    px_f, py_f = float(px), float(py)
    if px_f < 0 or py_f < 0 or px_f >= float(mosaic_w) or py_f >= float(mosaic_h):
        return "_blank_", 0.0, 0.0
    wf = float(w0)
    pad_top = float((mosaic_h - h0) // 2)
    back_y0 = float(mosaic_h - h0)
    if px_f < wf:
        if pad_top <= py_f < pad_top + float(h0):
            return "left", px_f, py_f - pad_top
        return "_blank_", px_f, py_f
    if px_f < 2.0 * wf:
        if py_f < float(h0):
            return "front", px_f - wf, py_f
        if py_f >= back_y0:
            return "back", px_f - wf, py_f - back_y0
        return "_blank_", px_f, py_f
    if pad_top <= py_f < pad_top + float(h0):
        return "right", px_f - 2.0 * wf, py_f - pad_top
    return "_blank_", px_f, py_f


def _pixel_to_world_human_explore_mosaic(
    pixel_xy_mosaic: tuple[float, float],
    tiles: Dict[str, Dict[str, Optional[str]]],
    pose_by_direction: Dict[str, Dict[str, Any]],
    *,
    horizontal_fov_degrees: float = 90.0,
    depth_scale: float = 1.0,
    neighborhood: int = 1,
) -> Optional[np.ndarray]:
    """(px, py) on Human three-column explore mosaic → world XYZ; same back-proj as ``location4human`` per tile."""
    front = tiles.get("front") or {}
    rgb0 = front.get("rgb")
    if not rgb0 or not Path(str(rgb0)).is_file():
        return None
    h0, w0 = _assert_explore_tile_rgb_path_hw("front", str(rgb0))
    px_m, py_m = float(pixel_xy_mosaic[0]), float(pixel_xy_mosaic[1])
    direction, lx, ly = _human_mosaic_pixel_to_direction_local(px_m, py_m, h0, w0)
    tile = tiles.get(direction)
    if not tile:
        return None
    rgb_path = tile.get("rgb")
    if not rgb_path or not Path(str(rgb_path)).is_file():
        return None
    explicit_depth = tile.get("depth")
    if explicit_depth and str(explicit_depth).strip() and Path(str(explicit_depth)).is_file():
        depth_kw: Optional[str] = str(explicit_depth)
    else:
        depth_kw = None
        if not Path(depth_path_inferred_from_rgb_path_l4h(str(rgb_path))).is_file():
            return None
    pose = pose_by_direction.get(direction)
    if pose is None:
        return None
    cam_world_pos, cam_world_rot = _explore_pose_to_cam_arrays_l4h(pose)
    h_i, w_i = _assert_explore_tile_rgb_path_hw(direction, str(rgb_path))
    intrinsic = _intrinsic_matrix_from_wh_fov_l4h(w_i, h_i, horizontal_fov_degrees)
    return pixel_to_world_from_paths(
        str(rgb_path),
        intrinsic,
        cam_world_pos,
        cam_world_rot,
        (lx, ly),
        depth_path=depth_kw,
        depth_scale=depth_scale,
        neighborhood=neighborhood,
    )


def explore_result_empty() -> Dict[str, Any]:
    """Same keys as a successful explore; used on failure or incomplete quadrants."""
    return {"mosaic_rgb_path": None, "rgb_paths": [], "camera_poses": []}


# Camera offset (must match base_agent_h.DEFAULT_CAMERA_OFFSET / spawn-time optical center)
DEFAULT_CAMERA_OFFSET = ts.Vector3(100, 0, 110)

# World navigation height (matches base_agent_vlm.Z_HEIGHT; perception z is not world height)
NAV_TARGET_Z = 1000.0

# Fixed planar moves for move_front/back/left/right (UU), same semantics as move_by
MOVE_CARDINAL_DISTANCE_UU = 200.0

# Horizontal nozzle offset ahead of actor root (UU; matches action_ol / engine ref)
EXTINGUISHER_FORWARD_LENGTH_UU = 57.0
# Vertical reference for pitch: nozzle height offset from actor root (matches action_ol notes)
EXTINGUISHER_PITCH_REF_Z_OFFSET_UU = 26.0

# Helpers: yaw from quaternion
def _get_yaw_from_quaternion(q: ts.Quaternion) -> float:
    """Yaw in radians from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw in radians from quaternion (x, y, z, w)."""
    return math.atan2(
        2 * (w * z + x * y),
        1 - 2 * (y * y + z * z),
    )


def _compute_relative_xy(
    agent_xy: tuple[float, float],
    quat_xyzw: tuple[float, float, float, float],
    target_xy: tuple[float, float],
) -> dict:
    """Relative XY from agent to target plus bearing offset in degrees. quat_xyzw = (x, y, z, w)."""
    x1, y1 = agent_xy
    x2, y2 = target_xy
    dx = x2 - x1
    dy = y2 - y1
    yaw = _quaternion_to_yaw(*quat_xyzw)
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    x_rel = dx * cos_yaw - dy * sin_yaw
    y_rel = dx * sin_yaw + dy * cos_yaw
    angle_rad = math.atan2(y_rel, x_rel)
    angle_deg = math.degrees(angle_rad)
    return {
        "relative_position": (x_rel, y_rel),
        "angle_deg": angle_deg,
    }


def _extinguisher_ab_from_target(tf: ts.Transform, target: ts.Vector3) -> tuple[float, float]:
    """
    Compute ``set_extinguisher_rotation(a, b)`` in degrees from agent pose and target world position.

    Geometry: 1) nozzle N0 = root + forward * L; 2) vector from N0 to target; 3) CCW yaw ``phi`` in the
    vehicle plane from forward to target; engine ``a`` is clockwise-positive: ``a = -deg(phi)`` (invert sign
    if left/right is swapped on hardware); 4) rotate horizontal offset (N0-A) by ``phi`` (i.e. ``-rad(a)``)
    to N1; 5) pitch ``b`` from ``atan2(dz, r_xy)`` on (T - N1).
    """
    quat_xyzw = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    yaw_ref = _quaternion_to_yaw(*quat_xyzw)
    ax = float(tf.location.x)
    ay = float(tf.location.y)
    az = float(tf.location.z)

    L = EXTINGUISHER_FORWARD_LENGTH_UU
    z_off = EXTINGUISHER_PITCH_REF_Z_OFFSET_UU

    tx, ty, tz = float(target.x), float(target.y), float(target.z)
    target_xy = (tx, ty)
    nozzle_xy = (ax, ay)
    rel = _compute_relative_xy(nozzle_xy, quat_xyzw, target_xy)
    x_rel, y_rel = rel["relative_position"]
    phi_rad = math.atan2(y_rel, x_rel)
    # Clockwise-positive horizontal angle for engine; geometry uses phi_rad
    a = math.degrees(phi_rad)

    n0x = ax + L * math.cos(yaw_ref)
    n0y = ay + L * math.sin(yaw_ref)
    n0z = az + z_off

    d0x = n0x - ax
    d0y = n0y - ay
    cos_g = math.cos(phi_rad)
    sin_g = math.sin(phi_rad)
    d1x = d0x * cos_g - d0y * sin_g
    d1y = d0x * sin_g + d0y * cos_g
    n1x = ax + d1x
    n1y = ay + d1y
    n1z = n0z

    wx = tx - n1x
    wy = ty - n1y
    wz = tz - n1z
    r_xy = math.hypot(wx, wy)
    if r_xy > 1e-8:
        pitch_rad = math.atan2(wz, r_xy)
    else:
        pitch_rad = 0.0
    b = math.degrees(pitch_rad)
    b = max(-90.0, min(90.0, b))
    return (a, b)


def world_position_from_yaw_distance_2d(
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    distance: float,
) -> Optional[tuple[float, float, float]]:
    """
    Polar offset in world XY, same convention as ``EmbodiedMAS/Try_Agent/try_sos_agent.calculate_sos_location``:
    x' = x + d*cos(ψ), y' = y + d*sin(ψ), z' = z with ψ = radians(yaw_deg).

    Treat ``yaw_deg`` as an absolute world heading on the XY plane; validate against engine SOS fields if needed.
    Returns None when ``distance <= 0``.
    """
    if distance <= 0:
        return None
    psi = math.radians(float(yaw_deg))
    d = float(distance)
    return (
        float(x) + d * math.cos(psi),
        float(y) + d * math.sin(psi),
        float(z),
    )


def _camera_pose_dict_from_transform(
    agent_transform: ts.Transform,
    camera_offset: ts.Vector3,
) -> Dict[str, Any]:
    """Optical center = agent location + offset; orientation = agent quaternion + yaw (deg), same as spawn."""
    off = camera_offset
    loc = agent_transform.location
    rot = agent_transform.rotation
    cam_x = float(loc.x) + float(off.x)
    cam_y = float(loc.y) + float(off.y)
    cam_z = float(loc.z) + float(off.z)
    yaw_rad = _get_yaw_from_quaternion(rot)
    yaw_deg = float(math.degrees(yaw_rad))
    return {
        "camera_position": {"x": cam_x, "y": cam_y, "z": cam_z},
        "yaw_deg": yaw_deg,
        "agent_quaternion": {
            "x": float(rot.x),
            "y": float(rot.y),
            "z": float(rot.z),
            "w": float(rot.w),
        },
    }


def _camera_pose_dict_for_explore_view(
    agent_transform: ts.Transform,
    camera_offset: ts.Vector3,
    yaw_deg: float,
    direction: str,
) -> Dict[str, Any]:
    """Camera-only explore: body stays at ``agent_transform``; ``yaw_deg`` is current camera yaw."""
    off = camera_offset
    loc = agent_transform.location
    rot = agent_transform.rotation
    cam_x = float(loc.x) + float(off.x)
    cam_y = float(loc.y) + float(off.y)
    cam_z = float(loc.z) + float(off.z)
    return {
        "camera_position": {"x": cam_x, "y": cam_y, "z": cam_z},
        "yaw_deg": float(yaw_deg),
        "agent_quaternion": {
            "x": float(rot.x),
            "y": float(rot.y),
            "z": float(rot.z),
            "w": float(rot.w),
        },
        "direction": direction,
    }


class ActionAPI:
    """
    Simulator-facing helpers split into:
    - Actions: move_to, move_by, extinguish, follow, explore
    - Perception lists via ``get_perception_object_list`` (explicit calls; ``explore`` does not fetch objects)
    """

    def __init__(
        self,
        context: WorldContext,
        observation_camera: Optional[ObservationCamera] = None,
        experiment_result: Optional[Any] = None,
    ):
        self._context = context
        self._conn = context.conn
        #: Optional ``Metric_Tool.evaluation.ExperimentResult`` for ``send_stop_follow`` npc counts
        self.experiment_result = experiment_result
        self._perception = PerceptionInfo(context)
        self._observation_camera = observation_camera
        self.perception_cache: dict = {}  # legacy perception dict cache
        # Simulator-aligned cache for get_perception_object_list / perception_evaluation (do not rely on for VL/VLM policy)
        self._perception_object_list: Optional[dict[str, Any]] = {
            "actor_info": [{
                "name": "Safe_Zone",
                "location": {"x": 200.0, "y": 0.0, "z": 1010.0},
                "bounding_box": {
                    "min": {"x": 150.0, "y": 50.0, "z": -50.0},
                    "max": {"x": 250.0, "y": 150.0, "z": 50.0}
                },
                "burning_state": False
            }],
            "npc_info": []
        }
        # Simplified list view: name + location only
        self._visual_perception_object_list: Optional[List[Dict[str, Any]]] = [{
            "name": "Safe_Zone",
            "location": {"x": 200.0, "y": 0.0, "z": 1010.0}
        }]
        # Latest explore RGB mosaic path (for human_agent_vlm)
        self.last_locator_visualization_paths: Optional[Dict[str, str]] = None
        # explore: per-direction paths front/left/back/right → rgb (required), depth (engine path, optional)
        self.last_explore_tile_paths: Dict[str, Dict[str, Optional[str]]] = {}
        # explore: camera poses matching order front→left→back→right (JSON-serializable dicts)
        self.last_explore_camera_poses: List[Dict[str, Any]] = []

    def _world_vector3_from_last_explore_mosaic(
        self, px: float, py: float
    ) -> Optional[ts.Vector3]:
        """Map last successful explore mosaic pixel → world XYZ via depth back-projection."""
        required = ("front", "left", "back", "right")
        tiles = self.last_explore_tile_paths
        if not tiles or not all(k in tiles for k in required):
            print(
                "[ActionAPI] pixel→world: need last_explore_tile_paths with "
                "front/left/back/right (run explore first)"
            )
            return None
        poses = self.last_explore_camera_poses
        if len(poses) < 4:
            print("[ActionAPI] pixel→world: incomplete last_explore_camera_poses")
            return None
        pose_by_direction = explore_poses_to_pose_by_direction(poses)
        arr = _pixel_to_world_human_explore_mosaic(
            (px, py),
            tiles,
            pose_by_direction,
            horizontal_fov_degrees=EXPLORE_CAMERA_HFOV_DEG,
        )
        if arr is None:
            print("[ActionAPI] pixel→world: human explore mosaic back-projection returned None")
            return None
        return ts.Vector3(float(arr[0]), float(arr[1]), float(arr[2]))

    # -------------------------------------------------------------------------
    # Simulator actions (movement, interaction, explore)
    # -------------------------------------------------------------------------

    # --- world_location_from_sos is retired (kept commented for reference; do not call) ---
    # async def world_location_from_sos(
    #     self,
    #     listener_actor_id: bytes | str | dict,
    #     sos_info: Dict[str, Any],
    # ) -> Optional[ts.Vector3]:
    #     orientations = sos_info.get("orientations") or []
    #     distances = sos_info.get("distances") or []
    #     if not orientations or not distances:
    #         print("[ActionAPI] world_location_from_sos: missing orientations or distances")
    #         return None
    #     o0 = orientations[0]
    #     if not isinstance(o0, dict):
    #         print("[ActionAPI] world_location_from_sos: orientations[0] is not a dict")
    #         return None
    #     try:
    #         yaw_deg = float(o0.get("yaw", 0.0))
    #         d = float(distances[0])
    #     except (TypeError, ValueError, IndexError):
    #         print("[ActionAPI] world_location_from_sos: invalid yaw or distance")
    #         return None
    #     if d <= 0:
    #         print(f"[ActionAPI] world_location_from_sos: distance must be > 0, got {d}")
    #         return None
    #     actor_key: bytes | str | dict = listener_actor_id
    #     if isinstance(listener_actor_id, dict):
    #         extracted = listener_actor_id.get("id")
    #         if extracted is None:
    #             extracted = listener_actor_id.get("guid")
    #         if extracted is not None:
    #             actor_key = extracted
    #     tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
    #     if not tf:
    #         print(f"[ActionAPI] world_location_from_sos: failed to get transform for {actor_key!r}")
    #         return None
    #     pt = world_position_from_yaw_distance_2d(
    #         tf.location.x, tf.location.y, tf.location.z, yaw_deg, d,
    #     )
    #     if pt is None:
    #         return None
    #     return ts.Vector3(pt[0], pt[1], pt[2])

    async def move_to(
        self,
        actor_id: bytes | str | dict,
        pixel_xy: tuple[float, float],
        *,
        timeout: float = 60.0,
        tolerance_uu: float = 50.0,
        orientation_mode: RLDemoOrientationMode = RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
        allow_partial: bool = True,
        speed_uu_per_sec: float = 100.0,
    ) -> Optional[dict]:
        """
        Navigate on the navmesh to a target.

        ``pixel_xy`` are mosaic coordinates on the latest ``explore`` RGB; Human mosaic mapping plus
        ``location4human.pixel_to_world_from_paths`` yield world XY; Z uses ``NAV_TARGET_Z``.

        Returns:
            ``navigate_to_location`` response dict, or None if back-projection fails.
        """
        actor_ref = actor_id
        if isinstance(actor_id, dict):
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            if extracted_id is not None:
                actor_id = extracted_id

        world = self._world_vector3_from_last_explore_mosaic(float(pixel_xy[0]), float(pixel_xy[1]))
        if world is None:
            print("[ActionAPI] move_to: failed to resolve pixel to world")
            return None

        target_location = ts.Vector3(world.x, world.y, NAV_TARGET_Z)

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_id, 0.0, 0.0, timeout=5.0
        )

        resp = await ts.UnaryAPI.navigate_to_location(
            self._conn,
            actor_id=actor_id,
            target_location=target_location,
            accept_radius=tolerance_uu,
            allow_partial=allow_partial,
            speed_uu_per_sec=speed_uu_per_sec,
            timeout=timeout,
        )

        print(f"[ActionAPI] move_to target_location: {target_location} (from pixel {pixel_xy})")

        await self.explore(actor_ref)

        return resp

    async def move_by(
        self,
        actor_id: bytes | str | dict,
        distance: float,
        angle: float,
        *,
        timeout: float = 60.0,
        tolerance_uu: float = 50.0,
        orientation_mode: RLDemoOrientationMode = RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
        allow_partial: bool = True,
        speed_uu_per_sec: float = 100.0,
    ) -> Optional[dict]:
        """
        Planar move in world XY relative to current heading: ``distance`` is engine units (UU) without extra scaling;
        ``angle`` is degrees (positive = left, negative = right).
        Move heading = agent yaw + angle (radians), then same polar offset as SOS helpers; target Z is ``NAV_TARGET_Z``.
        ``orientation_mode`` is kept for API parity with ``move_to`` and is ignored here.

        Returns:
            ``navigate_to_location`` response dict (aligned with action_ol), or None on failure.
        """
        _ = orientation_mode
        actor_ref = actor_id
        if isinstance(actor_id, dict):
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            if extracted_id is not None:
                actor_id = extracted_id

        d_eff = float(distance)
        tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_id)
        if not tf:
            print(f"[ActionAPI] move_by: failed to get transform for {actor_id!r}")
            return None

        phi = _get_yaw_from_quaternion(tf.rotation)
        theta = math.radians(float(angle))
        move_yaw = phi + theta
        tx = tf.location.x + d_eff * math.cos(move_yaw)
        ty = tf.location.y + d_eff * math.sin(move_yaw)
        target_location = ts.Vector3(tx, ty, NAV_TARGET_Z)

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_id, 0.0, 0.0, timeout=5.0
        )

        resp = await ts.UnaryAPI.navigate_to_location(
            self._conn,
            actor_id=actor_id,
            target_location=target_location,
            accept_radius=tolerance_uu,
            allow_partial=allow_partial,
            speed_uu_per_sec=speed_uu_per_sec,
            timeout=timeout,
        )

        print(
            f"[ActionAPI] move_by target_location: {target_location} "
            f"(distance={distance!r}, angle={angle!r}, d_eff={d_eff})"
        )

        await self.explore(actor_ref)

        return resp

    async def extinguish_fire(
        self,
        actor_id: bytes | str | dict,
        pixel_xy: Optional[tuple[float, float]] = None,
        *,
        timeout: float = 5.0,
    ) -> str:
        """
        Aim the extinguisher at ``pixel_xy`` (latest explore mosaic) and fire.

        Horizontal ``a`` / pitch ``b`` come from ``_extinguisher_ab_from_target`` (nozzle forward 57 UU,
        horizontal aim, rotated nozzle, triangle pitch — see that docstring).
        When ``pixel_xy`` is None the second ``set_extinguisher_rotation`` uses ``(0, 0)``.
        """
        actor_ref = actor_id
        if isinstance(actor_id, dict):
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            actor_key = extracted_id if extracted_id is not None else actor_id
        else:
            actor_key = actor_id

        target_location: Optional[ts.Vector3] = None
        if pixel_xy is not None:
            target_location = self._world_vector3_from_last_explore_mosaic(
                float(pixel_xy[0]), float(pixel_xy[1])
            )
            if target_location is None:
                return "Fail_NoDepthOrExplore"

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_key, 0.0, 0.0, timeout=timeout
        )

        tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if not tf:
            print("[ActionAPI] Error: failed to read current transform")
            return "Fail_CanotExtinguish"

        if target_location is None:
            a, b = 0.0, 0.0
        else:
            a, b = _extinguisher_ab_from_target(tf, target_location)

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn,
            actor_key,
            a,
            b,
            timeout=timeout,
        )

        result = await ts.UnaryAPI.extinguish_fire(self._conn, actor_key, timeout=timeout)
        await asyncio.sleep(10)

        await self.explore(actor_ref)

        if isinstance(result, bool):
            return "Success" if result else "Fail_CanotExtinguish"
        return result

    async def sendfollow(
        self,
        actor_id: bytes | str | dict,
    ) -> list[str]:
        """
        Send follow for the given agent.

        Args:
            actor_id: bytes, str, or dict containing ``id`` / ``guid``

        Returns:
            list[str] engine result list
        """
        actor_ref = actor_id
        # Extract id/guid when actor_id is a dict
        if isinstance(actor_id, dict):
            # Prefer "id", else "guid"
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            if extracted_id is not None:
                actor_id = extracted_id

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_id, 0.0, 0.0, timeout=5.0
        )
        
        result = await ts.UnaryAPI.sendfollow(self._conn, actor_id)
        print(f"[ActionAPI] sendfollow result: {result}")

        await self.explore(actor_ref)

        return result

    async def sendstopfollow(
        self,
        actor_id: bytes | str | dict,
    ) -> list[str]:
        """
        Send stop-follow for the given agent.

        Args:
            actor_id: bytes, str, or dict containing ``id`` / ``guid``

        Returns:
            list[str] engine result list
        """
        actor_ref = actor_id
        # Extract id/guid when actor_id is a dict
        if isinstance(actor_id, dict):
            # Prefer "id", else "guid"
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            if extracted_id is not None:
                actor_id = extracted_id

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_id, 0.0, 0.0, timeout=5.0
        )

        result = await ts.UnaryAPI.sendstopfollow(self._conn, actor_id)
        er = getattr(self, "experiment_result", None)
        if er is not None:
            await er.record_rescue_after_stop_follow(actor_id)
        await self.explore(actor_ref)

        return result

    async def wait(
        self,
        actor_id: bytes | str | dict,
        *,
        image_prefix: Optional[str] = "explore",
        image_format: str = "png",
        post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """Run ``explore`` once to refresh the mosaic and front snapshot."""
        await self.explore(
            actor_id,
            image_prefix=image_prefix,
            image_format=image_format,
            post_step_callback=post_step_callback,
        )

    async def explore(
        self,
        actor_id: bytes | str | dict,
        *,
        image_prefix: Optional[str] = "explore",
        image_format: str = "png",
        post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Four-way in-place sampling: **body pose fixed**; ``CaptureAPI.set_camera_pose`` rotates the mounted camera
        through front / left / back / right with ``capture_rgb_and_depth`` (no detections / false-color depth).
        Stitch RGB into a **three-column** mosaic; each tile **must** be ``640×480`` or ``ValueError`` is raised.
        Mosaic is 1920×1200: left column centers ``left``; middle column top ``front``, bottom ``back`` (white bar);
        right column centers ``right``; padding is white.

        On success fills ``last_locator_visualization_paths`` (``rgb`` mosaic only), ``last_explore_tile_paths``,
        ``last_explore_camera_poses`` (order front→left→back→right). Does **not** call ``move_to``,
        ``set_actor_transform``, or ``get_perception_object_list``; restores camera pose from entry.

        Args:
            actor_id: Agent id
            image_prefix: capture filename prefix
            image_format: capture format
            post_step_callback: invoked after each successful tile capture (not after restoring camera)

        Returns:
            ``{"mosaic_rgb_path", "rgb_paths", "camera_poses"}``; on failure the three fields are
            ``None`` / ``[]`` / ``[]``.
        """
        self.last_locator_visualization_paths = None
        self.last_explore_tile_paths = {}
        self.last_explore_camera_poses = []

        if self._observation_camera is None:
            print("[ActionAPI] explore: observation_camera is None")
            return explore_result_empty()

        actor_ref = actor_id
        if isinstance(actor_id, dict):
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            actor_key = extracted_id if extracted_id is not None else actor_id
        else:
            actor_key = actor_id

        vis_ref = actor_ref if isinstance(actor_ref, dict) else actor_key
        prefix = (image_prefix or "explore").strip() or "explore"

        # await ts.UnaryAPI.set_extinguisher_rotation(
        #     self._conn, actor_key, 0.0, 0.0, timeout=5.0
        # )

        initial_state = await ts.UnaryAPI.get_actor_state(self._conn, actor_key)
        if not initial_state:
            return explore_result_empty()

        initial_forward = initial_state.get("unit_forward_vector")
        if not initial_forward or not isinstance(initial_forward, ts.Vector3):
            initial_forward = ts.Vector3(1.0, 0.0, 0.0)

        initial_right = initial_state.get("unit_right_vector")
        if not initial_right or not isinstance(initial_right, ts.Vector3):
            initial_right = ts.Vector3(0.0, 1.0, 0.0)

        orientations: list[tuple[str, ts.Vector3]] = [
            ("front", initial_forward),
            ("left", ts.Vector3(-initial_right.x, -initial_right.y, -initial_right.z)),
            ("back", ts.Vector3(-initial_forward.x, -initial_forward.y, -initial_forward.z)),
            ("right", initial_right),
        ]

        initial_tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if not initial_tf:
            return explore_result_empty()
        initial_rotation = initial_tf.rotation
        euler0 = ts.math.quaternion_to_euler(initial_rotation, is_degree=True)
        roll_deg = float(euler0.x)
        pitch_deg = float(euler0.y)

        off = DEFAULT_CAMERA_OFFSET
        cam_loc = ts.Vector3(
            float(initial_tf.location.x) + float(off.x),
            float(initial_tf.location.y) + float(off.y),
            float(initial_tf.location.z) + float(off.z),
        )
        unit_scale = ts.Vector3(1.0, 1.0, 1.0)
        restore_cam_tf = ts.Transform(
            location=cam_loc,
            rotation=initial_rotation,
            scale=unit_scale,
        )

        camera_id = await self._observation_camera.resolve_camera_id(vis_ref, prefer_depth=True)
        if camera_id is None:
            print("[ActionAPI] explore: resolve_camera_id returned None")
            return explore_result_empty()

        def horizontal_dir(d: ts.Vector3) -> ts.Vector3:
            vx, vy = float(d.x), float(d.y)
            L = math.hypot(vx, vy)
            if L < 1e-6:
                return ts.Vector3(1.0, 0.0, 0.0)
            return ts.Vector3(vx / L, vy / L, 0.0)

        tiles: Dict[str, Dict[str, Optional[str]]] = {}
        camera_poses: List[Dict[str, Any]] = []

        async def _capture_tile(label: str, pose_yaw_deg: float) -> None:
            paths = await self._observation_camera.capture_rgb_and_depth(
                vis_ref,
                image_prefix=f"{prefix}_{label}",
                image_format=image_format,
            )
            rgb_p = paths.get("rgb") if paths else None
            if not rgb_p or not Path(rgb_p).is_file():
                print(f"[ActionAPI] explore: missing rgb for '{label}': {rgb_p!r}")
                return
            pose = _camera_pose_dict_for_explore_view(
                initial_tf,
                DEFAULT_CAMERA_OFFSET,
                pose_yaw_deg,
                label,
            )
            camera_poses.append(pose)
            depth_p = paths.get("depth") if paths else None
            if depth_p and not Path(depth_p).is_file():
                depth_p = None
            tiles[label] = {
                "rgb": str(Path(rgb_p).resolve()),
                "depth": str(Path(depth_p).resolve()) if depth_p else None,
            }

        mosaic_paths: list[str] = []
        required = ("front", "left", "back", "right")

        try:
            for turn_label, dir_vec in orientations:
                hd = horizontal_dir(dir_vec)
                yaw_deg = math.degrees(math.atan2(hd.y, hd.x))
                face_q = ts.math.euler_to_quaternion(
                    ts.Vector3(roll_deg, pitch_deg, yaw_deg),
                    is_degree=True,
                )
                cam_tf = ts.Transform(
                    location=cam_loc,
                    rotation=face_q,
                    scale=unit_scale,
                )
                ok = await ts.CaptureAPI.set_camera_pose(self._conn, camera_id, cam_tf)
                if not ok:
                    print(f"[ActionAPI] explore: set_camera_pose failed for '{turn_label}'")
                    break
                await _capture_tile(turn_label, yaw_deg)
                if post_step_callback:
                    await post_step_callback()
        finally:
            restored = await ts.CaptureAPI.set_camera_pose(
                self._conn, camera_id, restore_cam_tf
            )
            if not restored:
                print("[ActionAPI] explore: failed to restore camera pose after explore")

        if len(tiles) == 4 and all(k in tiles for k in required):
            first_rgb = Path(tiles["front"]["rgb"])
            out_dir = first_rgb.parent
            front_stem = first_rgb.stem
            front_key = f"{prefix}_front_"
            time_part = (
                front_stem[len(front_key) :]
                if front_stem.startswith(front_key)
                else front_stem
            )
            stem = f"{prefix}_mosaic_{time_part}"
            rgb_arr = self._mosaic_explore_quadrant_bgr(
                {k: tiles[k]["rgb"] for k in required}
            )
            if rgb_arr is not None:
                p_rgb = out_dir / f"{stem}_rgb.jpg"
                if cv2.imwrite(str(p_rgb), rgb_arr):
                    mosaic_paths = [str(p_rgb.resolve())]
                else:
                    print("[ActionAPI] explore: cv2.imwrite failed for mosaic rgb")
            else:
                print("[ActionAPI] explore: mosaic assembly failed (see logs above)")

        if mosaic_paths:
            self.last_locator_visualization_paths = {"rgb": mosaic_paths[0]}
            self.last_explore_tile_paths = {
                k: dict(tiles[k]) for k in required if k in tiles
            }
            self.last_explore_camera_poses = list(camera_poses)
            rgb_paths_ordered = [
                str(tiles[k]["rgb"]) for k in required if k in tiles and tiles[k].get("rgb")
            ]
            return {
                "mosaic_rgb_path": mosaic_paths[0],
                "rgb_paths": rgb_paths_ordered,
                "camera_poses": [dict(p) for p in camera_poses],
            }

        return explore_result_empty()

    # -------------------------------------------------------------------------
    # Perception (simulator object lists)
    # -------------------------------------------------------------------------

    def _mosaic_explore_quadrant_bgr(
        self,
        paths_by_label: Dict[str, Optional[str]],
    ) -> Optional[np.ndarray]:
        """
        Stitch front / left / back / right into the Human three-column BGR mosaic (matches ``_human_mosaic_pixel_to_direction_local``).
        Each BGR tile must be ``640×480`` (no rescaling; mismatch raises ``ValueError``).
        Canvas width ``3*w0``, height ``2*h0 + h0//2``, white background; middle column top ``front``, bottom ``back``;
        left/right columns center ``left`` / ``right``.
        """
        order = ("front", "left", "back", "right")
        for lb in order:
            p = paths_by_label.get(lb)
            if not p or not Path(p).is_file():
                print(f"[ActionAPI] explore mosaic: missing file for '{lb}': {p!r}")
                return None
        imgs: Dict[str, np.ndarray] = {}
        for lb in order:
            im = cv2.imread(str(paths_by_label[lb]))
            if im is None:
                print(f"[ActionAPI] explore mosaic: imread failed for '{lb}': {paths_by_label[lb]}")
                return None
            _assert_explore_tile_bgr_shape(lb, im)
            imgs[lb] = im
        h0, w0 = EXPLORE_TILE_HEIGHT, EXPLORE_TILE_WIDTH
        mosaic_h, mosaic_w = _human_explore_mosaic_size_hw(h0, w0)
        canvas = np.full((mosaic_h, mosaic_w, 3), 255, dtype=np.uint8)
        pad = (mosaic_h - h0) // 2
        canvas[0:h0, w0 : 2 * w0] = imgs["front"]
        canvas[mosaic_h - h0 : mosaic_h, w0 : 2 * w0] = imgs["back"]
        canvas[pad : pad + h0, 0:w0] = imgs["left"]
        canvas[pad : pad + h0, 2 * w0 : 3 * w0] = imgs["right"]
        return canvas

    async def get_perception_object_list(self, agent_id: Optional[bytes | str | dict] = None, timeout: float = 5.0) -> dict:
        """
        Pull simulator perception into ``perception_cache`` / ``_perception_object_list`` (same role as action_ol.ActionAPI).

        VL/VLM policies should read ``_visual_perception_object_list``; this call aligns caches and feeds
        ``EmbodiedMAS.Metric_Tool.perception_evaluation``. Do not treat the return value as the sole decision signal.

        Args:
            agent_id: Agent id; None yields an empty dict
            timeout: seconds

        Returns:
            Dict with ``actor_info`` / ``npc_info`` for logging and metric hooks.
        """
        if agent_id is None:
            print("[ActionAPI] Error: No agent_id provided")
            result = {"actor_info": [], "npc_info": []}
            self.perception_cache = result
            return result

        if isinstance(agent_id, dict):
            target_agent_id = agent_id.get("id")
            if target_agent_id is None:
                print("[ActionAPI] Error: Agent dict has no 'id' field")
                result = {"actor_info": [], "npc_info": []}
                self.perception_cache = result
                return result
        else:
            target_agent_id = agent_id

        result = await self._perception.get_perception(target_agent_id, timeout=timeout)

        simplified_cache = {
            "actor_info": [],
            "npc_info": []
        }

        actor_list = result.get("actor_info", [])
        for actor_info in actor_list:
            actor = actor_info.get("actor", {})
            actor_name = actor.get("name", "")
            actor_location = actor.get("location")
            actor_bounding_box = actor.get("bounding_box")
            actor_burning_state = actor_info.get("burning_state")
            simplified_cache["actor_info"].append({
                "name": actor_name,
                "location": actor_location,
                "bounding_box": actor_bounding_box,
                "burning_state": actor_burning_state
            })

        npc_list = result.get("npc_info", [])
        for npc_info in npc_list:
            object_info = npc_info.get("object_info", {})
            npc_name = object_info.get("name", "")
            npc_position = npc_info.get("position")
            simplified_cache["npc_info"].append({
                "name": npc_name,
                "location": npc_position
            })

        if self._perception_object_list is None:
            self._perception_object_list = {}

        for key, value in simplified_cache.items():
            if key not in self._perception_object_list:
                self._perception_object_list[key] = value
            else:
                existing_list = self._perception_object_list[key]
                if isinstance(existing_list, list) and isinstance(value, list):
                    name_to_item = {}
                    for item in existing_list:
                        name = item.get("name", "")
                        if name:
                            name_to_item[name] = item
                    for item in value:
                        name = item.get("name", "")
                        if name:
                            name_to_item[name] = item
                    self._perception_object_list[key] = list(name_to_item.values())
                else:
                    self._perception_object_list[key] = value

        self._perception.filter_simplified_object_list_cache(self._perception_object_list)

        return result
