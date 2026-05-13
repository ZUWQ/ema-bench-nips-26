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

# Ensure EmbodiedMAS imports when this file is run as a script
if __package__ is None or __package__ == "":
    import os
    import sys
    # Repo root (PythonClient): resolves EmbodiedMAS.* and tongsim
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

# Do not use `from EmbodiedMAS...` here: it runs EmbodiedMAS/__init__.py, which may import
# base_agent_vlm → action_vlm again and leave a partially initialized action_vlm (circular import).
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
    "embodiedmas_location4human_for_action_vlm",
    str(_embodied_root / "perception_yolo" / "location4human.py"),
)
if not _l4h_spec or not _l4h_spec.loader:
    raise ImportError(
        f"cannot load location4human from {_embodied_root / 'perception_yolo' / 'location4human.py'}"
    )
_l4h_module = importlib.util.module_from_spec(_l4h_spec)
_l4h_spec.loader.exec_module(_l4h_module)
explore_poses_to_pose_by_direction = _l4h_module.explore_poses_to_pose_by_direction
pixel_to_world_from_explore_mosaic_1280 = _l4h_module.pixel_to_world_from_explore_mosaic_1280

# Matches base_agent_vlm.AGENT_CAMERA_PARAMS horizontal FOV (back-projection)
EXPLORE_CAMERA_HFOV_DEG = 90.0


def explore_result_empty() -> Dict[str, Any]:
    """Placeholder explore dict with the same keys as a successful run."""
    return {"mosaic_rgb_path": None, "rgb_paths": [], "camera_poses": []}


# Camera offset (must match base_agent_vlm.DEFAULT_CAMERA_OFFSET / spawn-time optical center)
DEFAULT_CAMERA_OFFSET = ts.Vector3(100, 0, 110)

# Navigation Z (matches base_agent_vlm.Z_HEIGHT; perception elevation flags are not world height)
NAV_TARGET_Z = 1000.0

# Cardinal nudge distance for move_front/back/left/right (UU), same semantics as move_by
MOVE_CARDINAL_DISTANCE_UU = 200.0

# Extinguisher forward offset in UU (action_h / action_ol / engine reference)
EXTINGUISHER_FORWARD_LENGTH_UU = 57.0
# Vertical reference for pitch: nozzle height above actor root
EXTINGUISHER_PITCH_REF_Z_OFFSET_UU = 26.0

# Helper: yaw from quaternion
def _get_yaw_from_quaternion(q: ts.Quaternion) -> float:
    """Yaw in radians from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw (radians) from quaternion (x, y, z, w)."""
    return math.atan2(
        2 * (w * z + x * y),
        1 - 2 * (y * y + z * z),
    )


def _compute_relative_xy(
    agent_xy: tuple[float, float],
    quat_xyzw: tuple[float, float, float, float],
    target_xy: tuple[float, float],
) -> dict:
    """Relative XY offset from agent to target plus signed angle (deg); ``quat_xyzw`` = (x, y, z, w)."""
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
    Compute ``set_extinguisher_rotation(a, b)`` (degrees) from agent pose and world target.
    Geometry matches ``Human_Agent/action_h._extinguisher_ab_from_target``: N0 → horizontal ``a`` →
    rotate nozzle to N1 about vertical axis → pitch ``b`` from ``T - N1``.
    """
    quat_xyzw = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    yaw_ref = _quaternion_to_yaw(*quat_xyzw)
    ax = float(tf.location.x)
    ay = float(tf.location.y)
    az = float(tf.location.z)

    L = EXTINGUISHER_FORWARD_LENGTH_UU
    z_off = EXTINGUISHER_PITCH_REF_Z_OFFSET_UU

    n0x = ax + L * math.cos(yaw_ref)
    n0y = ay + L * math.sin(yaw_ref)
    n0z = az + z_off

    tx, ty, tz = float(target.x), float(target.y), float(target.z)
    rel = _compute_relative_xy((ax, ay), quat_xyzw, (tx, ty))
    x_rel, y_rel = rel["relative_position"]
    phi_rad = math.atan2(y_rel, x_rel)
    a = math.degrees(phi_rad)

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
    Polar offset in world XY (same as ``EmbodiedMAS/Try_Agent/try_sos_agent.calculate_sos_location``):
    x' = x + d·cos ψ, y' = y + d·sin ψ, z' = z with ψ = radians(yaw_deg).

    Treat ``yaw_deg`` as an absolute heading in the world XY plane; validate against your SOS payload if the engine differs.
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
    """Camera optical center = agent position + offset; orientation uses agent quaternion + yaw (deg)."""
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


class ActionAPI:
    """
    VLM action helpers (aligned with the human ``ActionAPI`` split).

    Front half: sim motion/interaction (``move_to``, ``move_by``, extinguish, follow, ``wait``); ``wait`` ends with ``explore``.
    Back half: ``get_perception_object_list`` (after the front tile capture only) to stay consistent with the sim and
    ``Metric_Tool.perception_evaluation`` dumps — object lists never feed the VLM policy/LLM.
    ``explore`` captures four headings + RGB mosaic; geometry uses ``perception_yolo/location4human``, not detector labels.
    """

    def __init__(
        self,
        context: WorldContext,
        observation_camera: Optional[ObservationCamera] = None,
        experiment_result: Optional[Any] = None,
    ):
        self._context = context
        self._conn = context.conn
        self.experiment_result = experiment_result
        self._perception = PerceptionInfo(context)
        self._observation_camera = observation_camera
        self.perception_cache: dict = {}  # Latest simplified perception snapshot
        # Rolling sim-aligned cache (matches action_ol; used by perception_evaluation)
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
        # Latest explore RGB mosaic path (single image for the LLM)
        self.last_locator_visualization_paths: Optional[Dict[str, str]] = None
        # Per-direction capture paths: front/left/back/right → rgb (required), depth (optional on-disk path)
        self.last_explore_tile_paths: Dict[str, Dict[str, Optional[str]]] = {}
        # Camera poses in front→left→back→right order (JSON-serializable dicts)
        self.last_explore_camera_poses: List[Dict[str, Any]] = []

    def _world_vector3_from_last_explore_mosaic(
        self, px: float, py: float
    ) -> Optional[ts.Vector3]:
        """Map the latest successful mosaic pixel to world coordinates via depth back-projection."""
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
        arr = pixel_to_world_from_explore_mosaic_1280(
            (px*2, py*2),
            tiles,
            pose_by_direction,
            horizontal_fov_degrees=EXPLORE_CAMERA_HFOV_DEG,
        )
        if arr is None:
            print("[ActionAPI] pixel→world: pixel_to_world_from_explore_mosaic_1280 returned None")
            return None
        return ts.Vector3(float(arr[0]), float(arr[1]), float(arr[2]))

    # -------------------------------------------------------------------------
    # Sim actions (locomotion, interaction, explore)
    # -------------------------------------------------------------------------

    async def world_location_from_sos(
        self,
        listener_actor_id: bytes | str | dict,
        sos_info: Dict[str, Any],
    ) -> Optional[ts.Vector3]:
        """
        Listener pose + SOS yaw (deg) and distance → target ``Vector3`` (z matches listener).
        ``sos_info`` mirrors ``receive_npc_sos``: ``orientations[0].yaw``, ``distances[0]``.
        """
        orientations = sos_info.get("orientations") or []
        distances = sos_info.get("distances") or []
        if not orientations or not distances:
            print("[ActionAPI] world_location_from_sos: missing orientations or distances")
            return None
        o0 = orientations[0]
        if not isinstance(o0, dict):
            print("[ActionAPI] world_location_from_sos: orientations[0] is not a dict")
            return None
        try:
            yaw_deg = float(o0.get("yaw", 0.0))
            d = float(distances[0])
        except (TypeError, ValueError, IndexError):
            print("[ActionAPI] world_location_from_sos: invalid yaw or distance")
            return None
        if d <= 0:
            print(f"[ActionAPI] world_location_from_sos: distance must be > 0, got {d}")
            return None

        actor_key: bytes | str | dict = listener_actor_id
        if isinstance(listener_actor_id, dict):
            extracted = listener_actor_id.get("id")
            if extracted is None:
                extracted = listener_actor_id.get("guid")
            if extracted is not None:
                actor_key = extracted

        tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if not tf:
            print(f"[ActionAPI] world_location_from_sos: failed to get transform for {actor_key!r}")
            return None

        pt = world_position_from_yaw_distance_2d(
            tf.location.x,
            tf.location.y,
            tf.location.z,
            yaw_deg,
            d,
        )
        if pt is None:
            return None
        return ts.Vector3(pt[0], pt[1], pt[2])

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
        Navigate on the navmesh using a mosaic pixel.

        ``pixel_xy`` indexes the latest ``explore`` 2×2 RGB mosaic; ``location4human`` back-projects to world XY and
        ``NAV_TARGET_Z`` supplies the navigation height.

        Returns:
            ``navigate_to_location`` response dict, or None if back-projection fails
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
        Planar move in world XY relative to current heading: ``distance`` is the true sim length (UU, no hidden scale);
        ``angle`` in degrees (left positive). Movement heading = agent yaw + angle (radians) with the same polar offset as SOS.
        Target Z is ``NAV_TARGET_Z``. ``orientation_mode`` exists only for API parity with ``move_to``.

        Returns:
            ``navigate_to_location`` response dict (action_ol-compatible) or None on failure.
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
        Aim the extinguisher using optional mosaic pixels, then spray.

        When ``pixel_xy`` references the latest mosaic, it back-projects to a world target and ``_extinguisher_ab_from_target``
        supplies ``(a, b)`` like ``Human_Agent/action_h``.
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
            print("[ActionAPI] Error: could not read current transform")
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
        await asyncio.sleep(15)

        await self.explore(actor_ref)

        if isinstance(result, bool):
            return "Success" if result else "Fail_CanotExtinguish"
        return result

    async def sendfollow(
        self,
        actor_id: bytes | str | dict,
    ) -> list[str]:
        """
        Issue follow for the agent.
        
        Args:
            actor_id: bytes, str, or dict with id/guid
        
        Returns:
            RPC result list
        """
        actor_ref = actor_id
        # Dict handles: prefer id then guid
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
        Issue stop-follow for the agent.
        
        Args:
            actor_id: bytes, str, or dict with id/guid
        
        Returns:
            RPC result list
        """
        actor_ref = actor_id
        # Dict handles: prefer id then guid
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
        """Sleep 5s, then ``explore`` to refresh mosaic + sim perception (front capture)."""
        await asyncio.sleep(5.0)
        await self.explore(
            actor_id,
            image_prefix=image_prefix,
            image_format=image_format,
            post_step_callback=post_step_callback,
        )

    async def get_camera_info_and_convert(
        self,
        agent_ref: bytes | str | dict,
        *,
        camera_offset: Optional[ts.Vector3] = None,
    ) -> dict[str, Any]:
        """Debug helper: camera pose bundle consistent with ``BaseAgentVLM``."""
        agent_id_str = agent_ref["id"] if isinstance(agent_ref, dict) else agent_ref
        agent_transform = await ts.UnaryAPI.get_actor_transform(self._conn, agent_id_str)
        if not agent_transform:
            raise RuntimeError(f"get_actor_transform failed for {agent_id_str!r}")
        offset = camera_offset or DEFAULT_CAMERA_OFFSET
        camera_pos = ts.Vector3(
            agent_transform.location.x + offset.x,
            agent_transform.location.y + offset.y,
            agent_transform.location.z + offset.z,
        )
        yaw_rad = _get_yaw_from_quaternion(agent_transform.rotation)
        yaw_deg = math.degrees(yaw_rad)
        orientation = (0.0, 0.0, yaw_deg)
        return {
            "camera_position": camera_pos,
            "camera_rotation": agent_transform.rotation,
            "agent_transform": agent_transform,
            "orientation": orientation,
            "world_coord": None,
        }

    async def explore(
        self,
        actor_id: bytes | str | dict,
        *,
        image_prefix: Optional[str] = "explore",
        image_format: str = "png",
        post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Teleport-based rotation sampling: fixed position, ``set_actor_transform`` to face front/left/back/right,
        ``capture_rgb_and_depth`` per heading (no detection / false-color depth); stitch four RGB tiles into a 2×2 mosaic.

        On success fills ``last_locator_visualization_paths`` (``rgb`` mosaic only), ``last_explore_tile_paths``,
        and ``last_explore_camera_poses`` (front→left→back→right). This path never calls ``move_to``.

        Args:
            actor_id: Agent handle
            image_prefix: Capture filename stem
            image_format: Image format for RGB tiles
            post_step_callback: Invoked after each perception (including initial front); not after restoring heading

        Returns:
            ``{"mosaic_rgb_path", "rgb_paths", "camera_poses"}``; on failure those become ``None`` / ``[]`` / ``[]``.
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

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_key, 0.0, 0.0, timeout=5.0
        )

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
        fixed_loc = initial_tf.location
        fixed_scale = initial_tf.scale
        euler0 = ts.math.quaternion_to_euler(initial_rotation, is_degree=True)
        roll_deg = float(euler0.x)
        pitch_deg = float(euler0.y)

        def horizontal_dir(d: ts.Vector3) -> ts.Vector3:
            vx, vy = float(d.x), float(d.y)
            L = math.hypot(vx, vy)
            if L < 1e-6:
                return ts.Vector3(1.0, 0.0, 0.0)
            return ts.Vector3(vx / L, vy / L, 0.0)

        tiles: Dict[str, Dict[str, Optional[str]]] = {}
        camera_poses: List[Dict[str, Any]] = []

        async def _capture_tile(label: str) -> None:
            paths = await self._observation_camera.capture_rgb_and_depth(
                vis_ref,
                image_prefix=f"{prefix}_{label}",
                image_format=image_format,
            )
            rgb_p = paths.get("rgb") if paths else None
            if not rgb_p or not Path(rgb_p).is_file():
                print(f"[ActionAPI] explore: missing rgb for '{label}': {rgb_p!r}")
                return
            tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
            if not tf:
                print(f"[ActionAPI] explore: no transform after capture for '{label}'")
                return
            pose = _camera_pose_dict_from_transform(tf, DEFAULT_CAMERA_OFFSET)
            pose["direction"] = label
            camera_poses.append(pose)
            depth_p = paths.get("depth") if paths else None
            if depth_p and not Path(depth_p).is_file():
                depth_p = None
            tiles[label] = {
                "rgb": str(Path(rgb_p).resolve()),
                "depth": str(Path(depth_p).resolve()) if depth_p else None,
            }
            if label == "front":
                await self.get_perception_object_list(actor_key)

        await _capture_tile("front")
        if post_step_callback:
            await post_step_callback()

        for turn_label, dir_vec in orientations[1:]:
            hd = horizontal_dir(dir_vec)
            yaw_deg = math.degrees(math.atan2(hd.y, hd.x))
            face_q = ts.math.euler_to_quaternion(
                ts.Vector3(roll_deg, pitch_deg, yaw_deg),
                is_degree=True,
            )
            ok = await ts.UnaryAPI.set_actor_transform(
                self._conn,
                actor_key,
                ts.Transform(
                    location=fixed_loc,
                    rotation=face_q,
                    scale=fixed_scale,
                ),
            )
            if not ok:
                break
            await _capture_tile(turn_label)
            if post_step_callback:
                await post_step_callback()

        mosaic_paths: list[str] = []
        required = ("front", "left", "back", "right")
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

        final_tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if final_tf:
            await ts.UnaryAPI.set_actor_transform(
                self._conn,
                actor_key,
                ts.Transform(
                    location=final_tf.location,
                    rotation=initial_rotation,
                    scale=final_tf.scale,
                ),
            )

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
    # Perception (sim object lists)
    # -------------------------------------------------------------------------

    def _mosaic_explore_quadrant_bgr(
        self,
        paths_by_label: Dict[str, Optional[str]],
    ) -> Optional[np.ndarray]:
        """
        Stitch front/left/back/right into a 2×2 BGR image:
        top-left front, bottom-left left, top-right back, bottom-right right.
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
            imgs[lb] = im
        h0, w0 = imgs["front"].shape[:2]
        for lb in order:
            h, w = imgs[lb].shape[:2]
            if (h, w) != (h0, w0):
                imgs[lb] = cv2.resize(imgs[lb], (w0, h0), interpolation=cv2.INTER_LINEAR)
        top = np.hstack([imgs["front"], imgs["back"]])
        bottom = np.hstack([imgs["left"], imgs["right"]])
        return np.vstack([top, bottom])

    async def get_perception_object_list(self, agent_id: Optional[bytes | str | dict] = None, timeout: float = 5.0) -> dict:
        """
        Pull embodied perception and refresh ``perception_cache`` / ``_perception_object_list`` (action_ol-compatible).

        Intended for ``Metric_Tool.perception_evaluation`` and other sim-aligned logs — not fed to the VLM/LLM.

        Args:
            agent_id: Agent handle; None yields an empty dict
            timeout: RPC timeout (seconds)

        Returns:
            Dict with ``actor_info`` and ``npc_info`` (observation tooling only).
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

        self.perception_cache = simplified_cache

        return result
