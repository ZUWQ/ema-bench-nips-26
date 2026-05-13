from __future__ import annotations

import asyncio
import math
from typing import Optional, Callable, Awaitable, Any, Dict

import tongsim as ts
from tongsim.core.world_context import WorldContext
from tongsim.type.rl_demo import RLDemoOrientationMode

# Ensure EmbodiedMAS imports when this file is run as a script
if __package__ is None or __package__ == "":
    import os
    import sys
    # Add EmbodiedMAS parent (PythonClient root) to sys.path when run standalone
    _embodiedmas_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if _embodiedmas_parent not in sys.path:
        sys.path.insert(0, _embodiedmas_parent)

from EmbodiedMAS.observation import PerceptionInfo

# Navigation target Z (aligned with VL/VLM base agents' Z_HEIGHT)
NAV_TARGET_Z = 1000.0

# Extinguisher horizontal forward offset in UU (matches Human_Agent/action_h / engine reference)
EXTINGUISHER_FORWARD_LENGTH_UU = 57.0
# Vertical reference for pitch: nozzle/reference height above actor root
EXTINGUISHER_PITCH_REF_Z_OFFSET_UU = 26.0


def _get_yaw_from_quaternion(q: ts.Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_position_from_yaw_distance_2d(
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    distance: float,
) -> Optional[tuple[float, float, float]]:
    """
    Polar offset in world XY (same convention as Try_Agent/try_sos_agent.calculate_sos_location).
    ``yaw_deg`` is an absolute heading in the world XY plane (degrees); returns None when ``distance <= 0``.
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


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Extract yaw (radians) from quaternion (x, y, z, w).
    """
    yaw = math.atan2(
        2 * (w * z + x * y),
        1 - 2 * (y * y + z * z),
    )
    return yaw


def _compute_relative_xy(
    agent_xy: tuple[float, float],
    quat_xyzw: tuple[float, float, float, float],
    target_xy: tuple[float, float],
) -> dict:
    """
    Relative XY offset from agent to target and signed angle (degrees) in the agent frame.
    ``quat_xyzw`` = (x, y, z, w).
    """
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
    Compute ``set_extinguisher_rotation(a, b)`` angles (degrees) from agent pose and a world target.
    Geometry matches ``Human_Agent/action_h._extinguisher_ab_from_target``.
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


class ActionAPI:
    """
    Action utilities extracted from EnvironmentWrapper.
    Provides simple movement helpers operating on a given connection/context.
    """

    def __init__(self, context: WorldContext, experiment_result: Optional[Any] = None):
        self._context = context
        self._conn = context.conn
        self.experiment_result = experiment_result
        self._perception = PerceptionInfo(context)
        self.perception_cache: dict = {}  # Latest simplified perception snapshot
        # Seed _perception_object_list with a static "Safe_Zone" entry
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

    async def world_location_from_sos(
        self,
        listener_actor_id: bytes | str | dict,
        sos_info: Dict[str, Any],
    ) -> Optional[ts.Vector3]:
        """Listener pose + SOS yaw (deg) and distance → world ``Vector3`` (z matches listener)."""
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
        Move in world XY relative to current heading. ``distance`` is the policy length (OL does **not** scale by 100);
        ``angle`` in degrees (left positive). Target Z is ``NAV_TARGET_Z``.
        """
        _ = orientation_mode
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
        move_yaw = phi + math.radians(float(angle))
        target_location = ts.Vector3(
            tf.location.x + d_eff * math.cos(move_yaw),
            tf.location.y + d_eff * math.sin(move_yaw),
            NAV_TARGET_Z,
        )

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
        await self.get_perception_object_list(actor_id)
        return resp

    async def move_to(
        self,
        actor_id: bytes | str | dict,
        target_location: ts.Vector3,
        *,
        timeout: float = 60.0,
        tolerance_uu: float = 50.0,
        orientation_mode: RLDemoOrientationMode = RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
        allow_partial: bool = True,
        speed_uu_per_sec: float = 100.0,
    ) -> Optional[dict]:
        """
        Navigate on the navmesh to ``target_location``.
        
        Args:
            actor_id: Agent handle
            target_location: Goal ``Vector3``
            timeout: Seconds
            tolerance_uu: Arrival radius (``accept_radius``)
            orientation_mode: Kept for API compatibility; not used by ``navigate_to_location`` here
            allow_partial: Allow partial arrival
            speed_uu_per_sec: Move speed (uu/s)
        
        Returns:
            ``navigate_to_location`` response dict (includes ``success``), or None on failure
        """
        # If actor_id is a dict, take "id" or "guid"
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
        
        # Call navigate_to_location
        resp = await ts.UnaryAPI.navigate_to_location(
            self._conn,
            actor_id=actor_id,
            target_location=target_location,
            accept_radius=tolerance_uu,
            allow_partial=allow_partial,
            speed_uu_per_sec=speed_uu_per_sec,
            timeout=timeout,
        )
        
        await self.get_perception_object_list(actor_id)

        return resp


    async def sendfollow(
        self,
        actor_id: bytes | str | dict,
    ) -> list[str]:
        """
        Issue follow for the given agent.
        
        Args:
            actor_id: bytes, str, or dict with id/guid
        
        Returns:
            RPC result list
        """
        # If actor_id is a dict, take "id" or "guid"
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

        # Refresh perception after the action
        # print(f"[ActionAPI] sendfollow result: {result}")
        await self.get_perception_object_list(actor_id)

        return result

    async def sendstopfollow(
        self,
        actor_id: bytes | str | dict,
    ) -> list[str]:
        """
        Issue stop-follow for the given agent.
        
        Args:
            actor_id: bytes, str, or dict with id/guid
        
        Returns:
            RPC result list
        """
        # If actor_id is a dict, take "id" or "guid"
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

        await self.get_perception_object_list(actor_id)

        result = await ts.UnaryAPI.sendstopfollow(self._conn, actor_id)
        er = getattr(self, "experiment_result", None)
        if er is not None:
            await er.record_rescue_after_stop_follow(actor_id)
        return result

    async def wait(self) -> None:
        """
        Sleep 5 seconds.
        
        Returns:
            None
        """
        await asyncio.sleep(5.0)

    async def extinguish_fire(
        self,
        actor_id: bytes | str | dict,
        target_location: ts.Vector3,
        *,
        timeout: float = 5.0,
    ) -> str:
        """
        Aim the extinguisher at ``target_location`` and spray.
        ``a``/``b`` come from ``_extinguisher_ab_from_target`` (same as ``Human_Agent/action_h``).
        """
        # 1. Resolve agent key
        if isinstance(actor_id, dict):
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            actor_key = extracted_id if extracted_id is not None else actor_id
        else:
            actor_key = actor_id

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_key, 0.0, 0.0, timeout=timeout
        )

        # 2. Agent transform (position + quaternion)
        tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
        if not tf:
            print("[ActionAPI] Error: could not read current transform")
            return "Fail_CanotExtinguish"

        a, b = _extinguisher_ab_from_target(tf, target_location)

        # 3. Set extinguisher yaw=a, pitch=b, then extinguish
        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn,
            actor_key,
            a,
            b,
            timeout=timeout,
        )

        result = await ts.UnaryAPI.extinguish_fire(self._conn, actor_key, timeout=timeout)
        await asyncio.sleep(10)

        await self.get_perception_object_list(actor_key)

        if isinstance(result, bool):
            return "Success" if result else "Fail_CanotExtinguish"
        return result

    # async def explore(
    #     self,
    #     actor_id: bytes | str | dict,
    #     *,
    #     post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    # ) -> None:
    #     """
    #     Legacy in-place rotation sampling: face front/left/back/right (no image capture).
        
    #     Args:
    #         actor_id: Agent handle
    #         post_step_callback: Async hook after each reorientation
    #     """
    #     # Original explore body kept under ``if False`` for reference; runtime uses explore2.
    #     if False:
    #         if isinstance(actor_id, dict):
    #             extracted_id = actor_id.get("id")
    #             if extracted_id is None:
    #                 extracted_id = actor_id.get("guid")
    #             actor_key = extracted_id if extracted_id is not None else actor_id
    #         else:
    #             actor_key = actor_id
    
    #         await ts.UnaryAPI.set_extinguisher_rotation(
    #             self._conn, actor_key, 0.0, 0.0, timeout=5.0
    #         )
    
    #         # Save initial heading
    #         initial_state = await ts.UnaryAPI.get_actor_state(self._conn, actor_key)
    #         if not initial_state:
    #             return
            
    #         initial_forward = initial_state.get("unit_forward_vector")
    #         if not initial_forward or not isinstance(initial_forward, ts.Vector3):
    #             initial_forward = ts.Vector3(1.0, 0.0, 0.0)
            
    #         initial_right = initial_state.get("unit_right_vector")
    #         if not initial_right or not isinstance(initial_right, ts.Vector3):
    #             initial_right = ts.Vector3(0.0, 1.0, 0.0)
    
    #         # Helper: normalize a vector
    #         def normalize(v: ts.Vector3) -> ts.Vector3:
    #             mag2 = v.x * v.x + v.y * v.y + v.z * v.z
    #             if mag2 <= 1e-8:
    #                 return ts.Vector3(1.0, 0.0, 0.0)
    #             inv = (mag2) ** -0.5
    #             return ts.Vector3(v.x * inv, v.y * inv, v.z * inv)
    
    #         # Four headings: front, left, back, right
    #         orientations: list[tuple[str, ts.Vector3]] = [
    #             ("front", initial_forward),
    #             ("left", ts.Vector3(-initial_right.x, -initial_right.y, -initial_right.z)),
    #             ("back", ts.Vector3(-initial_forward.x, -initial_forward.y, -initial_forward.z)),
    #             ("right", initial_right),
    #         ]
    
    #         move_distance = 1.0  # Tiny step used mainly to bias facing
    
    #         # Visit each heading in order
    #         for label, dir_vec in orientations:
    #             cur_tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
    #             if not cur_tf:
    #                 break
    
    #             # Nudge toward target heading (small translation for facing)
    #             fwd = normalize(dir_vec)
    #             target_location = ts.Vector3(
    #                 cur_tf.location.x + fwd.x * move_distance,
    #                 cur_tf.location.y + fwd.y * move_distance,
    #                 cur_tf.location.z + fwd.z * move_distance,
    #             )
                
    #             # Face the desired direction
    #             await self.move_to(
    #                 actor_id=actor_key,
    #                 target_location=target_location,
    #                 timeout=3.0,
    #                 tolerance_uu=0.1,
    #             )
    
    #             # Optional callback (e.g. map update)
    #             if post_step_callback:
    #                 await post_step_callback()
            
    #         # Restore initial heading
    #         final_tf = await ts.UnaryAPI.get_actor_transform(self._conn, actor_key)
    #         if final_tf:
    #             initial_forward_norm = normalize(initial_forward)
    #             restore_target = ts.Vector3(
    #                 final_tf.location.x + initial_forward_norm.x * move_distance,
    #                 final_tf.location.y + initial_forward_norm.y * move_distance,
    #                 final_tf.location.z + initial_forward_norm.z * move_distance,
    #             )
    #             await self.move_to(
    #                 actor_id=actor_key,
    #                 target_location=restore_target,
    #                 timeout=3.0,
    #                 tolerance_uu=0.1,
    #             )
    #     await self.explore2(actor_id, post_step_callback=post_step_callback)
    #     return

    async def explore(
        self,
        actor_id: bytes | str | dict,
        *,
        post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """
        Teleport-based rotation sampling: turn left/back/right, then restore initial front; position fixed via ``set_actor_transform``.

        Four perceptions: initial front, then left/back/right; no extra perception after returning to the start heading
        (deduped against the opening front sample).

        Args:
            actor_id: Agent handle
            post_step_callback: Like legacy explore — invoked after each perception (including initial front); not after final restore
        """
        if isinstance(actor_id, dict):
            extracted_id = actor_id.get("id")
            if extracted_id is None:
                extracted_id = actor_id.get("guid")
            actor_key = extracted_id if extracted_id is not None else actor_id
        else:
            actor_key = actor_id

        await ts.UnaryAPI.set_extinguisher_rotation(
            self._conn, actor_key, 0.0, 0.0, timeout=5.0
        )

        initial_state = await ts.UnaryAPI.get_actor_state(self._conn, actor_key)
        if not initial_state:
            return

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
            return
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

        await self.get_perception_object_list(actor_key)
        if post_step_callback:
            await post_step_callback()

        for _label, dir_vec in orientations[1:]:
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
            # await asyncio.sleep(1.0)
            await self.get_perception_object_list(actor_key)
            if post_step_callback:
                await post_step_callback()

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
            # await asyncio.sleep(1.0)

    async def get_perception_object_list(self, agent_id: Optional[bytes | str | dict] = None, timeout: float = 5.0) -> dict:
        """
        Fetch embodied perception for ``agent_id`` and merge into the simplified cache.

        Args:
            agent_id: Agent handle; must not be None (no default ``self._agent`` on ActionAPI)
            timeout: RPC timeout (seconds)

        Returns:
            Dict with "actor_info" and "npc_info"
        """
        if agent_id is None:
            print("[ActionAPI] Error: No agent_id provided")
            result = {"actor_info": [], "npc_info": []}
            self.perception_cache = result
            return result

        # Dict agent handles: require an id field
        if isinstance(agent_id, dict):
            target_agent_id = agent_id.get("id")
            if target_agent_id is None:
                print("[ActionAPI] Error: Agent dict has no 'id' field")
                result = {"actor_info": [], "npc_info": []}
                self.perception_cache = result
                return result
        else:
            target_agent_id = agent_id

        # Pull perception from the sim
        result = await self._perception.get_perception(target_agent_id, timeout=timeout)

        # Simplified cache keeps name + location (+ bbox / burning for actors)
        simplified_cache = {
            "actor_info": [],
            "npc_info": []
        }

        # actor_info: name, location, bounding_box, burning_state
        actor_list = result.get("actor_info", [])
        for actor_info in actor_list:
            actor = actor_info.get("actor", {})
            # actor_name = self._simplify_actor_name(actor.get("name", ""))
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

        # npc_info: name and location (position)
        npc_list = result.get("npc_info", [])
        for npc_info in npc_list:
            object_info = npc_info.get("object_info", {})
            npc_name = object_info.get("name", "")
            npc_position = npc_info.get("position")
            simplified_cache["npc_info"].append({
                "name": npc_name,
                "location": npc_position
            })

        # Merge into rolling _perception_object_list
        if simplified_cache:
            # Initialize container if needed
            if self._perception_object_list is None:
                self._perception_object_list = {}
            
            # Merge new perception data into cached data with deduplication
            # Deduplicate by ``name``; newer rows replace older ones with the same name
            for key, value in simplified_cache.items():
                if key not in self._perception_object_list:
                    self._perception_object_list[key] = value
                else:
                    # Existing key: merge lists, still keyed by name
                    existing_list = self._perception_object_list[key]
                    if isinstance(existing_list, list) and isinstance(value, list):
                        # Dict keyed by name — last write wins
                        name_to_item = {}
                        
                        # Seed from the previous list
                        for item in existing_list:
                            name = item.get("name", "")
                            if name:
                                name_to_item[name] = item
                        
                        # Overlay new rows (same name replaces)
                        for item in value:
                            name = item.get("name", "")
                            if name:
                                name_to_item[name] = item  # newer replaces older
                        
                        # Flatten back to a list
                        self._perception_object_list[key] = list(name_to_item.values())
                    else:
                        # Non-list values: overwrite
                        self._perception_object_list[key] = value

        # Destroyed actors were already filtered inside PerceptionInfo.get_perception; drop stale simplified-cache rows too
        self._perception.filter_simplified_object_list_cache(self._perception_object_list)

        return result
