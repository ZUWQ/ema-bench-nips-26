from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List, Union, Tuple, Dict

import tongsim as ts
from tongsim.core.world_context import WorldContext

if __package__ is None or __package__ == "":
    import os
    import sys
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    _ol_agent_dir = os.path.dirname(__file__)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    if _ol_agent_dir not in sys.path:
        sys.path.insert(0, _ol_agent_dir)
    # Import action_h directly when run as a script
    import action_h
    ActionAPI = action_h.ActionAPI
    explore_result_empty = action_h.explore_result_empty
else:
    from .action_h import ActionAPI, explore_result_empty
from EmbodiedMAS.observation import ObservationCamera, PerceptionInfo

# Blueprints used for spawning actors
DEFAULT_CAMERA_OFFSET = ts.Vector3(0, 0, 110)

# Fixed camera params for agent-mounted camera (CaptureAPI.create_camera params)
AGENT_CAMERA_PARAMS: dict[str, Any] = {
    "width": 640,
    "height": 480,
    "fov_degrees": 90.0,
    "enable_color": True,
    "enable_depth": True,
}

FireDog_BP = "/Game/Blueprint/BP_Firedog.BP_Firedog_C"
SaveDog_BP = "/Game/Blueprint/BP_Savedog.BP_Savedog_C"

# Navigation target height (decoupled from perception location.z elevation flags)
Z_HEIGHT = 1000


def _guid_str_to_bytes(guid_str: str) -> bytes:
    """Convert GUID string to FGuid bytes format."""
    s = guid_str.replace("-", "").strip()
    if len(s) != 32:
        return b""
    try:
        raw = bytes.fromhex(s)
        # FGuid: Data1(4) LE, Data2(2) LE, Data3(2) LE, Data4(8) BE
        return raw[0:4][::-1] + raw[4:6][::-1] + raw[6:8][::-1] + raw[8:16]
    except Exception:
        return b""


def _get_agent_id_bytes(agent_id: str | bytes) -> bytes | None:
    """Convert agent ID to bytes format for attach_parent."""
    if isinstance(agent_id, bytes) and len(agent_id) == 16:
        return agent_id
    if isinstance(agent_id, str):
        result = _guid_str_to_bytes(agent_id)
        return result if len(result) == 16 else None
    return None


class BaseAgentHuman:
    """
    Human-style agent using RGB/depth observation.
    Decisions use vision (RGB/depth) and perception info (PerceptionInfo).
    """
    
    # Subclasses should override this to choose a blueprint
    _blueprint: str = FireDog_BP

    def __init__(self, context: WorldContext, log_dir: Optional[Path] = None):
        self._context = context
        self._conn = context.conn
        self._obs_camera = ObservationCamera(context, log_dir=log_dir)
        self._actions = ActionAPI(context, observation_camera=self._obs_camera)
        self._perception = PerceptionInfo(context)
        self._agent: Optional[dict] = None
        self._agent_cameras: dict[str, bytes] = {}
        self._capture_camera_id: Optional[bytes | str] = None

    async def spawn_agent(
        self,
        *,
        location: Optional[ts.Vector3] = None,
        rotation: Optional[ts.Quaternion] = None,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        timeout: float = 5.0,
        camera_offset: Optional[ts.Vector3] = None,
    ) -> Optional[dict]:
        """
        Spawn an agent actor with a first-person camera attached.
        
        Args:
            location: Initial agent location
            rotation: Initial rotation (quaternion); default if None
            name: Agent display name
            tags: Agent tag list
            timeout: Timeout in seconds
            camera_offset: Camera offset relative to the agent root
        """
        # Build transform: use rotation when provided, else location-only
        transform = ts.Transform(location=location, rotation=rotation) if rotation else ts.Transform(location=location)
        agent = await ts.UnaryAPI.spawn_actor(
            self._conn, 
            blueprint=self._blueprint, 
            transform=transform, 
            name=name, 
            tags=tags, 
            timeout=timeout,
        )
        if not agent:
            return None

        agent_id_str = agent.get("id") if isinstance(agent, dict) else str(agent)

        agent_transform = await ts.UnaryAPI.get_actor_transform(self._conn, agent_id_str)

        # Calculate camera position
        offset = camera_offset or DEFAULT_CAMERA_OFFSET
        camera_pos = ts.Vector3(
            agent_transform.location.x + offset.x,
            agent_transform.location.y + offset.y,
            agent_transform.location.z + offset.z,
        )
        print(f"[BaseAgentVLM] camera_pos: {camera_pos}")

        # Create camera
        agent_id_bytes = _get_agent_id_bytes(agent_id_str)
        camera_id = await ts.CaptureAPI.create_camera(
            self._conn,
            transform=ts.Transform(location=camera_pos, rotation=agent_transform.rotation),
            params=dict(AGENT_CAMERA_PARAMS),
            capture_name=f"AgentCamera_{name or agent_id_str}",
            attach_parent=agent_id_bytes,
            attach_socket="",
            keep_world=True,
        )

        if camera_id:
            self._agent_cameras[agent_id_str] = camera_id
            if isinstance(agent, dict):
                agent["camera_id"] = camera_id
            self._obs_camera.register_agent_camera(agent_id_str, camera_id)

        self._agent = agent
        try:
            try:
                from EmbodiedMAS.Metric_Tool.perception_evaluation import (
                    record_query_info_snapshot as _record_query_info_snapshot,
                )
            except ImportError:
                from Metric_Tool.perception_evaluation import (
                    record_query_info_snapshot as _record_query_info_snapshot,
                )
            _qi = await ts.UnaryAPI.query_info(self._conn)
            _record_query_info_snapshot(self._agent, _qi)
        except Exception:
            pass
        return agent

    # ----------------
    # Action functionality
    # ----------------
    async def send_follow(self, actor: Optional[bytes | str | dict] = None) -> list[str]:
        """
        Send a follow command for the given agent.

        Returns:
            list[str]: NPC IDs that accepted the follow command
        """
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print("[BaseAgentVision] Error: Agent not spawned and no actor provided")
            return []
        return await self._actions.sendfollow(target_actor)

    async def send_stop_follow(self, actor: Optional[bytes | str | dict] = None) -> list[str]:
        """
        Send a stop-follow command for the given agent.

        Returns:
            list[str]: NPC IDs that accepted the stop-follow command
        """
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print("[BaseAgentVision] Error: Agent not spawned and no actor provided")
            return []
        return await self._actions.sendstopfollow(target_actor)

    async def move_to(
        self,
        pixel_xy: tuple[float, float],
        *,
        timeout: float = 60.0,
        tolerance_uu: float = 50.0,
        orientation_mode = None,
    ) -> Optional[dict]:
        """
        Navigate to a target: ``pixel_xy`` are mosaic pixel coordinates from the latest ``explore``.

        Returns:
            Response dict from navigate_to_location, or None if back-projection fails.
        """
        if not self._agent:
            print("[BaseAgentHuman] Error: Agent not spawned")
            return None
        return await self._actions.move_to(
            actor_id=self._agent,
            pixel_xy=pixel_xy,
            timeout=timeout,
            tolerance_uu=tolerance_uu,
            orientation_mode=orientation_mode,
        )

    async def move_by(
        self,
        distance: float,
        angle: float,
        *,
        timeout: float = 60.0,
        tolerance_uu: float = 50.0,
        orientation_mode=None,
    ) -> Optional[dict]:
        """
        Move relative to current heading; semantics match ``ActionAPI.move_by``;
        ``angle`` is in degrees (positive = left, negative = right).
        """
        return await self._actions.move_by(
            self._agent,
            distance,
            angle,
            timeout=timeout,
            tolerance_uu=tolerance_uu,
            orientation_mode=orientation_mode,
        )

    async def explore(
        self,
        *,
        image_prefix: Optional[str] = "explore",
        image_format: str = "png",
    ) -> Dict[str, Any]:
        """
        In-place four-way sampling: capture RGB per facing (front/left/back/right); engine depth paths
        go to ``ActionAPI.last_explore_tile_paths``. Tiles are stitched into one **three-column** RGB mosaic
        (1920×1200). Each tile **must** be 640×480 or ``explore`` raises ``ValueError``.
        Layout: left column center = left; middle column top = front, bottom = back (white bar between);
        right column center = right; remaining area is white. Camera poses per facing in
        ``ActionAPI.last_explore_camera_poses``.

        Args:
            image_prefix: Filename prefix for captures (default ``explore``)
            image_format: ``png`` or ``jpg``

        Returns:
            ``{"mosaic_rgb_path", "rgb_paths", "camera_poses"}``; empty structure on failure.
            ``ActionAPI.last_locator_visualization_paths`` holds only ``rgb`` on success.
        """
        if not self._agent:
            print("[BaseAgentHuman] Error: Agent not spawned")
            return explore_result_empty()

        return await self._actions.explore(
            actor_id=self._agent,
            image_prefix=image_prefix,
            image_format=image_format,
            post_step_callback=None,
        )

    async def wait(
        self,
        *,
        image_prefix: Optional[str] = "explore",
        image_format: str = "png",
    ) -> None:
        if not self._agent:
            print("[BaseAgentHuman] Error: Agent not spawned")
            return
        await self._actions.wait(
            self._agent,
            image_prefix=image_prefix,
            image_format=image_format,
            post_step_callback=None,
        )

    async def ensure_capture_camera(self, params: Optional[dict[str, Any]] = None) -> Optional[bytes | str]:
        """Ensure a capture camera exists for general snapshots."""
        if self._capture_camera_id:
            return self._capture_camera_id

        params = params or {
            "width": 640,
            "height": 480,
            "fov_degrees": 90.0,
            "enable_color": True,
            "enable_depth": True
        }

        self._capture_camera_id = await ts.CaptureAPI.create_camera(
            self._conn,
            transform=ts.Transform(location=ts.Vector3(200, -700, 300)),
            params=params,
            capture_name="BaseAgentHuman_CaptureCam",
        )
        return self._capture_camera_id

    async def capture_snapshot(self, include_color: bool = True, include_depth: bool = False) -> Optional[dict[str, Any]]:
        """Capture a snapshot using the capture camera."""
        camera_id = await self.ensure_capture_camera()
        if not camera_id:
            return None
        return await ts.CaptureAPI.capture_snapshot(
            self._conn, camera_id, include_color=include_color, include_depth=include_depth, timeout_seconds=1.0
        )

    async def destroy_capture_camera(self) -> None:
        """Destroy the capture camera."""
        if self._capture_camera_id:
            try:
                await ts.CaptureAPI.destroy_camera(self._conn, self._capture_camera_id)
            finally:
                self._capture_camera_id = None

    # ----------------
    # Image saving
    # ----------------
    async def save_images(
        self,
        agent_id: bytes | str | dict | None = None,
        *,
        image_prefix: Optional[str] = None,
        image_format: str = "png",
        save_rgb: bool = True,
        save_depth: bool = False,
    ) -> dict[str, Optional[str]]:
        """
        Save first-person RGB and/or depth for an agent to disk via ``ObservationCamera``.

        Args:
            agent_id: Agent id (dict, str, or bytes); defaults to ``self._agent``
            image_prefix: Filename prefix; default if None
            image_format: RGB format ``png`` or ``jpg`` (RGB only)
            save_rgb: Whether to save RGB
            save_depth: Whether to save depth

        Returns:
            Dict with ``rgb`` / ``depth`` file paths or None when skipped/failed.
            Example: ``{"rgb": "/path/to/image.png", "depth": "/path/to/depth.exr"}``
        """
        agent = agent_id if agent_id is not None else self._agent
        if agent is None:
            return {"rgb": None, "depth": None}

        results: dict[str, Optional[str]] = {"rgb": None, "depth": None}

        if save_rgb and save_depth:
            results.update(
                await self._obs_camera.capture_rgb_and_depth(
                    agent,
                    image_prefix=image_prefix,
                    image_format=image_format,
                )
            )
        else:
            if save_rgb:
                results["rgb"] = await self._obs_camera.capture_rgb_image(
                    agent,
                    image_prefix=image_prefix,
                    image_format=image_format,
                )
            if save_depth:
                results["depth"] = await self._obs_camera.capture_depth_image(
                    agent,
                    image_prefix=image_prefix,
                )

        return results

    # ----------------
    # Perception functionality
    # ----------------
    
    async def find_actor_by_name(
        self, 
        name_pattern: str, 
        agent_id: Optional[bytes | str | dict] = None,
    ) -> Optional[dict]:
        """
        Find an actor by substring name in the cached perception list.

        Args:
            name_pattern: Substring match (e.g. ``TV`` matches ``**TV**``)
            agent_id: Agent id; defaults to ``self._agent``

        Returns:
            Dict with ``name`` and ``location`` (Vector3 or dict), or None.
        """
        object_list = self._actions._visual_perception_object_list
        if not object_list:
            return None

        for obj_info in object_list:
            obj_name = obj_info.get("name", "")
            # print(f"[BaseAgentHuman] name {obj_name}")

            if name_pattern.lower() in obj_name.lower():
                return obj_info
        
        return None
    
    async def get_actor_position(
        self,
        name_pattern: str,
        agent_id: Optional[bytes | str | dict] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get actor world position (x, y, z) by name substring.

        Args:
            name_pattern: Substring match
            agent_id: Agent id; defaults to ``self._agent``

        Returns:
            ``(x, y, z)`` if found; ``z`` defaults to 0.0 when missing; None if not found.
        """
        actor_info = await self.find_actor_by_name(name_pattern, agent_id)
        if actor_info is None:
            return None

        location = actor_info.get("location")
        if location is None:
            return None

        if hasattr(location, "x") and hasattr(location, "y"):
            z = float(getattr(location, "z", 0.0))
            return (float(location.x), float(location.y), z)
        if isinstance(location, dict):
            return (
                float(location.get("x", 0.0)),
                float(location.get("y", 0.0)),
                float(location.get("z", 0.0)),
            )

        return None

    def _get_actor_id(self, actor: Optional[bytes | str | dict], class_name: str) -> Union[tuple[bytes | str | dict, bytes | str | dict], tuple[None, str]]:
        """
        Resolve ``actor`` to ``(target_actor, actor_id)`` for RPC helpers.

        Args:
            actor: Agent id (bytes, str, dict with ``id``/``guid``); defaults to ``self._agent``
            class_name: Class name for error logs

        Returns:
            ``(target_actor, actor_id)`` on success, or ``(None, error_token)`` on failure.
        """
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print(f"[{class_name}] Error: Agent not spawned and no actor provided")
            return None, "Fail_NoAgent"
        
        # If actor is a dict, pull id/guid
        actor_id = target_actor
        if isinstance(target_actor, dict):
            actor_id = target_actor.get("id") or target_actor.get("guid")
            if actor_id is None:
                print(f"[{class_name}] Error: Actor dict has no 'id' or 'guid' field")
                return None, "Fail_InvalidActor"
        
        return target_actor, actor_id

    # ----------------
    # LLM-related helper methods
    # ----------------
    
    async def get_agent_position_and_forward(self) -> Optional[dict[str, Any]]:
        """Get agent position and forward direction."""
        if not self._agent:
            return None
        agent_id = self._agent.get("id")
        if not agent_id:
            return None
        agent_tf = await ts.UnaryAPI.get_actor_transform(self._conn, agent_id)
        if agent_tf:
            forward_vector = self._get_forward_vector(agent_tf.rotation)
            return {
                "x": agent_tf.location.x,
                "y": agent_tf.location.y,
                "z": agent_tf.location.z,
                "forward": forward_vector,
            }
        return None

    def _get_forward_vector(self, rotation: ts.Quaternion) -> ts.Vector3:
        """
        Forward vector from quaternion rotation.
        In Unreal Engine, local forward is +X.

        Args:
            rotation: Orientation quaternion

        Returns:
            World-space forward ``Vector3`` with tiny components zeroed.
        """
        # Local forward is +X
        local_forward = ts.Vector3(1.0, 0.0, 0.0)
        # Rotate by quaternion
        world_forward = rotation * local_forward

        # Zero near-epsilon components
        threshold = 1e-4
        x = 0.0 if abs(world_forward.x) < threshold else world_forward.x
        y = 0.0 if abs(world_forward.y) < threshold else world_forward.y
        z = 0.0 if abs(world_forward.z) < threshold else world_forward.z

        return ts.Vector3(x, y, z)

    def format_agent_position_text(self, agent_position: Optional[dict[str, Any]]) -> str:
        """Format agent position into text string for LLM (x,y,z in simulation units, no scaling)."""
        if not agent_position:
            return "AGENT POSITION: Unknown"

        pos_x = float(agent_position.get("x", 0.0))
        pos_y = float(agent_position.get("y", 0.0))
        pos_z = float(agent_position.get("z", 0.0))
        forward = agent_position.get("forward")
        if forward:
            forward_x = int(round(forward.x))
            forward_y = int(round(forward.y))
            return (
                f"AGENT POSITION: Current location at (x={pos_x:.1f}, y={pos_y:.1f}, z={pos_z:.1f}), "
                f"facing forward ({forward_x},{forward_y})"
            )
        return f"AGENT POSITION: Current location at (x={pos_x:.1f}, y={pos_y:.1f}, z={pos_z:.1f})"

    def format_perception_cache(self) -> str:
        """Format _visual_perception_object_list for LLM (simulation coordinates, no scaling)."""
        if not self._actions._visual_perception_object_list:
            return "No perception information available."

        lines = ["PERCEPTION INFORMATION (Nearby Objects and NPCs):"]

        # New list layout: iterate entries directly
        object_list = self._actions._visual_perception_object_list
        if object_list:
            lines.append(f"  Objects ({len(object_list)} found):")
            for obj_info in object_list:
                name = obj_info.get("name", "Unknown")
                location = obj_info.get("location")

                if location:
                    if hasattr(location, "x") and hasattr(location, "y"):
                        x = float(location.x)
                        y = float(location.y)
                        if hasattr(location, "z"):
                            z = float(location.z)
                            loc_str = f"({x:.1f}, {y:.1f}, {z:.1f})"
                        else:
                            loc_str = f"({x:.1f}, {y:.1f})"
                    elif isinstance(location, dict):
                        x = float(location.get("x", 0.0))
                        y = float(location.get("y", 0.0))
                        if "z" in location:
                            z = float(location["z"])
                            loc_str = f"({x:.1f}, {y:.1f}, {z:.1f})"
                        else:
                            loc_str = f"({x:.1f}, {y:.1f})"
                    else:
                        loc_str = str(location)
                else:
                    loc_str = "Unknown location"

                lines.append(f"    - {name}: {loc_str}")
        else:
            lines.append("  Objects: None")

        return "\n".join(lines)

    def format_object_name_list_text(self) -> str:
        """Format object names only (no coordinates) for multimodal LLM; list source matches format_perception_cache."""
        if not self._actions._visual_perception_object_list:
            return "No perception information available."

        object_list = self._actions._visual_perception_object_list
        print(f"[BaseAgentHuman] object_list: {object_list}")
        lines = ["OBJECT NAMES (detected nearby):"]
        lines.append(f"  Count: {len(object_list)}")
        for obj_info in object_list:
            name = obj_info.get("name", "Unknown")
            lines.append(f"    - {name}")
        return "\n".join(lines)

    async def close(self) -> None:
        """Clean up all resources."""
        await self.destroy_capture_camera()
        for agent_id, camera_id in list(self._agent_cameras.items()):
            try:
                await ts.CaptureAPI.destroy_camera(self._conn, camera_id)
            except Exception:
                pass
            finally:
                self._obs_camera.unregister_agent_camera(agent_id)
        self._agent_cameras.clear()


class FireAgent(BaseAgentHuman):
    """
    Firefighting agent (inherits ``BaseAgentHuman``) with extinguish support.
    """
    _blueprint = FireDog_BP

    async def extinguish_fire(
        self,
        actor: Optional[bytes | str | dict] = None,
        *,
        pixel_xy: Optional[tuple[float, float]] = None,
        timeout: float = 5.0,
    ) -> str:
        """
        Run extinguish for the resolved agent.

        ``pixel_xy`` comes from the latest ``explore`` mosaic; if omitted, ``ActionAPI`` uses nozzle pose (0, 0).
        """
        target_actor, actor_id = self._get_actor_id(actor, "FireAgent")
        if target_actor is None:
            return actor_id  # error token

        result = await self._actions.extinguish_fire(
            actor_id=target_actor,
            pixel_xy=pixel_xy,
            timeout=timeout,
        )
        if isinstance(result, bool):
            return "Success" if result else "Fail_CanotExtinguish"
        return result


class LimitedWaterFireAgent(FireAgent):
    """
    Vision-language fire agent with limited water; capacity/recover come from ``UnaryAPI.set_extinguisher``
    after spawn (parent camera setup completes first).
    """

    def __init__(
        self,
        context: WorldContext,
        *,
        water_capacity: int = 100,
        recover_time: int = 10,
        log_dir: Optional[Path] = None,
    ):
        super().__init__(context, log_dir=log_dir)
        self._water_capacity = water_capacity
        self._recover_time = recover_time

    async def spawn_agent(
        self,
        *,
        location: Optional[ts.Vector3] = None,
        rotation: Optional[ts.Quaternion] = None,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        timeout: float = 5.0,
        camera_offset: Optional[ts.Vector3] = None,
    ) -> Optional[dict]:
        agent = await super().spawn_agent(
            location=location,
            rotation=rotation,
            name=name,
            tags=tags,
            timeout=timeout,
            camera_offset=camera_offset,
        )
        if not agent:
            return None
        target_actor, actor_id = self._get_actor_id(self._agent, "LimitedWaterFireAgent")
        ok = await ts.UnaryAPI.set_extinguisher(
            self._conn,
            actor_id,
            self._water_capacity,
            self._recover_time,
            timeout=timeout,
        )
        if ok is None or ok is False:
            print(
                f"[LimitedWaterFireAgent] Warning: set_extinguisher failed "
                f"(water_capacity={self._water_capacity}, recover_time={self._recover_time})"
            )
        return agent

    async def reapply_extinguisher_config(self, timeout: float = 5.0) -> bool:
        """Re-send current ``water_capacity`` / ``recover_time`` to the simulator (debug)."""
        target_actor, actor_id = self._get_actor_id(self._agent, "LimitedWaterFireAgent")
        if target_actor is None:
            return False
        ok = await ts.UnaryAPI.set_extinguisher(
            self._conn,
            actor_id,
            self._water_capacity,
            self._recover_time,
            timeout=timeout,
        )
        return bool(ok)


class SaveAgent(BaseAgentHuman):
    """
    Rescue agent using the SaveDog blueprint.
    """
    _blueprint = SaveDog_BP