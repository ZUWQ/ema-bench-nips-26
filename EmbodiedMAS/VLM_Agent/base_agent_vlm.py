from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Union

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
    # Import action_vlm when run as a loose script
    import action_vlm
    ActionAPI = action_vlm.ActionAPI
else:
    from .action_vlm import ActionAPI
from EmbodiedMAS.observation import ObservationCamera

# Blueprints used for spawning actors
DEFAULT_CAMERA_OFFSET = ts.Vector3(0, 0, 110)

# Fixed camera params for ``CaptureAPI.create_camera``
AGENT_CAMERA_PARAMS: dict[str, Any] = {
    "width": 640,
    "height": 640,
    "fov_degrees": 90.0,
    "enable_color": True,
    "enable_depth": True,
}

FireDog_BP = "/Game/Blueprint/BP_Firedog.BP_Firedog_C"
SaveDog_BP = "/Game/Blueprint/BP_Savedog.BP_Savedog_C"

# Navigation target height (decoupled from perception ``location.z`` elevation flags)
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


class BaseAgentVLM:
    """
    Vision-language agent driven by RGB/depth mosaics and proprioceptive state rather than raw object-name lists.
    """
    
    # Subclasses override to pick a blueprint
    _blueprint: str = FireDog_BP

    def __init__(self, context: WorldContext, log_dir: Optional[Path] = None):
        self._context = context
        self._conn = context.conn
        self._obs_camera = ObservationCamera(context, log_dir=log_dir)
        self._actions = ActionAPI(context, observation_camera=self._obs_camera)
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
        Spawn an agent with a first-person capture camera.
        
        Args:
            location: Initial world position
            rotation: Initial quaternion; default pose if None
            name: Display name
            tags: Actor tags
            timeout: RPC timeout (seconds)
            camera_offset: Camera offset from the actor root
        """
        # Build Transform: include rotation when provided
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
    # Camera information
    # ----------------
    
    async def get_camera_info_and_convert(
        self,
        agent_id: Optional[bytes | str | dict] = None,
        *,
        camera_offset: Optional[ts.Vector3] = None,
    ) -> dict[str, Any]:
        """
        Camera pose metadata (same contract as ``ActionAPI.get_camera_info_and_convert``).

        Args:
            agent_id: Agent handle; None uses ``self._agent``
            camera_offset: Offset from actor root; ``DEFAULT_CAMERA_OFFSET`` when None

        Returns:
            Dict with camera_position, camera_rotation, agent_transform, orientation; ``world_coord`` is always None
        """
        target_agent = agent_id if agent_id is not None else self._agent
        if target_agent is None:
            raise ValueError("No agent_id provided and self._agent is None")

        return await self._actions.get_camera_info_and_convert(
            target_agent,
            camera_offset=camera_offset,
        )

    # ----------------
    # Action functionality
    # ----------------
    async def send_follow(self, actor: Optional[bytes | str | dict] = None) -> list[str]:
        """Issue a follow command for the given agent.
        
        Returns:
            List of NPC ids that accepted the follow RPC
        """
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print("[BaseAgentVision] Error: Agent not spawned and no actor provided")
            return []
        return await self._actions.sendfollow(target_actor)

    async def send_stop_follow(self, actor: Optional[bytes | str | dict] = None) -> list[str]:
        """Issue stop-follow for the given agent.
        
        Returns:
            List of NPC ids that acknowledged stop-follow
        """
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print("[BaseAgentVision] Error: Agent not spawned and no actor provided")
            return []
        return await self._actions.sendstopfollow(target_actor)

    async def wait(self, actor: Optional[bytes | str | dict] = None) -> None:
        """Sleep 5 seconds, then run ``explore`` to refresh the mosaic and sim-aligned perception."""
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print("[BaseAgentVLM] Error: Agent not spawned and no actor provided")
            return
        await self._actions.wait(target_actor)

    async def move_to(
        self,
        pixel_xy: tuple[float, float],
        *,
        timeout: float = 60.0,
        tolerance_uu: float = 50.0,
        orientation_mode = None,
    ) -> Optional[dict]:
        """
        Navigate using ``pixel_xy`` on the latest ``explore`` 2×2 mosaic (px, py).

        Returns:
            ``navigate_to_location`` response dict, or None if back-projection fails
        """
        if not self._agent:
            print("[BaseAgentVLM] Error: Agent not spawned")
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
        Move relative to current heading; semantics match ``ActionAPI.move_by`` (degrees, left positive).
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
        post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        In-place rotation sampling (**not** an LLM-exposed action): used after other moves, inside ``wait``, or during init.

        Returns:
            ``{"mosaic_rgb_path", "rgb_paths", "camera_poses"}`` with empty values on failure
        """
        if not self._agent:
            print("[BaseAgentVLM] Error: Agent not spawned")
            return {"mosaic_rgb_path": None, "rgb_paths": [], "camera_poses": []}

        return await self._actions.explore(
            self._agent,
            image_prefix=image_prefix,
            image_format=image_format,
            post_step_callback=post_step_callback,
        )

    async def ensure_capture_camera(self, params: Optional[dict[str, Any]] = None) -> Optional[bytes | str]:
        """Ensure a capture camera exists for general snapshots."""
        if self._capture_camera_id:
            return self._capture_camera_id

        params = params or {
            "width": 640,
            "height": 640,
            "fov_degrees": 90.0,
            "enable_color": True,
            "enable_depth": True
        }

        self._capture_camera_id = await ts.CaptureAPI.create_camera(
            self._conn,
            transform=ts.Transform(location=ts.Vector3(200, -700, 300)),
            params=params,
            capture_name="BaseAgentVLM_CaptureCam",
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
        Save first-person RGB and/or depth via ``ObservationCamera``.
        
        Args:
            agent_id: dict/str/bytes handle; None uses ``self._agent``
            image_prefix: Filename stem; default when None
            image_format: ``png`` or ``jpg`` for RGB
            save_rgb / save_depth: toggles
        
        Returns:
            ``{"rgb": path|None, "depth": path|None}``
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

    def _get_actor_id(self, actor: Optional[bytes | str | dict], class_name: str) -> Union[tuple[bytes | str | dict, bytes | str | dict], tuple[None, str]]:
        """
        Resolve ``actor`` into ``(target_actor, actor_id)`` for ActionAPI calls.
        
        Args:
            actor: Agent handle; None uses ``self._agent``
            class_name: For log messages
        
        Returns:
            ``(target_actor, actor_id)`` or ``(None, error_token)``
        """
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print(f"[{class_name}] Error: Agent not spawned and no actor provided")
            return None, "Fail_NoAgent"
        
        # Dict handles: read id/guid
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
        Forward vector from quaternion rotation (Unreal local +X).
        Args:
            rotation: Orientation quaternion
        Returns:
            World-space forward ``Vector3`` with tiny components zeroed
        """
        # Local +X is forward
        local_forward = ts.Vector3(1.0, 0.0, 0.0)
        # Rotate into world space
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


class FireAgent(BaseAgentVLM):
    """
    Firefighting VLM agent subclass.
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
        Extinguish toward ``pixel_xy`` on the latest mosaic; omit pixels to keep the (0,0) nozzle pose.
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
    VLM firefighter with tank limits configured via ``SetExtinguisher``; calls ``UnaryAPI.set_extinguisher`` after spawn (post camera rig).
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
        """Re-send ``water_capacity`` / ``recover_time`` to the sim (debug)."""
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


class SaveAgent(BaseAgentVLM):
    """
    Rescue VLM agent (SaveDog blueprint).
    """
    _blueprint = SaveDog_BP