from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, Optional, List, Union, Tuple, Callable, Awaitable

import numpy as np

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
    # Import action_ol directly when run as a loose script
    import action_ol
    ActionAPI = action_ol.ActionAPI
else:
    from .action_ol import ActionAPI

from EmbodiedMAS.observation import PerceptionInfo

# Blueprints used for spawning actors
FireDog_BP = "/Game/Blueprint/BP_Firedog.BP_Firedog_C"
SaveDog_BP = "/Game/Blueprint/BP_Savedog.BP_Savedog_C"


class BaseAgentOnlyLanguage:
    """
    Language-only agent: no RGB/depth vision.
    Decides from ``PerceptionInfo`` text/summary only.
    """
    
    # Subclasses should override to choose a blueprint path
    _blueprint: str = FireDog_BP

    def __init__(self, context: WorldContext):
        self._context = context
        self._conn = context.conn
        self._actions = ActionAPI(context)
        self._perception = PerceptionInfo(context)
        self._agent: Optional[dict] = None

    async def spawn_agent(
        self,
        *,
        location: Optional[ts.Vector3] = None,
        rotation: Optional[ts.Quaternion] = None,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        timeout: float = 5.0,
    ) -> Optional[dict]:
        """
        Spawn an agent actor (without camera).
        
        Args:
            location: Initial world position
            rotation: Initial orientation quaternion; default rotation if None
            name: Display name
            tags: Actor tags
            timeout: RPC timeout in seconds
        """
        # Build Transform: use rotation when provided, else location only
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
        """Tell the given agent to start following."""
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print("[BaseAgentOnlyLanguage] Error: Agent not spawned and no actor provided")
            return []
        return await self._actions.sendfollow(target_actor)

    async def send_stop_follow(self, actor: Optional[bytes | str | dict] = None) -> list[str]:
        """Tell the given agent to stop following."""
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print("[BaseAgentOnlyLanguage] Error: Agent not spawned and no actor provided")
            return []
        return await self._actions.sendstopfollow(target_actor)


    async def wait(self, actor: Optional[bytes | str | dict] = None) -> None:
        """Wait 5 seconds."""
        await self._actions.wait()

    async def move_to(
        self,
        target_location: ts.Vector3,
        *,
        timeout: float = 60.0,
        tolerance_uu: float = 50.0,
        orientation_mode = None,
    ) -> Optional[dict]:
        """
        Navigate to ``target_location`` (no image capture on this path).
        
        Args:
            target_location: Goal ``Vector3``
            timeout: Seconds
            tolerance_uu: Arrival radius
            orientation_mode: Facing mode; None uses defaults
        
        Returns:
            ``navigate_to_location`` response dict (includes ``success``), or None on failure
        """
        return await self._actions.move_to(
            actor_id=self._agent,
            target_location=target_location,
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
        """Move relative to current heading; ``distance`` is world units (OL does not scale by 100); ``angle`` in degrees (left positive)."""
        if not self._agent:
            print("[BaseAgentOnlyLanguage] Error: Agent not spawned")
            return None
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
        post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """
        In-place rotation sampling: face front / left / back / right (no image capture).
        
        Args:
            post_step_callback: Optional async hook after each heading change
        
        Returns:
            None
        """
        if not self._agent:
            print("[BaseAgentOnlyLanguage] Error: Agent not spawned")
            return
        
        # Delegate to ActionAPI.explore
        await self._actions.explore(
            actor_id=self._agent,
            post_step_callback=post_step_callback,
        )

    async def explore2(
        self,
        *,
        post_step_callback: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """
        Four-way look-around via ``set_actor_transform``: four perceptions (start front + left/back/right); no extra perception after returning to front.
        """
        if not self._agent:
            print("[BaseAgentOnlyLanguage] Error: Agent not spawned")
            return
        await self._actions.explore2(
            actor_id=self._agent,
            post_step_callback=post_step_callback,
        )

    # ----------------
    # Perception functionality
    # ----------------
    
    async def find_actor_by_name(
        self, 
        name_pattern: str, 
        agent_id: Optional[bytes | str | dict] = None,
    ) -> Optional[dict]:
        """
        Find an actor whose name matches ``name_pattern`` in the cached simplified perception dict.
        
        Args:
            name_pattern: Substring match (e.g. "TV" matches names containing "TV")
            agent_id: Agent id; None uses ``self._agent``
        
        Returns:
            Dict with "name" and "location" when found, else None
        """
        # Read from simplified perception cache
        if self._actions._perception_object_list is None:
            self._actions._perception_object_list = {}
        actor_list = self._actions._perception_object_list.get("actor_info", [])
        
        # print(f"[BaseAgentOnlyLanguage] actor_list {actor_list}")
        for actor_info in actor_list:
            actor_name = actor_info.get("name", "")
            # print(f"[BaseAgentOnlyLanguage] name {actor_name}")

            if name_pattern.lower() in actor_name.lower():
                return actor_info
        
        return None
    
    async def get_actor_position(
        self,
        name_pattern: str,
        agent_id: Optional[bytes | str | dict] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Return (x, y) for the first actor matching ``name_pattern``.
        
        Args:
            name_pattern: Substring match
            agent_id: Agent id; None uses ``self._agent``
        
        Returns:
            (x, y) tuple or None
        """
        actor_info = await self.find_actor_by_name(name_pattern, agent_id)
        if actor_info is None:
            return None
        
        # Read location from simplified cache entry
        location = actor_info.get("location")
        if location is None:
            return None
        
        # ``location`` may be a Vector3-like object or a dict
        if hasattr(location, 'x') and hasattr(location, 'y'):
            return (location.x, location.y)
        elif isinstance(location, dict):
            return (location.get("x", 0.0), location.get("y", 0.0))
        
        return None

    def _get_actor_id(self, actor: Optional[bytes | str | dict], class_name: str) -> Union[tuple[bytes | str | dict, bytes | str | dict], tuple[None, str]]:
        """
        Resolve ``actor`` into ``(target_actor, actor_id)`` for ActionAPI calls.
        
        Args:
            actor: Agent handle (bytes, str, dict with id/guid); None uses ``self._agent``
            class_name: For log messages
        
        Returns:
            ``(target_actor, actor_id)`` on success, or ``(None, error_token)`` on failure
        """
        target_actor = actor if actor is not None else self._agent
        if target_actor is None:
            print(f"[{class_name}] Error: Agent not spawned and no actor provided")
            return None, "Fail_NoAgent"
        
        # Dict actors: prefer id/guid field
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
        Forward vector from a quaternion rotation.
        In Unreal, local +X is forward.
        Args:
            rotation: Orientation quaternion
        Returns:
            World-space forward ``Vector3`` with tiny components zeroed
        """
        # Local forward is +X
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
        """Format agent position into text string for LLM."""
        if not agent_position:
            return "AGENT POSITION: Unknown"
        
        pos_x = agent_position.get("x")
        pos_y = agent_position.get("y")
        pos_z = agent_position.get("z")
        forward = agent_position.get("forward")
        if forward:
            forward_x = int(round(forward.x))
            forward_y = int(round(forward.y))
            return f"AGENT POSITION: Current location at (x={pos_x:.1f}, y={pos_y:.1f}, z={pos_z:.1f}), facing forward ({forward_x},{forward_y})"
        else:
            return f"AGENT POSITION: Current location at (x={pos_x:.1f}, y={pos_y:.1f}, z={pos_z:.1f})"

    def format_perception_cache(self) -> str:
        """Format _perception_object_list into a readable text string for LLM."""
        if not self._actions._perception_object_list:
            return "No perception information available."

        lines = ["PERCEPTION INFORMATION (Nearby Objects and NPCs):"]

        # Format actor_info
        actor_list = self._actions._perception_object_list.get("actor_info", [])
        if actor_list:
            lines.append(f"  Actors ({len(actor_list)} found):")
            print(f"[BaseAgentOnlyLanguage] number of actors: {len(actor_list)}")
            for i, actor_info in enumerate(actor_list):
                name = actor_info.get("name", "Unknown")  # Name is already simplified in ActionAPI
                location = actor_info.get("location")
                bounding_box = actor_info.get("bounding_box")
                burning = actor_info.get("burning_state", False)
                if location:
                    if hasattr(location, 'x') and hasattr(location, 'y') and hasattr(location, 'z'):
                        loc_str = f"({location.x:.1f}, {location.y:.1f}, {location.z:.1f})"
                    elif isinstance(location, dict):
                        x = location.get("x")
                        y = location.get("y")
                        z = location.get("z")
                        loc_str = f"({x}, {y}, {z})"
                    else:
                        loc_str = str(location)
                else:
                    loc_str = "Unknown location"

                # Add bounding box info if available
                bbox_str = ""
                if bounding_box and isinstance(bounding_box, dict):
                    min_point = bounding_box.get("min")
                    max_point = bounding_box.get("max")
                    if min_point and max_point:
                        try:
                            if hasattr(min_point, 'x') and hasattr(max_point, 'x'):
                                size_x = max_point.x - min_point.x
                                size_y = max_point.y - min_point.y
                                size_z = max_point.z - min_point.z
                                bbox_str = f"({size_x:.1f}, {size_y:.1f}, {size_z:.1f})"
                            elif isinstance(min_point, dict) and isinstance(max_point, dict):
                                size_x = max_point.get("x") - min_point.get("x")
                                size_y = max_point.get("y") - min_point.get("y")
                                size_z = max_point.get("z") - min_point.get("z")
                                bbox_str = f"({size_x:.1f}, {size_y:.1f}, {size_z:.1f})"
                        except:
                            bbox_str = ", bbox: available"

                lines.append(f"    - {name}: {loc_str}, size: {bbox_str}, burning: {burning}")
        else:
            lines.append("  Actors: None")

        # Format npc_info
        npc_list = self._actions._perception_object_list.get("npc_info", [])
        if npc_list:
            lines.append(f"  NPCs ({len(npc_list)} found):")
            for i, npc_info in enumerate(npc_list):
                name = npc_info.get("name", "Unknown")
                location = npc_info.get("location")
                if location:
                    if hasattr(location, 'x') and hasattr(location, 'y') and hasattr(location, 'z'):
                        loc_str = f"({location.x:.1f}, {location.y:.1f}, {location.z:.1f})"
                    elif isinstance(location, dict):
                        x = location.get("x")
                        y = location.get("y")
                        z = location.get("z")
                        loc_str = f"({x}, {y}, {z})"
                    else:
                        loc_str = str(location)
                else:
                    loc_str = "Unknown location"
                lines.append(f"    - {name}: {loc_str}")
        else:
            lines.append("  NPCs: None")

        return "\n".join(lines)


class FireAgent(BaseAgentOnlyLanguage):
    """
    Firefighting agent subclass; can extinguish fires.
    """
    _blueprint = FireDog_BP

    async def extinguish_fire(
        self, 
        actor: Optional[bytes | str | dict] = None, 
        *, 
        target_location: ts.Vector3,
        timeout: float = 5.0
    ) -> str:
        """
        Aim and extinguish toward ``target_location``.
        
        Args:
            actor: Agent handle; None uses ``self._agent``
            target_location: World point to face before spraying
            timeout: Seconds
        """
        target_actor, actor_id = self._get_actor_id(actor, "FireAgent")
        if target_actor is None:
            return actor_id  # error token string
        
        # ActionAPI.orients the extinguisher and fires
        result = await self._actions.extinguish_fire(
            actor_id=target_actor,
            target_location=target_location,
            timeout=timeout
        )
        
        # Normalize bool to legacy string status
        if isinstance(result, bool):
            return "Success" if result else "Fail_CanotExtinguish"
        return result


class LimitedWaterFireAgent(FireAgent):
    """
    Fire agent with tank/recover parameters configured in-sim via ``SetExtinguisher``.
    After spawn, calls ``UnaryAPI.set_extinguisher`` automatically.
    """

    def __init__(
        self,
        context: WorldContext,
        *,
        water_capacity: int = 100,
        recover_time: int = 10,
    ):
        super().__init__(context)
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
    ) -> Optional[dict]:
        agent = await super().spawn_agent(
            location=location,
            rotation=rotation,
            name=name,
            tags=tags,
            timeout=timeout,
        )

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
        """Re-send current ``water_capacity`` / ``recover_time`` to the sim (debug helper)."""
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


class SaveAgent(BaseAgentOnlyLanguage):
    """
    Rescue agent subclass; provides masks to NPCs.
    """
    _blueprint = SaveDog_BP