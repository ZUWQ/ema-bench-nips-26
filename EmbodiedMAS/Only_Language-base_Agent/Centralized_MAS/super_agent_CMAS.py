from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict

import tongsim as ts
from tongsim.core.world_context import WorldContext
from openai import OpenAI

# Allow running as a script and as a module
if __package__ is None or __package__ == "":
    import os
    import sys
    _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _ol_agent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    if _ol_agent_dir not in sys.path:
        sys.path.insert(0, _ol_agent_dir)
    try:
        from Metric_Tool import llm_scene_timing
    except ImportError:
        from EmbodiedMAS.Metric_Tool import llm_scene_timing  # type: ignore
    # Import base_agent_only_language directly when resolving paths as a script
    import base_agent_only_language
    FireAgent = base_agent_only_language.FireAgent
    SaveAgent = base_agent_only_language.SaveAgent
    LimitedWaterFireAgent = base_agent_only_language.LimitedWaterFireAgent
    import llm_config  # API profiles live here; do not commit secrets (see Automation_runner privacy note)
    import llm_config
    get_llm_config = llm_config.get_llm_config
else:
    from ..base_agent_only_language import FireAgent, SaveAgent, LimitedWaterFireAgent  # type: ignore
    from ...llm_config import get_llm_config  # type: ignore
    from ...Metric_Tool import llm_scene_timing  # type: ignore

# Global z-height constant for agent spawning
Z_HEIGHT = 1000

_DEFAULT_PROMPT_ENV: dict[str, Any] = {
    "num_agents": 3,
    "num_fire_agents": 2,
    "num_rescue_agents": 1,
    "num_civilians": 4,
    "num_fires": "unknown",
    "other_info": "",
}


class SuperLLMController:
    """
    Centralized LLM planner for multiple robots.
    Calls OpenAI using fused perception text from every robot each step.
    """

    def __init__(
        self,
        max_memory_size: int = 3,
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        # Load LLM configuration
        self._llm_config = get_llm_config()
        
        # Initialize OpenAI client with configuration
        client_kwargs = self._llm_config.get_client_kwargs()
        self._client = OpenAI(**client_kwargs)
        
        # Fully centralized super — assemble_prompt("super", "CENTRAL", env)
        self._system_prompt = self._load_system_prompt(prompt_environment)
        
        # Initialize memory for action history (one entry per super step, aggregated per_agent)
        self._memory: list[dict[str, Any]] = []
        self._step_counter = 0
        self._max_memory_size = max_memory_size
        
        # Store current step's summary (only the last step)
        self._current_summary: Optional[str] = None
        self._current_high_level_plan: Optional[str] = None

        self._agent_id = agent_id
        self._session_id: Optional[str] = session_id
        self._session_memory_path: Optional[Path] = None
        self._session_timing_path: Optional[Path] = None
        self._session_memory_header_written: bool = False
        self._session_timing_header_written: bool = False
        self._step_timing_events: list[dict[str, Any]] = []
    
    def _load_system_prompt(self, prompt_environment: Optional[dict[str, Any]]) -> str:
        """assemble_prompt(super, CENTRAL, environment) — see EmbodiedMAS/prompt/verify_prompt.py."""
        import os
        import sys

        _mas_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _mas_root not in sys.path:
            sys.path.insert(0, _mas_root)
        from prompt.prompt_assembler import assemble_prompt

        env = dict(_DEFAULT_PROMPT_ENV)
        if prompt_environment:
            env.update(prompt_environment)
        return assemble_prompt("super", "CENTRAL", env)

    def clear_step_timing(self) -> None:
        self._step_timing_events.clear()

    def record_timing_event(self, event: dict[str, Any]) -> None:
        self._step_timing_events.append(event)

    def flush_session_log(self, log_round: int, agent_id: Optional[str] = None) -> Optional[Path]:
        """Append MEMORY to session file and TIMING to a separate file in the same logs directory."""
        log_dir = llm_scene_timing.resolve_benchmark_log_dir()
        aid = agent_id if agent_id is not None else self._agent_id
        if self._session_id is None:
            self._session_id = llm_scene_timing.ensure_session_id(None)
        sid = self._session_id
        mem_path = llm_scene_timing.session_memory_path(log_dir, aid, sid)
        tim_path = llm_scene_timing.session_timing_path(log_dir, aid, sid)
        self._session_memory_path = mem_path
        self._session_timing_path = tim_path

        if not self._memory:
            memory_lines = ["(empty)"]
        else:
            memory_lines = [self._format_memory_entry(mem).strip() for mem in self._memory]

        mem_header: Optional[str] = None
        if not self._session_memory_header_written:
            self._session_memory_header_written = True
            mem_header = (
                f"SESSION kind=memory agent_id={aid!r} session_id={sid!r} "
                f"file_started={datetime.now().isoformat()}"
            )

        tim_header: Optional[str] = None
        if not self._session_timing_header_written:
            self._session_timing_header_written = True
            tim_header = (
                f"SESSION kind=timing agent_id={aid!r} session_id={sid!r} "
                f"file_started={datetime.now().isoformat()}"
            )

        llm_scene_timing.append_memory_block(
            mem_path,
            log_round=log_round,
            memory_lines=memory_lines,
            session_header=mem_header,
        )
        llm_scene_timing.append_timing_block(
            tim_path,
            log_round=log_round,
            timing_events=list(self._step_timing_events),
            session_header=tim_header,
        )
        self._step_timing_events.clear()
        return mem_path

    @staticmethod
    def _robot_id_sort_key(rid: str) -> tuple[int, str]:
        try:
            return (int(rid.replace("agent_", "")), rid)
        except ValueError:
            return (0, rid)

    def _format_memory_entry(self, mem: dict[str, Any], is_detailed: bool = False) -> str:
        """One line per super step: all agents' actions and feedback under per_agent."""
        parts: list[str] = [f"step={mem.get('step')}"]
        per_agent = mem.get("per_agent")
        if isinstance(per_agent, dict) and per_agent:
            agent_bits: list[str] = []
            for rid in sorted(per_agent.keys(), key=self._robot_id_sort_key):
                info = per_agent[rid]
                if not isinstance(info, dict):
                    continue
                action = info.get("action", "?")
                sub = [f"{rid}:{action}"]
                if action_params := info.get("action_params"):
                    if (tn := action_params.get("target_name")) is not None:
                        sub.append(f"params={tn}")
                if "move_to_success" in info and info.get("move_to_success") is not None:
                    sub.append(f"move_ok={info['move_to_success']}")
                elif "move_to_success" in info:
                    sub.append("move_ok=None")
                if "sendfollow_received" in info:
                    sub.append(f"follow_rx={info['sendfollow_received']}")
                if "sendstopfollow_received" in info:
                    sub.append(f"stopfollow_rx={info['sendstopfollow_received']}")
                if (er := info.get("extinguish_result")) is not None:
                    sub.append(f"extinguish={er!r}")
                agent_bits.append("[" + " ".join(sub) + "]")
            if agent_bits:
                parts.append("agents=" + " ".join(agent_bits))
        if (p := mem.get("high_level_plan")):
            parts.append(f"plan={p!r}")
        if (s := mem.get("summary")):
            parts.append(f"summary={s!r}")
        return "  " + " | ".join(parts)

    def _format_memory(self) -> str:
        """Format memory (action history) into a readable text string."""
        if not self._memory:
            return "No previous actions."

        lines = [f"Previous actions (last {len(self._memory)} actions):"]
        for mem in self._memory:
            lines.append(self._format_memory_entry(mem))

        return "\n".join(lines)

    def _parse_json_from_response(self, response_text: str) -> Any:
        """Parse JSON from LLM response text, handling markdown code blocks."""
        response_text = response_text.strip()

        # Extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()

        return json.loads(response_text)


    def _parse_response(self, response_text: str, robot_ids: List[str]) -> Dict[str, dict[str, Any]]:
        """
        Parse response from LLM based on centralized format.
        
        Expected format:
        {
          "summary": "...",
          "high_level_plan": "...",
          "robot_actions": {
            "agent_1": {
              "action": "...",
              "action_parameter": {...}
            },
            "agent_2": {...},
            ...
          }
        }
        
        Args:
            response_text: Raw response text from LLM containing JSON
            robot_ids: List of robot IDs that should have actions assigned
            
        Returns:
            Dictionary mapping robot_id to action dictionary
        """
        response_text = response_text.strip()
        robot_actions: Dict[str, dict[str, Any]] = {}
        
        # Parse JSON from response
        response_json = self._parse_json_from_response(response_text)
        
        if not isinstance(response_json, dict):
            print(f"[SuperLLMController] Response is not a JSON object: {type(response_json)}")
            return {}
        
        # Extract fields from centralized format
        summary = response_json.get("summary", "")
        high_level_plan = response_json.get("high_level_plan", "")
        
        # Save current step's summary and plan
        self._current_summary = summary if summary else None
        self._current_high_level_plan = high_level_plan if high_level_plan else None
        
        # Extract robot_actions
        robot_actions_dict = response_json.get("robot_actions", {})
        
        if not isinstance(robot_actions_dict, dict):
            print(f"[SuperLLMController] robot_actions is not a dictionary: {type(robot_actions_dict)}")
            return {}
        
        # Parse each robot's action
        for robot_id in robot_ids:
            if robot_id not in robot_actions_dict:
                print(f"[SuperLLMController] Warning: No action assigned to {robot_id}")
                continue
            
            robot_action_data = robot_actions_dict[robot_id]
            if not isinstance(robot_action_data, dict):
                print(f"[SuperLLMController] Action data for {robot_id} is not a dictionary")
                continue
            
            action = robot_action_data.get("action")
            action_parameter = robot_action_data.get("action_parameter")
            
            if not action:
                print(f"[SuperLLMController] Warning: No action specified for {robot_id}")
                continue
            
            # Build action dictionary
            action_dict = {
                "action": action,
                "high_level_plan": high_level_plan,
                "summary": summary,
            }
            
            # Parse action parameters based on action type
            if action == "move_to":
                if action_parameter and isinstance(action_parameter, dict):
                    target_x = action_parameter.get("x")
                    target_y = action_parameter.get("y")
                    if target_x is not None and target_y is not None:
                        action_dict["target_x"] = target_x
                        action_dict["target_y"] = target_y
                    if (tn := action_parameter.get("name")) is not None:
                        action_dict["target_name"] = tn
            elif action == "extinguish_fire":
                if action_parameter and isinstance(action_parameter, dict):
                    if (tn := action_parameter.get("name")) is not None:
                        action_dict["target_name"] = tn

            robot_actions[robot_id] = action_dict
        
        return robot_actions

    def _build_llm_content(
        self,
        all_agent_positions: Dict[str, str],
        global_perception_text: str,
        robot_ids: List[str]
    ) -> list[dict[str, Any]]:
        """Build content items for LLM call with multi-robot information."""
        content_items = []

        # Memory
        content_items.append({"type": "text", "text": f"MEMORY:\n{self._format_memory()}"})

        # Active Robot List
        robot_list_str = ", ".join([f'"{rid}"' for rid in robot_ids])
        content_items.append({"type": "text", "text": f"ACTIVE ROBOT LIST: [{robot_list_str}]"})

        # All agent positions
        position_lines = ["ROBOT STATES:"]
        for robot_id, position_text in all_agent_positions.items():
            position_lines.append(f"  {robot_id}: {position_text}")
        content_items.append({"type": "text", "text": "\n".join(position_lines)})

        # Global perception cache
        content_items.append({"type": "text", "text": global_perception_text})

        return content_items

    async def _call_llm(
        self,
        all_agent_positions: Dict[str, str],
        global_perception_text: str,
        robot_ids: List[str]
    ) -> Dict[str, dict[str, Any]]:
        """
        Call LLM with multi-robot perception information.
        
        Args:
            all_agent_positions: Dictionary mapping robot_id to position text
            global_perception_text: Combined perception information from all robots
            robot_ids: List of active robot IDs
            
        Returns:
            Dictionary mapping robot_id to action dictionary
        """
        content_items = self._build_llm_content(all_agent_positions, global_perception_text, robot_ids)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content_items}
        ]

        # Run OpenAI API call with configuration
        model = self._llm_config.get_model()
        
        def _sync_call():
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
            )

        completion = await asyncio.to_thread(_sync_call)
        response_text = completion.choices[0].message.content

        if response_text is None:
            raise ValueError("LLM returned None response")

        print(f"[SuperLLMController] LLM Response: {response_text}")
        return self._parse_response(response_text, robot_ids)

    def get_current_summary(self) -> Optional[str]:
        """Get the current step's summary."""
        return self._current_summary

    def get_current_high_level_plan(self) -> Optional[str]:
        """Get the current step's high-level plan."""
        return self._current_high_level_plan

    def save_memory(self, last_step_result: Optional[dict[str, Any]] = None, agent_id: Optional[str] = None) -> Optional[Path]:
        """Legacy alias: append one session block (log_round=0 if no step context)."""
        return self.flush_session_log(log_round=0, agent_id=agent_id)


class SuperLLMAgent:
    """
    Centralized multi-robot LLM driver managing several ``FireAgent`` / ``SaveAgent`` / ``LimitedWaterFireAgent`` instances.
    ``SuperLLMController`` plans once per step, then actions execute in parallel across robots.
    """
    
    def __init__(
        self,
        context: WorldContext,
        n_agents: int = 3,
        controller: Optional[SuperLLMController] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        memory_log_agent_id: str = "super_agent",
        *,
        water_capacity: int = 1,
        recover_time: int = 10,
    ):
        """
        Initialize Super LLM Agent.
        
        Args:
            context: WorldContext instance
            n_agents: Number of agents to manage (default: 3)
            controller: Optional SuperLLMController instance, if None will create a new one
            prompt_environment: Optional overrides for prompt placeholder slots
            session_id: Optional fixed session id for append-only memory/timing logs
            memory_log_agent_id: agent_id label in session log filenames (default: super_agent)
            water_capacity: SetExtinguisher capacity for each ``FD_WL`` agent (default: 5)
            recover_time: Water recovery seconds for each ``FD_WL`` agent (default: 10)
        """
        self._context = context
        self._conn = context.conn
        self._n_agents = n_agents
        self._water_capacity = int(water_capacity)
        self._recover_time = int(recover_time)
        self._memory_log_agent_id = memory_log_agent_id
        self._log_round = 0
        self.controller = controller if controller is not None else SuperLLMController(
            prompt_environment=prompt_environment,
            session_id=session_id,
            agent_id=memory_log_agent_id,
        )
        if controller is not None:
            controller._agent_id = memory_log_agent_id
            if session_id is not None:
                controller._session_id = session_id
        
        # Store agent instances, keyed by robot_id (agent_1, agent_2, ...)
        self._agents: Dict[str, FireAgent | SaveAgent | LimitedWaterFireAgent] = {}
        self._agent_types: Dict[str, str] = {}
        self._step_counter = 0

    @staticmethod
    def _valid_actions_for_agent_type(agent_type: str) -> set[str]:
        """Allowed tool names for FD / FD_WL vs SD (matches single-agent OL)."""
        fire_actions = {
            "move_to",
            "extinguish_fire",
            "send_follow",
            "send_stop_follow",
            "explore",
            "wait",
        }
        if agent_type in ("FD", "FD_WL"):
            return fire_actions
        if agent_type == "SD":
            return fire_actions - {"extinguish_fire"}
        return fire_actions
    
    async def spawn_agents(
        self,
        agent_positions: List[Dict[str, Any]],
        agent_types: List[str]
    ) -> None:
        """
        Spawn n FireAgent, LimitedWaterFireAgent (FD_WL), or SaveAgent instances.
        
        Args:
            agent_positions: Required list of position dictionaries for each agent.
                            Each dict should have "x", "y", "z", and optionally "rotation"
            agent_types: Required list of agent types for each agent.
                        Each element should be "FD", "FD_WL", or "SD"
        """
        for i in range(self._n_agents):
            robot_id = f"agent_{i+1}"
            agent_type = agent_types[i]
            
            if agent_type == "FD":
                agent = FireAgent(self._context)
            elif agent_type == "FD_WL":
                agent = LimitedWaterFireAgent(
                    self._context,
                    water_capacity=self._water_capacity,
                    recover_time=self._recover_time,
                )
            elif agent_type == "SD":
                agent = SaveAgent(self._context)
            else:
                raise ValueError(
                    f"Invalid agent_type: {agent_type}. Must be 'FD', 'FD_WL', or 'SD'"
                )
            
            # Get agent position and rotation
            pos_config = agent_positions[i]
            location = ts.Vector3(
                pos_config.get("x"),
                pos_config.get("y"),
                pos_config.get("z")
            )
            rotation_config = pos_config.get("rotation")
            if rotation_config:
                rotation = ts.math.euler_to_quaternion(
                    ts.Vector3(rotation_config),
                    is_degree=True
                )
            else:
                rotation = None

            # Spawn agent
            agent_actor = await agent.spawn_agent(
                name=f"SuperLLMAgent_{i+1}",
                location=location,
                rotation=rotation
            )
            
            agent._agent = agent_actor
            self._agents[robot_id] = agent
            self._agent_types[robot_id] = agent_type
            print(f"[SuperLLMAgent] Spawned {robot_id}: {agent_actor.get('id') if isinstance(agent_actor, dict) else agent_actor}")

    
    def _get_robot_ids(self) -> List[str]:
        """Get list of active robot IDs."""
        return [f"agent_{i+1}" for i in range(self._n_agents) if f"agent_{i+1}" in self._agents]
    
    async def _collect_all_perceptions(self) -> Dict[str, str]:
        """
        Collect perception information from all agents.
        
        Returns:
            Dictionary mapping robot_id to perception text
        """
        perceptions = {}
        for robot_id, agent in self._agents.items():
            if agent._agent:
                await agent._actions.get_perception_object_list(agent._agent)
                perception_text = agent.format_perception_cache()
                perceptions[robot_id] = perception_text
        return perceptions
    
    def _merge_perceptions(self, all_perceptions: Dict[str, str]) -> str:
        """
        Merge perception information from all agents into a global view.
        
        Args:
            all_perceptions: Dictionary mapping robot_id to perception text
            
        Returns:
            Combined perception text
        """
        lines = ["PERCEPTION INFORMATION (Global View - All Robots):"]
        
        # Collect all unique objects (by name only - new data replaces old data if name matches)
        all_objects = {}  # key: name, value: object_info
        
        for robot_id, perception_text in all_perceptions.items():
            agent = self._agents[robot_id]
            if agent._actions._perception_object_list:
                actor_list = agent._actions._perception_object_list.get("actor_info", [])
                for actor_info in actor_list:
                    name = actor_info.get("name", "Unknown")
                    if name:
                        # Same object name → keep the newest observation
                        all_objects[name] = actor_info
        
        # Format merged objects
        if all_objects:
            lines.append(f"  Actors ({len(all_objects)} unique objects found):")
            for name, actor_info in all_objects.items():
                location = actor_info.get("location")
                bounding_box = actor_info.get("bounding_box")
                burning = actor_info.get("burning_state", False)
                
                # Extract location for display
                if location:
                    if hasattr(location, 'x') and hasattr(location, 'y'):
                        x, y = location.x, location.y
                    elif isinstance(location, dict):
                        x, y = location.get("x", 0.0), location.get("y", 0.0)
                    else:
                        x, y = 0.0, 0.0
                else:
                    x, y = 0.0, 0.0
                
                loc_str = f"({x:.1f}, {y:.1f})"
                
                bbox_str = ""
                if bounding_box and isinstance(bounding_box, dict):
                    min_point = bounding_box.get("min")
                    max_point = bounding_box.get("max")
                    if min_point and max_point:
                        try:
                            if hasattr(min_point, 'x') and hasattr(max_point, 'x'):
                                size_x = max_point.x - min_point.x
                                size_y = max_point.y - min_point.y
                                bbox_str = f"({size_x:.1f}, {size_y:.1f})"
                            elif isinstance(min_point, dict) and isinstance(max_point, dict):
                                size_x = max_point.get("x", 0) - min_point.get("x", 0)
                                size_y = max_point.get("y", 0) - min_point.get("y", 0)
                                bbox_str = f"({size_x:.1f}, {size_y:.1f})"
                        except:
                            bbox_str = ", bbox: available"
                
                lines.append(f"    - {name}: {loc_str}, size: {bbox_str}, burning: {burning}")
        else:
            lines.append("  Actors: None")
        
        # Collect all NPCs (by name only - new data replaces old data if name matches)
        all_npcs = {}
        for robot_id, perception_text in all_perceptions.items():
            agent = self._agents[robot_id]
            if agent._actions._perception_object_list:
                npc_list = agent._actions._perception_object_list.get("npc_info", [])
                for npc_info in npc_list:
                    name = npc_info.get("name", "Unknown")
                    if name:
                        # Same object name → keep the newest observation
                        all_npcs[name] = npc_info
        
        if all_npcs:
            lines.append(f"  NPCs ({len(all_npcs)} unique NPCs found):")
            for name, npc_info in all_npcs.items():
                location = npc_info.get("location")
                if location:
                    if hasattr(location, 'x') and hasattr(location, 'y'):
                        loc_str = f"({location.x:.1f}, {location.y:.1f})"
                    elif isinstance(location, dict):
                        x = location.get("x", 0.0)
                        y = location.get("y", 0.0)
                        loc_str = f"({x:.1f}, {y:.1f})"
                    else:
                        loc_str = str(location)
                else:
                    loc_str = "Unknown location"
                lines.append(f"    - {name}: {loc_str}")
        else:
            lines.append("  NPCs: None")
        
        return "\n".join(lines)
    
    async def _collect_all_positions(self) -> Dict[str, str]:
        """
        Collect position information from all agents.
        
        Returns:
            Dictionary mapping robot_id to position text
        """
        positions = {}
        for robot_id, agent in self._agents.items():
            if agent._agent:
                agent_position = await agent.get_agent_position_and_forward()
                position_text = agent.format_agent_position_text(agent_position)
                positions[robot_id] = position_text
        return positions
    
    async def step(self) -> None:
        """
        Perform one step: collect all agent perceptions, call LLM for centralized decision,
        then execute actions in parallel.
        """
        robot_ids = self._get_robot_ids()
        if not robot_ids:
            print("[SuperLLMAgent] No agents available")
            return

        self._log_round += 1
        self.controller.clear_step_timing()
        
        # Update perception information for all agents
        print("[SuperLLMAgent] Collecting perceptions from all agents...")
        all_perceptions = await self._collect_all_perceptions()
        global_perception_text = self._merge_perceptions(all_perceptions)
        # print(f"[SuperLLMAgent] Global perception text: {global_perception_text}")

        # Collect all agent positions
        all_positions = await self._collect_all_positions()
        
        # Call LLM for centralized decision
        print("[SuperLLMAgent] Calling LLM for centralized decision...")
        robot_actions = await self.controller._call_llm(
            all_agent_positions=all_positions,
            global_perception_text=global_perception_text,
            robot_ids=robot_ids
        )
        
        if not robot_actions:
            print("[SuperLLMAgent] No actions received from LLM")
            self.controller.flush_session_log(
                log_round=self._log_round,
                agent_id=self._memory_log_agent_id,
            )
            return
        
        # Execute actions in parallel
        await self._execute_actions_parallel(robot_actions)
        self.controller.flush_session_log(
            log_round=self._log_round,
            agent_id=self._memory_log_agent_id,
        )
    
    async def _execute_actions_parallel(self, robot_actions: Dict[str, dict[str, Any]]) -> None:
        """
        Execute actions for all robots in parallel.
        
        Args:
            robot_actions: Dictionary mapping robot_id to action dictionary
        """
        self._step_counter += 1
        current_step = self._step_counter

        per_agent: Dict[str, dict[str, Any]] = {}
        for robot_id, action_dict in robot_actions.items():
            if robot_id not in self._agents:
                print(f"[SuperLLMAgent] Warning: Unknown robot_id {robot_id}")
                continue

            agent_type = self._agent_types.get(robot_id)
            action = action_dict.get("action")
            if action not in self._valid_actions_for_agent_type(agent_type):
                print(
                    f"[SuperLLMAgent] Warning: Invalid action for {robot_id} ({agent_type}): {action}, not recorded"
                )
                continue

            action_params: dict[str, Any] = {}
            if action == "move_to":
                target_x = action_dict.get("target_x")
                target_y = action_dict.get("target_y")
                target_name = action_dict.get("target_name")
                if target_x is not None and target_y is not None:
                    action_params = {
                        "target_x": target_x,
                        "target_y": target_y,
                        "target_z": Z_HEIGHT,
                        "target_name": target_name,
                    }
            elif action == "extinguish_fire":
                target_name = action_dict.get("target_name")
                action_params = {"target_name": target_name}

            per_agent[robot_id] = {
                "action": action,
                "action_params": action_params,
            }

        if per_agent:
            memory_entry: dict[str, Any] = {
                "step": current_step,
                "summary": self.controller.get_current_summary() or "",
                "high_level_plan": self.controller.get_current_high_level_plan() or "",
                "per_agent": per_agent,
                "timestamp": datetime.now().isoformat(),
            }
            self.controller._memory.append(memory_entry)
            while len(self.controller._memory) > self.controller._max_memory_size:
                self.controller._memory.pop(0)

        def _update_memory(robot_id: str, step: int, **kwargs):
            for mem in reversed(self.controller._memory):
                if mem.get("step") != step:
                    continue
                pa = mem.get("per_agent")
                if not isinstance(pa, dict) or robot_id not in pa:
                    continue
                pa[robot_id].update(kwargs)
                break
        
        # Create tasks for parallel execution
        async def execute_agent_action(robot_id: str, action_dict: dict[str, Any]):
            """Execute action for a single agent."""
            if robot_id not in self._agents:
                print(f"[SuperLLMAgent] Error: Robot {robot_id} not found")
                return
            
            agent = self._agents[robot_id]
            if not agent._agent:
                print(f"[SuperLLMAgent] Error: Agent {robot_id} not spawned")
                return

            agent_type = self._agent_types.get(robot_id)
            action = action_dict.get("action")
            if action not in self._valid_actions_for_agent_type(agent_type):
                print(f"[SuperLLMAgent] Skipping invalid action for {robot_id} ({agent_type}): {action}")
                return

            if action == "move_to":
                name = action_dict.get("target_name")
                print(f"[SuperLLMAgent] Executing {robot_id}: {action} to {name}")
            else:
                print(f"[SuperLLMAgent] Executing {robot_id}: {action}")
            
            try:
                if action == "move_to":
                    target_x = action_dict.get("target_x")
                    target_y = action_dict.get("target_y")
                    if target_x is None or target_y is None:
                        print(f"[SuperLLMAgent] Error: {robot_id} move_to missing target_x or target_y")
                        return
                    
                    target_location = ts.Vector3(float(target_x), float(target_y), Z_HEIGHT)
                    nav = await agent.move_to(
                        target_location=target_location,
                        timeout=60.0,
                        tolerance_uu=50.0
                    )
                    move_ok = nav.get("success") if isinstance(nav, dict) else None
                    _update_memory(robot_id, current_step, move_to_success=move_ok)
                    if move_ok is False:
                        print(f"[SuperLLMAgent] Warning: {robot_id} move_to success=False")
                
                elif action == "extinguish_fire":
                    # Find burning object location from perception
                    if agent._actions._perception_object_list:
                        actor_list = agent._actions._perception_object_list.get("actor_info", [])
                        target_location = None
                        for actor_info in actor_list:
                            burning_state = actor_info.get("burning_state")
                            if burning_state:
                                location = actor_info.get("location")
                                if location is not None:
                                    if hasattr(location, 'x') and hasattr(location, 'y'):
                                        target_location = ts.Vector3(location.x, location.y, Z_HEIGHT)
                                    elif isinstance(location, dict):
                                        target_location = ts.Vector3(
                                            location.get("x"),
                                            location.get("y"),
                                            Z_HEIGHT
                                        )
                                break
                        
                        if target_location is None:
                            print(f"[SuperLLMAgent] Error: {robot_id} no burning object found for extinguish_fire")
                            return
                        
                        result = await agent.extinguish_fire(
                            actor=agent._agent,
                            target_location=target_location,
                            timeout=5.0
                        )
                        print(f"[SuperLLMAgent] {robot_id} extinguish_fire result: {result}")
                        _update_memory(robot_id, current_step, extinguish_result=result)
                
                elif action == "send_follow":
                    result = await agent.send_follow()
                    print(f"[SuperLLMAgent] {robot_id} send_follow result: {result}")
                    _update_memory(
                        robot_id,
                        current_step,
                        sendfollow_received=bool(result),
                    )

                elif action == "send_stop_follow":
                    result = await agent.send_stop_follow()
                    print(f"[SuperLLMAgent] {robot_id} send_stop_follow result: {result}")
                    _update_memory(
                        robot_id,
                        current_step,
                        sendstopfollow_received=bool(result),
                    )
                
                elif action == "explore":
                    # Explore action: wait a bit to simulate exploration
                    await agent.explore()
                    print(f"[SuperLLMAgent] {robot_id} explore completed")
                    _update_memory(robot_id, current_step, explore_result="Success")
                
                elif action == "wait":
                    await agent.wait()
                    print(f"[SuperLLMAgent] {robot_id} wait completed")
                    _update_memory(robot_id, current_step, wait_result="Success")
                
                else:
                    print(f"[SuperLLMAgent] Warning: Unknown action {action} for {robot_id}")
            
            except Exception as e:
                print(f"[SuperLLMAgent] Error executing {action} for {robot_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Execute all actions in parallel
        tasks = [execute_agent_action(robot_id, action_dict) 
                 for robot_id, action_dict in robot_actions.items() 
                 if robot_id in self._agents]
        
        await asyncio.gather(*tasks, return_exceptions=True)
