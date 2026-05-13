from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Callable, Coroutine

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
    # Import base_agent_only_language directly when resolving paths as a script
    import base_agent_only_language
    FireAgent = base_agent_only_language.FireAgent
    SaveAgent = base_agent_only_language.SaveAgent
    import llm_config  # keep API keys out of git; use env vars / untracked profiles
    import llm_config
    get_llm_config = llm_config.get_llm_config
    try:
        from Metric_Tool import llm_scene_timing
    except ImportError:
        from EmbodiedMAS.Metric_Tool import llm_scene_timing  # type: ignore
else:
    from ..base_agent_only_language import FireAgent, SaveAgent  # type: ignore
    from ...llm_config import get_llm_config  # type: ignore
    from ...Metric_Tool import llm_scene_timing  # type: ignore

# Global z-height constant for agent spawning
Z_HEIGHT = 1000

# Default environment slots for prompt_assembler (aligned with Single_agent/llm_agent.py / EmbodiedMAS/prompt)
_DEFAULT_PROMPT_ENV: dict[str, Any] = {
    "num_agents": 1,
    "num_fire_agents": 1,
    "num_rescue_agents": 0,
    "num_civilians": 0,
    "num_fires": "unknown",
    "other_info": "",
}


def _role_for_prompt_type(prompt_type: str) -> str:
    """Map legacy FD/SD codes to prompt_assembler roles (fire / rescue)."""
    if prompt_type == "SD":
        return "rescue"
    return "fire"


class LLMControllerDMAS:
    """
    LLM-powered controller that uses OpenAI API to decide actions based on perception information.
    Uses perception data (objects, NPCs) instead of visual observations.
    """

    def __init__(
        self,
        max_memory_size: int = 3,
        agent_id: Optional[str] = None,
        prompt_type: str = "FD",
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
        # Load LLM configuration
        self._llm_config = get_llm_config()
        
        # Initialize OpenAI client with configuration
        client_kwargs = self._llm_config.get_client_kwargs()
        self._client = OpenAI(**client_kwargs)
        
        # EmbodiedMAS/prompt — same API as EmbodiedMAS/prompt/verify_prompt.py
        self._system_prompt = self._load_system_prompt(prompt_type, prompt_environment)
        
        # Initialize memory for action history
        self._memory: list[dict[str, Any]] = []
        self._step_counter = 0
        self._max_memory_size = max_memory_size
        
        # Store agent_id for logging
        self._agent_id = agent_id

        self._session_id: Optional[str] = session_id
        self._session_memory_path: Optional[Path] = None
        self._session_timing_path: Optional[Path] = None
        self._session_memory_header_written: bool = False
        self._session_timing_header_written: bool = False
        self._step_timing_events: list[dict[str, Any]] = []
        
        # Store current step's summary (only the last step)
        self._current_summary: Optional[str] = None
        self._high_level_plan: Optional[str] = None

    def clear_step_timing(self) -> None:
        self._step_timing_events.clear()

    def record_timing_event(self, event: dict[str, Any]) -> None:
        self._step_timing_events.append(event)

    def flush_session_log(self, log_round: int, agent_id: Optional[str] = None) -> Optional[Path]:
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
        aid_log = f" {self._agent_id}" if self._agent_id else ""
        # print(f"[LLMController{aid_log}] Memory log: {mem_path}")
        # print(f"[LLMController{aid_log}] Timing log: {tim_path}")
        return mem_path
    
    def _load_system_prompt(
        self,
        prompt_type: str,
        prompt_environment: Optional[dict[str, Any]],
    ) -> str:
        """assemble_prompt(role, coordination, environment) — decentralized single/worker DMAS."""
        import os
        import sys

        _mas_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _mas_root not in sys.path:
            sys.path.insert(0, _mas_root)
        from prompt.prompt_assembler import assemble_prompt

        role = _role_for_prompt_type(prompt_type)
        env = dict(_DEFAULT_PROMPT_ENV)
        if prompt_environment:
            env.update(prompt_environment)
        return assemble_prompt(role, "DISTRIBUTED", env)

    def _format_memory_entry(self, mem: dict[str, Any], is_detailed: bool = False) -> str:
        """Compact single-line style for LLM MEMORY and disk logs."""
        parts: list[str] = [
            f"step={mem.get('step')}",
            f"action={mem.get('action')}",
        ]
        act = mem.get("action")
        if action_params := mem.get("action_params"):
            if act == "move_by":
                d, a = action_params.get("distance"), action_params.get("angle")
                if d is not None or a is not None:
                    parts.append(f"params=d={d!r},a={a!r}")
            elif (tn := action_params.get("target_name")) is not None:
                parts.append(f"params={tn}")
        if "move_to_success" in mem and mem.get("move_to_success") is not None:
            parts.append(f"move_ok={mem['move_to_success']}")
        elif "move_to_success" in mem:
            parts.append("move_ok=None")
        if "sendfollow_received" in mem:
            parts.append(f"follow_rx={mem['sendfollow_received']}")
        # if "sendstopfollow_received" in mem:
        #     parts.append(f"stopfollow_rx={mem['sendstopfollow_received']}")
        # if (er := mem.get("extinguish_result")) is not None:
        #     parts.append(f"extinguish={er}")
        if (p := mem.get("high_level_plan")):
            parts.append(f"plan={p!r}")
        if (s := mem.get("summary")):
            parts.append(f"summary={s!r}")
        return "  " + " | ".join(parts)

    def _format_memory(self) -> str:
        """Format memory (action history) into a readable text string for LLMVisionAgent."""
        if not self._memory:
            return "No previous actions."

        lines = [f"Previous actions (last {len(self._memory)} actions):"]
        for mem in self._memory:
            lines.append(self._format_memory_entry(mem))

        return "\n".join(lines)

    def _parse_json_from_response(self, response_text: str) -> Any:
        """Parse JSON from LLM response text."""
        text = response_text.strip()
        
        # Strip markdown fences if present
        # Handle ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            # Skip first line (```json or ```)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Drop trailing ```
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        return json.loads(text)

    def _parse_response(self, response_text: str) -> list[dict[str, Any]]:
        """
        Parse response from LLM based on new format.
        
        New format: Response is a JSON object with:
        - summary: summary of observations
        - high_level_plan: brief description of strategy
        - action: one action from Action Space ("move_to", "extinguish_fire", "send_follow", "send_stop_follow", "explore", "wait")
        - action_parameter: parameters for the action (for move_to: {"x": float, "y": float})
        
        Args:
            response_text: Raw response text from LLM containing JSON
            
        Returns:
            List of action dictionaries, each with keys: action, why, high_level_plan, target_x, target_y
        """
        response_text = response_text.strip()
        action_list: list[dict[str, Any]] = []
        
        # Parse JSON from response
        response_json = self._parse_json_from_response(response_text)
        
        if not isinstance(response_json, dict):
            agent_id_str = f" {self._agent_id}" if self._agent_id else ""
            print(f"[LLMController{agent_id_str}] Response is not a JSON object: {type(response_json)}")
            return []
        
        # Extract fields from new format
        summary = response_json.get("summary", "")
        high_level_plan = response_json.get("high_level_plan", "")
        action = response_json.get("action")
        action_parameter = response_json.get("action_parameter")
        
        # Save current step's summary (only the last step)
        self._current_summary = summary if summary else None
        self._high_level_plan = high_level_plan if high_level_plan else None

        # Parse action based on type
        if action in ["move_to", "extinguish_fire"]:
            # Extract target coordinates from action_parameter
            target_x = action_parameter.get("x")
            target_y = action_parameter.get("y")
            target_name = action_parameter.get("name")

            action_list.append({
                "action": action,
                "target_x": target_x,
                "target_y": target_y,
                "target_name": target_name,
                "high_level_plan": high_level_plan,
                "summary": summary
            })
        elif action == "move_by":
            try:
                ap = action_parameter
                if not isinstance(ap, dict):
                    raise TypeError
                dist = float(ap["distance"])
                ang = float(ap["angle"])
            except (TypeError, KeyError, ValueError):
                agent_id_str = f" {self._agent_id}" if self._agent_id else ""
                print(
                    f"[LLMController{agent_id_str}] move_by missing or invalid distance/angle: "
                    f"{action_parameter!r}"
                )
                return []
            action_list.append({
                "action": "move_by",
                "distance": dist,
                "angle": ang,
                "high_level_plan": high_level_plan,
                "summary": summary,
            })
        elif action in ["explore", "send_follow", "send_stop_follow", "wait"]:
            action_list.append({
                "action": action,
                "high_level_plan": high_level_plan,
                "summary": summary
            })
        else:
            agent_id_str = f" {self._agent_id}" if self._agent_id else ""
            print(f"[LLMController{agent_id_str}] Unknown action: {action}, action_parameter: {action_parameter}")
            return []

        return action_list

    def _build_llm_content(self, agent_position_text: str, perception_text: str,
                          other_agent_info: Optional[str], sos_info: Optional[str] = None) -> list[dict[str, Any]]:
        """Build content items for LLM call."""
        content_items = []

        # Memory
        content_items.append({"type": "text", "text": f"MEMORY:\n{self._format_memory()}"})

        # Agent position
        content_items.append({"type": "text", "text": agent_position_text})

        # Perception cache
        content_items.append({"type": "text", "text": perception_text})

        # Other agent info
        if other_agent_info:
            content_items.append({"type": "text", "text": f"INFORMATION FROM OTHER AGENTS:\n{other_agent_info}"})

        # SOS info
        if sos_info:
            content_items.append({"type": "text", "text": f"SOS SIGNAL:\n{sos_info}"})

        # # Observation prompt
        # obs_text = "OBSERVATION: Please analyze the current perception information and decide on the next action following the EXECUTE format."
        # content_items.append({"type": "text", "text": obs_text})

        return content_items

    async def _call_llm(self, agent_position_text: str, perception_text: str,
                       other_agent_info: Optional[str] = None, agent_idx: Optional[int] = None,
                       sos_info: Optional[str] = None) -> list[dict[str, Any]]:
        """Call LLM with perception information, memory, position, and optional other agent information."""
        content_items = self._build_llm_content(agent_position_text, perception_text, other_agent_info, sos_info=sos_info)

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
                max_tokens=500,
            )

        completion = await asyncio.to_thread(_sync_call)
        response_text = completion.choices[0].message.content

        if response_text is None:
            raise ValueError("LLM returned None response")

        # Use agent_id if available, otherwise use agent_idx, otherwise empty
        if self._agent_id:
            agent_id_str = f" {self._agent_id}"
        elif agent_idx is not None:
            agent_id_str = f" agent_{agent_idx}"
        else:
            agent_id_str = ""
        print(f"[LLMController{agent_id_str}] LLM Response: {response_text}")
        return self._parse_response(response_text)

    def get_current_summary_and_high_level_plan(self) -> tuple[Optional[str], Optional[str]]:
        """Get the current step's summary and high_level_plan."""
        return self._current_summary, self._high_level_plan

    def save_memory(self, last_step_result: Optional[dict[str, Any]] = None, agent_id: Optional[str] = None) -> Optional[Path]:
        return self.flush_session_log(log_round=0, agent_id=agent_id)


class LLMAgentDMAS:
    """
    Base class for LLM-powered agents that integrate LLMController and BaseAgentOnlyLanguage.
    Uses LLMController for decision-making and BaseAgentOnlyLanguage for action execution.
    """
    
    def __init__(
        self,
        context: WorldContext,
        agent: FireAgent | SaveAgent,
        controller: Optional[LLMControllerDMAS] = None,
        agent_id: Optional[str] = None,
        prompt_type: str = "FD",
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        pause_during_llm: bool = False,
        max_memory_size: int = 3,
    ):
        self.context = context
        self.agent = agent
        self._agent_id = agent_id
        self._pause_during_llm = pause_during_llm
        self._log_round = 0
        self.controller = controller if controller is not None else LLMControllerDMAS(
            max_memory_size=max_memory_size,
            agent_id=agent_id,
            prompt_type=prompt_type,
            prompt_environment=prompt_environment,
            session_id=session_id,
        )
        # If controller was provided, update its agent_id
        if controller is not None:
            controller._agent_id = agent_id
            controller._max_memory_size = max_memory_size
            if session_id is not None:
                controller._session_id = session_id
        self._step_counter = 0
        self._current_action_task: Optional[asyncio.Task] = None
        self._sos_listener_task: Optional[asyncio.Task] = None
        self._pending_sos_messages: list[dict[str, Any]] = []
        self._execution_interrupted_by_sos = False
        self._is_replanning_from_sos = False

    def _get_agent_id(self) -> str:
        """Get runtime actor GUID for grpc APIs (prefer spawned actor id)."""
        actor = self.agent._agent
        return actor.get("id")

    def _get_agent_name(self) -> str:
        """Get agent name from explicit config or runtime actor."""
        return self._agent_id

    def _format_sos_for_llm(self, sos_messages: list[dict[str, Any]]) -> Optional[str]:
        """Format SOS messages into concise LLM input text."""
        if not sos_messages:
            return None
        lines = []
        for i, sos in enumerate(sos_messages[-3:], start=1):
            agent_id = sos.get("agent_id", "unknown")
            orientations = sos.get("orientations", [])
            distances = sos.get("distances", [])
            yaw = orientations[0].get("yaw") if orientations else None
            distance = distances[0] if distances else None
            lines.append(f"- SOS {i}: source={agent_id}, yaw={yaw}, distance={distance}")
        return "\n".join(lines)

    def _consume_pending_sos_text(self) -> Optional[str]:
        """Consume queued SOS messages and convert them into prompt text."""
        if not self._pending_sos_messages:
            return None
        messages = list(self._pending_sos_messages)
        self._pending_sos_messages.clear()
        return self._format_sos_for_llm(messages)

    async def _ensure_sos_listener_started(self) -> None:
        """Start SOS listener once after agent is spawned."""
        if self._sos_listener_task and not self._sos_listener_task.done():
            return
        if not self.agent._agent:
            return
        agent_id = self._get_agent_id()
        agent_name = self._get_agent_name()
        self._sos_listener_task = asyncio.create_task(self._listen_sos(agent_id))
        print(f"[SOS Listener] Started for agent {agent_name}, id: {agent_id}")

    async def _listen_sos(self, agent_id: str) -> None:
        """Listen SOS signals and interrupt current action when needed."""
        try:
            async for sos in ts.UnaryAPI.receive_npc_sos(self.context.conn, [agent_id]):
                self._pending_sos_messages.append(sos)
                while len(self._pending_sos_messages) > 10:
                    self._pending_sos_messages.pop(0)
                print(f"[SOS Listener] Received SOS: {sos}")
                if self._current_action_task and not self._current_action_task.done():
                    self._execution_interrupted_by_sos = True
                    self._current_action_task.cancel()
        except asyncio.CancelledError:
            print("[SOS Listener] Cancelled")
            raise
        except Exception as e:
            print(f"[SOS Listener] Error while listening SOS: {e}")

    async def _run_cancellable_action(self, coro: Coroutine[Any, Any, Any], action_name: str) -> Any:
        """Run action as a cancellable task so SOS can interrupt it."""
        action_task = asyncio.create_task(coro)
        self._current_action_task = action_task
        try:
            return await action_task
        except asyncio.CancelledError:
            if self._execution_interrupted_by_sos:
                print(f"[LLMAgent] Action interrupted by SOS: {action_name}")
                return None
            raise
        finally:
            if self._current_action_task is action_task:
                self._current_action_task = None

    async def _collect_prompt_inputs(self) -> tuple[str, str]:
        """Refresh perception and return prompt-ready position/perception text."""
        await self.agent._actions.get_perception_object_list(self.agent._agent)
        agent_position = await self.agent.get_agent_position_and_forward()
        agent_position_text = self.agent.format_agent_position_text(agent_position)
        perception_text = self.agent.format_perception_cache()
        return agent_position_text, perception_text
    
    def get_current_summary_and_high_level_plan(self) -> tuple[Optional[str], Optional[str]]:
        """Get the current step's summary from the controller."""
        return self.controller.get_current_summary_and_high_level_plan()
    
    def _get_valid_actions(self) -> set[str]:
        """Get valid actions for this agent type. Should be overridden by subclasses."""
        return {
            "move_to",
            "move_by",
            "send_follow",
            "send_stop_follow",
            "explore",
            "wait",
        }

    def _record_action_duration(
        self,
        current_step: int,
        action_index: int,
        action: str,
        t0_wall: str,
        t0_mono: float,
    ) -> None:
        self.controller.record_timing_event(
            {
                "name": "action_execution",
                "action": action,
                "step": current_step,
                "action_index": action_index,
                "start": t0_wall,
                "end": datetime.now().isoformat(),
                "duration_sec": round(time.monotonic() - t0_mono, 6),
            }
        )
    
    async def _execute_specific_action(self, action: str, action_dict: dict[str, Any], 
                                      current_step: int, action_index: int,
                                      _update_memory: Callable[..., None]) -> None:
        """
        Execute agent-specific actions. Should be overridden by subclasses.
        
        Args:
            action: Action name
            action_dict: Action dictionary
            current_step: Current step number
            action_index: Action index
            _update_memory: Function to update memory
        """
        pass
    
    async def step(self, other_agent_info: Optional[str] = None, agent_idx: Optional[int] = None) -> None:
        """
        Perform one step: use controller to decide on actions, then execute them on the agent.
        
        Args:
            other_agent_info: Optional information from other agents to include in LLM prompt
            agent_idx: Optional agent index for logging purposes
        """
        if not self.agent._agent:
            print(f"[LLMAgent] Error: Agent not spawned")
            return
        await self._ensure_sos_listener_started()
        self._execution_interrupted_by_sos = False

        self._log_round += 1
        self.controller.clear_step_timing()

        agent_position_text, perception_text = await self._collect_prompt_inputs()

        async def _do_main_llm() -> list[dict[str, Any]]:
            return await self.controller._call_llm(
                agent_position_text=agent_position_text,
                perception_text=perception_text,
                other_agent_info=other_agent_info,
                agent_idx=agent_idx,
                sos_info=self._consume_pending_sos_text(),
            )

        if self._pause_during_llm:
            action_list, llm_ev = await llm_scene_timing.run_with_scene_paused(
                self.agent._conn, _do_main_llm
            )
            self.controller.record_timing_event(llm_ev)
        else:
            t0w = datetime.now().isoformat()
            t0m = time.monotonic()
            action_list = await _do_main_llm()
            self.controller.record_timing_event(
                {
                    "name": "llm_thinking",
                    "start": t0w,
                    "end": datetime.now().isoformat(),
                    "duration_sec": round(time.monotonic() - t0m, 6),
                }
            )

        if not action_list:
            self.controller.flush_session_log(log_round=self._log_round, agent_id=self._agent_id)
            return

        await self._execute_action(action_list)

        # Keep logic simple: one immediate replan after a SOS-triggered interrupt.
        if self._execution_interrupted_by_sos and not self._is_replanning_from_sos:
            self._is_replanning_from_sos = True
            try:
                agent_position_text, perception_text = await self._collect_prompt_inputs()
                sos_info = self._consume_pending_sos_text() or "SOS received during last action; replan now."

                async def _do_replan_llm() -> list[dict[str, Any]]:
                    return await self.controller._call_llm(
                        agent_position_text=agent_position_text,
                        perception_text=perception_text,
                        other_agent_info=other_agent_info,
                        agent_idx=agent_idx,
                        sos_info=sos_info,
                    )

                if self._pause_during_llm:
                    replanned_actions, ev2 = await llm_scene_timing.run_with_scene_paused(
                        self.agent._conn, _do_replan_llm, replan=True
                    )
                    self.controller.record_timing_event(ev2)
                else:
                    t0w = datetime.now().isoformat()
                    t0m = time.monotonic()
                    replanned_actions = await _do_replan_llm()
                    self.controller.record_timing_event(
                        {
                            "name": "llm_thinking",
                            "replan": True,
                            "start": t0w,
                            "end": datetime.now().isoformat(),
                            "duration_sec": round(time.monotonic() - t0m, 6),
                        }
                    )

                self._execution_interrupted_by_sos = False
                if replanned_actions:
                    await self._execute_action(replanned_actions)
            finally:
                self._is_replanning_from_sos = False

        self.controller.flush_session_log(log_round=self._log_round, agent_id=self._agent_id)
    
    async def _execute_action(self, action_list: list[dict[str, Any]]) -> None:
        """Execute actions based on LLM's decision."""
        if not self.agent._agent or not action_list:
            print(f"[LLMAgent] No agent or empty action list")
            return
        
        self._step_counter += 1
        current_step = self._step_counter
        
        # Helper function to update memory entry
        def _update_memory(step: int, action_index: int, **kwargs):
            for mem in reversed(self.controller._memory):
                if mem.get("step") == step and mem.get("action_index") == action_index:
                    mem.update(kwargs)
                    break
        
        # Create memory entries
        memory_entries = []
        for i, action_dict in enumerate(action_list):
            action = action_dict.get("action")
            action_params = {}
            if action == "move_to":
                target_x, target_y = action_dict.get("target_x"), action_dict.get("target_y")
                target_name = action_dict.get("target_name")
                action_params = {
                    "target_x": target_x,
                    "target_y": target_y,
                    "target_z": Z_HEIGHT,
                    "target_name": target_name,
                }
            elif action == "extinguish_fire":
                target_x, target_y = action_dict.get("target_x"), action_dict.get("target_y")
                target_name = action_dict.get("target_name")
                action_params = {
                    "target_x": target_x,
                    "target_y": target_y,
                    "target_z": 1010.0,
                    "target_name": target_name,
                }
            elif action == "move_by":
                action_params = {
                    "distance": action_dict.get("distance"),
                    "angle": action_dict.get("angle"),
                }

            memory_entries.append({
                "step": current_step,
                "action_index": i + 1,
                "action": action,
                "high_level_plan": action_dict.get("high_level_plan", ""),
                "summary": action_dict.get("summary", ""),
                "action_params": action_params,
            })
        
        self.controller._memory.extend(memory_entries)
        while len(self.controller._memory) > self.controller._max_memory_size:
            self.controller._memory.pop(0)
        
        # Execute actions
        valid_actions = self._get_valid_actions()
        
        for i, action_dict in enumerate(action_list):
            action = action_dict.get("action")
            if action not in valid_actions:
                print(f"[LLMAgent] Skipping invalid action: {action}")
                continue
            
            action_index = i + 1
            print(f"[LLMAgent] Executing action: {action}")
            t0w = datetime.now().isoformat()
            t0m = time.monotonic()

            if action == "move_to":
                target_x, target_y = action_dict.get("target_x"), action_dict.get("target_y")
                if target_x is None or target_y is None:
                    print(f"[LLMAgent] Error: move_to missing target_x or target_y")
                    continue
                
                target_location = ts.Vector3(float(target_x), float(target_y), Z_HEIGHT)
                move_result = await self._run_cancellable_action(
                    self.agent.move_to(
                        target_location=target_location,
                        timeout=60.0,
                        tolerance_uu=50.0
                    ),
                    action_name="move_to",
                )
                if move_result is None and self._execution_interrupted_by_sos:
                    return
                if move_result is None:
                    continue
                nav = move_result
                move_ok = nav.get("success") if isinstance(nav, dict) else None
                if move_ok is False:
                    print(f"[LLMAgent] move_to reported success=False")
                _update_memory(current_step, action_index, move_to_success=move_ok)
                self._record_action_duration(current_step, action_index, action, t0w, t0m)

            elif action == "move_by":
                dist = action_dict.get("distance")
                ang = action_dict.get("angle")
                if dist is None or ang is None:
                    print("[LLMAgent] Error: move_by missing distance or angle")
                    continue
                move_result = await self._run_cancellable_action(
                    self.agent.move_by(
                        float(dist),
                        float(ang),
                        timeout=60.0,
                        tolerance_uu=50.0,
                    ),
                    action_name="move_by",
                )
                if move_result is None and self._execution_interrupted_by_sos:
                    return
                if move_result is None:
                    continue
                nav = move_result
                move_ok = nav.get("success") if isinstance(nav, dict) else None
                if move_ok is False:
                    print("[LLMAgent] move_by reported success=False")
                _update_memory(current_step, action_index, move_to_success=move_ok)
                self._record_action_duration(current_step, action_index, action, t0w, t0m)

            elif action == "send_follow":
                result = await self.agent.send_follow()
                print(f"[LLMAgent] Send follow result: {result}")
                _update_memory(
                    current_step,
                    action_index,
                    sendfollow_received=bool(result),
                )
                self._record_action_duration(current_step, action_index, action, t0w, t0m)

            elif action == "send_stop_follow":
                result = await self.agent.send_stop_follow()
                print(f"[LLMAgent] Send stop follow result: {result}")
                _update_memory(
                    current_step,
                    action_index,
                    sendstopfollow_received=bool(result),
                )
                self._record_action_duration(current_step, action_index, action, t0w, t0m)

            elif action == "explore":
                explore_result = await self._run_cancellable_action(
                    self.agent.explore(),
                    action_name="explore",
                )
                if explore_result is None and self._execution_interrupted_by_sos:
                    return
                print(f"[LLMAgent] Explore completed")
                _update_memory(current_step, action_index, explore_result="Success")
                self._record_action_duration(current_step, action_index, action, t0w, t0m)
            
            elif action == "wait":
                await self.agent.wait()
                print(f"[LLMAgent] Wait completed")
                _update_memory(current_step, action_index, wait_result="Success")
                self._record_action_duration(current_step, action_index, action, t0w, t0m)
            
            else:
                await self._execute_specific_action(action, action_dict, current_step, action_index, _update_memory)
                self._record_action_duration(current_step, action_index, action, t0w, t0m)


class LLMAgentDMASFire(LLMAgentDMAS):
    """
    LLM-powered Fire Agent that integrates LLMController and FireAgent.
    Uses LLMController for decision-making and FireAgent for action execution.
    """
    
    def __init__(
        self,
        context: WorldContext,
        controller: Optional[LLMControllerDMAS] = None,
        agent_id: Optional[str] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        pause_during_llm: bool = False,
        max_memory_size: int = 3,
    ):
        agent = FireAgent(context)
        super().__init__(
            context,
            agent,
            controller,
            agent_id=agent_id,
            prompt_type="FD",
            prompt_environment=prompt_environment,
            session_id=session_id,
            pause_during_llm=pause_during_llm,
            max_memory_size=max_memory_size,
        )
    
    def _get_valid_actions(self) -> set[str]:
        """Get valid actions for Fire Agent."""
        return {
            "move_to",
            "move_by",
            "extinguish_fire",
            "send_follow",
            "send_stop_follow",
            "explore",
            "wait",
        }


    async def _execute_specific_action(self, action: str, action_dict: dict[str, Any], 
                                      current_step: int, action_index: int,
                                      _update_memory: Callable[..., None]) -> None:
        """Execute Fire Agent specific actions."""
        if action == "extinguish_fire":
            target_x, target_y = action_dict.get("target_x"), action_dict.get("target_y")
            if target_x is None or target_y is None:
                print(f"[LLMAgent] Error: extinguish_fire missing target_x or target_y")
                return
            
            target_location = ts.Vector3(float(target_x), float(target_y), 1010.0)
            result = await self.agent.extinguish_fire(
                actor=self.agent._agent, 
                target_location=target_location,
                timeout=5.0
            )
            print(f"[LLMAgentFire] Extinguish fire result: {result}")
            _update_memory(current_step, action_index, extinguish_result=result)


class LLMAgentDMASSave(LLMAgentDMAS):
    """
    LLM-powered Save Agent that integrates LLMController and SaveAgent.
    Uses LLMController for decision-making and SaveAgent for action execution.
    """
    
    def __init__(
        self,
        context: WorldContext,
        controller: Optional[LLMControllerDMAS] = None,
        agent_id: Optional[str] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        pause_during_llm: bool = False,
        max_memory_size: int = 3,
    ):
        agent = SaveAgent(context)
        super().__init__(
            context,
            agent,
            controller,
            agent_id=agent_id,
            prompt_type="SD",
            prompt_environment=prompt_environment,
            session_id=session_id,
            pause_during_llm=pause_during_llm,
            max_memory_size=max_memory_size,
        )
    
    
    def _get_valid_actions(self) -> set[str]:
        """Get valid actions for Save Agent."""
        return {
            "move_to",
            "move_by",
            "send_follow",
            "send_stop_follow",
            "explore",
            "wait",
        }
