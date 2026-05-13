from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import datetime
import tempfile
from pathlib import Path
from typing import Optional, Any, List, Callable

import cv2
import numpy as np
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
    # Import base_agent_vlm directly when resolving paths as a script
    import base_agent_vlm
    FireAgent = base_agent_vlm.FireAgent
    SaveAgent = base_agent_vlm.SaveAgent
    LimitedWaterFireAgent = base_agent_vlm.LimitedWaterFireAgent
    import llm_config  # keep API keys out of git; use env vars / untracked profiles
    import llm_config
    get_llm_config = llm_config.get_llm_config
else:
    from ..base_agent_vlm import FireAgent, SaveAgent, LimitedWaterFireAgent  # type: ignore
    from ...llm_config import get_llm_config  # type: ignore
    from ...Metric_Tool import llm_scene_timing  # type: ignore

# Default environment slots for prompt_assembler (aligned with EmbodiedMAS/prompt/verify_prompt.py)
_DEFAULT_PROMPT_ENV: dict[str, Any] = {
    "num_agents": 1,
    "num_fire_agents": 1,
    "num_rescue_agents": 0,
    "num_civilians": 0,
    "num_fires": "unknown",
    "other_info": "",
}


def _merge_water_limit_prompt_environment(
    water_capacity: int,
    recover_time: int,
    prompt_environment: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Merge P2 ``other_info`` describing ``SetExtinguisher`` tank limits."""
    merged: dict[str, Any] = {**_DEFAULT_PROMPT_ENV, **(prompt_environment or {})}
    note = (
        f"Limited extinguisher water: water_capacity={water_capacity}, recover_time={recover_time}s. "
        "Use extinguish_fire only when necessary."
    )
    existing = str(merged.get("other_info", "")).strip()
    merged["other_info"] = f"{existing} {note}".strip() if existing else note
    return merged


def _role_for_prompt_type(prompt_type: str) -> str:
    """Map legacy FD/SD codes to prompt_assembler roles (fire / rescue)."""
    if prompt_type == "SD":
        return "rescue"
    return "fire"


def _local_image_path_to_image_url_part(path: str) -> Optional[dict[str, Any]]:
    """Build OpenAI multimodal image_url content part from a local file. Requires a vision-capable model."""
    p = Path(path)
    if not p.is_file():
        return None
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    raw = p.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _resize_local_image_uniform(src: str | Path, *, scale: float = 0.5) -> Optional[str]:
    """
    Uniformly scale an on-disk image, write a temporary PNG, and return its path.

    E.g. ``scale=0.5`` maps 1280×1280 → 640×640. Uses ``numpy.fromfile`` + ``cv2.imdecode`` so non-ASCII paths work.
    Returns ``None`` on failure.
    """
    path = Path(src)
    if not path.is_file():
        return None
    if not (0.0 < scale < 1.0):
        return str(path.resolve())
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if nw == w and nh == h:
        return str(path.resolve())
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", out)
    if not ok or buf is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix="vlm_perception_", delete=False
        ) as tf:
            tf.write(buf.tobytes())
        return tf.name
    except OSError:
        return None


class LLMControllerVisionLanguage:
    """
    LLM controller that fuses memory, pose, and optional explore mosaics before calling OpenAI.
    No simulated object-name list is injected; targets are mosaic pixel coordinates.
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
        
        # Load system prompt via EmbodiedMAS/prompt (VL observation modules: obs="VL")
        self._system_prompt = self._load_system_prompt(prompt_type, prompt_environment)
        
        # Initialize memory for action history
        self._memory: list[dict[str, Any]] = []
        self._step_counter = 0
        self._max_memory_size = max_memory_size
        self._agent_id = agent_id
        self._current_summary: Optional[str] = None
        self._high_level_plan: Optional[str] = None

        self._session_id: Optional[str] = session_id
        self._session_memory_path: Optional[Path] = None
        self._session_timing_path: Optional[Path] = None
        self._session_memory_header_written: bool = False
        self._session_timing_header_written: bool = False
        self._step_timing_events: list[dict[str, Any]] = []

    def clear_step_timing(self) -> None:
        self._step_timing_events.clear()

    def record_timing_event(self, event: dict[str, Any]) -> None:
        self._step_timing_events.append(event)

    def flush_session_log(self, log_round: int, agent_id: Optional[str] = None) -> Optional[Path]:
        """Append MEMORY to session file and TIMING to a separate file (aligned with Only-Language llm_agent)."""
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

    def _prompt_assembler_role(self, prompt_type: str) -> str:
        return _role_for_prompt_type(prompt_type)

    def _load_system_prompt(
        self,
        prompt_type: str,
        prompt_environment: Optional[dict[str, Any]],
    ) -> str:
        """assemble_prompt(role, DISTRIBUTED, environment, obs='VLM') — see EmbodiedMAS/prompt/verify_prompt.py."""
        import os
        import sys

        _mas_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _mas_root not in sys.path:
            sys.path.insert(0, _mas_root)
        from prompt.prompt_assembler import assemble_prompt

        role = self._prompt_assembler_role(prompt_type)
        env = dict(_DEFAULT_PROMPT_ENV)
        if prompt_environment:
            env.update(prompt_environment)
        return assemble_prompt(role, "DISTRIBUTED", env, obs="VLM")

    def _format_memory_entry(self, mem: dict[str, Any], is_detailed: bool = False) -> str:
        """Compact single-line style for LLM MEMORY and disk logs (aligned with Only-Language llm_agent)."""
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
            elif act == "move_to":
                x, y, nm = (
                    action_params.get("x"),
                    action_params.get("y"),
                    action_params.get("name"),
                )
                if x is not None or y is not None or nm is not None:
                    parts.append(f"params=x={x!r},y={y!r},name={nm!r}")
            elif (tn := action_params.get("target_name")) is not None:
                parts.append(f"params={tn}")
        if "move_to_success" in mem and mem.get("move_to_success") is not None:
            parts.append(f"move_ok={mem['move_to_success']}")
        elif "move_to_success" in mem:
            parts.append("move_ok=None")
        if "sendfollow_received" in mem:
            parts.append(f"follow_rx={mem['sendfollow_received']}")
        if (p := mem.get("high_level_plan")):
            parts.append(f"plan={p!r}")
        if (s := mem.get("summary")):
            parts.append(f"summary={s!r}")
        return "  " + " | ".join(parts)

    def _format_memory(self) -> str:
        """Format memory (action history) into a readable text string for LLMVisionLanguageAgent."""
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

    def _parse_response(self, response_text: str) -> list[dict[str, Any]]:
        """
        Parse response from LLM based on new format.

        New format: JSON object with summary, high_level_plan, action, action_parameter.
        """
        response_text = response_text.strip()
        action_list: list[dict[str, Any]] = []

        response_json = self._parse_json_from_response(response_text)

        if not isinstance(response_json, dict):
            agent_id_str = f" {self._agent_id}" if self._agent_id else ""
            print(f"[LLMControllerVisionLanguage{agent_id_str}] Response is not a JSON object: {type(response_json)}")
            return []

        summary = response_json.get("summary", "")
        high_level_plan = response_json.get("high_level_plan", "")
        action = response_json.get("action")
        action_parameter = response_json.get("action_parameter")

        self._current_summary = summary if summary else None
        self._high_level_plan = high_level_plan if high_level_plan else None

        def _pixel_xy_from_parameter(ap: Any) -> Optional[tuple[float, float]]:
            """Parse mosaic pixel dicts supporting ``{\"x\",\"y\"}`` or ``{\"px\",\"py\"}``."""
            if not isinstance(ap, dict):
                return None
            if "x" in ap and "y" in ap:
                try:
                    return (float(ap["x"]), float(ap["y"]))
                except (TypeError, ValueError):
                    return None
            if "px" in ap and "py" in ap:
                try:
                    return (float(ap["px"]), float(ap["py"]))
                except (TypeError, ValueError):
                    return None
            return None

        if action == "move_to":
            pxy = _pixel_xy_from_parameter(action_parameter)
            if pxy is None:
                agent_id_str = f" {self._agent_id}" if self._agent_id else ""
                print(
                    f"[LLMControllerVisionLanguage{agent_id_str}] "
                    f"move_to requires action_parameter with x,y or px,py: {action_parameter!r}"
                )
                return []
            move_to_name: Optional[str] = None
            if isinstance(action_parameter, dict):
                raw_name = action_parameter.get("name")
                if raw_name is not None and str(raw_name).strip():
                    move_to_name = str(raw_name).strip()
            action_list.append({
                "action": action,
                "pixel_xy": pxy,
                "name": move_to_name,
                "high_level_plan": high_level_plan,
                "summary": summary,
            })
        elif action == "extinguish_fire":
            pxy = _pixel_xy_from_parameter(action_parameter)
            action_list.append({
                "action": action,
                "pixel_xy": pxy,
                "high_level_plan": high_level_plan,
                "summary": summary,
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
                    f"[LLMControllerVisionLanguage{agent_id_str}] "
                    f"move_by missing or invalid distance/angle: {action_parameter!r}"
                )
                return []
            action_list.append({
                "action": "move_by",
                "distance": dist,
                "angle": ang,
                "high_level_plan": high_level_plan,
                "summary": summary,
            })
        elif action in ["send_follow", "send_stop_follow", "wait"]:
            action_list.append({
                "action": action,
                "high_level_plan": high_level_plan,
                "summary": summary,
            })
        else:
            agent_id_str = f" {self._agent_id}" if self._agent_id else ""
            print(f"[LLMControllerVisionLanguage{agent_id_str}] Unknown action: {action}, action_parameter: {action_parameter}")
            return []

        return action_list

    def _build_llm_content(
        self,
        agent_position_text: str,
        perception_text: str,
        other_agent_info: Optional[str],
        perception_image_paths: Optional[List[str]] = None,
    ) -> list[dict[str, Any]]:
        """Build content items for LLM call (optional local images as data URLs; requires a vision-capable model)."""
        content_items = []

        content_items.append({"type": "text", "text": f"MEMORY:\n{self._format_memory()}"})
        print(f"[LLMControllerVisionLanguage] Memory text: {self._format_memory()}")

        content_items.append({"type": "text", "text": agent_position_text})

        if perception_text.strip():
            content_items.append({"type": "text", "text": perception_text})

        if other_agent_info:
            content_items.append({"type": "text", "text": f"INFORMATION FROM OTHER AGENTS:\n{other_agent_info}"})

        if perception_image_paths:
            content_items.append({
                "type": "text",
                "text": "RECENT EXPLORE MOSAIC (single RGB image, front-left-back-right quadrants):",
            })
            for path in perception_image_paths:
                part = _local_image_path_to_image_url_part(path)
                if part is not None:
                    content_items.append(part)
                else:
                    print(f"[LLMControllerVisionLanguage] Skip missing perception image: {path}")

        # # Observation prompt (covered by assembled P9; keep commented like Only-Language llm_agent)
        # obs_text = "OBSERVATION: Please analyze the current perception information and decide on the next action following the EXECUTE format."
        # content_items.append({"type": "text", "text": obs_text})

        return content_items

    async def _call_llm(
        self,
        agent_position_text: str,
        perception_text: str,
        other_agent_info: Optional[str] = None,
        agent_idx: Optional[int] = None,
        perception_image_paths: Optional[List[str]] = None,
    ) -> list[dict[str, Any]]:
        """Call LLM with perception information, memory, position, optional other agent info, optional images."""
        content_items = self._build_llm_content(
            agent_position_text,
            perception_text,
            other_agent_info,
            perception_image_paths=perception_image_paths,
        )

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
        print(f"[LLMControllerVisionLanguage{agent_id_str}] LLM Response: {response_text}")
        return self._parse_response(response_text)

    def get_current_summary_and_high_level_plan(self) -> tuple[Optional[str], Optional[str]]:
        return self._current_summary, self._high_level_plan

    def save_memory(self, last_step_result: Optional[dict[str, Any]] = None, agent_id: Optional[str] = None) -> Optional[Path]:
        """Legacy alias: append one session block (uses log_round=0 if no step context)."""
        return self.flush_session_log(log_round=0, agent_id=agent_id)


class LLMControllerVisionLanguageWaterLimit(LLMControllerVisionLanguage):
    """``limited_water_fire`` role with ``obs=VLM``."""

    def _prompt_assembler_role(self, prompt_type: str) -> str:
        return "limited_water_fire"


class LLMAgent:
    """
    Base class for LLM-powered agents that integrate LLMController and BaseAgentVLM.
    Uses LLMController for decision-making and BaseAgentVLM for action execution.
    """
    
    def __init__(
        self,
        context: WorldContext,
        agent: FireAgent | SaveAgent | LimitedWaterFireAgent,
        controller: Optional[LLMControllerVisionLanguage] = None,
        agent_id: Optional[str] = None,
        prompt_type: str = "FD",
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        pause_during_llm: bool = False,
        max_memory_size: int = 3,
    ):
        """
        Initialize LLM Agent.
        
        Args:
            context: WorldContext instance
            agent: FireAgent | SaveAgent instance
            controller: Optional LLMControllerVLM instance, if None will create a new one
            agent_id: Optional agent ID for logging
            prompt_type: Prompt type ("FD" for Fire, "SD" for Save)
            prompt_environment: Optional overrides for prompt_assembler environment slots
        """
        self.agent = agent
        self._agent_id = agent_id
        self._pause_during_llm = pause_during_llm
        self._log_round = 0
        self.controller = controller if controller is not None else LLMControllerVisionLanguage(
            max_memory_size=max_memory_size,
            agent_id=agent_id,
            prompt_type=prompt_type,
            prompt_environment=prompt_environment,
            session_id=session_id,
        )
        if controller is not None:
            controller._agent_id = agent_id
            controller._max_memory_size = max_memory_size
            if session_id is not None:
                controller._session_id = session_id
        self._step_counter = 0
    
    def get_current_summary_and_high_level_plan(self) -> tuple[Optional[str], Optional[str]]:
        return self.controller.get_current_summary_and_high_level_plan()

    def _get_log_prefix(self) -> str:
        """Get log prefix for this agent type. Should be overridden by subclasses."""
        return "[LLMAgent]"
    
    def _get_valid_actions(self) -> set[str]:
        """Get valid actions for this agent type. Should be overridden by subclasses."""
        return {
            "move_to",
            "move_by",
            "send_follow",
            "send_stop_follow",
            "wait",
        }
    
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

        self._log_round += 1
        self.controller.clear_step_timing()

        agent_position = await self.agent.get_agent_position_and_forward()
        agent_position_text = self.agent.format_agent_position_text(agent_position)
        perception_text = ""
        print(f"[LLMAgent] Agent position text: {agent_position_text}")

        perception_image_paths: List[str] = []
        vis = getattr(self.agent._actions, "last_locator_visualization_paths", None)
        if vis:
            p = vis.get("rgb")
            if p and Path(p).is_file():
                scaled = _resize_local_image_uniform(p, scale=0.5)
                perception_image_paths.append(scaled if scaled is not None else p)
            elif p:
                print(f"[LLMAgent] Mosaic RGB path not on disk: {p}")
        else:
            print("[LLMAgent] No last_locator_visualization_paths (needs prior explore or a wait/move that triggers explore).")

        async def _do_llm() -> list[dict[str, Any]]:
            return await self.controller._call_llm(
                agent_position_text=agent_position_text,
                perception_text=perception_text,
                other_agent_info=other_agent_info,
                agent_idx=agent_idx,
                perception_image_paths=perception_image_paths or None,
            )

        if self._pause_during_llm:
            action_list, llm_ev = await llm_scene_timing.run_with_scene_paused(
                self.agent._conn, _do_llm
            )
            self.controller.record_timing_event(llm_ev)
        else:
            t0w = datetime.now().isoformat()
            t0 = time.monotonic()
            action_list = await _do_llm()
            self.controller.record_timing_event(
                {
                    "name": "llm_thinking",
                    "start": t0w,
                    "end": datetime.now().isoformat(),
                    "duration_sec": round(time.monotonic() - t0, 6),
                }
            )

        if action_list:
            await self._execute_action(action_list)

        self.controller.flush_session_log(log_round=self._log_round, agent_id=self._agent_id)
    
    async def _execute_action(self, action_list: list[dict[str, Any]]) -> None:
        """
        Execute actions based on LLM's decision.
        """
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
            action_params: dict[str, Any] = {}
            if action == "move_to":
                pxy = action_dict.get("pixel_xy")
                if isinstance(pxy, (tuple, list)) and len(pxy) >= 2:
                    x_val, y_val = float(pxy[0]), float(pxy[1])
                else:
                    x_val, y_val = None, None
                action_params = {
                    "x": x_val,
                    "y": y_val,
                    "name": action_dict.get("name"),
                }
            elif action == "extinguish_fire":
                action_params = {"pixel_xy": action_dict.get("pixel_xy")}
            elif action == "move_by":
                dist, ang = action_dict.get("distance"), action_dict.get("angle")
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
            
            try:
                if action == "move_to":
                    pxy = action_dict.get("pixel_xy")
                    if not pxy or not isinstance(pxy, (tuple, list)) or len(pxy) < 2:
                        print("[LLMAgent] Error: move_to missing pixel_xy")
                        continue
                    nav = await self.agent.move_to(
                        (float(pxy[0]), float(pxy[1])),
                        timeout=60.0,
                        tolerance_uu=10.0,
                    )
                    move_ok = nav.get("success") if isinstance(nav, dict) else None
                    _update_memory(current_step, action_index, move_to_success=move_ok)
                    if isinstance(nav, dict) and not nav.get("success"):
                        print(
                            f"[LLMAgent] Warning: Navigation failed: {nav.get('message')!r}"
                        )

                elif action == "move_by":
                    dist = action_dict.get("distance")
                    ang = action_dict.get("angle")
                    if dist is None or ang is None:
                        print("[LLMAgent] Error: move_by missing distance or angle")
                        continue
                    nav = await self.agent.move_by(
                        float(dist),
                        float(ang),
                        timeout=60.0,
                        tolerance_uu=10.0,
                    )
                    move_ok = nav.get("success") if isinstance(nav, dict) else None
                    _update_memory(current_step, action_index, move_to_success=move_ok)
                    if isinstance(nav, dict) and not nav.get("success"):
                        print(
                            f"[LLMAgent] Warning: Navigation failed: {nav.get('message')!r}"
                        )

                elif action == "send_follow":
                    follow_result = await self.agent.send_follow()
                    _update_memory(
                        current_step,
                        action_index,
                        sendfollow_received=bool(follow_result),
                    )

                elif action == "send_stop_follow":
                    await self.agent.send_stop_follow()

                elif action == "wait":
                    await self.agent.wait()
                    print(f"[LLMAgent] Wait completed")
                
                else:
                    # Execute agent-specific actions
                    await self._execute_specific_action(action, action_dict, current_step, action_index, _update_memory)
            
            except Exception as e:
                print(f"[LLMAgent] Error executing {action}: {e}")
                continue


class LLMAgentVisionLanguageFire(LLMAgent):
    """
    LLM-powered Vision Language Fire Agent that integrates LLMController and FireAgent.
    Uses LLMController for decision-making and FireAgent for action execution.
    """
    
    def __init__(
        self,
        context: WorldContext,
        controller: Optional[LLMControllerVisionLanguage] = None,
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
    
    def _get_log_prefix(self) -> str:
        """Get log prefix for Vision Language Fire Agent."""
        return "[LLMAgentVisionLanguageFire]"
    
    def _get_valid_actions(self) -> set[str]:
        """Get valid actions for Vision Language Fire Agent."""
        return {
            "move_to",
            "move_by",
            "extinguish_fire",
            "send_follow",
            "send_stop_follow",
            "wait",
        }
    
    async def _execute_specific_action(self, action: str, action_dict: dict[str, Any], 
                                      current_step: int, action_index: int,
                                      _update_memory: Callable[..., None]) -> None:
        """Execute Vision Language Fire Agent specific actions."""
        if action == "extinguish_fire":
            pxy = action_dict.get("pixel_xy")
            pfx = self._get_log_prefix()
            pixel_arg: Optional[tuple[float, float]] = None
            if pxy is not None:
                if isinstance(pxy, (tuple, list)) and len(pxy) >= 2:
                    pixel_arg = (float(pxy[0]), float(pxy[1]))
                else:
                    print(f"{pfx} Error: extinguish_fire invalid pixel_xy: {pxy!r}")
                    return

            result = await self.agent.extinguish_fire(
                actor=self.agent._agent,
                pixel_xy=pixel_arg,
                timeout=5.0,
            )
            print(f"{pfx} Extinguish fire result: {result}")


class LLMAgentVisionLanguageFireWaterLimit(LLMAgentVisionLanguageFire):
    """Limited-water VLM firefighter: ``LimitedWaterFireAgent`` + ``limited_water_fire`` prompt and ``other_info``."""

    def __init__(
        self,
        context: WorldContext,
        controller: Optional[LLMControllerVisionLanguage] = None,
        agent_id: Optional[str] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        pause_during_llm: bool = False,
        max_memory_size: int = 3,
        *,
        water_capacity: int = 5,
        recover_time: int = 10,
    ):
        env = _merge_water_limit_prompt_environment(water_capacity, recover_time, prompt_environment)
        agent = LimitedWaterFireAgent(
            context, water_capacity=water_capacity, recover_time=recover_time
        )
        ctrl = controller
        if ctrl is None:
            ctrl = LLMControllerVisionLanguageWaterLimit(
                max_memory_size=max_memory_size,
                agent_id=agent_id,
                prompt_type="FD",
                prompt_environment=env,
                session_id=session_id,
            )
        LLMAgent.__init__(
            self,
            context,
            agent,
            ctrl,
            agent_id=agent_id,
            prompt_type="FD",
            prompt_environment=env,
            session_id=session_id,
            pause_during_llm=pause_during_llm,
            max_memory_size=max_memory_size,
        )

    def _get_log_prefix(self) -> str:
        return "[LLMAgentVisionLanguageFireWaterLimit]"


class LLMAgentVisionLanguageSave(LLMAgent):
    """
    LLM-powered Save Agent that integrates LLMController and SaveAgent.
    Uses LLMController for decision-making and SaveAgent for action execution.
    """
    
    def __init__(
        self,
        context: WorldContext,
        controller: Optional[LLMControllerVisionLanguage] = None,
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
    
    def _get_log_prefix(self) -> str:
        """Get log prefix for Vision Language Save Agent."""
        return "[LLMAgentVisionLanguageSave]"
    
    def _get_valid_actions(self) -> set[str]:
        """Get valid actions for Vision Language Save Agent."""
        return {
            "move_to",
            "move_by",
            "send_follow",
            "send_stop_follow",
            "wait",
        }
