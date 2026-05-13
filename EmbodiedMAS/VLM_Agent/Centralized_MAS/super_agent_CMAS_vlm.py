from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict

import tongsim as ts
from tongsim.core.world_context import WorldContext
from openai import OpenAI
import time

# Allow running as a script and as a module
if __package__ is None or __package__ == "":
    import os
    import sys
    _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _vlm_agent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    if _vlm_agent_dir not in sys.path:
        sys.path.insert(0, _vlm_agent_dir)
    try:
        from Metric_Tool import llm_scene_timing
    except ImportError:
        from EmbodiedMAS.Metric_Tool import llm_scene_timing  # type: ignore
    import base_agent_vlm
    FireAgent = base_agent_vlm.FireAgent
    SaveAgent = base_agent_vlm.SaveAgent
    LimitedWaterFireAgent = base_agent_vlm.LimitedWaterFireAgent
    import llm_config  # API profiles belong in env/untracked files — never commit secrets
    get_llm_config = llm_config.get_llm_config
else:
    from ..base_agent_vlm import (  # type: ignore
        FireAgent,
        SaveAgent,
        LimitedWaterFireAgent,
    )
    from ...llm_config import get_llm_config  # type: ignore
    from ...Metric_Tool import llm_scene_timing  # type: ignore

Z_HEIGHT = 1000
BURN_TIME = 1

_DEFAULT_PROMPT_ENV: dict[str, Any] = {
    "num_agents": 3,
    "num_fire_agents": 2,
    "num_rescue_agents": 1,
    "num_civilians": 4,
    "num_fires": "unknown",
    "other_info": "",
}


def _local_image_path_to_image_url_part(path: str) -> Optional[dict[str, Any]]:
    """Same as Single_agent/llm_agent_vlm: wrap a local image file as an OpenAI multimodal part."""
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


def _pixel_xy_from_parameter(ap: Any) -> Optional[tuple[float, float]]:
    """Mosaic pixel keys: ``{\"x\",\"y\"}`` or ``{\"px\",\"py\"}``."""
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


class SuperLLMController:
    """
    Centralized VLM planner: ``assemble_prompt("super", "CENTRAL", obs="VLM")`` with per-robot ``robot_actions``;
    user messages may mix text plus each robot's observation image (same recipe as ``llm_agent_vlm``).
    """

    def __init__(
        self,
        max_memory_size: int = 3,
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        self._llm_config = get_llm_config()
        client_kwargs = self._llm_config.get_client_kwargs()
        self._client = OpenAI(**client_kwargs)
        self._system_prompt = self._load_system_prompt(prompt_environment)
        self._memory: list[dict[str, Any]] = []
        self._step_counter = 0
        self._max_memory_size = max_memory_size
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
        return assemble_prompt("super", "CENTRAL", env, obs="VLM")

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
        return mem_path

    @staticmethod
    def _robot_id_sort_key(rid: str) -> tuple[int, str]:
        try:
            return (int(rid.replace("agent_", "")), rid)
        except ValueError:
            return (0, rid)

    def _format_memory_entry(self, mem: dict[str, Any], is_detailed: bool = False) -> str:
        """One line per super step; per_agent aligned with super_agent_CMAS_vl / single VLM."""
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
                    if action == "move_by":
                        d = action_params.get("distance")
                        a = action_params.get("angle")
                        if d is not None or a is not None:
                            sub.append(f"params=d={d!r},a={a!r}")
                    elif (pxy := action_params.get("pixel_xy")) is not None:
                        sub.append(f"params=pixel_xy={pxy!r}")
                if "move_to_success" in info and info.get("move_to_success") is not None:
                    sub.append(f"move_ok={info['move_to_success']}")
                elif "move_to_success" in info:
                    sub.append("move_ok=None")
                if "sendfollow_received" in info:
                    sub.append(f"follow_rx={info['sendfollow_received']}")
                agent_bits.append("[" + " ".join(sub) + "]")
            if agent_bits:
                parts.append("agents=" + " ".join(agent_bits))
        if (p := mem.get("high_level_plan")):
            parts.append(f"plan={p!r}")
        if (s := mem.get("summary")):
            parts.append(f"summary={s!r}")
        return "  " + " | ".join(parts)

    def _format_memory(self) -> str:
        if not self._memory:
            return "No previous actions."
        lines = [f"Previous actions (last {len(self._memory)} actions):"]
        for mem in self._memory:
            lines.append(self._format_memory_entry(mem))
        return "\n".join(lines)

    def _parse_json_from_response(self, response_text: str) -> Any:
        response_text = response_text.strip()
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
        Parse centralized JSON; each robot's ``action`` / ``action_parameter`` matches Single_agent/llm_agent_vlm
        (``move_to`` / ``extinguish_fire`` use mosaic pixels x,y or px,py).
        """
        response_text = response_text.strip()
        robot_actions: Dict[str, dict[str, Any]] = {}
        response_json = self._parse_json_from_response(response_text)

        if not isinstance(response_json, dict):
            print(f"[SuperLLMController-VLM] Response is not a JSON object: {type(response_json)}")
            return {}

        summary = response_json.get("summary", "")
        high_level_plan = response_json.get("high_level_plan", "")
        self._current_summary = summary if summary else None
        self._current_high_level_plan = high_level_plan if high_level_plan else None

        robot_actions_dict = response_json.get("robot_actions", {})
        if not isinstance(robot_actions_dict, dict):
            print(f"[SuperLLMController-VLM] robot_actions is not a dictionary: {type(robot_actions_dict)}")
            return {}

        for robot_id in robot_ids:
            if robot_id not in robot_actions_dict:
                print(f"[SuperLLMController-VLM] Warning: No action assigned to {robot_id}")
                continue

            robot_action_data = robot_actions_dict[robot_id]
            if not isinstance(robot_action_data, dict):
                print(f"[SuperLLMController-VLM] Action data for {robot_id} is not a dictionary")
                continue

            action = robot_action_data.get("action")
            action_parameter = robot_action_data.get("action_parameter")
            if not action:
                print(f"[SuperLLMController-VLM] Warning: No action specified for {robot_id}")
                continue

            action_dict: dict[str, Any] = {
                "action": action,
                "high_level_plan": high_level_plan,
                "summary": summary,
            }

            if action in ("move_to", "extinguish_fire"):
                action_dict["pixel_xy"] = _pixel_xy_from_parameter(action_parameter)
            elif action == "move_by":
                action_dict["distance"] = float(action_parameter["distance"])
                action_dict["angle"] = float(action_parameter["angle"])

            robot_actions[robot_id] = action_dict

        return robot_actions

    def _build_llm_content(
        self,
        all_agent_positions: Dict[str, str],
        global_perception_text: str,
        robot_ids: List[str],
        robot_image_paths: Optional[Dict[str, List[str]]] = None,
    ) -> list[dict[str, Any]]:
        content_items: list[dict[str, Any]] = []
        content_items.append({"type": "text", "text": f"MEMORY:\n{self._format_memory()}"})
        print(f"[SuperLLMController-VLM] Memory text: {self._format_memory()}")

        robot_list_str = ", ".join([f'"{rid}"' for rid in robot_ids])
        content_items.append({"type": "text", "text": f"ACTIVE ROBOT LIST: [{robot_list_str}]"})

        position_lines = ["ROBOT STATES:"]
        for robot_id, position_text in all_agent_positions.items():
            position_lines.append(f"  {robot_id}: {position_text}")
        content_items.append({"type": "text", "text": "\n".join(position_lines)})

        if global_perception_text.strip():
            content_items.append({"type": "text", "text": global_perception_text})

        if robot_image_paths:
            for rid in robot_ids:
                paths = robot_image_paths.get(rid) or []
                if not paths:
                    continue
                content_items.append({
                    "type": "text",
                    "text": f"RECENT EXPLORE MOSAIC for {rid} (single RGB):",
                })
                for path in paths:
                    part = _local_image_path_to_image_url_part(path)
                    if part is not None:
                        content_items.append(part)
                    else:
                        print(f"[SuperLLMController-VLM] Skip missing perception image: {path}")

        return content_items

    async def _call_llm(
        self,
        all_agent_positions: Dict[str, str],
        global_perception_text: str,
        robot_ids: List[str],
        robot_image_paths: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, dict[str, Any]]:
        content_items = self._build_llm_content(
            all_agent_positions,
            global_perception_text,
            robot_ids,
            robot_image_paths=robot_image_paths,
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content_items},
        ]
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
        print(f"[SuperLLMController-VLM] LLM Response: {response_text}")
        return self._parse_response(response_text, robot_ids)

    def get_current_summary(self) -> Optional[str]:
        return self._current_summary

    def get_current_high_level_plan(self) -> Optional[str]:
        return self._current_high_level_plan

    def save_memory(self, last_step_result: Optional[dict[str, Any]] = None, agent_id: Optional[str] = None) -> Optional[Path]:
        return self.flush_session_log(log_round=0, agent_id=agent_id)


class SuperLLMAgent:
    """
    Centralized VLM multi-robot stack: no simulated object-name list is injected into the LLM; each robot contributes
    pose plus ``last_locator_visualization_paths`` (RGB mosaic); ``move_to`` / ``extinguish_fire`` consume pixel coords.
    """

    def __init__(
        self,
        context: WorldContext,
        n_agents: int = 3,
        controller: Optional[SuperLLMController] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        memory_log_agent_id: str = "super_vlm_agent",
        *,
        water_capacity: int = 5,
        recover_time: int = 10,
    ):
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
        self._agents: Dict[str, FireAgent | SaveAgent | LimitedWaterFireAgent] = {}
        self._agent_types: Dict[str, str] = {}
        self._step_counter = 0

    @staticmethod
    def _valid_actions_for_agent_type(agent_type: str) -> set[str]:
        base = {
            "move_to",
            "move_by",
            "send_follow",
            "send_stop_follow",
            "wait",
        }
        if agent_type in ("FD", "FD_WL"):
            return base | {"extinguish_fire"}
        return base

    async def spawn_agents(
        self,
        agent_positions: List[Dict[str, Any]],
        agent_types: List[str],
    ) -> None:
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

            pos_config = agent_positions[i]
            location = ts.Vector3(
                pos_config.get("x"),
                pos_config.get("y"),
                pos_config.get("z"),
            )
            rotation_config = pos_config.get("rotation")
            if rotation_config:
                rotation = ts.math.euler_to_quaternion(
                    ts.Vector3(rotation_config),
                    is_degree=True,
                )
            else:
                rotation = None

            agent_actor = await agent.spawn_agent(
                name=f"SuperVLMAgent_{i+1}",
                location=location,
                rotation=rotation,
            )
            agent._agent = agent_actor
            self._agents[robot_id] = agent
            self._agent_types[robot_id] = agent_type
            print(
                f"[SuperLLMAgent-VLM] Spawned {robot_id}: "
                f"{agent_actor.get('id') if isinstance(agent_actor, dict) else agent_actor}"
            )

    def _get_robot_ids(self) -> List[str]:
        return [f"agent_{i+1}" for i in range(self._n_agents) if f"agent_{i+1}" in self._agents]

    def _collect_robot_image_paths(self) -> Dict[str, List[str]]:
        """Same selection as ``llm_agent_vlm.step``: RGB mosaic paths only."""
        result: Dict[str, List[str]] = {}
        for robot_id, agent in self._agents.items():
            if not agent._agent:
                continue
            paths: List[str] = []
            vis = getattr(agent._actions, "last_locator_visualization_paths", None)
            if vis:
                p = vis.get("rgb")
                if p and Path(p).is_file():
                    paths.append(p)
                elif p:
                    print(f"[SuperLLMAgent-VLM] {robot_id} mosaic RGB path not on disk: {p}")
            else:
                print(
                    f"[SuperLLMAgent-VLM] {robot_id}: no last_locator_visualization_paths "
                    "(run explore first)."
                )
            result[robot_id] = paths
        return result

    async def _collect_all_positions(self) -> Dict[str, str]:
        positions: Dict[str, str] = {}
        for robot_id, agent in self._agents.items():
            if agent._agent:
                agent_position = await agent.get_agent_position_and_forward()
                positions[robot_id] = agent.format_agent_position_text(agent_position)
        return positions

    async def step(self) -> None:
        robot_ids = self._get_robot_ids()
        if not robot_ids:
            print("[SuperLLMAgent-VLM] No agents available")
            return

        self._log_round += 1
        self.controller.clear_step_timing()

        print("[SuperLLMAgent-VLM] Collecting robot positions and mosaic images...")
        global_text = ""
        all_positions = await self._collect_all_positions()
        robot_image_paths = self._collect_robot_image_paths()

        print("[SuperLLMAgent-VLM] Calling LLM for centralized decision...")
        robot_actions = await self.controller._call_llm(
            all_agent_positions=all_positions,
            global_perception_text=global_text,
            robot_ids=robot_ids,
            robot_image_paths=robot_image_paths,
        )
        if not robot_actions:
            print("[SuperLLMAgent-VLM] No actions received from LLM")
            self.controller.flush_session_log(
                log_round=self._log_round,
                agent_id=self._memory_log_agent_id,
            )
            return
        await self._execute_actions_parallel(robot_actions)
        self.controller.flush_session_log(
            log_round=self._log_round,
            agent_id=self._memory_log_agent_id,
        )

    async def _execute_actions_parallel(self, robot_actions: Dict[str, dict[str, Any]]) -> None:
        self._step_counter += 1
        current_step = self._step_counter

        per_agent: Dict[str, dict[str, Any]] = {}
        for robot_id, action_dict in robot_actions.items():
            if robot_id not in self._agents:
                print(f"[SuperLLMAgent-VLM] Warning: Unknown robot_id {robot_id}")
                continue

            agent_type = self._agent_types.get(robot_id, "FD")
            action = action_dict.get("action")
            if action not in self._valid_actions_for_agent_type(agent_type):
                print(
                    f"[SuperLLMAgent-VLM] Warning: Invalid action for {robot_id} ({agent_type}): {action}, not recorded"
                )
                continue

            action_params: dict[str, Any] = {}
            if action in ("move_to", "extinguish_fire"):
                action_params = {"pixel_xy": action_dict.get("pixel_xy")}
            elif action == "move_by":
                dist, ang = action_dict.get("distance"), action_dict.get("angle")
                action_params = {"distance": float(dist), "angle": float(ang)}

            per_agent[robot_id] = {"action": action, "action_params": action_params}

        if per_agent:
            self.controller._memory.append({
                "step": current_step,
                "summary": self.controller.get_current_summary() or "",
                "high_level_plan": self.controller.get_current_high_level_plan() or "",
                "per_agent": per_agent,
            })
            while len(self.controller._memory) > self.controller._max_memory_size:
                self.controller._memory.pop(0)

        def _update_memory(rid: str, step: int, **kwargs):
            for mem in reversed(self.controller._memory):
                if mem.get("step") != step:
                    continue
                pa = mem.get("per_agent")
                if not isinstance(pa, dict) or rid not in pa:
                    continue
                pa[rid].update(kwargs)
                break

        async def execute_agent_action(robot_id: str, action_dict: dict[str, Any]):
            if robot_id not in self._agents:
                print(f"[SuperLLMAgent-VLM] Error: Robot {robot_id} not found")
                return
            agent = self._agents[robot_id]
            if not agent._agent:
                print(f"[SuperLLMAgent-VLM] Error: Agent {robot_id} not spawned")
                return

            agent_type = self._agent_types.get(robot_id, "FD")
            valid = self._valid_actions_for_agent_type(agent_type)
            action = action_dict.get("action")
            if action not in valid:
                print(f"[SuperLLMAgent-VLM] Skipping invalid action for {robot_id} ({agent_type}): {action}")
                return

            if action == "move_to":
                pxy = action_dict.get("pixel_xy")
                print(f"[SuperLLMAgent-VLM] Executing {robot_id}: {action} pixel_xy={pxy!r}")
            else:
                print(f"[SuperLLMAgent-VLM] Executing {robot_id}: {action}")

            try:
                if action == "move_to":
                    pxy = action_dict.get("pixel_xy")
                    if not pxy or not isinstance(pxy, (tuple, list)) or len(pxy) < 2:
                        print(f"[SuperLLMAgent-VLM] Error: {robot_id} move_to missing pixel_xy")
                        return
                    nav = await agent.move_to(
                        (float(pxy[0]), float(pxy[1])),
                        timeout=60.0,
                        tolerance_uu=50.0,
                    )
                    move_ok = nav.get("success") if isinstance(nav, dict) else None
                    _update_memory(robot_id, current_step, move_to_success=move_ok)
                    if isinstance(nav, dict) and not nav.get("success"):
                        print(
                            f"[SuperLLMAgent-VLM] Warning: {robot_id} navigation failed: {nav.get('message')!r}"
                        )

                elif action == "move_by":
                    dist = action_dict.get("distance")
                    ang = action_dict.get("angle")
                    dist_f = float(dist)
                    ang_f = float(ang)
                    nav = await agent.move_by(
                        dist_f,
                        ang_f,
                        timeout=60.0,
                        tolerance_uu=50.0,
                    )
                    move_ok = nav.get("success") if isinstance(nav, dict) else None
                    _update_memory(robot_id, current_step, move_to_success=move_ok)
                    if isinstance(nav, dict) and not nav.get("success"):
                        print(
                            f"[SuperLLMAgent-VLM] Warning: {robot_id} navigation failed: {nav.get('message')!r}"
                        )

                elif action == "extinguish_fire":
                    pxy = action_dict.get("pixel_xy")
                    pixel_arg: Optional[tuple[float, float]] = None
                    if pxy is not None:
                        if isinstance(pxy, (tuple, list)) and len(pxy) >= 2:
                            pixel_arg = (float(pxy[0]), float(pxy[1]))
                        else:
                            print(f"[SuperLLMAgent-VLM] Error: {robot_id} invalid pixel_xy: {pxy!r}")
                            return
                    result = await agent.extinguish_fire(
                        actor=agent._agent,
                        pixel_xy=pixel_arg,
                        timeout=5.0,
                    )
                    print(f"[SuperLLMAgent-VLM] {robot_id} extinguish_fire result: {result}")

                elif action == "send_follow":
                    result = await agent.send_follow()
                    print(f"[SuperLLMAgent-VLM] {robot_id} send_follow result: {result}")
                    _update_memory(
                        robot_id,
                        current_step,
                        sendfollow_received=bool(result),
                    )

                elif action == "send_stop_follow":
                    result = await agent.send_stop_follow()
                    print(f"[SuperLLMAgent-VLM] {robot_id} send_stop_follow result: {result}")

                elif action == "wait":
                    await agent.wait()
                    print(f"[SuperLLMAgent-VLM] {robot_id} wait completed")

                else:
                    print(f"[SuperLLMAgent-VLM] Warning: Unknown action {action} for {robot_id}")

            except Exception as e:
                print(f"[SuperLLMAgent-VLM] Error executing {action} for {robot_id}: {e}")
                import traceback
                traceback.print_exc()

        tasks = [
            execute_agent_action(rid, ad)
            for rid, ad in robot_actions.items()
            if rid in self._agents
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


async def demo_run(context: WorldContext, n_agents: int = 3, max_steps: int = 10) -> None:
    print(f"[SuperLLMAgent-VLM] Starting demo with {n_agents} agents, {max_steps} steps")
    await ts.UnaryAPI.refresh_actors_map(context.conn)
    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(BURN_TIME)

    custom_positions = [
        {"x": -400, "y": 0, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
        {"x": -200, "y": -100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
        {"x": -200, "y": 100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
    ]
    agent_types = ["FD", "FD", "FD"]

    super_agent = SuperLLMAgent(context, n_agents=n_agents)
    print("[SuperLLMAgent-VLM] Spawning agents...")
    await super_agent.spawn_agents(custom_positions, agent_types)
    print(f"[SuperLLMAgent-VLM] Spawned {len(super_agent._agents)} agents")

    print("[SuperLLMAgent-VLM] Initial explore (mosaic + front perception)...")
    for _rid, agent in super_agent._agents.items():
        if agent._agent:
            await agent.explore()

    print("[SuperLLMAgent-VLM] Centralized control loop...")
    for step in range(max_steps):
        print(f"\n[SuperLLMAgent-VLM] === Step {step + 1} ===")
        await super_agent.step()

    print("\n[SuperLLMAgent-VLM] Demo completed")


def main() -> None:
    GRPC_ENDPOINT = "127.0.0.1:5726"
    print("[SuperLLMAgent-VLM] Connecting to TongSim ...")
    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        ue.context.sync_run(demo_run(ue.context, n_agents=3, max_steps=50))


if __name__ == "__main__":
    try:
        from Metric_Tool.llm_token_evaluation import install as _install_llm_token_metrics
    except ImportError:
        from EmbodiedMAS.Metric_Tool.llm_token_evaluation import (  # type: ignore
            install as _install_llm_token_metrics,
        )
    _install_llm_token_metrics()
    try:
        from Metric_Tool.perception_evaluation import install_perception_evaluation as _install_pe
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import (  # type: ignore
            install_perception_evaluation as _install_pe,
        )
    _install_pe()
    main()
