from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Callable

from tongsim.core.world_context import WorldContext

# Allow running as a script and as a module
if __package__ is None or __package__ == "":
    import os
    import sys

    _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _vl_agent_dir = os.path.dirname(os.path.dirname(__file__))
    _single_dir = os.path.dirname(__file__)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    if _vl_agent_dir not in sys.path:
        sys.path.insert(0, _vl_agent_dir)
    if _single_dir not in sys.path:
        sys.path.insert(0, _single_dir)

    import base_agent_h
    FireAgent = base_agent_h.FireAgent
    LimitedWaterFireAgent = base_agent_h.LimitedWaterFireAgent
    SaveAgent = base_agent_h.SaveAgent

    import human_ui_vlm
    HumanDecisionRequest = human_ui_vlm.HumanDecisionRequest
    HumanDecisionResult = human_ui_vlm.HumanDecisionResult
    TkHumanDecisionProvider = human_ui_vlm.TkHumanDecisionProvider
    WebHumanDecisionProvider = human_ui_vlm.WebHumanDecisionProvider
    load_prompt_text = human_ui_vlm.load_prompt_text
else:
    from ..base_agent_h import FireAgent, LimitedWaterFireAgent, SaveAgent  # type: ignore
    from .human_ui_vlm import (
        HumanDecisionRequest,
        HumanDecisionResult,
        TkHumanDecisionProvider,
        WebHumanDecisionProvider,
        load_prompt_text,
    )


class HumanControllerVisionLanguage:
    """
    Human-input controller that mirrors VLM input/output schema.

    It exposes `_call_llm` to stay API-compatible with existing agent loops,
    but internally it shows observations to a participant and waits for input.
    """

    def __init__(
        self,
        max_memory_size: int = 100,
        agent_id: Optional[str] = None,
        prompt_type: str = "FD",
        prompt_environment: Optional[dict[str, Any]] = None,
        decision_provider: Optional[Any] = None,
    ):
        self._memory: list[dict[str, Any]] = []
        self._step_counter = 0
        self._max_memory_size = max_memory_size
        self._agent_id = agent_id
        self._current_summary: Optional[str] = None
        self._current_high_level_plan: Optional[str] = None

        self._observation_space_text, self._action_space_text = self._load_spaces(prompt_type)
        self._system_prompt_text = self._load_system_prompt(prompt_type, prompt_environment)
        self._scenario_mode, self._scenario_constraints = self._build_mode_info(prompt_type, prompt_environment)
        self._decision_provider = (
            decision_provider
            if decision_provider is not None
            else WebHumanDecisionProvider(agent_id=agent_id or "anonymous")
        )

    def _build_mode_info(
        self,
        prompt_type: str,
        prompt_environment: Optional[dict[str, Any]],
    ) -> tuple[str, str]:
        env = prompt_environment or {}
        if prompt_type == "FD_WL":
            wc = env.get("water_capacity", "unknown")
            rt = env.get("recover_time", "unknown")
            return "FD_WL", f"Limited water enabled: water_capacity={wc}, recover_time={rt}s."
        if prompt_type == "SD":
            return "SD", "Rescue mode; avoid extinguish_fire actions."
        return "FD", "Standard fire mode."

    def _load_spaces(self, prompt_type: str) -> tuple[str, str]:
        project_root = Path(__file__).resolve().parents[2]
        observation_path = project_root / "prompt" / "P7VLM2_Observation_Space.txt"
        action_path = project_root / "prompt" / "P8VLM2_Action_Space.txt"

        obs_text = load_prompt_text(str(observation_path))
        act_text = load_prompt_text(str(action_path))

        if prompt_type == "SD":
            # Rescue agents do not use extinguish_fire in current settings.
            act_text += "\n\n[NOTE] In SD mode, avoid 'extinguish_fire'."

        return obs_text, act_text

    def _load_system_prompt(
        self,
        prompt_type: str,
        prompt_environment: Optional[dict[str, Any]],
    ) -> str:
        import os
        import sys

        _mas_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _mas_root not in sys.path:
            sys.path.insert(0, _mas_root)
        from prompt.prompt_assembler import assemble_prompt

        role = "fire"
        if prompt_type == "SD":
            role = "rescue"
        elif prompt_type == "FD_WL":
            role = "limited_water_fire"

        env: dict[str, Any] = {
            "num_agents": 1,
            "num_fire_agents": 1,
            "num_rescue_agents": 0,
            "num_civilians": 0,
            "num_fires": "unknown",
            "other_info": "",
        }
        if prompt_environment:
            env.update(prompt_environment)

        return assemble_prompt(role, "DISTRIBUTED", env, obs="VLM")

    def _format_memory_entry(self, mem: dict[str, Any], is_detailed: bool = False) -> str:
        """Same as LLMControllerVisionLanguage: compact one line for MEMORY text and logs."""
        _ = is_detailed
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
            elif act in {"move_to", "extinguish_fire"}:
                px = action_params.get("pixel_x")
                py = action_params.get("pixel_y")
                if px is not None and py is not None:
                    parts.append(f"params=px={px!r},py={py!r}")
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
        if not self._memory:
            return "No previous actions."
        lines = [f"Previous actions (last {len(self._memory)} actions):"]
        for mem in self._memory:
            lines.append(self._format_memory_entry(mem))
        return "\n".join(lines)

    def _parse_human_result(self, result):
        action = (result.action or "").strip()
        summary = result.summary or ""
        high_level_plan = result.high_level_plan or ""
        action_parameter = result.action_parameter

        self._current_summary = summary if summary else None
        self._current_high_level_plan = high_level_plan if high_level_plan else None

        def _move_by_from_parameter(ap: Any) -> Optional[tuple[float, float]]:
            if not isinstance(ap, dict):
                return None
            try:
                distance = float(ap["distance"])
                angle = float(ap["angle"])
                return distance, angle
            except (TypeError, KeyError, ValueError):
                return None

        def _pixel_from_parameter(ap: Any) -> Optional[tuple[float, float]]:
            if ap is None:
                return None
            if isinstance(ap, (list, tuple)) and len(ap) >= 2:
                try:
                    return (float(ap[0]), float(ap[1]))
                except (TypeError, ValueError):
                    return None
            if isinstance(ap, dict):
                for kx, ky in (("x", "y"), ("px", "py"), ("pixel_x", "pixel_y")):
                    if kx in ap and ky in ap:
                        try:
                            return (float(ap[kx]), float(ap[ky]))
                        except (TypeError, ValueError):
                            return None
                pix = ap.get("pixel")
                if isinstance(pix, (list, tuple)) and len(pix) >= 2:
                    try:
                        return (float(pix[0]), float(pix[1]))
                    except (TypeError, ValueError):
                        return None
            return None

        if action == "move_to":
            pixel = _pixel_from_parameter(action_parameter)
            if pixel is None:
                print(
                    "[HumanControllerVisionLanguage] Missing or invalid pixel for action=move_to; skipping."
                )
                return []
            px, py = pixel
            return [
                {
                    "action": "move_to",
                    "pixel_x": px,
                    "pixel_y": py,
                    "high_level_plan": high_level_plan,
                    "summary": summary,
                }
            ]

        if action == "extinguish_fire":
            pixel = _pixel_from_parameter(action_parameter)
            row: dict[str, Any] = {
                "action": "extinguish_fire",
                "high_level_plan": high_level_plan,
                "summary": summary,
            }
            if pixel is not None:
                px, py = pixel
                row["pixel_x"] = px
                row["pixel_y"] = py
            return [row]

        if action == "move_by":
            move_by = _move_by_from_parameter(action_parameter)
            if move_by is None:
                print("[HumanControllerVisionLanguage] Missing or invalid move_by parameters; skipping.")
                return []
            distance, angle = move_by
            return [
                {
                    "action": "move_by",
                    "distance": distance,
                    "angle": angle,
                    "high_level_plan": high_level_plan,
                    "summary": summary,
                }
            ]

        if action in [
            "explore",
            "send_follow",
            "send_stop_follow",
            "wait",
        ]:
            return [
                {
                    "action": action,
                    "high_level_plan": high_level_plan,
                    "summary": summary,
                }
            ]

        print(f"[HumanControllerVisionLanguage] Unknown action={action}; skipping.")
        return []

    async def _call_llm(
        self,
        agent_position_text: str,
        perception_text: str,
        other_agent_info: Optional[str] = None,
        agent_idx: Optional[int] = None,
        perception_image_paths: Optional[List[str]] = None,
    ) -> list[dict[str, Any]]:
        """Compatibility method name retained to avoid changing loop code."""
        self._step_counter += 1

        request = HumanDecisionRequest(
            step=self._step_counter,
            observation_text=perception_text,
            agent_position_text=agent_position_text,
            memory_text=self._format_memory(),
            other_agent_info=other_agent_info,
            perception_image_paths=perception_image_paths or [],
            observation_space_text=self._observation_space_text,
            action_space_text=self._action_space_text,
            system_prompt_text=self._system_prompt_text,
            scenario_mode=self._scenario_mode,
            scenario_constraints=self._scenario_constraints,
            agent_id=self._agent_id,
        )

        if hasattr(self._decision_provider, "async_decide") and callable(self._decision_provider.async_decide):
            result = await self._decision_provider.async_decide(request)
        else:
            def _sync_decide():
                return self._decision_provider.decide(request)

            result = await asyncio.to_thread(_sync_decide)

        if self._agent_id:
            actor_str = self._agent_id
        elif agent_idx is not None:
            actor_str = f"agent_{agent_idx}"
        else:
            actor_str = "anonymous"

        print(
            f"[HumanControllerVisionLanguage {actor_str}] action={result.action}, "
            f"action_parameter={result.action_parameter}, summary={result.summary!r}"
        )

        return self._parse_human_result(result)

    def get_current_summary(self) -> Optional[str]:
        return self._current_summary

    def get_current_summary_and_high_level_plan(self) -> tuple[Optional[str], Optional[str]]:
        return self._current_summary, self._current_high_level_plan

    def save_memory(self, last_step_result: Optional[dict[str, Any]] = None, agent_id: Optional[str] = None) -> Optional[Path]:
        log_dir = Path("memory_logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_id_str = f"_{agent_id}" if agent_id else ""
        save_path = log_dir / f"human_vl_memory{agent_id_str}_{timestamp}.txt"

        lines = ["=" * 80, "Short-term MEMORY (Action History)", "=" * 80, ""]
        if not self._memory:
            lines.append("No previous actions.")
        else:
            lines.append(f"Total actions in memory: {len(self._memory)}")
            lines.append("")
            for mem in self._memory:
                lines.append(self._format_memory_entry(mem).strip())
                lines.append("")

        save_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[HumanControllerVisionLanguage] Memory saved to: {save_path}")
        return save_path


class HumanVisionAgent:
    """Base class for human-driven vision-language agents."""

    def __init__(
        self,
        context: WorldContext,
        agent,
        controller: Optional[HumanControllerVisionLanguage] = None,
        agent_id: Optional[str] = None,
        prompt_type: str = "FD",
        prompt_environment: Optional[dict[str, Any]] = None,
        decision_provider: Optional[Any] = None,
    ):
        self.agent = agent
        self._agent_id = agent_id
        self.controller = controller if controller is not None else HumanControllerVisionLanguage(
            agent_id=agent_id,
            prompt_type=prompt_type,
            prompt_environment=prompt_environment,
            decision_provider=decision_provider,
        )
        if controller is not None:
            controller._agent_id = agent_id
        self._step_counter = 0

    def _get_valid_actions(self) -> set[str]:
        return {
            "move_to",
            "move_by",
            "send_follow",
            "send_stop_follow",
            "explore",
            "wait",
        }

    def _format_perception_text(self) -> str:
        """Build OBJECT NAMES from simulator cache (default step unused; same as llm_agent_vlm.step)."""
        actions = getattr(self.agent, "_actions", None)
        perception_cache = getattr(actions, "_perception_object_list", None)

        names: list[str] = []
        if isinstance(perception_cache, dict):
            for key in ("actor_info", "npc_info"):
                entries = perception_cache.get(key)
                if not isinstance(entries, list):
                    continue
                for item in entries:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if isinstance(name, str) and name and name not in names:
                        names.append(name)

        if names:
            lines = ["OBJECT NAMES (detected nearby):", f"  Count: {len(names)}"]
            lines.extend([f"    - {name}" for name in names])
            return "\n".join(lines)

        return self.agent.format_object_name_list_text()

    async def _execute_specific_action(
        self,
        action: str,
        action_dict: dict[str, Any],
        current_step: int,
        action_index: int,
        _update_memory: Callable[..., None],
    ) -> None:
        # Subclasses can extend; default no-op.
        _ = action
        _ = action_dict
        _ = current_step
        _ = action_index
        _ = _update_memory
        return

    def get_current_summary(self) -> Optional[str]:
        return self.controller.get_current_summary()

    def get_current_summary_and_high_level_plan(self) -> tuple[Optional[str], Optional[str]]:
        return self.controller.get_current_summary_and_high_level_plan()

    async def step(self, other_agent_info: Optional[str] = None, agent_idx: Optional[int] = None) -> None:
        if not self.agent._agent:
            print("[HumanVisionAgent] Error: Agent not spawned")
            return

        if self._step_counter == 0:
            try:
                await self.agent.explore()
                print("[HumanVisionAgent] Initial explore completed before first decision.")
            except Exception as e:
                print(f"[HumanVisionAgent] Warning: initial explore failed: {e}")

        agent_position = await self.agent.get_agent_position_and_forward()
        agent_position_text = self.agent.format_agent_position_text(agent_position)
        # Matches EmbodiedMAS/VLM_Agent/Single_agent/llm_agent_vlm.py: do not inject simulator object names; decisions use mosaic RGB.
        perception_text = ""
        print(f"[HumanVisionAgent] Agent position text: {agent_position_text}")

        perception_image_paths: List[str] = []
        vis = getattr(self.agent._actions, "last_locator_visualization_paths", None)
        if vis:
            for key in ("rgb",):
                p = vis.get(key)
                if p and Path(p).is_file():
                    perception_image_paths.append(p)
                elif p:
                    print(f"[HumanVisionAgent] Perception image path not on disk: {p}")

        action_list = await self.controller._call_llm(
            agent_position_text=agent_position_text,
            perception_text=perception_text,
            other_agent_info=other_agent_info,
            agent_idx=agent_idx,
            perception_image_paths=perception_image_paths or None,
        )

        if action_list:
            await self._execute_action(action_list)

    async def _execute_action(self, action_list: list[dict[str, Any]]) -> None:
        if not self.agent._agent or not action_list:
            print("[HumanVisionAgent] No agent or empty action list")
            return

        self._step_counter += 1
        current_step = self._step_counter

        def _update_memory(step: int, action_index: int, **kwargs):
            for mem in reversed(self.controller._memory):
                if mem.get("step") == step and mem.get("action_index") == action_index:
                    mem.update(kwargs)
                    break

        memory_entries = []
        for i, action_dict in enumerate(action_list):
            action = action_dict.get("action")
            action_params: dict[str, Any] = {}
            if action == "move_to":
                action_params = {
                    "pixel_x": action_dict.get("pixel_x"),
                    "pixel_y": action_dict.get("pixel_y"),
                }
            elif action == "extinguish_fire":
                action_params = {
                    "pixel_x": action_dict.get("pixel_x"),
                    "pixel_y": action_dict.get("pixel_y"),
                }
            elif action == "move_by":
                action_params = {
                    "distance": action_dict.get("distance"),
                    "angle": action_dict.get("angle"),
                }

            memory_entries.append(
                {
                    "step": current_step,
                    "action_index": i + 1,
                    "action": action,
                    "high_level_plan": action_dict.get("high_level_plan", ""),
                    "summary": action_dict.get("summary", ""),
                    "action_params": action_params,
                }
            )

        self.controller._memory.extend(memory_entries)
        while len(self.controller._memory) > self.controller._max_memory_size:
            self.controller._memory.pop(0)

        valid_actions = self._get_valid_actions()

        for i, action_dict in enumerate(action_list):
            action = action_dict.get("action")
            if action not in valid_actions:
                print(f"[HumanVisionAgent] Skipping invalid action: {action}")
                continue

            action_index = i + 1
            print(f"[HumanVisionAgent] Executing action {action_index}/{len(action_list)}: {action}")

            try:
                if action == "move_to":
                    pixel_x = action_dict.get("pixel_x")
                    pixel_y = action_dict.get("pixel_y")
                    if pixel_x is None or pixel_y is None:
                        print("[HumanVisionAgent] Error: move_to missing pixel_x or pixel_y")
                        continue

                    nav = await self.agent.move_to(
                        pixel_xy=(float(pixel_x), float(pixel_y)),
                        timeout=60.0,
                        tolerance_uu=50.0,
                    )
                    move_ok = nav.get("success") if isinstance(nav, dict) else None
                    _update_memory(current_step, action_index, move_to_success=move_ok)
                    if isinstance(nav, dict) and not nav.get("success"):
                        print(
                            f"[HumanVisionAgent] Warning: Navigation failed: {nav.get('message')!r}"
                        )

                elif action == "move_by":
                    distance = action_dict.get("distance")
                    angle = action_dict.get("angle")
                    if distance is None or angle is None:
                        print("[HumanVisionAgent] Error: move_by missing distance or angle")
                        continue

                    nav = await self.agent.move_by(
                        float(distance),
                        float(angle),
                        timeout=60.0,
                        tolerance_uu=50.0,
                    )
                    move_ok = nav.get("success") if isinstance(nav, dict) else None
                    _update_memory(current_step, action_index, move_to_success=move_ok)
                    if isinstance(nav, dict) and not nav.get("success"):
                        print(
                            f"[HumanVisionAgent] Warning: Navigation failed: {nav.get('message')!r}"
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

                elif action == "explore":
                    await self.agent.explore()

                elif action == "wait":
                    await self.agent.wait()

                else:
                    await self._execute_specific_action(action, action_dict, current_step, action_index, _update_memory)

            except Exception as e:
                print(f"[HumanVisionAgent] Error executing {action}: {e}")
                continue


class HumanVisionFireAgent(HumanVisionAgent):
    def __init__(
        self,
        context: WorldContext,
        controller: Optional[HumanControllerVisionLanguage] = None,
        agent_id: Optional[str] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        decision_provider: Optional[Any] = None,
    ):
        agent = FireAgent(context)
        super().__init__(
            context,
            agent,
            controller,
            agent_id=agent_id,
            prompt_type="FD",
            prompt_environment=prompt_environment,
            decision_provider=decision_provider,
        )

    def _get_valid_actions(self) -> set[str]:
        return {
            "move_to",
            "move_by",
            "extinguish_fire",
            "send_follow",
            "send_stop_follow",
            "explore",
            "wait",
        }

    async def _execute_specific_action(
        self,
        action: str,
        action_dict: dict[str, Any],
        current_step: int,
        action_index: int,
        _update_memory: Callable[..., None],
    ) -> None:
        _ = _update_memory
        if action != "extinguish_fire":
            return

        pixel_x = action_dict.get("pixel_x")
        pixel_y = action_dict.get("pixel_y")
        if pixel_x is not None and pixel_y is not None:
            result = await self.agent.extinguish_fire(
                actor=self.agent._agent,
                pixel_xy=(float(pixel_x), float(pixel_y)),
                timeout=5.0,
            )
            print(f"[HumanVisionFireAgent] Extinguish result: {result}")
            return

        result = await self.agent.extinguish_fire(
            actor=self.agent._agent,
            timeout=5.0,
        )
        print(f"[HumanVisionFireAgent] Extinguish result: {result}")


class HumanVisionSaveAgent(HumanVisionAgent):
    def __init__(
        self,
        context: WorldContext,
        controller: Optional[HumanControllerVisionLanguage] = None,
        agent_id: Optional[str] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        decision_provider: Optional[Any] = None,
    ):
        agent = SaveAgent(context)
        super().__init__(
            context,
            agent,
            controller,
            agent_id=agent_id,
            prompt_type="SD",
            prompt_environment=prompt_environment,
            decision_provider=decision_provider,
        )


class HumanVisionFireWaterLimitAgent(HumanVisionFireAgent):
    def __init__(
        self,
        context: WorldContext,
        controller: Optional[HumanControllerVisionLanguage] = None,
        agent_id: Optional[str] = None,
        prompt_environment: Optional[dict[str, Any]] = None,
        *,
        water_capacity: int = 5,
        recover_time: int = 10,
        decision_provider: Optional[Any] = None,
    ):
        agent = LimitedWaterFireAgent(
            context,
            water_capacity=water_capacity,
            recover_time=recover_time,
        )
        HumanVisionAgent.__init__(
            self,
            context,
            agent,
            controller,
            agent_id=agent_id,
            prompt_type="FD_WL",
            prompt_environment=prompt_environment,
            decision_provider=decision_provider,
        )
