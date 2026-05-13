from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Allow running as a script and as a module
if __package__ is None or __package__ == "":
    import os
    import sys

    _single_dir = os.path.dirname(__file__)
    if _single_dir not in sys.path:
        sys.path.insert(0, _single_dir)

    import human_ui_vlm
    HumanDecisionRequest = human_ui_vlm.HumanDecisionRequest
    HumanDecisionResult = human_ui_vlm.HumanDecisionResult
    ScriptedDecisionProvider = human_ui_vlm.ScriptedDecisionProvider
    TkHumanDecisionProvider = human_ui_vlm.TkHumanDecisionProvider
    load_prompt_text = human_ui_vlm.load_prompt_text

    _workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if _workspace_root not in sys.path:
        sys.path.insert(0, _workspace_root)
    from prompt.prompt_assembler import assemble_prompt
else:
    from .human_ui_vlm import (
        HumanDecisionRequest,
        HumanDecisionResult,
        ScriptedDecisionProvider,
        TkHumanDecisionProvider,
        load_prompt_text,
    )
    from prompt.prompt_assembler import assemble_prompt


@dataclass
class MockObservation:
    step: int
    agent_position_text: str
    perception_text: str
    image_paths: list[str]
    other_agent_info: Optional[str] = None


def make_default_mock_observations(total_steps: int = 5) -> list[MockObservation]:
    """Construct lightweight mock observations aligned with P7VLM2 inputs."""
    data: list[MockObservation] = []
    objects_by_step = [
        ["Fire_01", "Wall_1", "CorridorSign"],
        ["Fire_01", "Door_A", "Table_3"],
        ["Fire_01", "FireExtinguisherBox", "Chair_2"],
        ["Fire_01", "Window_2", "ExitSign"],
        ["Fire_01", "SafeZoneMarker", "Wall_2"],
    ]

    for i in range(total_steps):
        obj_names = objects_by_step[min(i, len(objects_by_step) - 1)]
        perception_lines = ["OBJECT NAMES (detected nearby):", f"  Count: {len(obj_names)}"]
        perception_lines.extend([f"    - {name}" for name in obj_names])

        data.append(
            MockObservation(
                step=i + 1,
                agent_position_text=(
                    f"AGENT POSITION: Current location at (x={-1400 + i * 120:.1f}, "
                    f"y={i * 35:.1f}, z=1000.0), facing forward (1,0)"
                ),
                perception_text="\n".join(perception_lines),
                image_paths=[
                    f"mock_images/step_{i + 1:02d}_rgb.png",
                    f"mock_images/step_{i + 1:02d}_seg.png",
                    f"mock_images/step_{i + 1:02d}_depth.png",
                ],
                other_agent_info=None,
            )
        )

    return data


class MockHumanBaselineRunner:
    """No-render dry-run runner for validating human action pipeline."""

    def __init__(self, decision_provider: Optional[Any] = None, prompt_root: Optional[Path] = None):
        self._decision_provider = decision_provider if decision_provider is not None else TkHumanDecisionProvider(
            title="Human Baseline Mock"
        )
        root = prompt_root or Path(__file__).resolve().parents[2] / "prompt"
        self._obs_space = load_prompt_text(str(root / "P7VLM2_Observation_Space.txt"))
        self._act_space = load_prompt_text(str(root / "P8VLM2_Action_Space.txt"))
        self._system_prompt = assemble_prompt(
            role="fire",
            coordination="DISTRIBUTED",
            environment={
                "num_agents": 1,
                "num_fire_agents": 1,
                "num_rescue_agents": 0,
                "num_civilians": 0,
                "num_fires": 1,
                "other_info": "",
            },
            obs="VLM",
        )
        self._memory: list[dict[str, Any]] = []

    def _memory_text(self) -> str:
        if not self._memory:
            return "No previous actions."
        return "\n".join(
            [
                "Previous actions:",
                *[
                    f"- Step {m['step']}: action={m['action']}, summary={m['summary']}"
                    for m in self._memory
                ],
            ]
        )

    def _normalize(self, result):
        action = result.action.strip()
        if action not in {
            "move_to",
            "move_by",
            "extinguish_fire",
            "explore",
            "send_follow",
            "send_stop_follow",
            "wait",
        }:
            action = "wait"

        action_parameter = result.action_parameter or None
        pixel_x: Optional[float] = None
        pixel_y: Optional[float] = None
        move_by_distance: Optional[float] = None
        move_by_angle: Optional[float] = None

        if action == "move_to":
            try:
                if not isinstance(action_parameter, dict):
                    raise ValueError("missing pixel dict")
                pixel_x = float(action_parameter.get("pixel_x", action_parameter.get("x")))
                pixel_y = float(action_parameter.get("pixel_y", action_parameter.get("y")))
            except (TypeError, ValueError):
                action = "wait"
        elif action == "extinguish_fire":
            if isinstance(action_parameter, dict) and (
                ("pixel_x" in action_parameter and "pixel_y" in action_parameter)
                or ("x" in action_parameter and "y" in action_parameter)
            ):
                try:
                    pixel_x = float(action_parameter.get("pixel_x", action_parameter.get("x")))
                    pixel_y = float(action_parameter.get("pixel_y", action_parameter.get("y")))
                except (TypeError, ValueError):
                    action = "wait"
        elif action == "move_by":
            try:
                if not isinstance(action_parameter, dict):
                    raise ValueError("missing move_by dict")
                move_by_distance = float(action_parameter.get("distance"))
                move_by_angle = float(action_parameter.get("angle"))
            except (TypeError, ValueError):
                action = "wait"

        out = {
            "summary": result.summary or f"Selected action: {action}",
            "high_level_plan": result.high_level_plan or "",
            "action": action,
            "action_parameter": (
                {
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                }
                if action in {"move_to", "extinguish_fire"} and pixel_x is not None and pixel_y is not None
                else (
                    {
                        "distance": move_by_distance,
                        "angle": move_by_angle,
                    }
                    if action == "move_by" and move_by_distance is not None and move_by_angle is not None
                    else None
                )
            ),
        }
        return out

    def run(self, observations: list[MockObservation]) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        for obs in observations:
            req = HumanDecisionRequest(
                step=obs.step,
                observation_text=obs.perception_text,
                agent_position_text=obs.agent_position_text,
                memory_text=self._memory_text(),
                other_agent_info=obs.other_agent_info,
                perception_image_paths=obs.image_paths,
                observation_space_text=self._obs_space,
                action_space_text=self._act_space,
                system_prompt_text=self._system_prompt,
            )
            raw = self._decision_provider.decide(req)
            normalized = self._normalize(raw)

            decisions.append(normalized)
            self._memory.append(
                {
                    "step": obs.step,
                    "action": normalized["action"],
                    "summary": normalized["summary"],
                }
            )
        return decisions


def _demo_scripted_run() -> None:
    scripted = ScriptedDecisionProvider(
        [
            HumanDecisionResult(
                summary="Move closer to fire",
                high_level_plan="Navigate to fire source",
                action="move_to",
                action_parameter={"x": 320, "y": 320},
            ),
            HumanDecisionResult(
                summary="Extinguish now",
                high_level_plan="Suppress nearby fire",
                action="extinguish_fire",
                action_parameter={"x": 320, "y": 320},
            ),
            HumanDecisionResult(
                summary="Observe after extinguish",
                high_level_plan="Collect new observations",
                action="explore",
                action_parameter=None,
            ),
        ]
    )
    runner = MockHumanBaselineRunner(decision_provider=scripted)
    obs = make_default_mock_observations(total_steps=3)
    decisions = runner.run(obs)
    print("Decisions from scripted mock run:")
    for i, d in enumerate(decisions, start=1):
        print(f"Step {i}: {d}")


if __name__ == "__main__":
    _demo_scripted_run()
