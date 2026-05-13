from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script and as a module.
if __package__ is None or __package__ == "":
    import os
    import sys

    _dmas_dir = os.path.dirname(__file__)
    _vlm_root = os.path.dirname(_dmas_dir)
    _workspace_root = os.path.dirname(_vlm_root)
    _single_dir = os.path.join(_vlm_root, "Single_agent")
    for _p in (_workspace_root, _vlm_root, _dmas_dir, _single_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from Human_Agent.Single_agent import human_ui_vlm
    HumanDecisionRequest = human_ui_vlm.HumanDecisionRequest
    HumanDecisionResult = human_ui_vlm.HumanDecisionResult
    ScriptedDecisionProvider = human_ui_vlm.ScriptedDecisionProvider
    TkHumanDecisionProvider = human_ui_vlm.TkHumanDecisionProvider
    load_prompt_text = human_ui_vlm.load_prompt_text

    from prompt.prompt_assembler import assemble_prompt
else:
    from ..Single_agent.human_ui_vlm import (
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
    agent_id: str
    agent_position_text: str
    perception_text: str
    image_paths: List[str]


@dataclass
class MockMessage:
    sender: str
    content: Dict[str, Any]


class MockMessageRouter:
    def __init__(self):
        self._boxes: Dict[str, asyncio.Queue] = {}

    def register(self, agent_id: str, q: asyncio.Queue) -> None:
        self._boxes[agent_id] = q

    async def broadcast_all(self, msg: MockMessage) -> None:
        if not self._boxes:
            return
        await asyncio.gather(*(q.put(msg) for q in self._boxes.values()))


class LockedDecisionProvider:
    """Serialize GUI prompts across multiple agents."""

    def __init__(self, inner_provider: Any, shared_lock: Optional[Any] = None):
        import threading

        self._inner = inner_provider
        self._lock = shared_lock if shared_lock is not None else threading.Lock()

    def decide(self, request: Any) -> Any:
        with self._lock:
            return self._inner.decide(request)


class MockHumanDMASAgent:
    def __init__(
        self,
        agent_id: str,
        decision_provider: Any,
        router: MockMessageRouter,
        prompt_root: Optional[Path] = None,
    ):
        self.agent_id = agent_id
        self._decision_provider = decision_provider
        self._router = router

        self._box: asyncio.Queue = asyncio.Queue()
        self._router.register(agent_id, self._box)

        root = prompt_root or Path(__file__).resolve().parents[2] / "prompt"
        self._obs_space = load_prompt_text(str(root / "P7VLM2_Observation_Space.txt"))
        self._act_space = load_prompt_text(str(root / "P8VLM2_Action_Space.txt"))
        self._system_prompt = assemble_prompt(
            role="fire",
            coordination="DISTRIBUTED",
            environment={
                "num_agents": 2,
                "num_fire_agents": 2,
                "num_rescue_agents": 0,
                "num_civilians": 0,
                "num_fires": 1,
                "other_info": "",
            },
            obs="VLM",
        )

        self.memory: List[Dict[str, Any]] = []
        self.current_summary: Optional[str] = None
        self.current_high_level_plan: Optional[str] = None
        self.received_other_info: List[Optional[str]] = []

    async def broadcast(self, content: Dict[str, Any]) -> None:
        await self._router.broadcast_all(MockMessage(self.agent_id, content))

    def _drain_inbox(self) -> List[MockMessage]:
        msgs: List[MockMessage] = []
        while True:
            try:
                msgs.append(self._box.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    def consume_other_agent_info(self) -> Optional[str]:
        messages = self._drain_inbox()
        if not messages:
            return None

        parts: List[str] = []
        for msg in messages:
            if msg.sender == self.agent_id:
                continue
            msg_type = msg.content.get("type")
            if msg_type == "summary":
                parts.append(
                    f"Agent {msg.sender} **Summary**: {msg.content.get('summary', '')} "
                    f"**High_level_plan**: {msg.content.get('high_level_plan', '')}"
                )
            elif msg_type == "position":
                loc = msg.content.get("location") or {}
                if isinstance(loc, dict) and {"x", "y", "z"} <= loc.keys():
                    parts.append(
                        f"Agent {msg.sender} position: ({loc['x']:.1f}, {loc['y']:.1f}, {loc['z']:.1f})"
                    )
            else:
                parts.append(f"Agent {msg.sender}: {msg.content}")
        return "\n".join(parts) if parts else None

    def _memory_text(self) -> str:
        if not self.memory:
            return "No previous actions."
        lines = ["Previous actions:"]
        for m in self.memory:
            parts = [
                f"step={m['step']}",
                f"action={m['action']}",
                f"summary={m['summary']}",
            ]
            if "move_to_success" in m:
                parts.append(f"move_ok={m['move_to_success']}")
            if "sendfollow_received" in m:
                parts.append(f"follow_rx={m['sendfollow_received']}")
            lines.append("- " + " | ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _normalize(result: Any) -> Dict[str, Any]:
        action = (result.action or "").strip()
        if action not in {
            "move_to",
            "move_by",
            "extinguish_fire",
            "explore",
            "send_follow",
            "send_stop_follow",
        }:
            action = "explore"

        action_parameter = result.action_parameter or None
        pixel_x = None
        pixel_y = None
        move_by_distance = None
        move_by_angle = None
        if action == "move_to":
            try:
                if not isinstance(action_parameter, dict):
                    raise ValueError("missing pixel dict")
                pixel_x = int(round(float(action_parameter.get("pixel_x", action_parameter.get("x")))))
                pixel_y = int(round(float(action_parameter.get("pixel_y", action_parameter.get("y")))))
            except (TypeError, ValueError):
                action = "explore"
        elif action == "extinguish_fire":
            try:
                if not isinstance(action_parameter, dict):
                    raise ValueError("missing pixel dict")
                pixel_x = int(round(float(action_parameter.get("pixel_x", action_parameter.get("x")))))
                pixel_y = int(round(float(action_parameter.get("pixel_y", action_parameter.get("y")))))
            except (TypeError, ValueError):
                action = "explore"
        elif action == "move_by":
            try:
                if not isinstance(action_parameter, dict):
                    raise ValueError("missing move_by dict")
                move_by_distance = float(action_parameter.get("distance"))
                move_by_angle = float(action_parameter.get("angle"))
            except (TypeError, ValueError):
                action = "wait"

        return {
            "summary": result.summary or f"Selected action: {action}",
            "high_level_plan": result.high_level_plan or "",
            "action": action,
            "action_parameter":
                (
                    {
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                    }
                    if pixel_x is not None and pixel_y is not None
                    else None
                )
                if action in {"move_to", "extinguish_fire"}
                else (
                    {
                        "distance": move_by_distance,
                        "angle": move_by_angle,
                    }
                    if action == "move_by" and move_by_distance is not None and move_by_angle is not None
                    else None
                ),
        }

    async def step_once(self, observation: MockObservation, other_agent_info: Optional[str]) -> Dict[str, Any]:
        request = HumanDecisionRequest(
            step=observation.step,
            observation_text=observation.perception_text,
            agent_position_text=observation.agent_position_text,
            memory_text=self._memory_text(),
            other_agent_info=other_agent_info,
            perception_image_paths=observation.image_paths,
            observation_space_text=self._obs_space,
            action_space_text=self._act_space,
            system_prompt_text=self._system_prompt,
            scenario_mode="FD",
            scenario_constraints="Mock decentralized test.",
            agent_id=self.agent_id,
        )

        if hasattr(self._decision_provider, "async_decide") and callable(self._decision_provider.async_decide):
            raw = await self._decision_provider.async_decide(request)
        else:
            raw = await asyncio.to_thread(self._decision_provider.decide, request)
        normalized = self._normalize(raw)

        self.current_summary = normalized["summary"]
        self.current_high_level_plan = normalized["high_level_plan"]
        self.memory.append(
            {
                "step": observation.step,
                "action": normalized["action"],
                "summary": normalized["summary"],
            }
        )
        return normalized


def make_default_mock_observation(agent_id: str, step: int) -> MockObservation:
    names = ["Fire_01", f"Corridor_{agent_id}", f"Marker_{step}"]
    return MockObservation(
        step=step,
        agent_id=agent_id,
        agent_position_text=(
            f"AGENT POSITION: Current location at (x={-1400 + step * 80:.1f}, "
            f"y={step * 20:.1f}, z=1000.0), facing forward (1,0)"
        ),
        perception_text="\n".join(
            [
                "OBJECT NAMES (detected nearby):",
                f"  Count: {len(names)}",
                *[f"    - {n}" for n in names],
                "",
                "DISTRESS SIGNALS: none",
            ]
        ),
        image_paths=[
            f"mock_images/{agent_id}_step_{step:02d}_rgb.jpg",
            f"mock_images/{agent_id}_step_{step:02d}_seg.jpg",
            f"mock_images/{agent_id}_step_{step:02d}_depth.jpg",
        ],
    )


class MockDMASHumanRunner:
    """Async decentralized runner for fast validation without TongSim."""

    def __init__(
        self,
        agents: List[MockHumanDMASAgent],
        *,
        inter_step_delay: float = 0.01,
        observation_factory: Optional[Any] = None,
    ):
        self.agents = agents
        self.inter_step_delay = inter_step_delay
        self.observation_factory = observation_factory

    async def run(self, max_steps: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        logs: Dict[str, List[Dict[str, Any]]] = {agent.agent_id: [] for agent in self.agents}

        async def _loop(agent: MockHumanDMASAgent, agent_idx: int) -> None:
            for step in range(1, max_steps + 1):
                if self.observation_factory is None:
                    obs = make_default_mock_observation(agent.agent_id, step)
                else:
                    obs = self.observation_factory(agent.agent_id, step)
                other_info = agent.consume_other_agent_info()
                agent.received_other_info.append(other_info)

                decision = await agent.step_once(obs, other_info)
                logs[agent.agent_id].append(decision)

                await agent.broadcast(
                    {
                        "type": "summary",
                        "summary": decision["summary"],
                        "high_level_plan": decision["high_level_plan"],
                        "step": step,
                    }
                )
                await agent.broadcast(
                    {
                        "type": "position",
                        "location": {
                            "x": -1400 + 90 * step + agent_idx * 20,
                            "y": 15 * step + agent_idx * 30,
                            "z": 1000.0,
                        },
                        "step": step,
                    }
                )

                await asyncio.sleep(self.inter_step_delay)

        await asyncio.gather(*[asyncio.create_task(_loop(a, idx)) for idx, a in enumerate(self.agents)])
        return logs


def _demo_scripted_run() -> None:
    router = MockMessageRouter()
    provider_a = ScriptedDecisionProvider(
        [
            HumanDecisionResult(
                summary="Move closer",
                high_level_plan="Approach Fire_01",
                action="move_to",
                action_parameter={"name": "Fire_01"},
            ),
            HumanDecisionResult(
                summary="Pause",
                high_level_plan="Hold",
                action="wait",
                action_parameter=None,
            ),
        ]
    )
    provider_b = ScriptedDecisionProvider(
        [
            HumanDecisionResult(
                summary="Explore corridor",
                high_level_plan="Scan first",
                action="explore",
                action_parameter=None,
            ),
            HumanDecisionResult(
                summary="Extinguish fire",
                high_level_plan="Suppress now",
                action="extinguish_fire",
                action_parameter={"name": "Fire_01"},
            ),
        ]
    )

    agents = [
        MockHumanDMASAgent("A0", provider_a, router),
        MockHumanDMASAgent("A1", provider_b, router),
    ]
    logs = asyncio.run(MockDMASHumanRunner(agents).run(max_steps=2))

    print("Mock decentralized run completed.")
    for aid, decisions in logs.items():
        print(f"{aid}: {decisions}")


if __name__ == "__main__":
    _demo_scripted_run()
