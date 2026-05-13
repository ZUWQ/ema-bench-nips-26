from __future__ import annotations

import asyncio
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import tongsim as ts
from tongsim.core.world_context import WorldContext
import time

# Same path setup as Single_agent/SA_ol: EmbodiedMAS parent, OL root, this package, Single_agent (for `import llm_agent`)
_cfmas_dir = os.path.dirname(os.path.abspath(__file__))
_ol_agent_dir = os.path.dirname(_cfmas_dir)
_parent_dir = os.path.dirname(_ol_agent_dir)
_single_agent_dir = os.path.join(_ol_agent_dir, "Single_agent")
for _p in (_parent_dir, _ol_agent_dir, _cfmas_dir, _single_agent_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_agent
import llm_agent_SD

LLMAgentDMASFire = llm_agent.LLMAgentDMASFire
LLMAgentDMASSave = llm_agent_SD.LLMAgentDMASSave
LLMAgentDMASFireWaterLimit = llm_agent.LLMAgentDMASFireWaterLimit
LLMControllerDMAS = llm_agent.LLMControllerDMAS
# Optional: swap LLMAgentDMASFire / LLMAgentDMASSave inside DMASAgent for
# Single_agent.llm_agent_mid_replan LLMAgentDMASFireMidReplan / LLMAgentDMASSaveMidReplan,
# and set mid_action_llm_enabled=True, mid_action_llm_after_sec=..., etc.

try:
    from Metric_Tool.evaluation import (
        ExperimentResult,
        attach_experiment_result_to_base_agent,
    )
    from Metric_Tool.llm_token_evaluation import flush_token_summary
except ImportError:
    from EmbodiedMAS.Metric_Tool.evaluation import (  # type: ignore
        ExperimentResult,
        attach_experiment_result_to_base_agent,
    )
    from EmbodiedMAS.Metric_Tool.llm_token_evaluation import (  # type: ignore
        flush_token_summary,
    )

# Global z-height constant for agent spawning
Z_HEIGHT = 1000
BURN_TIME = 1

# Default when demo_run(..., scene_id=...) omits it; never read sys.argv at import time —
# DMAS_benchmark_runner loads this module via importlib while argv still holds the runner's --backend, etc.
CURRENT_SCENE_ID = 0

def _default_prompt_env(agent_types: List[str]) -> Dict[str, Any]:
    """Match CMAS_ol._default_prompt_env; firefighter count includes FD and FD_WL."""
    return {
        "num_agents": len(agent_types),
        "num_fire_agents": sum(1 for t in agent_types if t in ("FD", "FD_WL")),
        "num_rescue_agents": sum(1 for t in agent_types if t == "SD"),
        "num_civilians": 0,
        "num_fires": "unknown",
    }


# ---------- Messaging ----------
@dataclass
class Message:
    sender: str
    content: Dict[str, Any]


class MessageRouter:
    def __init__(self):
        self._boxes: Dict[str, asyncio.Queue] = {}

    def clear(self) -> None:
        """Clear mailboxes before rerunning a demo in-process (stale agent ids)."""
        self._boxes.clear()

    def register(self, aid: str, q: asyncio.Queue):
        self._boxes[aid] = q

    async def broadcast_all(self, msg: Message) -> None:
        if not self._boxes:
            return
        await asyncio.gather(*(q.put(msg) for q in self._boxes.values()))

# Process-global message router
router = MessageRouter()

# ---------- DMASAgent ----------
class DMASAgent:
    """
    Distributed Multi-Agent with communication capabilities.
    Uses composition to wrap LLMAgent/LLMAgentFire/LLMAgentSave instances.
    """
    def __init__(
        self,
        agent_id: str,
        context: WorldContext,
        agent_type: str = "FD",
        controller: Optional[LLMControllerDMAS] = None,
        prompt_environment: Optional[Dict[str, Any]] = None,
        *,
        water_capacity: int = 1,
        recover_time: int = 10,
    ):
        """
        Initialize DMASAgent.
        
        Args:
            agent_id: Agent identifier
            context: WorldContext instance
            agent_type: "FD" | "SD" | "FD_WL" (limited water; see llm_agent.LLMAgentDMASFireWaterLimit)
            controller: Optional LLMController instance
            prompt_environment: Placeholders for prompt_assembler (aligned with SA_ol / verify_prompt)
            water_capacity / recover_time: Only for agent_type=="FD_WL" → LLMAgentDMASFireWaterLimit
        """
        # Instantiate the matching LLMAgent subclass (same pattern as SA_ol)
        if agent_type == "FD":
            self.llm_agent = LLMAgentDMASFire(
                context,
                controller,
                prompt_environment=prompt_environment,
            )
        elif agent_type == "SD":
            self.llm_agent = LLMAgentDMASSave(
                context,
                controller,
                prompt_environment=prompt_environment,
            )
        elif agent_type == "FD_WL":
            self.llm_agent = LLMAgentDMASFireWaterLimit(
                context,
                controller,
                prompt_environment=prompt_environment,
                water_capacity=water_capacity,
                recover_time=recover_time,
            )
        else:
            raise ValueError(
                f"Unknown agent_type: {agent_type}. Must be 'FD', 'SD', or 'FD_WL'"
            )

        self.agent_type = agent_type

        # Communication state
        self.agent_id = agent_id
        self.box = asyncio.Queue()
        router.register(agent_id, self.box)

        # Convenience handles
        self.agent = self.llm_agent.agent
        self.controller = self.llm_agent.controller

    async def broadcast(self, content: Dict[str, Any]):
        """Broadcast to every mailbox (including self; ``get_other_agent_info`` drops self-sent traffic)."""
        await router.broadcast_all(Message(self.agent_id, content))

    def drain_inbox(self) -> List[Message]:
        """Drain this step's inbox using ``QueueEmpty`` instead of ``empty()`` to avoid asyncio races."""
        msgs: List[Message] = []
        while True:
            try:
                msgs.append(self.box.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    def get_other_agent_info(self) -> Optional[str]:
        """Format queued broadcasts for the LLM (summary / discussion / position, etc.)."""
        messages = self.drain_inbox()
        if not messages:
            return None

        info_parts: List[str] = []
        # print(f"[DMAS_ol{self.agent_id}] Messages: {messages}")
        for msg in messages:
            if msg.sender == self.agent_id:
                continue
            msg_type = msg.content.get("type")
            if msg_type == "summary":
                summary = msg.content.get("summary", "")
                high_level_plan = msg.content.get("high_level_plan", "")
                if summary:
                    info_parts.append(f"Agent {msg.sender} **Summary**: {summary} **High_level_plan**: {high_level_plan}")
            elif msg_type == "position":
                loc = msg.content.get("location") or {}
                step_n = msg.content.get("step", "")
                if isinstance(loc, dict) and {"x", "y", "z"} <= loc.keys():
                    x, y, z = float(loc["x"]), float(loc["y"]), float(loc["z"])
                    step_suffix = f" step={step_n}" if step_n != "" else ""
                    info_parts.append(
                        f"Agent {msg.sender} position: ({x:.1f}, {y:.1f}, {z:.1f}){step_suffix}"
                    )
                else:
                    info_parts.append(f"Agent {msg.sender}: {msg.content}")
            else:
                info_parts.append(f"Agent {msg.sender}: {msg.content}")

        return "\n".join(info_parts) if info_parts else None

    async def step_with_communication(self, agent_idx: Optional[int] = None) -> None:
        """One step: drain inbox, then run the LLM controller step."""
        other_agent_info = self.get_other_agent_info()
        # print(f"[DMAS_ol{self.agent_id}] Other agent info: {other_agent_info}")
        await self.llm_agent.step(other_agent_info=other_agent_info, agent_idx=agent_idx)

# ---------- DMAS_ol manager ----------
class DMAS_ol:
    """
    Distributed Multi-Agent System manager for llm_agent_ol agents.
    """
    def __init__(
        self,
        context: WorldContext,
        agent_positions: Optional[List[Dict[str, Any]]] = None,
        agent_types: Optional[List[str]] = None,
        prompt_environment: Optional[Dict[str, Any]] = None,
        *,
        water_capacity: int = 1,
        recover_time: int = 10,
    ):
        """
        Initialize DMAS_ol.
        
        Args:
            context: WorldContext instance
            agent_positions: Optional list of position configs for each agent
            agent_types: Optional list of roles ("FD" / "SD" / "FD_WL")
            prompt_environment: Shared prompt placeholders for each LLM controller (SA_ol convention)
            water_capacity / recover_time: ``SetExtinguisher`` params for every FD_WL agent (SA_ol convention)
        """
        self.context = context
        self.agents: List[DMASAgent] = []
        self.tasks: List[asyncio.Task] = []
        self.agent_positions = agent_positions
        self.agent_types = agent_types
        self.prompt_environment = prompt_environment
        self.water_capacity = water_capacity
        self.recover_time = recover_time

    async def spawn_agents(self, n_agents: int = 3):
        """Spawn ``n_agents`` DMAS agents."""
        router.clear()
        for i in range(n_agents):
            # Resolve role string
            agent_type = self.agent_types[i]
            
            # Build wrapper
            agent = DMASAgent(
                f"DMAS_Agent_{agent_type}_{i}",
                self.context,
                agent_type=agent_type,
                prompt_environment=self.prompt_environment,
                water_capacity=self.water_capacity,
                recover_time=self.recover_time,
            )

            # Spawn pose from config
            pos_config = self.agent_positions[i]
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

            # Delegate to the wrapped BaseAgent.spawn_agent
            agent_actor = await agent.llm_agent.agent.spawn_agent(
                name=f"DMAS_Agent_{agent_type}_{i}",
                location=location,
                rotation=rotation
            )

            if agent_actor:
                agent.llm_agent.agent._agent = agent_actor
                self.agents.append(agent)
                print(f"[DMAS_ol] Spawned agent {i}: {agent.agent_id} (type: {agent_type})")
            else:
                print(f"[DMAS_ol] Failed to spawn agent {i}")

    async def run_agents(
        self,
        max_steps: int = 10,
        time_limit_sec: Optional[float] = None,
        hard_timeout_grace_sec: float = 60.0,
    ) -> bool:
        """Run all agent loops concurrently.

        Returns True on natural completion (``max_steps`` exhausted / normal exit) and False when cancelled by the hard timeout.
        Soft limits are checked between steps; ``asyncio.wait_for`` enforces the hard cap.
        """
        start_ts = time.monotonic()
        timed_out = False

        async def agent_loop(agent: DMASAgent, agent_idx: int):
            try:
                for step in range(max_steps):
                    if (
                        time_limit_sec is not None
                        and (time.monotonic() - start_ts) >= float(time_limit_sec)
                    ):
                        print(
                            f"[DMAS_ol] Agent {agent_idx} soft time-limit "
                            f"reached at step {step + 1}/{max_steps}"
                        )
                        break
                    print(f"[DMAS_ol] Agent {agent_idx} step {step + 1}/{max_steps}")

                    await agent.step_with_communication(agent_idx=agent_idx)

                    current_summary, high_level_plan = agent.llm_agent.get_current_summary_and_high_level_plan()
                    if current_summary and high_level_plan:
                        await agent.broadcast({
                                "type": "summary",
                                "from": agent.agent_id,
                                "summary": current_summary,
                                "high_level_plan": high_level_plan,
                                "step": step + 1,
                            })

                    aid = agent.llm_agent.agent._agent.get("id") if agent.llm_agent.agent._agent else None
                    if aid:
                        agent_tf = await ts.UnaryAPI.get_actor_transform(self.context.conn, aid)
                        if agent_tf:
                            await agent.broadcast({
                                "type": "position",
                                "location": {
                                    "x": agent_tf.location.x,
                                    "y": agent_tf.location.y,
                                    "z": agent_tf.location.z,
                                },
                                "step": step + 1,
                            })

                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                print(f"[DMAS_ol] Agent {agent_idx} cancelled (hard timeout)")
                raise
            except Exception as e:
                print(f"[DMAS_ol] Agent {agent_idx} error: {e}")
                traceback.print_exc()

        # Launch parallel agent tasks
        self.tasks = [asyncio.create_task(agent_loop(agent, i)) for i, agent in enumerate(self.agents)]

        if time_limit_sec is None:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        else:
            hard_timeout = float(time_limit_sec) + float(hard_timeout_grace_sec)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=hard_timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                print(
                    f"[DMAS_ol] Hard timeout {hard_timeout:.1f}s reached, "
                    f"cancelling agent loops"
                )
                for t in self.tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*self.tasks, return_exceptions=True)

        return not timed_out

    async def stop(self):
        """Await spawned asyncio tasks (errors swallowed per task)."""
        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Memory is saved after each step in LLMAgentDMAS._execute_action

# ---------- Demo entrypoint ----------
async def demo_run(
    context: WorldContext,
    n_agents: int = 3,
    max_steps: int = 10,
    agent_positions: Optional[List[Dict[str, Any]]] = None,
    agent_types: Optional[List[str]] = None,
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    water_capacity: int = 1,
    recover_time: int = 10,
    plot_output_dir: Optional[Path] = None,
    burn_time: Optional[float] = None,
    scene_id: Optional[int] = None,
    task_id: str = "",
    time_limit_sec: Optional[float] = None,
    hard_timeout_grace_sec: float = 60.0,
):
    """DMAS_ol demo with the same evaluation pipeline as CMAS_ol (``Metric_Tool.evaluation.ExperimentResult``).

    ``agent_types`` may mix "FD" / "SD" / "FD_WL"; FD_WL uses ``llm_agent.LLMAgentDMASFireWaterLimit``.
    When ``prompt_environment`` is None, ``_default_prompt_env(agent_types)`` fills defaults.
    ``plot_output_dir`` / ``burn_time`` default to a relative plots folder and ``BURN_TIME`` unless DMAS_benchmark_runner overrides them.
    """
    if agent_types is None:
        agent_types = ["FD"] * n_agents
    if prompt_environment is None:
        prompt_environment = _default_prompt_env(agent_types)

    logical = task_id.strip() or f"DMAS_ol_n{n_agents}"

    print(f"[DMAS_ol] Starting demo with {n_agents} agents for {max_steps} steps")

    _plots = (
        plot_output_dir
        if plot_output_dir is not None
        else Path("./data_save/experiment_plots")
    )
    _sid = CURRENT_SCENE_ID if scene_id is None else int(scene_id)
    result = ExperimentResult(
        task_type="fire_rescue",
        task_id=logical,
        scene_id=_sid,
    ).bind_context(
        context,
        update_interval=1.0,
        plot_output_dir=_plots,
        enable_live_display=False,
        agent_state_update_interval=0.5,
    )
    print("\nRefresh_map_actors")
    await ts.UnaryAPI.refresh_actors_map(context.conn)
    result.initial_property_value = await ts.UnaryAPI.get_obj_residual(context.conn)
    burned_state = await ts.UnaryAPI.get_burned_area(context.conn)
    print(f"burned_state:{burned_state}")
    result.initial_property_num = int((burned_state or {}).get("total_num", 0))

    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(float(burn_time) if burn_time is not None else BURN_TIME)

    dmas = DMAS_ol(
        context,
        agent_positions,
        agent_types,
        prompt_environment=prompt_environment,
        water_capacity=water_capacity,
        recover_time=recover_time,
    )
    completed_ok = False
    finished_naturally = True
    try:
        try:
            await dmas.spawn_agents(n_agents)

            if len(dmas.agents) == 0:
                print("[DMAS_ol] No agents spawned, exiting")
            else:
                print(f"[DMAS_ol] Successfully spawned {len(dmas.agents)} agents")

                for agent in dmas.agents:
                    actor = agent.llm_agent.agent._agent
                    if not actor:
                        continue
                    await agent.llm_agent.agent._actions.get_perception_object_list(actor)
                    aid = actor.get("id")
                    result.register_agents([aid], [agent.agent_id])
                    if agent.agent_type == "FD_WL":
                        result.set_agent_water_setting(
                            aid, float(water_capacity), float(recover_time)
                        )
                    else:
                        result.set_agent_water_setting(aid, None, None)

                for _a in dmas.agents:
                    attach_experiment_result_to_base_agent(_a.llm_agent.agent, result)

                await result.start_async()

                finished_naturally = await dmas.run_agents(
                    max_steps,
                    time_limit_sec=time_limit_sec,
                    hard_timeout_grace_sec=hard_timeout_grace_sec,
                )
                completed_ok = True

        except KeyboardInterrupt:
            print("[DMAS_ol] Interrupted by user")
        except Exception as e:
            print(f"[DMAS_ol] Demo error: {e}")
            traceback.print_exc()
        finally:
            await dmas.stop()
            if completed_ok:
                if finished_naturally:
                    stop_reason = "Experiment completed"
                else:
                    stop_reason = (
                        f"Time limit reached "
                        f"(soft={time_limit_sec}s, hard_grace={hard_timeout_grace_sec}s)"
                    )
            else:
                stop_reason = "Interrupted or error"
            # Main-thread Ctrl+C → TongSim.__exit__ cancels this coroutine; without shield, stop_async may lose the race to CancelledError before save()
            if result.is_running():
                try:
                    await asyncio.shield(
                        result.stop_async(
                            success=completed_ok,
                            reason=stop_reason,
                        )
                    )
                except asyncio.CancelledError:
                    try:
                        if result.end_time <= 0:
                            result.end_time = time.time()
                        result.success = completed_ok
                        result.termination_reason = (
                            "Interrupted or error"
                            if not completed_ok
                            else "Cancelled during shutdown"
                        )
                        result.save()
                    except Exception as e:
                        print(f"[WARN] Emergency experiment save failed: {e}")
                    raise

        print("[INFO] Experiment ended")
        for _aid, am in result._agents.items():
            print(f"  Agent {am.name}:")
            print(f"    - Distance: {am.distance_traveled:.2f}cm")
            print(f"    - Water: {am.water_used:.2f}")
            print(f"    - Rescued: {am.npcs_rescued}")
            print(f"    - Extinguished: {am.extinguished_objects}")
        final_metrics = result.calculate_metrics()
        print("\n[FINAL METRICS]")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
    finally:
        try:
            flush_token_summary()
        except Exception as e:
            print(f"[WARN] LLM token summary flush failed: {e}")

    print("[DMAS_ol] Demo completed")

def main():
    """CLI entrypoint."""
    GRPC_ENDPOINT = "127.0.0.1:5726"
    print("[DMAS_ol] Connecting to TongSim ...")

    # Optional custom spawn positions
    # custom_positions = [
    #     {"x": 1250, "y": -100, "z": Z_HEIGHT, "rotation": {"pitch": 0, "yaw": 45, "roll": 0}},
    #     {"x": 1100, "y": 0, "z": Z_HEIGHT, "rotation": {"pitch": 0, "yaw": 0, "roll": 90}},
    #     {"x": 1500, "y": 80, "z": Z_HEIGHT, "rotation": {"pitch": 0, "yaw": -45, "roll": 270}},
    # ]
    custom_positions = [
        {"x": 400, "y": 0, "z": Z_HEIGHT, "rotation": (0, 0, 0)},
        {"x": 200, "y": -100, "z": Z_HEIGHT, "rotation": (0, 0, 0)},
        {"x": 200, "y": 100, "z": Z_HEIGHT, "rotation": (0, 0, 0)},
        {"x": 0, "y": 0, "z": Z_HEIGHT, "rotation": (0, 0, 0)},
        # {"x": -100, "y": -150, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
    ]

    # Optional custom role list: "FD" firefighter | "FD_WL" limited water (LLMAgentDMASFireWaterLimit) | "SD" rescue
    # custom_types = ["FD", "FD", "FD"]  # Example: change one slot to "FD_WL" for limited-water runs
    custom_types = ["FD", "FD", "FD","SD"]  # Example: change one slot to "FD_WL" for limited-water runs

    prompt_environment = _default_prompt_env(custom_types)

    cli_scene_id = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cli_scene_id = int(sys.argv[1])

    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        # Custom poses/types or None for caller defaults
        ue.context.sync_run(
            demo_run(
                ue.context,
                n_agents=len(custom_types),
                max_steps=20,
                agent_positions=custom_positions,
                agent_types=custom_types,
                prompt_environment=prompt_environment,
                scene_id=cli_scene_id,
            )
        )

if __name__ == "__main__":
    try:
        from Metric_Tool.llm_token_evaluation import install as _install_llm_token_metrics
    except ImportError:
        from EmbodiedMAS.Metric_Tool.llm_token_evaluation import (  # type: ignore
            install as _install_llm_token_metrics,
        )
    _install_llm_token_metrics()
    try:
        from Metric_Tool.perception_evaluation import install_perception_evaluation as _install_perception_evaluation
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import (  # type: ignore
            install_perception_evaluation as _install_perception_evaluation,
        )
    _install_perception_evaluation()
    main()
