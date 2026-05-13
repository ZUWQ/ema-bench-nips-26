from __future__ import annotations

import asyncio
import os
import socket
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tongsim as ts
from tongsim.core.world_context import WorldContext
import time

# Align path handling with DMAS_vlm so this file can run as script or module.
_cfmas_dir = os.path.dirname(os.path.abspath(__file__))
_vlm_agent_dir = os.path.dirname(_cfmas_dir)
_parent_dir = os.path.dirname(_vlm_agent_dir)
_parent_parent_dir = os.path.dirname(_parent_dir)
_single_agent_dir = os.path.join(_vlm_agent_dir, "Single_agent")
for _p in (_parent_parent_dir, _parent_dir, _vlm_agent_dir, _cfmas_dir, _single_agent_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __package__ is None or __package__ == "":
    from Human_Agent.Single_agent import human_agent_vlm
    from Human_Agent.Single_agent import human_ui_vlm
    from Human_Agent.Single_agent.web_server_ui import (
        ensure_web_ui_server_started,
        web_ui_agent_url,
    )  

    HumanVisionFireAgent = human_agent_vlm.HumanVisionFireAgent
    HumanVisionSaveAgent = human_agent_vlm.HumanVisionSaveAgent
    HumanVisionFireWaterLimitAgent = human_agent_vlm.HumanVisionFireWaterLimitAgent
    WebHumanDecisionProvider = human_ui_vlm.WebHumanDecisionProvider
else:
    from ..Single_agent.human_agent_vlm import (
        HumanVisionFireAgent,
        HumanVisionSaveAgent,
        HumanVisionFireWaterLimitAgent,
    )
    from ..Single_agent.human_ui_vlm import WebHumanDecisionProvider
    from ..Single_agent.web_server_ui import (
        ensure_web_ui_server_started,
        web_ui_agent_url,
    )

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
    from EmbodiedMAS.Metric_Tool.llm_token_evaluation import flush_token_summary  # type: ignore

Z_HEIGHT = 1000
BURN_TIME = 1


def _get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


def _is_port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


def _choose_web_port(host: str, preferred_port: int) -> int:
    if _is_port_available(host, preferred_port):
        return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        fallback_port = int(s.getsockname()[1])

    print(
        f"[WARN] DMAS_vl_human web UI port {preferred_port} is already in use. "
        f"Using fallback port {fallback_port}."
    )
    print("[WARN] Open the newly printed agent URLs for this run.")
    return fallback_port


def _print_web_ui_agent_pages(agent_ids: List[str], *, port: int) -> None:
    lan_ip = _get_lan_ip()
    print("[DMAS_vl_human] Open these pages in separate browsers/devices:")
    for aid in agent_ids:
        print(f"  {aid} local: {web_ui_agent_url(aid, host='127.0.0.1', port=port)}")
        print(f"  {aid} LAN  : {web_ui_agent_url(aid, host=lan_ip, port=port)}")
    print("[DMAS_vl_human] Submit decisions through the web pages. The script will wait for all submissions.")


def _default_prompt_env(agent_types: List[str]) -> Dict[str, Any]:
    """Default prompt environment aligned with DMAS_vlm/DMAS_ol semantics."""
    return {
        "num_agents": len(agent_types),
        "num_fire_agents": sum(1 for t in agent_types if t in ("FD", "FD_WL")),
        "num_rescue_agents": sum(1 for t in agent_types if t == "SD"),
        "num_civilians": 0,
        "num_fires": "unknown",
        "other_info": "",
    }


def _install_perception_metric_hooks() -> None:
    """Same as DMAS_benchmark_runner: patch loaded ActionAPI for perception logging; safe to call multiple times."""
    try:
        from Metric_Tool.perception_evaluation import (
            install_perception_evaluation as _install,
        )
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import (  # type: ignore
            install_perception_evaluation as _install,
        )
    _install()


@dataclass
class Message:
    sender: str
    content: Dict[str, Any]


class MessageRouter:
    def __init__(self):
        self._boxes: Dict[str, asyncio.Queue] = {}

    def clear(self) -> None:
        self._boxes.clear()

    def register(self, aid: str, q: asyncio.Queue) -> None:
        self._boxes[aid] = q

    async def broadcast_all(self, msg: Message) -> None:
        if not self._boxes:
            return
        await asyncio.gather(*(q.put(msg) for q in self._boxes.values()))


router = MessageRouter()


class HumanDMASAgent:
    """Human-driven decentralized agent wrapper with mailbox communication."""

    def __init__(
        self,
        agent_id: str,
        context: WorldContext,
        agent_type: str = "FD",
        prompt_environment: Optional[Dict[str, Any]] = None,
        *,
        water_capacity: int = 1,
        recover_time: int = 10,
        decision_provider: Optional[Any] = None,
    ):
        if agent_type == "FD":
            self.human_agent = HumanVisionFireAgent(
                context,
                agent_id=agent_id,
                prompt_environment=prompt_environment,
                decision_provider=decision_provider,
            )
        elif agent_type == "SD":
            self.human_agent = HumanVisionSaveAgent(
                context,
                agent_id=agent_id,
                prompt_environment=prompt_environment,
                decision_provider=decision_provider,
            )
        elif agent_type == "FD_WL":
            self.human_agent = HumanVisionFireWaterLimitAgent(
                context,
                agent_id=agent_id,
                prompt_environment=prompt_environment,
                water_capacity=water_capacity,
                recover_time=recover_time,
                decision_provider=decision_provider,
            )
        else:
            raise ValueError(
                f"Unknown agent_type: {agent_type}. Must be 'FD', 'SD', or 'FD_WL'"
            )

        self.agent_type = agent_type
        self.agent_id = agent_id
        self.box: asyncio.Queue = asyncio.Queue()
        router.register(agent_id, self.box)

        self.agent = self.human_agent.agent
        self.controller = self.human_agent.controller

    async def broadcast(self, content: Dict[str, Any]) -> None:
        await router.broadcast_all(Message(self.agent_id, content))

    def drain_inbox(self) -> List[Message]:
        msgs: List[Message] = []
        while True:
            try:
                msgs.append(self.box.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    def get_other_agent_info(self) -> Optional[str]:
        messages = self.drain_inbox()
        if not messages:
            return None

        info_parts: List[str] = []
        for msg in messages:
            if msg.sender == self.agent_id:
                continue
            msg_type = msg.content.get("type")
            if msg_type == "summary":
                summary = msg.content.get("summary", "")
                high_level_plan = msg.content.get("high_level_plan", "")
                if summary:
                    info_parts.append(
                        f"Agent {msg.sender} **Summary**: {summary} **High_level_plan**: {high_level_plan}"
                    )
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
        other_agent_info = self.get_other_agent_info()
        await self.human_agent.step(other_agent_info=other_agent_info, agent_idx=agent_idx)


class DMAS_vl_human:
    """Decentralized multi-agent human baseline manager for VLM observation/action spaces."""

    def __init__(
        self,
        context: WorldContext,
        agent_positions: Optional[List[Dict[str, Any]]] = None,
        agent_types: Optional[List[str]] = None,
        prompt_environment: Optional[Dict[str, Any]] = None,
        *,
        water_capacity: int = 1,
        recover_time: int = 10,
        decision_providers: Optional[List[Any]] = None,
        auto_start_web_ui: bool = True,
        web_ui_host: str = "0.0.0.0",
        web_ui_port: int = 8080,
    ):
        self.context = context
        self.agents: List[HumanDMASAgent] = []
        self.tasks: List[asyncio.Task] = []
        self.agent_positions = agent_positions
        self.agent_types = agent_types
        self.prompt_environment = prompt_environment
        self.water_capacity = water_capacity
        self.recover_time = recover_time
        self.decision_providers = decision_providers
        self.auto_start_web_ui = auto_start_web_ui
        self.web_ui_host = web_ui_host
        self.web_ui_port = web_ui_port

    def _provider_for(self, idx: int, agent_id: str) -> Any:
        if self.decision_providers is not None and idx < len(self.decision_providers):
            return self.decision_providers[idx]

        return WebHumanDecisionProvider(
            agent_id=agent_id,
            # Server startup is centralized in spawn_agents to avoid importing another UI module path.
            auto_start_server=False,
            host=self.web_ui_host,
            port=self.web_ui_port,
        )

    async def spawn_agents(self, n_agents: int = 3) -> None:
        if self.auto_start_web_ui:
            ensure_web_ui_server_started(host=self.web_ui_host, port=self.web_ui_port)

        router.clear()
        for i in range(n_agents):
            agent_type = self.agent_types[i]
            agent_id = f"DMAS_HUMAN_VL_Agent_{agent_type}_{i}"
            agent = HumanDMASAgent(
                agent_id,
                self.context,
                agent_type=agent_type,
                prompt_environment=self.prompt_environment,
                water_capacity=self.water_capacity,
                recover_time=self.recover_time,
                decision_provider=self._provider_for(i, agent_id),
            )

            pos_config = self.agent_positions[i]
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

            agent_actor = await agent.human_agent.agent.spawn_agent(
                name=f"DMAS_HUMAN_VL_Agent_{agent_type}_{i}",
                location=location,
                rotation=rotation,
            )

            if agent_actor:
                agent.human_agent.agent._agent = agent_actor
                self.agents.append(agent)
                print(f"[DMAS_vl_human] Spawned agent {i}: {agent.agent_id} (type: {agent_type})")
            else:
                print(f"[DMAS_vl_human] Failed to spawn agent {i}")

    async def run_agents(
        self,
        max_steps: int = 10,
        time_limit_sec: Optional[float] = None,
        hard_timeout_grace_sec: float = 60.0,
    ) -> bool:
        """Run all agents in parallel; True if finished normally, False if cancelled by hard timeout."""
        start_ts = time.monotonic()
        timed_out = False

        async def agent_loop(agent: HumanDMASAgent, agent_idx: int) -> None:
            try:
                try:
                    await agent.human_agent.agent.explore(image_format="jpg")
                    actor = agent.human_agent.agent._agent
                    if actor:
                        # Explicit fetch for metric hooks; explore does not call get_perception_object_list.
                        await agent.human_agent.agent._actions.get_perception_object_list(actor)
                except Exception as e:
                    print(
                        f"[DMAS_vl_human] WARN: initial explore/perception warmup failed for agent {agent_idx}: {e}"
                    )
                    actor = agent.human_agent.agent._agent
                    if actor:
                        # Same: metric/cache warmup, not Human prompt text.
                        await agent.human_agent.agent._actions.get_perception_object_list(actor)
                for step in range(max_steps):
                    if (
                        time_limit_sec is not None
                        and (time.monotonic() - start_ts) >= float(time_limit_sec)
                    ):
                        print(
                            f"[DMAS_vl_human] Agent {agent_idx} soft time-limit "
                            f"reached at step {step + 1}/{max_steps}"
                        )
                        break
                    print(f"[DMAS_vl_human] Agent {agent_idx} step {step + 1}/{max_steps}")

                    await agent.step_with_communication(agent_idx=agent_idx)

                    actor = agent.human_agent.agent._agent
                    if actor:
                        try:
                            await agent.human_agent.agent._actions.get_perception_object_list(
                                actor
                            )
                        except Exception as pe:
                            print(
                                f"[DMAS_vl_human] WARN: perception snapshot failed "
                                f"agent {agent_idx} step {step + 1}: {pe}"
                            )

                    current_summary, high_level_plan = (
                        agent.human_agent.get_current_summary_and_high_level_plan()
                    )
                    if current_summary and high_level_plan is not None:
                        await agent.broadcast(
                            {
                                "type": "summary",
                                "from": agent.agent_id,
                                "summary": current_summary,
                                "high_level_plan": high_level_plan,
                                "step": step + 1,
                            }
                        )

                    actor = agent.human_agent.agent._agent
                    aid = actor.get("id") if actor else None
                    if aid:
                        agent_tf = await ts.UnaryAPI.get_actor_transform(self.context.conn, aid)
                        if agent_tf:
                            await agent.broadcast(
                                {
                                    "type": "position",
                                    "location": {
                                        "x": agent_tf.location.x,
                                        "y": agent_tf.location.y,
                                        "z": agent_tf.location.z,
                                    },
                                    "step": step + 1,
                                }
                            )

                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                print(f"[DMAS_vl_human] Agent {agent_idx} cancelled (hard timeout)")
                raise
            except Exception as e:
                print(f"[DMAS_vl_human] Agent {agent_idx} error: {e}")
                traceback.print_exc()

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
                    f"[DMAS_vl_human] Hard timeout {hard_timeout:.1f}s reached, "
                    f"cancelling agent loops"
                )
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self.tasks, return_exceptions=True)
        return not timed_out

    async def stop(self) -> None:
        await asyncio.gather(*self.tasks, return_exceptions=True)


async def demo_run(
    context: WorldContext,
    n_agents: int = 3,
    max_steps: int = 100,
    agent_positions: Optional[List[Dict[str, Any]]] = None,
    agent_types: Optional[List[str]] = None,
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    water_capacity: int = 1,
    recover_time: int = 10,
    experiment_id: str = "exp_dmas_vl_human_001",
    plot_output_dir: Optional[Path] = None,
    burn_time: Optional[float] = None,
    decision_providers: Optional[List[Any]] = None,
    auto_start_web_ui: bool = True,
    web_ui_host: str = "0.0.0.0",
    web_ui_port: int = 8080,
    time_limit_sec: Optional[float] = None,
    hard_timeout_grace_sec: float = 60.0,
) -> None:
    if agent_types is None:
        agent_types = ["FD"] * n_agents
    if prompt_environment is None:
        prompt_environment = _default_prompt_env(agent_types)

    selected_web_port = (
        _choose_web_port(web_ui_host, web_ui_port)
        if auto_start_web_ui
        else web_ui_port
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_id = f"{experiment_id}_{run_ts}"

    print(f"[DMAS_vl_human] Starting demo with {n_agents} agents for {max_steps} steps")

    _install_perception_metric_hooks()

    _plots = (
        plot_output_dir
        if plot_output_dir is not None
        else Path("./data_save/experiment_plots")
    )
    result = ExperimentResult(
        task_id=experiment_id,
        task_type="fire_rescue",
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
    result.initial_property_num = burned_state["total_num"]

    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(float(burn_time) if burn_time is not None else BURN_TIME)

    dmas = DMAS_vl_human(
        context,
        agent_positions,
        agent_types,
        prompt_environment=prompt_environment,
        water_capacity=water_capacity,
        recover_time=recover_time,
        decision_providers=decision_providers,
        auto_start_web_ui=auto_start_web_ui,
        web_ui_host=web_ui_host,
        web_ui_port=selected_web_port,
    )
    completed_ok = False
    finished_naturally = True
    try:
        try:
            await dmas.spawn_agents(n_agents)

            if len(dmas.agents) == 0:
                print("[DMAS_vl_human] No agents spawned, exiting")
            else:
                print(f"[DMAS_vl_human] Successfully spawned {len(dmas.agents)} agents")
                _print_web_ui_agent_pages([agent.agent_id for agent in dmas.agents], port=selected_web_port)

                for agent in dmas.agents:
                    actor = agent.human_agent.agent._agent
                    if not actor:
                        continue
                    # Pull simulator perception before experiment registration for metrics/logging; not Web UI copy.
                    await agent.human_agent.agent._actions.get_perception_object_list(actor)
                    aid = actor.get("id")
                    result.register_agents([aid], [agent.agent_id])
                    if agent.agent_type == "FD_WL":
                        result.set_agent_water_setting(
                            aid, float(water_capacity), float(recover_time)
                        )
                    else:
                        result.set_agent_water_setting(aid, None, None)

                for _a in dmas.agents:
                    attach_experiment_result_to_base_agent(_a.agent, result)

                await result.start_async()
                finished_naturally = await dmas.run_agents(
                    max_steps,
                    time_limit_sec=time_limit_sec,
                    hard_timeout_grace_sec=hard_timeout_grace_sec,
                )
                completed_ok = True

        except KeyboardInterrupt:
            print("[DMAS_vl_human] Interrupted by user")
        except Exception as e:
            print(f"[DMAS_vl_human] Demo error: {e}")
            traceback.print_exc()
        finally:
            await dmas.stop()
            for agent in dmas.agents:
                try:
                    agent.controller.save_memory(agent_id=agent.agent_id)
                except Exception as e:
                    print(f"[WARN] Failed to save memory for {agent.agent_id}: {e}")

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
            print(f"[WARN] LLM token eval save failed: {e}")

    print("[DMAS_vl_human] Demo completed")


def main() -> None:
    grpc_endpoint = "127.0.0.1:5726"
    web_host = "0.0.0.0"
    web_port = int(os.environ.get("EMBODIED_WEB_UI_PORT", "8080"))
    print("[DMAS_vl_human] Connecting to TongSim ...")

    rotation = (0, 0, -180)
    custom_positions = [
        {"x": -300, "y": 100, "z": Z_HEIGHT, "rotation": rotation},
        {"x": -300, "y": -100, "z": Z_HEIGHT, "rotation": rotation},
        {"x": 300, "y": 100, "z": Z_HEIGHT, "rotation": rotation},
        {"x": 300, "y": -100, "z": Z_HEIGHT, "rotation": rotation},
            ]

    custom_types = ["FD", "FD", "FD", "SD"]
    prompt_environment = _default_prompt_env(custom_types)

    print("[DMAS_vl_human] Web UI agent pages will be printed after agents are spawned.")

    with ts.TongSim(grpc_endpoint=grpc_endpoint) as ue:
        ue.context.sync_run(
            demo_run(
                ue.context,
                n_agents=len(custom_types),
                max_steps=500,
                agent_positions=custom_positions,
                agent_types=custom_types,
                prompt_environment=prompt_environment,
                water_capacity=1,
                recover_time=10,
                auto_start_web_ui=True,
                web_ui_host=web_host,
                web_ui_port=web_port,
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
    _install_perception_metric_hooks()
    main()
