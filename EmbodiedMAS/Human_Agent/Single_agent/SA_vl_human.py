from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any

import tongsim as ts
from tongsim.core.world_context import WorldContext

# Allow running as a script and as a module
if __package__ is None or __package__ == "":
    import os
    import sys

    _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _single_dir = os.path.dirname(__file__)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    if _single_dir not in sys.path:
        sys.path.insert(0, _single_dir)

    import human_agent_vlm
    HumanVisionFireAgent = human_agent_vlm.HumanVisionFireAgent
    HumanVisionFireWaterLimitAgent = human_agent_vlm.HumanVisionFireWaterLimitAgent
    HumanVisionSaveAgent = human_agent_vlm.HumanVisionSaveAgent
else:
    from .human_agent_vlm import HumanVisionFireAgent, HumanVisionFireWaterLimitAgent, HumanVisionSaveAgent


Z_HEIGHT = 1000
BURN_TIME = 15


async def demo_run_human(
    context: WorldContext,
    max_steps: int = 10,
    agent_location: Optional[ts.Vector3] = None,
    agent_rotation: Optional[ts.Quaternion] = None,
    agent_type: str = "FD",
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    water_capacity: int = 5,
    recover_time: int = 10,
) -> None:
    """Single-agent human baseline loop with identical observations/actions to VLM mode."""

    await ts.UnaryAPI.refresh_actors_map(context.conn)
    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(BURN_TIME)

    print(f"[SA_vl_human] Starting single-agent human baseline for {max_steps} steps")

    if agent_type == "FD":
        human_agent = HumanVisionFireAgent(
            context,
            agent_id="SA_HUMAN_VL_FireAgent_0",
            prompt_environment=prompt_environment,
        )
    elif agent_type == "SD":
        human_agent = HumanVisionSaveAgent(
            context,
            agent_id="SA_HUMAN_VL_SaveAgent_0",
            prompt_environment=prompt_environment,
        )
    elif agent_type == "FD_WL":
        human_agent = HumanVisionFireWaterLimitAgent(
            context,
            agent_id="SA_HUMAN_VL_FireWaterLimit_0",
            prompt_environment=prompt_environment,
            water_capacity=water_capacity,
            recover_time=recover_time,
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}, expected FD, SD, or FD_WL")

    agent_actor = await human_agent.agent.spawn_agent(
        name=f"SA_HUMAN_VL_{agent_type}Agent_0",
        location=agent_location,
        rotation=agent_rotation,
    )

    if not agent_actor:
        print("[SA_vl_human] Failed to spawn agent, exiting")
        return

    print(f"[SA_vl_human] Spawned agent: {agent_actor.get('name', 'Unknown')}")

    print("[SA_vl_human] Initial perception update (explore + simulator perception)...")
    await human_agent.agent.explore(image_format="jpg")
    # explore does not fetch object lists; explicit call for Metric_Tool / perception_evaluation hooks (not Human prompts).
    await human_agent.agent._actions.get_perception_object_list(human_agent.agent._agent)

    for step in range(max_steps):
        print(f"[SA_vl_human] === Step {step + 1}/{max_steps} ===")
        await human_agent.step(agent_idx=0, other_agent_info=None)

        current_summary = human_agent.get_current_summary()
        if current_summary:
            print(f"[SA_vl_human] Step {step + 1} summary: {current_summary}")

        await asyncio.sleep(0.1)

    print("[SA_vl_human] Demo completed")
    human_agent.controller.save_memory(agent_id=f"SA_HUMAN_VL_{agent_type}Agent_0")


def main() -> None:
    grpc_endpoint = "127.0.0.1:5726"
    print("[SA_vl_human] Connecting to TongSim ...")

    agent_location = ts.Vector3(-1400, 0, Z_HEIGHT)
    agent_rotation = ts.math.euler_to_quaternion(ts.Vector3(0, 0, -45), is_degree=True)

    # FD for fire-extinguishing baseline; switch to SD or FD_WL when needed.
    agent_type = "FD"
    water_capacity = 1
    recover_time = 10

    if agent_type in ("FD", "FD_WL"):
        num_fire_agents = 1
        num_rescue_agents = 0
        num_fires = 1
    elif agent_type == "SD":
        num_fire_agents = 0
        num_rescue_agents = 1
        num_fires = 1
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")

    prompt_environment = {
        "num_agents": num_fire_agents + num_rescue_agents,
        "num_fire_agents": num_fire_agents,
        "num_rescue_agents": num_rescue_agents,
        "num_civilians": 0,
        "num_fires": num_fires,
    }
    if agent_type == "FD_WL":
        prompt_environment["water_capacity"] = water_capacity
        prompt_environment["recover_time"] = recover_time
        prompt_environment["other_info"] = (
            f"Limited extinguisher water: water_capacity={water_capacity}, recover_time={recover_time}s. "
            "Use extinguish_fire only when necessary."
        )

    with ts.TongSim(grpc_endpoint=grpc_endpoint) as ue:
        ue.context.sync_run(
            demo_run_human(
                ue.context,
                max_steps=50,
                agent_location=agent_location,
                agent_rotation=agent_rotation,
                agent_type=agent_type,
                prompt_environment=prompt_environment,
                water_capacity=water_capacity,
                recover_time=recover_time,
            )
        )


if __name__ == "__main__":
    try:
        from Metric_Tool.llm_token_evaluation import install as _install_llm_token_metrics
    except ImportError:
        from EmbodiedMAS.Metric_Tool.llm_token_evaluation import install as _install_llm_token_metrics  # type: ignore

    _install_llm_token_metrics()

    try:
        from Metric_Tool.perception_evaluation import install_perception_evaluation as _install_perception_dump
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import install_perception_evaluation as _install_perception_dump  # type: ignore

    _install_perception_dump()
    main()
