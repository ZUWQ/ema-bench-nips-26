import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import tongsim as ts
from tongsim.core.world_context import WorldContext
import time

# Allow running as a script and as a module
if __package__ is None or __package__ == "":
    import os
    import sys
    _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _vl_agent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    if _vl_agent_dir not in sys.path:
        sys.path.insert(0, _vl_agent_dir)
    import llm_agent_vlm
    LLMAgentVisionLanguageFire = llm_agent_vlm.LLMAgentVisionLanguageFire
    LLMAgentVisionLanguageFireWaterLimit = llm_agent_vlm.LLMAgentVisionLanguageFireWaterLimit
    import llm_agent_vlm_SD
    LLMAgentVisionLanguageSave = llm_agent_vlm_SD.LLMAgentVisionLanguageSave
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
else:
    from .llm_agent_vlm import LLMAgentVisionLanguageFire, LLMAgentVisionLanguageFireWaterLimit
    from .llm_agent_vlm_SD import LLMAgentVisionLanguageSave
    from ...Metric_Tool.evaluation import (
        ExperimentResult,
        attach_experiment_result_to_base_agent,
    )
    from ...Metric_Tool.llm_token_evaluation import flush_token_summary

Z_HEIGHT = 1000
BURN_TIME = 1


async def demo_run(
    context: WorldContext,
    max_steps: int = 10,
    agent_location: Optional[ts.Vector3] = None,
    agent_rotation: Optional[ts.Quaternion] = None,
    agent_type: str = "FD",
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    water_capacity: int = 5,
    recover_time: int = 10,
    task_id: str = "sa_vlm",
    plot_output_dir: Optional[Path] = None,
):
    """
    Single-agent vision-language demo (see ``main`` in this module).

    Args:
        context: ``WorldContext`` instance
        max_steps: Max controller steps
        agent_location / agent_rotation: Spawn pose
        agent_type: "FD" firefighter (``llm_agent_vlm``) | "SD" rescue + SOS (``llm_agent_vlm_SD``) |
            "FD_WL" limited water (``llm_agent_vlm.LLMAgentVisionLanguageFireWaterLimit``)
        prompt_environment: ``prompt_assembler`` placeholders; None uses each llm_agent's defaults
        water_capacity / recover_time: Only when ``agent_type=="FD_WL"``
        task_id: Stored in ``ExperimentResult.task_id`` and plot/JSON naming
        plot_output_dir: Metrics figure directory; default ``./data_save/experiment_plots``
    """
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logical = task_id.strip() or "sa_vlm"
    session_id = f"{logical}_{run_ts}"

    _plots = (
        plot_output_dir
        if plot_output_dir is not None
        else Path("./data_save/experiment_plots")
    )

    result = ExperimentResult(
        task_type="fire_rescue",
        task_id=logical,
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
    time.sleep(BURN_TIME)

    print(f"[SA_vlm] Starting single agent VL demo for {max_steps} steps")

    if agent_type == "FD":
        llm_agent = LLMAgentVisionLanguageFire(
            context,
            agent_id="SA_VL_FireAgent_0",
            prompt_environment=prompt_environment,
            session_id=session_id,
            pause_during_llm=False,
        )
    elif agent_type == "SD":
        llm_agent = LLMAgentVisionLanguageSave(
            context,
            agent_id="SA_VL_SaveAgent_0",
            prompt_environment=prompt_environment,
            session_id=session_id,
            pause_during_llm=False,
        )
    elif agent_type == "FD_WL":
        llm_agent = LLMAgentVisionLanguageFireWaterLimit(
            context,
            agent_id="SA_VL_FireWaterLimit_0",
            prompt_environment=prompt_environment,
            session_id=session_id,
            pause_during_llm=False,
            water_capacity=water_capacity,
            recover_time=recover_time,
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}, expected FD, SD, or FD_WL")
    print(f"[SA_vlm] Created LLM VL {agent_type} Agent")

    agent_actor = await llm_agent.agent.spawn_agent(
        name=f"SA_VL_{agent_type}Agent_0",
        location=agent_location,
        rotation=agent_rotation,
    )

    memory_label = f"SA_VL_{agent_type}Agent_0"
    completed_ok = False
    try:
        if not agent_actor:
            print("[SA_vlm] Failed to spawn agent, exiting")
            return

        print(f"[SA_vlm] Successfully spawned agent: {agent_actor.get('name', 'Unknown')}")

        print("[SA_vlm] Initial explore (mosaic + simulation perception on front)...")
        await llm_agent.agent.explore()

        aid = agent_actor.get("id")
        reg_name = getattr(llm_agent, "_agent_id", None) or memory_label
        result.register_agents([aid], [reg_name])
        if agent_type == "FD_WL":
            result.set_agent_water_setting(aid, float(water_capacity), float(recover_time))
        else:
            result.set_agent_water_setting(aid, None, None)

        attach_experiment_result_to_base_agent(llm_agent.agent, result)

        await result.start_async()

        for step in range(max_steps):
            print(f"[SA_vlm] === Step {step + 1}/{max_steps} ===")
            await llm_agent.step(agent_idx=0, other_agent_info=None)

            current_summary, high_level_plan = llm_agent.get_current_summary_and_high_level_plan()
            if current_summary and high_level_plan:
                print(f"[SA_vlm] Step {step + 1} summary: {current_summary} high_level_plan: {high_level_plan}")

            await asyncio.sleep(0.1)
        completed_ok = True
    except KeyboardInterrupt:
        print("[SA_vlm] Interrupted by user")
    except Exception as e:
        print(f"[SA_vlm] Demo error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if result.is_running():
            await result.stop_async(
                success=completed_ok,
                reason="Experiment completed" if completed_ok else "Interrupted or error",
            )

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

    try:
        flush_token_summary()
    except Exception as e:
        print(f"[WARN] LLM token summary flush failed: {e}")

    llm_agent.controller.save_memory(agent_id=memory_label)
    print("[SA_vlm] Saved memory")
    print("[SA_vlm] Demo completed")


def main() -> None:
    GRPC_ENDPOINT = "127.0.0.1:5726"
    print("[SA_vlm] Connecting to TongSim ...")

    # agent_location = ts.Vector3(-1400, 0, Z_HEIGHT)
    # agent_rotation = ts.math.euler_to_quaternion(ts.Vector3(0, 0, -45), is_degree=True)
    agent_location = ts.Vector3(0, 0, Z_HEIGHT)
    agent_rotation = ts.math.euler_to_quaternion(ts.Vector3(0, 0, -180), is_degree=True)

    agent_type = "FD"

    if agent_type in ("FD", "FD_WL"):
        num_fire_agents = 1
        num_rescue_agents = 0
    elif agent_type == "SD":
        num_fire_agents = 0
        num_rescue_agents = 1
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")

    prompt_environment = {
        "num_agents": num_fire_agents + num_rescue_agents,
        "num_fire_agents": num_fire_agents,
        "num_rescue_agents": num_rescue_agents,
        "num_civilians": 0,
        "num_fires": 1,
    }

    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        ue.context.sync_run(
            demo_run(
                ue.context,
                max_steps=20,
                agent_location=agent_location,
                agent_rotation=agent_rotation,
                agent_type=agent_type,
                prompt_environment=prompt_environment,
                water_capacity=1,
                recover_time=10,
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
