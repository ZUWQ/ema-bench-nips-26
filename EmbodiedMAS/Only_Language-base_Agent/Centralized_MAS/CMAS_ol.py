"""Language-only centralized MAS entry: drives ``SuperLLMAgent`` (``super_llm_agent_CMAS.py``)."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import tongsim as ts
from tongsim.core.world_context import WorldContext

if __package__ is None or __package__ == "":
    import os
    import sys

    _parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    _ol_agent_dir = os.path.dirname(os.path.dirname(__file__))
    _cmas_dir = os.path.dirname(__file__)
    for _p in (_parent_dir, _ol_agent_dir, _cmas_dir):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import super_agent_CMAS

    SuperLLMAgent = super_agent_CMAS.SuperLLMAgent
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
    from .super_agent_CMAS import SuperLLMAgent
    from ...Metric_Tool.evaluation import (
        ExperimentResult,
        attach_experiment_result_to_base_agent,
    )
    from ...Metric_Tool.llm_token_evaluation import flush_token_summary

Z_HEIGHT = 1000
BURN_TIME = 1


def _default_prompt_env(agent_types: List[str]) -> Dict[str, Any]:
    return {
        "num_agents": len(agent_types),
        "num_fire_agents": sum(1 for t in agent_types if t == "FD"),
        "num_rescue_agents": sum(1 for t in agent_types if t == "SD"),
        "num_civilians": 0,
        "num_fires": "unknown",
    }


async def demo_run(
    context: WorldContext,
    n_agents: int = 3,
    max_steps: int = 10,
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    task_id: str = "exp_new_metrics_001",
) -> None:
    logical = task_id.strip() or "exp_new_metrics_001"

    # Mirrors the removed super_llm_agent_CMAS.demo_run wiring: Metric_Tool.evaluation.ExperimentResult
    result = ExperimentResult(
        task_type="fire_rescue",
        task_id=logical,
    ).bind_context(
        context,
        update_interval=1.0,
        plot_output_dir=Path("./data_save/experiment_plots"),
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

    custom_positions = [
        {"x": -400, "y": 0, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
        {"x": -200, "y": -100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
        {"x": -200, "y": 100, "z": Z_HEIGHT, "rotation": (0, 0, 180)},
    ]
    agent_types = ["FD", "FD", "FD"]
    if prompt_environment is None:
        prompt_environment = _default_prompt_env(agent_types)

    super_agent: Optional[SuperLLMAgent] = None
    completed_ok = False
    try:
        try:
            super_agent = SuperLLMAgent(
                context, n_agents=n_agents, prompt_environment=prompt_environment
            )
            await super_agent.spawn_agents(custom_positions, agent_types)
            for robot_id, agent in super_agent._agents.items():
                if not agent._agent:
                    continue
                await agent._actions.get_perception_object_list(agent._agent)
                aid = agent._agent.get("id")
                result.register_agents([aid], [robot_id])
                result.set_agent_water_setting(aid, None, None)

            for _rid, _ag in super_agent._agents.items():
                if _ag is not None and getattr(_ag, "_agent", None):
                    attach_experiment_result_to_base_agent(_ag, result)

            await result.start_async()

            for step in range(max_steps):
                print(f"\n[CMAS_ol] === Step {step + 1}/{max_steps} ===")
                await super_agent.step()
            completed_ok = True
        except KeyboardInterrupt:
            print("[CMAS_ol] Interrupted by user")
        except Exception as e:
            print(f"[CMAS_ol] Demo error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Memory is saved after each step in SuperLLMAgent._execute_actions_parallel
            # Main-thread Ctrl+C → TongSim.__exit__ cancels this coroutine; without shield, stop_async may lose the race to CancelledError before save()
            if result.is_running():
                try:
                    await asyncio.shield(
                        result.stop_async(
                            success=completed_ok,
                            reason="Experiment completed"
                            if completed_ok
                            else "Interrupted or error",
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

    print("[CMAS_ol] Demo completed")


def main() -> None:
    GRPC_ENDPOINT = "127.0.0.1:5726"
    print("[CMAS_ol] Connecting to TongSim ...")
    pe = _default_prompt_env(["FD", "FD", "FD"])
    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        ue.context.sync_run(
            demo_run(ue.context, n_agents=3, max_steps=30, prompt_environment=pe)
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
