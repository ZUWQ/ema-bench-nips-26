"""
Single-agent benchmark entry: mirrors SA_ol (ExperimentResult / flush_token_summary).
Agent class is chosen per backend; initial perception is dispatched here by backend.

Examples (from PythonClient repo root)::

    python -m EmbodiedMAS.ExperimentRunning.SA_benchmark_runner --backend OL --agent-type FD
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import tongsim as ts
from tongsim.core.world_context import WorldContext

_EMBODIED_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_CLIENT_ROOT = _EMBODIED_ROOT.parent

# Cache by path so the same Single_agent file is not exec'd twice (importlib uses distinct module names).
_single_agent_module_by_path: Dict[str, Any] = {}

# Automation contract: fixed water capacity and recover time; not exposed as CLI flags.
_BENCHMARK_WATER_CAPACITY = 1
_BENCHMARK_RECOVER_TIME = 30


def _ensure_python_client_on_path() -> None:
    s = str(_PYTHON_CLIENT_ROOT.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _single_agent_dir(backend: str) -> Path:
    mapping = {
        "OL": _EMBODIED_ROOT / "Only_Language-base_Agent" / "Single_agent",
        "VL": _EMBODIED_ROOT / "Vision_Language-base_Agent" / "Single_agent",
        "VLM": _EMBODIED_ROOT / "Vision_VLM_Agent" / "Single_agent",
    }
    if backend not in mapping:
        raise ValueError(f"Unknown backend {backend!r}, expected OL, VL, or VLM")
    p = mapping[backend]
    if not p.is_dir():
        raise FileNotFoundError(f"Single_agent directory not found: {p}")
    return p


def _prepare_single_agent_imports(backend: str) -> None:
    """Prepend the backend's ``Single_agent`` directory on ``sys.path`` (one backend per process recommended)."""
    d = _single_agent_dir(backend)
    ds = str(d.resolve())
    try:
        sys.path.remove(ds)
    except ValueError:
        pass
    sys.path.insert(0, ds)


def _load_single_agent_module(backend: str, stem: str) -> Any:
    """Load ``Single_agent/{stem}.py`` explicitly (helps static analyzers; avoids implicit ``sys.path``)."""
    path = _single_agent_dir(backend) / f"{stem}.py"
    key = str(path.resolve())
    cached = _single_agent_module_by_path.get(key)
    if cached is not None:
        return cached
    mod_name = f"_embodied_sa_{backend.upper()}_{stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    _single_agent_module_by_path[key] = mod
    return mod


def _class_from_single_agent(backend: str, stem: str, class_name: str) -> Type[Any]:
    mod = _load_single_agent_module(backend, stem)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise AttributeError(f"{stem}.py has no attribute {class_name!r}")
    return cls


async def _run_initial_perception(backend: str, llm_agent: Any) -> None:
    """Same behavior as each backend's ``Single_agent/perception_bootstrap``."""
    b = backend.upper()
    if b == "OL":
        await llm_agent.agent._actions.get_perception_object_list(llm_agent.agent._agent)
    elif b in ("VL", "VLM"):
        await llm_agent.agent.explore()
    else:
        raise ValueError(f"Unknown backend: {backend!r}, expected OL, VL, or VLM")


def _get_backend_bundle(
    backend: str,
) -> Tuple[Type[Any], Type[Any], Type[Any]]:
    _prepare_single_agent_imports(backend)

    if backend == "OL":
        return (
            _class_from_single_agent(backend, "llm_agent", "LLMAgentDMASFire"),
            _class_from_single_agent(backend, "llm_agent_SD", "LLMAgentDMASSave"),
            _class_from_single_agent(
                backend, "llm_agent", "LLMAgentDMASFireWaterLimit"
            ),
        )
    elif backend == "VL":
        return (
            _class_from_single_agent(backend, "llm_agent_vl", "LLMAgentVisionLanguageFire"),
            _class_from_single_agent(
                backend, "llm_agent_vl_SD", "LLMAgentVisionLanguageSave"
            ),
            _class_from_single_agent(
                backend,
                "llm_agent_vl",
                "LLMAgentVisionLanguageFireWaterLimit",
            ),
        )
    elif backend == "VLM":
        return (
            _class_from_single_agent(backend, "llm_agent_vlm", "LLMAgentVisionLanguageFire"),
            _class_from_single_agent(
                backend, "llm_agent_vlm_SD", "LLMAgentVisionLanguageSave"
            ),
            _class_from_single_agent(
                backend,
                "llm_agent_vlm",
                "LLMAgentVisionLanguageFireWaterLimit",
            ),
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}, expected OL, VL, or VLM")

def _agent_naming(
    backend: str, agent_type: str, num: int = 0
) -> Tuple[str, str, str]:
    """(llm_constructor_agent_id, spawn_actor_name, memory_label); all ``{type}_{BACKEND}_Agent_{num}``."""
    if agent_type not in ("FD", "SD", "FD_WL"):
        raise ValueError(f"Unknown agent_type: {agent_type!r}, expected FD, SD, or FD_WL")
    base = f"{agent_type}_{backend.upper()}_Agent_{num}"
    return base, base, base


def _data_save_root() -> Path:
    env = os.environ.get("EMBODIED_BENCHMARK_DATA_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent / "data_save"
    root.mkdir(parents=True, exist_ok=True)
    return root


async def demo_run(
    context: WorldContext,
    backend: str,
    max_steps: int = 10,
    agent_location: Optional[ts.Vector3] = None,
    agent_rotation: Optional[ts.Quaternion] = None,
    agent_type: str = "FD",
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    burn_time: Optional[float] = None,
    task_id: str = "",
    scene_id: int = 0,
    max_memory_size: int = 3,
) -> None:
    """
    Run a single-agent experiment (Metric_Tool parity with SA_ol; initial perception in
    ``_run_initial_perception``).

    Args:
        backend: ``OL`` | ``VL`` | ``VLM``
        burn_time: Seconds for ``time.sleep`` after ignition.
    """
    backend = backend.upper()
    _ensure_python_client_on_path()
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

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logical_task_id = task_id.strip() or f"{backend}_{agent_type}"
    session_id = f"{logical_task_id}_{run_ts}"

    plots_dir = _data_save_root() / "experiment_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    result = ExperimentResult(
        task_type="fire_rescue",
        task_id=logical_task_id,
        scene_id=scene_id,
    ).bind_context(
        context,
        update_interval=1.0,
        plot_output_dir=plots_dir,
        enable_live_display=False,
        agent_state_update_interval=0.5,
    )
    bt = float(burn_time)
    log_p = f"[SA_bench|{backend}|{agent_type}]"

    FireCls, SaveCls, WlCls = _get_backend_bundle(backend)
    # perception_evaluation only patches ActionAPI already in sys.modules; load this backend's llm_agent chain first.
    try:
        from Metric_Tool.perception_evaluation import (
            install_perception_evaluation as _install_perception_evaluation,
        )
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import (  # type: ignore
            install_perception_evaluation as _install_perception_evaluation,
        )

    _install_perception_evaluation()

    ctor_id, spawn_name, memory_label = _agent_naming(backend, agent_type)

    print("\nRefresh_map_actors")
    await ts.UnaryAPI.refresh_actors_map(context.conn)
    result.initial_property_value = await ts.UnaryAPI.get_obj_residual(context.conn)
    burned_state = await ts.UnaryAPI.get_burned_area(context.conn)
    print(f"burned_state:{burned_state}")
    result.initial_property_num = burned_state["total_num"]

    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(bt)

    print(f"{log_p} Starting single agent demo for {max_steps} steps")

    agent_kwargs: Dict[str, Any] = {
        "session_id": session_id,
        "pause_during_llm": False,
        "max_memory_size": max_memory_size,
    }

    if agent_type == "FD":
        llm_agent = FireCls(
            context,
            agent_id=ctor_id,
            prompt_environment=prompt_environment,
            **agent_kwargs,
        )
    elif agent_type == "SD":
        llm_agent = SaveCls(
            context,
            agent_id=ctor_id,
            prompt_environment=prompt_environment,
            **agent_kwargs,
        )
    elif agent_type == "FD_WL":
        llm_agent = WlCls(
            context,
            agent_id=ctor_id,
            prompt_environment=prompt_environment,
            **agent_kwargs,
            water_capacity=_BENCHMARK_WATER_CAPACITY,
            recover_time=_BENCHMARK_RECOVER_TIME,
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}, expected FD, SD, or FD_WL")
    print(f"{log_p} Created LLM {agent_type} Agent")

    agent_actor = await llm_agent.agent.spawn_agent(
        name=spawn_name,
        location=agent_location,
        rotation=agent_rotation,
    )

    completed_ok = False
    try:
        print(f"{log_p} Successfully spawned agent: {agent_actor.get('name')}")
        print(f"{log_p} Initial perception update...")
        await _run_initial_perception(backend, llm_agent)

        aid = agent_actor.get("id")
        reg_name = getattr(llm_agent, "_agent_id", None) or memory_label
        result.register_agents([aid], [reg_name])
        if agent_type == "FD_WL":
            result.set_agent_water_setting(
                aid, float(_BENCHMARK_WATER_CAPACITY), float(_BENCHMARK_RECOVER_TIME)
            )
        else:
            result.set_agent_water_setting(aid, None, None)

        attach_experiment_result_to_base_agent(llm_agent.agent, result)

        await result.start_async()

        for step in range(max_steps):
            print(f"{log_p} === Step {step + 1}/{max_steps} ===")
            await llm_agent.step(agent_idx=0)
            await asyncio.sleep(0.1)
        completed_ok = True
    except KeyboardInterrupt:
        print(f"{log_p} Interrupted by user")
    except Exception as e:
        print(f"{log_p} Demo error: {e}")
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
    print(f"{log_p} Demo completed")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-agent unified benchmark (OL / VL / VLM).")
    p.add_argument(
        "--backend",
        type=lambda s: s.upper(),
        choices=("OL", "VL", "VLM"),
        default="OL",
        help="Only Language-base / Vision Language-base / Vision VLM Agents (default: OL).",
    )
    p.add_argument(
        "--agent-type",
        choices=("FD", "SD", "FD_WL"),
        default="FD",
        help="Fire / rescue / fire with water limit (default: FD).",
    )
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument(
        "--burn-time",
        type=float,
        default=15,
        help="Seconds after start_to_burn; default 15 (OL/VL/VLM).",
    )
    p.add_argument("--num-fires", type=int, default=1)
    p.add_argument(
        "--task-id",
        default="",
        help="Logical task id from automation config; saved in evaluation JSON.",
    )
    p.add_argument(
        "--scene-id",
        type=int,
        default=0,
        help="Scene id for metrics JSON (UE scene is set via SceneConfig.json).",
    )
    p.add_argument(
        "--max-memory-size",
        type=int,
        default=3,
        help="Short-term action memory length for the LLM controller (default: 3).",
    )
    p.add_argument(
        "--start-euler",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        default=(0.0, 0.0, -180.0),
        help="Spawn orientation as Euler angles in degrees (roll pitch yaw order for Vector3; default: 0 0 -180).",
    )
    return p.parse_args()


def main() -> None:
    GRPC_ENDPOINT = "127.0.0.1:5726"
    args = _parse_args()
    _ensure_python_client_on_path()

    z_height = 1000
    agent_location = ts.Vector3(-300, 100, z_height)
    rx, ry, rz = args.start_euler
    agent_rotation = ts.math.euler_to_quaternion(
        ts.Vector3(rx, ry, rz), is_degree=True
    )

    if args.agent_type in ("FD", "FD_WL"):
        num_fire_agents = 1
        num_rescue_agents = 0
    else:
        num_fire_agents = 0
        num_rescue_agents = 1

    prompt_environment = {
        "num_agents": num_fire_agents + num_rescue_agents,
        "num_fire_agents": num_fire_agents,
        "num_rescue_agents": num_rescue_agents,
        "num_civilians": 0,
        "num_fires": args.num_fires,
    }

    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        ue.context.sync_run(
            demo_run(
                ue.context,
                backend=args.backend,
                max_steps=args.max_steps,
                agent_location=agent_location,
                agent_rotation=agent_rotation,
                agent_type=args.agent_type,
                prompt_environment=prompt_environment,
                burn_time=args.burn_time,
                task_id=args.task_id,
                scene_id=args.scene_id,
                max_memory_size=args.max_memory_size,
            )
        )


if __name__ == "__main__":
    _ensure_python_client_on_path()
    try:
        from Metric_Tool.llm_token_evaluation import install as _install_llm_token_metrics
    except ImportError:
        from EmbodiedMAS.Metric_Tool.llm_token_evaluation import (  # type: ignore
            install as _install_llm_token_metrics,
        )

    _install_llm_token_metrics()
    main()
