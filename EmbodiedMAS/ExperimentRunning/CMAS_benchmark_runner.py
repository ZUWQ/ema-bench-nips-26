"""
Centralized MAS benchmark entry: mirrors CMAS_ol (ExperimentResult /
flush_token_summary / shield teardown). Loads ``SuperLLMAgent`` from
``super_agent_CMAS*.py`` under each backend's ``Centralized_MAS`` package.

Examples (from PythonClient repo root)::

    python -m EmbodiedMAS.ExperimentRunning.CMAS_benchmark_runner --backend OL
    python -m EmbodiedMAS.ExperimentRunning.CMAS_benchmark_runner --backend VLM --burn-time 15
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import tongsim as ts
from tongsim.core.world_context import WorldContext

_EMBODIED_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_CLIENT_ROOT = _EMBODIED_ROOT.parent

Z_HEIGHT = 1000

_BENCHMARK_WATER_CAPACITY = 1
_BENCHMARK_RECOVER_TIME = 30

# Same default layout as DMAS_ol main; only the first n_agents entries are used
DEFAULT_AGENT_POSITIONS: List[Dict[str, Any]] = [
    {"x": -300, "y": 100, "z": Z_HEIGHT},
    {"x": -300, "y": -100, "z": Z_HEIGHT},
    {"x": 300, "y": 100, "z": Z_HEIGHT},
    {"x": 300, "y": -100, "z": Z_HEIGHT},
]

_super_agent_module_by_path: Dict[str, Any] = {}


def _ensure_python_client_on_path() -> None:
    s = str(_PYTHON_CLIENT_ROOT.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _centralized_mas_dir(backend: str) -> Path:
    mapping = {
        "OL": _EMBODIED_ROOT / "Only_Language-base_Agent" / "Centralized_MAS",
        "VL": _EMBODIED_ROOT / "Vision_Language-base_Agent" / "Centralized_MAS",
        "VLM": _EMBODIED_ROOT / "Vision_VLM_Agent" / "Centralized_MAS",
    }
    if backend not in mapping:
        raise ValueError(f"Unknown backend {backend!r}, expected OL, VL, or VLM")
    p = mapping[backend]
    if not p.is_dir():
        raise FileNotFoundError(f"Centralized_MAS directory not found: {p}")
    return p


def _super_agent_module_path(backend: str) -> Path:
    stem = {
        "OL": "super_agent_CMAS",
        "VL": "super_agent_CMAS_vl",
        "VLM": "super_agent_CMAS_vlm",
    }[backend]
    path = _centralized_mas_dir(backend) / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Super agent module not found: {path}")
    return path


def _prepare_centralized_imports(backend: str) -> None:
    d = _centralized_mas_dir(backend)
    ds = str(d.resolve())
    try:
        sys.path.remove(ds)
    except ValueError:
        pass
    sys.path.insert(0, ds)


def _load_super_agent_class(backend: str) -> Type[Any]:
    path = _super_agent_module_path(backend)
    key = str(path.resolve())
    cached = _super_agent_module_by_path.get(key)
    if cached is None:
        mod_name = f"_embodied_cmas_{backend.upper()}_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        _super_agent_module_by_path[key] = mod
        cached = mod
    cls = getattr(cached, "SuperLLMAgent", None)
    if cls is None:
        raise AttributeError(f"{path.name} has no attribute 'SuperLLMAgent'")
    return cls


def _data_save_root() -> Path:
    env = os.environ.get("EMBODIED_BENCHMARK_DATA_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent / "data_save"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _default_prompt_env(agent_types: List[str]) -> Dict[str, Any]:
    return {
        "num_agents": len(agent_types),
        "num_fire_agents": sum(1 for t in agent_types if t in ("FD", "FD_WL")),
        "num_rescue_agents": sum(1 for t in agent_types if t == "SD"),
        "num_civilians": 0,
        "num_fires": "unknown",
    }


def _merge_fd_wl_other_info(
    pe: Dict[str, Any],
    agent_types: List[str],
    water_capacity: int,
    recover_time: int,
) -> None:
    """Align with single-agent water hint: when FD_WL is present, append to other_info."""
    if "FD_WL" not in agent_types:
        return
    note = (
        f"Limited extinguisher water: water_capacity={water_capacity}, "
        f"recover_time={recover_time}s. Use extinguish_fire only when necessary."
    )
    existing = str(pe.get("other_info", "")).strip()
    pe["other_info"] = f"{existing} {note}".strip() if existing else note


async def _run_initial_perception_centralized(backend: str, super_agent: Any) -> None:
    for _rid, agent in super_agent._agents.items():
        if not agent._agent:
            continue
        if backend == "OL":
            await agent._actions.get_perception_object_list(agent._agent)
        else:
            await agent.explore()


def _parse_agent_types(s: str, n_agents: int) -> List[str]:
    parts = [p.strip().upper() for p in s.split(",") if p.strip()]
    if len(parts) != n_agents:
        raise ValueError(
            f"--agent-types must have {n_agents} comma-separated entries, got {len(parts)}"
        )
    for p in parts:
        if p not in ("FD", "SD", "FD_WL"):
            raise ValueError(f"Invalid agent type {p!r}, expected FD, SD, or FD_WL")
    return parts


async def demo_run(
    context: WorldContext,
    backend: str,
    n_agents: int = 3,
    max_steps: int = 10,
    agent_positions: Optional[List[Dict[str, Any]]] = None,
    agent_types: Optional[List[str]] = None,
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    burn_time: float = 15,
    num_fires: Optional[Any] = None,
    task_id: str = "",
    scene_id: int = 0,
    start_rotation_euler_deg: Tuple[float, float, float] = (0.0, 0.0, -180.0),
) -> None:
    backend = backend.upper()
    if n_agents > len(DEFAULT_AGENT_POSITIONS):
        raise ValueError(
            f"n_agents={n_agents} exceeds DEFAULT_AGENT_POSITIONS length "
            f"({len(DEFAULT_AGENT_POSITIONS)})"
        )

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

    if agent_types is None:
        agent_types = ["FD"] * n_agents
    if len(agent_types) != n_agents:
        raise ValueError(f"agent_types length {len(agent_types)} != n_agents {n_agents}")

    if agent_positions is None:
        agent_positions = DEFAULT_AGENT_POSITIONS[:n_agents]
    elif len(agent_positions) != n_agents:
        raise ValueError(
            f"agent_positions length {len(agent_positions)} != n_agents {n_agents}"
        )

    # rot_t = tuple(start_rotation_euler_deg)
    rot_t = (0, 0, -180)
    agent_positions = [{**dict(p), "rotation": rot_t} for p in agent_positions]

    types_tag = "-".join(agent_types)
    logical_task_id = task_id.strip() or f"CMAS_{backend}_n{n_agents}_{types_tag}"

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
    pe: Dict[str, Any]
    if prompt_environment is None:
        pe = _default_prompt_env(agent_types)
    else:
        pe = dict(prompt_environment)
    if num_fires is not None:
        pe["num_fires"] = num_fires
    _merge_fd_wl_other_info(
        pe, agent_types, _BENCHMARK_WATER_CAPACITY, _BENCHMARK_RECOVER_TIME
    )

    log_p = f"[CMAS_bench|{backend}]"

    _prepare_centralized_imports(backend)
    SuperLLMAgent = _load_super_agent_class(backend)

    try:
        from Metric_Tool.perception_evaluation import (
            install_perception_evaluation as _install_perception_evaluation,
        )
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import (  # type: ignore
            install_perception_evaluation as _install_perception_evaluation,
        )
    _install_perception_evaluation()

    print("\nRefresh_map_actors")
    await ts.UnaryAPI.refresh_actors_map(context.conn)
    result.initial_property_value = await ts.UnaryAPI.get_obj_residual(context.conn)
    burned_state = await ts.UnaryAPI.get_burned_area(context.conn)
    print(f"burned_state:{burned_state}")
    result.initial_property_num = burned_state["total_num"]

    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(float(burn_time))

    super_agent: Optional[Any] = None
    completed_ok = False
    try:
        try:
            super_agent = SuperLLMAgent(
                context,
                n_agents=n_agents,
                prompt_environment=pe,
                water_capacity=_BENCHMARK_WATER_CAPACITY,
                recover_time=_BENCHMARK_RECOVER_TIME,
            )
            await super_agent.spawn_agents(agent_positions, agent_types)
            await _run_initial_perception_centralized(backend, super_agent)

            for idx in range(n_agents):
                robot_id = f"agent_{idx + 1}"
                agent = super_agent._agents.get(robot_id)
                if not agent or not agent._agent:
                    continue
                aid = agent._agent.get("id")
                result.register_agents([aid], [robot_id])
                if agent_types[idx] == "FD_WL":
                    result.set_agent_water_setting(
                        aid,
                        float(_BENCHMARK_WATER_CAPACITY),
                        float(_BENCHMARK_RECOVER_TIME),
                    )
                else:
                    result.set_agent_water_setting(aid, None, None)

            for _rid, _ag in super_agent._agents.items():
                if _ag is not None and getattr(_ag, "_agent", None):
                    attach_experiment_result_to_base_agent(_ag, result)

            await result.start_async()

            for step in range(max_steps):
                print(f"\n{log_p} === Step {step + 1}/{max_steps} ===")
                await super_agent.step()
            completed_ok = True
        except KeyboardInterrupt:
            print(f"{log_p} Interrupted by user")
        except Exception as e:
            print(f"{log_p} Demo error: {e}")
            import traceback

            traceback.print_exc()
        finally:
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

    print(f"{log_p} Demo completed")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Centralized MAS unified benchmark (OL / VL / VLM SuperLLMAgent)."
    )
    p.add_argument(
        "--backend",
        type=lambda s: s.upper(),
        choices=("OL", "VL", "VLM"),
        default="OL",
        help="Only Language / Vision-Language / VLM centralized super agent (default: OL).",
    )
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument(
        "--agent-types",
        default="",
        help="Comma-separated FD/FD_WL/SD, length must equal --n-agents (default: all FD).",
    )
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument(
        "--burn-time",
        type=float,
        default=15,
        help="Seconds after start_to_burn; default 15 (OL/VL/VLM).",
    )
    p.add_argument(
        "--num-fires",
        default=None,
        help="If set, overrides prompt_environment num_fires (int or leave as string).",
    )
    p.add_argument(
        "--task-id",
        default="",
        help="Logical task id from automation; saved in evaluation JSON.",
    )
    p.add_argument(
        "--scene-id",
        type=int,
        default=0,
        help="Scene id for metrics (UE scene from SceneConfig.json).",
    )
    p.add_argument(
        "--start-euler",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        default=(0.0, 0.0, -180.0),
        help="Spawn orientation as Euler angles in degrees for every agent (default: 0 0 -180).",
    )
    return p.parse_args()


def main() -> None:
    GRPC_ENDPOINT = "127.0.0.1:5726"
    args = _parse_args()
    _ensure_python_client_on_path()

    n_agents = args.n_agents
    if args.agent_types.strip():
        agent_types = _parse_agent_types(args.agent_types, n_agents)
    else:
        agent_types = ["FD"] * n_agents

    num_fires_cli: Optional[Any] = None
    if args.num_fires is not None:
        s = str(args.num_fires).strip()
        if s.isdigit():
            num_fires_cli = int(s)
        else:
            try:
                num_fires_cli = int(float(s))
            except ValueError:
                num_fires_cli = s


    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        ue.context.sync_run(
            demo_run(
                ue.context,
                backend=args.backend,
                n_agents=n_agents,
                max_steps=args.max_steps,
                agent_positions=None,
                agent_types=agent_types,
                prompt_environment=None,
                burn_time=args.burn_time,
                num_fires=num_fires_cli,
                task_id=args.task_id,
                scene_id=args.scene_id,
                start_rotation_euler_deg=tuple(args.start_euler),
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
