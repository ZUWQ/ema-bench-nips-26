"""
Human DMAS baseline entry point: invokes only ``Human_Agent/Decentralized_MAS/DMAS_vl_human.py``.

Examples (from PythonClient repo root)::

    python -m EmbodiedMAS.Human_Agent.Running.DMAS_human_runner --n-agents 4 --agent-types FD,FD,FD,SD
    python -m EmbodiedMAS.Human_Agent.Running.DMAS_human_runner --max-steps 50 --burn-time 15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tongsim as ts
from tongsim.core.world_context import WorldContext

_HUMAN_AGENT_ROOT = Path(__file__).resolve().parent.parent
_EMBODIED_ROOT = _HUMAN_AGENT_ROOT.parent
_PYTHON_CLIENT_ROOT = _EMBODIED_ROOT.parent

Z_HEIGHT = 1000

_BENCHMARK_WATER_CAPACITY = 1
_BENCHMARK_RECOVER_TIME = 30

DEFAULT_AGENT_POSITIONS: List[Dict[str, Any]] = [
    {"x": -300, "y": 100, "z": Z_HEIGHT},
    {"x": -300, "y": -100, "z": Z_HEIGHT},
    {"x": 300, "y": 100, "z": Z_HEIGHT},
    {"x": 300, "y": -100, "z": Z_HEIGHT},
]


def _ensure_python_client_on_path() -> None:
    s = str(_PYTHON_CLIENT_ROOT.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


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
        "other_info": "",
    }


def _merge_fd_wl_other_info(
    pe: Dict[str, Any],
    agent_types: List[str],
    water_capacity: int,
    recover_time: int,
) -> None:
    if "FD_WL" not in agent_types:
        return
    note = (
        f"Limited extinguisher water: water_capacity={water_capacity}, "
        f"recover_time={recover_time}s. Use extinguish_fire only when necessary."
    )
    existing = str(pe.get("other_info", "")).strip()
    pe["other_info"] = f"{existing} {note}".strip() if existing else note


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
    n_agents: int = 3,
    max_steps: int = 10,
    agent_positions: Optional[List[Dict[str, Any]]] = None,
    agent_types: Optional[List[str]] = None,
    prompt_environment: Optional[Dict[str, Any]] = None,
    *,
    burn_time: float = 1.0,
    num_fires: Optional[Any] = None,
    task_id: str = "",
    scene_id: int = 0,
    start_rotation_euler_deg: Tuple[float, float, float] = (0.0, 0.0, -180.0),
    auto_start_web_ui: bool = True,
    web_ui_host: str = "0.0.0.0",
    web_ui_port: int = 8080,
    time_limit_sec: Optional[float] = None,
    hard_timeout_grace_sec: float = 60.0,
) -> None:
    if n_agents > len(DEFAULT_AGENT_POSITIONS):
        raise ValueError(
            f"n_agents={n_agents} exceeds DEFAULT_AGENT_POSITIONS length "
            f"({len(DEFAULT_AGENT_POSITIONS)})"
        )

    _ensure_python_client_on_path()

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

    plots_dir = _data_save_root() / "experiment_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    from EmbodiedMAS.Human_Agent.Decentralized_MAS.DMAS_vl_human import (  # noqa: WPS433
        demo_run as _vl_human_demo_run,
    )

    await _vl_human_demo_run(
        context,
        n_agents=n_agents,
        max_steps=max_steps,
        agent_positions=agent_positions,
        agent_types=agent_types,
        prompt_environment=pe,
        water_capacity=_BENCHMARK_WATER_CAPACITY,
        recover_time=_BENCHMARK_RECOVER_TIME,
        burn_time=burn_time,
        plot_output_dir=plots_dir,
        experiment_id=task_id.strip() or "exp_dmas_vl_human",
        auto_start_web_ui=auto_start_web_ui,
        web_ui_host=web_ui_host,
        web_ui_port=web_ui_port,
        time_limit_sec=time_limit_sec,
        hard_timeout_grace_sec=hard_timeout_grace_sec,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DMAS human runner (only DMAS_vl_human)."
    )
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument(
        "--agent-types",
        default="",
        help="Comma-separated FD/FD_WL/SD, length must equal --n-agents (default: all FD).",
    )
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument(
        "--burn-time",
        type=float,
        default=15,
        help="Seconds after start_to_burn; default 15.",
    )
    p.add_argument(
        "--num-fires",
        default=None,
        help="If set, overrides prompt_environment num_fires (int or string).",
    )
    p.add_argument(
        "--task-id",
        default="",
        help="Experiment id prefix; will be suffixed with timestamp by DMAS_vl_human.",
    )
    p.add_argument(
        "--scene-id",
        type=int,
        default=0,
        help="(Compat) DMAS_vl_human handles metrics internally; runner ignores this.",
    )
    p.add_argument(
        "--start-euler",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        default=(0.0, 0.0, -180.0),
        help="(Compat) rotation is fixed at 0 0 -180 (aligned with benchmark runner).",
    )
    p.add_argument(
        "--no-web-ui",
        action="store_true",
        help="Disable auto-start web UI server (requires custom decision_providers).",
    )
    p.add_argument(
        "--web-ui-host",
        default="0.0.0.0",
        help="Web UI server host (default: 0.0.0.0).",
    )
    p.add_argument(
        "--web-ui-port",
        type=int,
        default=int(os.environ.get("EMBODIED_WEB_UI_PORT", "8080")),
        help="Web UI server port (default: env EMBODIED_WEB_UI_PORT or 8080).",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=300,
        help=(
            "Wall-clock seconds for run_agents. When set, terminates the experiment "
            "by time (soft check at step boundary). Set --max-steps very large to "
            "make time the only stop condition. Unset = step-based termination."
        ),
    )
    p.add_argument(
        "--hard-timeout-grace",
        type=float,
        default=5.0,
        help=(
            "Extra seconds added on top of --time-limit as a hard cap "
            "(asyncio.wait_for); cancels stuck agent loops if soft check misses."
        ),
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
                auto_start_web_ui=not args.no_web_ui,
                web_ui_host=args.web_ui_host,
                web_ui_port=args.web_ui_port,
                time_limit_sec=args.time_limit,
                hard_timeout_grace_sec=args.hard_timeout_grace,
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
    try:
        from Metric_Tool.perception_evaluation import (
            install_perception_evaluation as _install_perception_evaluation,
        )
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import (  # type: ignore
            install_perception_evaluation as _install_perception_evaluation,
        )
    _install_perception_evaluation()
    main()
