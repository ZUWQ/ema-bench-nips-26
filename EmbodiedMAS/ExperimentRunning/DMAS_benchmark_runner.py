"""
Unified decentralized-MAS benchmark entry: loads ``demo_run`` from
``DMAS_ol.py`` / ``DMAS_vl.py`` / ``DMAS_vlm.py`` under each backend's ``Decentralized_MAS``
package. CLI layout matches ``CMAS_benchmark_runner``.

Examples (from PythonClient repo root)::

    python -m EmbodiedMAS.ExperimentRunning.DMAS_benchmark_runner --backend OL --n-agents 3
    python -m EmbodiedMAS.ExperimentRunning.DMAS_benchmark_runner --backend VL --max-steps 10
    python -m EmbodiedMAS.ExperimentRunning.DMAS_benchmark_runner --backend VLM --burn-time 15
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tongsim as ts
from tongsim.core.world_context import WorldContext

_EMBODIED_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_CLIENT_ROOT = _EMBODIED_ROOT.parent

Z_HEIGHT = 1000

_BENCHMARK_WATER_CAPACITY = 1
_BENCHMARK_RECOVER_TIME = 30

DEFAULT_AGENT_POSITIONS: List[Dict[str, Any]] = [
    {"x": -300, "y": 100, "z": Z_HEIGHT},
    {"x": -300, "y": -100, "z": Z_HEIGHT},
    {"x": 300, "y": 100, "z": Z_HEIGHT},
    {"x": 300, "y": -100, "z": Z_HEIGHT},
    {"x": 0, "y": 100, "z": Z_HEIGHT},
    {"x": 0, "y": -100, "z": Z_HEIGHT},
]

_dmas_module_by_path: Dict[str, Any] = {}


def _load_llm_profiles(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"llm profile file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"llm profile file format error: {path} top level must be object")
    out: Dict[str, Dict[str, str]] = {}
    for alias, item in raw.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"llm profile alias invalid: {alias!r}")
        if not isinstance(item, dict):
            raise ValueError(f"llm profile {alias!r} must be object")
        out[alias.strip()] = item
    return out


def _resolve_llm_profile_env(llm_profile: str, llm_profiles_path: Path) -> Dict[str, str]:
    profiles = _load_llm_profiles(llm_profiles_path)
    profile = profiles.get(llm_profile)
    if profile is None:
        raise ValueError(
            f"llm_profile={llm_profile!r} not defined in {llm_profiles_path}"
        )
    required = {
        "OPENAI_API_KEY": "api_key",
        "OPENAI_BASE_URL": "base_url",
        "OPENAI_MODEL": "model",
    }
    env_map: Dict[str, str] = {}
    for env_key, field in required.items():
        val = profile.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                f"llm profile {llm_profile!r} missing required field {field!r} (used for {env_key})"
            )
        env_map[env_key] = val.strip()

    extra_kwargs = profile.get("chat_completion_extra_kwargs")
    if extra_kwargs is not None:
        if not isinstance(extra_kwargs, dict):
            raise ValueError(
                f"llm profile {llm_profile!r} chat_completion_extra_kwargs must be object"
            )
        env_map["OPENAI_CHAT_COMPLETION_EXTRA_KWARGS"] = json.dumps(
            extra_kwargs, ensure_ascii=False
        )
    return env_map


def _ensure_python_client_on_path() -> None:
    s = str(_PYTHON_CLIENT_ROOT.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _decentralized_mas_dir(backend: str) -> Path:
    mapping = {
        "OL": _EMBODIED_ROOT / "Only_Language-base_Agent" / "Decentralized_MAS",
        "VL": _EMBODIED_ROOT / "Vision_Language-base_Agent" / "Decentralized_MAS",
        "VLM": _EMBODIED_ROOT / "VLM_Agent" / "Decentralized_MAS",
    }
    if backend not in mapping:
        raise ValueError(f"Unknown backend {backend!r}, expected OL, VL, or VLM")
    p = mapping[backend]
    if not p.is_dir():
        raise FileNotFoundError(f"Decentralized_MAS directory not found: {p}")
    return p


def _dmas_module_path(backend: str) -> Path:
    stem = {"OL": "DMAS_ol", "VL": "DMAS_vl", "VLM": "DMAS_vlm"}[backend]
    path = _decentralized_mas_dir(backend) / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"DMAS module not found: {path}")
    return path


def _load_dmas_module(backend: str) -> Any:
    path = _dmas_module_path(backend)
    key = str(path.resolve())
    cached = _dmas_module_by_path.get(key)
    if cached is not None:
        return cached
    mod_name = f"_embodied_dmas_{backend.upper()}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec_module: otherwise cls.__module__ is missing,
    # and Python 3.14+ dataclasses can raise AttributeError during @dataclass processing.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    _dmas_module_by_path[key] = mod
    return mod


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
    backend: str,
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
    time_limit_sec: Optional[float] = None,
    hard_timeout_grace_sec: float = 60.0,
) -> None:
    backend = backend.upper()
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

    dmas_mod = _load_dmas_module(backend)
    # DMAS is loaded dynamically; install hooks after load so ActionAPI gets the perception-logging patch.
    try:
        from Metric_Tool.perception_evaluation import (
            install_perception_evaluation as _install_perception_evaluation,
        )
    except ImportError:
        from EmbodiedMAS.Metric_Tool.perception_evaluation import (  # type: ignore
            install_perception_evaluation as _install_perception_evaluation,
        )
    _install_perception_evaluation()
    fn = getattr(dmas_mod, "demo_run", None)
    if fn is None:
        raise AttributeError(f"{_dmas_module_path(backend).name} has no demo_run")

    await fn(
        context,
        n_agents=n_agents,
        max_steps=max_steps,
        agent_positions=agent_positions,
        agent_types=agent_types,
        prompt_environment=pe,
        water_capacity=_BENCHMARK_WATER_CAPACITY,
        recover_time=_BENCHMARK_RECOVER_TIME,
        plot_output_dir=plots_dir,
        burn_time=burn_time,
        scene_id=scene_id,
        task_id=task_id,
        time_limit_sec=time_limit_sec,
        hard_timeout_grace_sec=hard_timeout_grace_sec,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decentralized MAS unified benchmark (OL / VL / VLM DMAS_ol|vl|vlm)."
    )
    p.add_argument(
        "--backend",
        type=lambda s: s.upper(),
        choices=("OL", "VL", "VLM"),
        default="OL",
        help="Only Language / Vision-Language / VLM decentralized MAS (default: OL).",
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
        help="Seconds after start_to_burn; default 15 (OL/VL/VLM).",
    )
    p.add_argument(
        "--num-fires",
        default=None,
        help="If set, overrides prompt_environment num_fires (int or string).",
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
    # p.add_argument(
    #     "--llm-profile",
    #     default="qwen7b",
    #     help=(
    #         "LLM profile alias (e.g. qwen72b). If set, reads key/url/model from "
    #         "llm_profiles.json and exports OPENAI_* for this process."
    #     ),
    # )
    # p.add_argument(
    #     "--llm-profiles-path",
    #     default=str(Path(__file__).resolve().parent / "llm_profiles.json"),
    #     help="Path to llm_profiles.json (default: ExperimentRunning/llm_profiles.json).",
    # )
    return p.parse_args()


def main() -> None:
    GRPC_ENDPOINT = "127.0.0.1:5726"
    args = _parse_args()
    _ensure_python_client_on_path()
    # if args.llm_profile.strip():
    #     profile_env = _resolve_llm_profile_env(
    #         llm_profile=args.llm_profile.strip(),
    #         llm_profiles_path=Path(args.llm_profiles_path).expanduser().resolve(),
    #     )
    #     os.environ.update(profile_env)
    #     print(
    #         f"[DMAS benchmark] LLM profile={args.llm_profile.strip()} "
    #         f"model={profile_env.get('OPENAI_MODEL')}"
    #     )

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
