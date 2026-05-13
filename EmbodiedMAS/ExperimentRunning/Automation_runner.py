#!/usr/bin/env python3
"""
TongSim Docker automation: runs ``tasks`` from JSON in order. Each task writes outputs
under ``ExperimentRunning/tasks/<backend>/<task_id>/`` (evaluation, ``memory_logs/``
for memory/timing, ``llm_tokens/``, and ``explore_perception/`` JSONL from
``Metric_Tool.perception_evaluation`` via ``perception_evaluation_DIR``).
``EMBODIED_BENCHMARK_DATA_ROOT`` / ``EMBODIED_BENCHMARK_LOG_DIR`` are set accordingly.

Runners: ``SA`` | ``CMAS`` | ``DMAS`` | ``DMAS_WO`` — each delegates to the matching
benchmark module.

Top level must include ``max_memory_size`` (SA runner only) and ``burn_time`` (seconds
to sleep after ``start_to_burn``), shared across tasks. Each task must have ``backend``
(OL/VL/VLM). Optional per-task fields: ``start_euler_z_deg`` (float, spawn yaw in degrees,
default -180; X/Y fixed at 0, forwarded as ``--start-euler 0 0 <z>``) and ``fire_num``
for SceneConfig and each runner's ``--num-fires``.

The committed ``benchmark_tasks.json`` uses placeholder paths and example image names.
Use ``benchmark_tasks.local.json`` for real machine paths and images (it is in
``.gitignore`` — do not commit). If that file exists and ``--config`` is omitted, it is
chosen by default; otherwise ``benchmark_tasks.json`` is used.

Secrets: copy ``llm_profiles.example.json`` to ``llm_profiles.json`` and fill in values
(do not commit ``llm_profiles.json``).

Privacy / repository hygiene: if history ever contained host-specific absolute paths,
private registry hostnames, or API keys, rewrite Git history with ``git filter-repo``
or BFG before pushing to a public remote. Old commits remain cloneable and can still
expose that sensitive material.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_EXPERIMENT_RUNNING_DIR = Path(__file__).resolve().parent
_PYTHON_CLIENT_ROOT = _EXPERIMENT_RUNNING_DIR.parent.parent
_LOCAL_BENCHMARK_CONFIG = _EXPERIMENT_RUNNING_DIR / "benchmark_tasks.local.json"
_BENCHMARK_TASKS_JSON = _EXPERIMENT_RUNNING_DIR / "benchmark_tasks.json"
_DEFAULT_CONFIG_PATH = (
    _LOCAL_BENCHMARK_CONFIG
    if _LOCAL_BENCHMARK_CONFIG.is_file()
    else _BENCHMARK_TASKS_JSON
)
_DEFAULT_LLM_PROFILES_PATH = _EXPERIMENT_RUNNING_DIR / "llm_profiles.json"
_TASKS_DIRNAME = "tasks"

_SA_MODULE = "EmbodiedMAS.ExperimentRunning.SA_benchmark_runner"
_CMAS_MODULE = "EmbodiedMAS.ExperimentRunning.CMAS_benchmark_runner"
_DMAS_MODULE = "EmbodiedMAS.ExperimentRunning.DMAS_benchmark_runner"
_DMAS_WO_MODULE = "EmbodiedMAS.ExperimentRunning.DMAS_benchmark_runner_wo"

logger = logging.getLogger(__name__)

container_process: Optional[subprocess.Popen] = None
container_name: Optional[str] = None
attempted_runs = 0


def _safe_task_dirname(task_id: str) -> str:
    return task_id.replace(os.sep, "_").replace("/", "_").strip() or "unnamed_task"


def _safe_backend_dirname(backend: str) -> str:
    """Sanitized directory segment matching JSON ``backend`` (OL/VL/VLM)."""
    b = (backend or "").strip().upper()
    if b in ("OL", "VL", "VLM"):
        return b
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in b) or "UNKNOWN"


def task_workspace_dir(backend: str, task_id: str) -> Path:
    return (
        _EXPERIMENT_RUNNING_DIR
        / _TASKS_DIRNAME
        / _safe_backend_dirname(backend)
        / _safe_task_dirname(task_id)
    )


def benchmark_data_save_dir(backend: str, task_id: str) -> Path:
    p = task_workspace_dir(backend, task_id)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def setup_task_logging(backend: str, task_id: str) -> None:
    log_path = task_workspace_dir(backend, task_id) / "tongsim_automation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def setup_pre_task_logging() -> None:
    """Stdout-only logging before Docker and per-task log files are set up."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root.addHandler(sh)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_llm_profiles(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid LLM profile file {path}: top level must be a JSON object")
    out: Dict[str, Dict[str, str]] = {}
    for alias, item in raw.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"Invalid LLM profile alias: {alias!r}")
        if not isinstance(item, dict):
            raise ValueError(f"LLM profile {alias!r} must be a JSON object")
        out[alias.strip()] = item  # Field validation deferred to resolve_task_llm_env
    return out


def resolve_task_llm_env(
    task: Mapping[str, Any],
    llm_profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    alias_raw = task.get("llm_profile")
    if alias_raw is None:
        return {}
    alias = str(alias_raw).strip()
    if not alias:
        raise ValueError(f"task_id={task.get('task_id')!r}: llm_profile must be non-empty")
    profile = llm_profiles.get(alias)
    if profile is None:
        raise ValueError(
            f"task_id={task.get('task_id')!r}: llm_profile={alias!r} is not defined in llm_profiles.json"
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
                f"llm profile {alias!r} missing required field {field!r} (used for {env_key})"
            )
        env_map[env_key] = val.strip()

    extra_kwargs = profile.get("chat_completion_extra_kwargs")
    if extra_kwargs is not None:
        if not isinstance(extra_kwargs, dict):
            raise ValueError(
                f"llm profile {alias!r} chat_completion_extra_kwargs must be object"
            )
        env_map["OPENAI_CHAT_COMPLETION_EXTRA_KWARGS"] = json.dumps(
            extra_kwargs, ensure_ascii=False
        )
    return env_map


def task_backend(task: Mapping[str, Any]) -> str:
    """Per-task backend (OL/VL/VLM); matches on-disk layout and runner ``--backend``."""
    return str(task["backend"]).strip().upper()


def task_start_euler_z_deg(task: Mapping[str, Any]) -> float:
    """Spawn yaw in degrees (Z only); euler X/Y are always 0."""
    v = task.get("start_euler_z_deg")
    if v is None:
        return -180.0
    return float(v)


def _docker_ps_has_container(name: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return name in r.stdout
    except Exception:
        return False


def build_sa_runner_argv(
    cfg: Mapping[str, Any],
    sa: Mapping[str, Any],
    task: Mapping[str, Any],
    task_id: str,
    scene_id: int,
    fire_num: int,
) -> List[str]:
    rz = task_start_euler_z_deg(task)
    return [
        "--backend",
        task_backend(task),
        "--agent-type",
        str(sa["agent_type"]),
        "--max-steps",
        str(sa["max_steps"]),
        "--burn-time",
        str(cfg["burn_time"]),
        "--num-fires",
        str(fire_num),
        "--max-memory-size",
        str(cfg["max_memory_size"]),
        "--task-id",
        task_id,
        "--scene-id",
        str(scene_id),
        "--start-euler",
        "0",
        "0",
        str(rz),
    ]


def build_cmas_runner_argv(
    cfg: Mapping[str, Any],
    d: Mapping[str, Any],
    task: Mapping[str, Any],
    task_id: str,
    scene_id: int,
    fire_num: int,
) -> List[str]:
    rz = task_start_euler_z_deg(task)
    argv: List[str] = [
        "--backend",
        task_backend(task),
        "--n-agents",
        str(d["n_agents"]),
        "--max-steps",
        str(d["max_steps"]),
        "--burn-time",
        str(cfg["burn_time"]),
        "--num-fires",
        str(fire_num),
        "--task-id",
        task_id,
        "--scene-id",
        str(scene_id),
        "--start-euler",
        "0",
        "0",
        str(rz),
    ]
    at = d.get("agent_types")
    if isinstance(at, str) and at.strip():
        argv.extend(["--agent-types", at])
    return argv


def build_dmas_runner_argv(
    cfg: Mapping[str, Any],
    d: Mapping[str, Any],
    task: Mapping[str, Any],
    task_id: str,
    scene_id: int,
    fire_num: int,
) -> List[str]:
    return build_cmas_runner_argv(cfg, d, task, task_id, scene_id, fire_num)


def task_scene_extra(task: Dict[str, Any]) -> Dict[str, Any]:
    raw = task.get("scene_config_extra") or task.get("scene_config") or {}
    return raw if isinstance(raw, dict) else {}


def resolve_runner_argv(
    cfg: Dict[str, Any],
    task: Dict[str, Any],
    task_id: str,
    scene_id: int,
    fire_num: int,
) -> Tuple[str, List[str]]:
    runner = str(task.get("runner") or "SA").upper()
    if runner == "SA":
        return _SA_MODULE, build_sa_runner_argv(
            cfg, task["sa_benchmark"], task, task_id, scene_id, fire_num
        )
    if runner == "CMAS":
        return _CMAS_MODULE, build_cmas_runner_argv(
            cfg, task["cmas_benchmark"], task, task_id, scene_id, fire_num
        )
    if runner == "DMAS":
        return _DMAS_MODULE, build_dmas_runner_argv(
            cfg, task["dmas_benchmark"], task, task_id, scene_id, fire_num
        )
    if runner in ("DMAS_WO", "DMASWO", "DMAS-WO"):
        return _DMAS_WO_MODULE, build_dmas_runner_argv(
            cfg, task["dmas_benchmark"], task, task_id, scene_id, fire_num
        )
    raise ValueError(f"Unknown runner: {runner!r}")


def signal_handler(signum, frame):
    logger.info("\nReceived interrupt signal, cleaning up...")
    cleanup()
    sys.exit(0)


def cleanup() -> None:
    global container_process, container_name
    if container_process and container_process.poll() is None:
        logger.info("Stopping Docker container...")
        try:
            container_process.terminate()
            container_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            container_process.kill()
            container_process.wait()
        except Exception as e:
            logger.error(f"Error stopping container: {e}")
        finally:
            container_process = None
    if container_name:
        try:
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                timeout=5,
            )
            logger.info(f"Forced stopped container: {container_name}")
        except Exception:
            pass
    container_name = None


def update_host_scene_config(
    host_path: str,
    scene_id: int,
    fire_num: int,
    npc_num: int,
    extra: Dict[str, Any],
) -> bool:
    logger.info(
        f"📝 Writing SceneConfig: SceneID={scene_id}, FireNum={fire_num}, NPCNum={npc_num}"
    )
    try:
        p = Path(host_path)
        with open(p, encoding="utf-8") as f:
            config = json.load(f)
        config["SceneID"] = scene_id
        config["FireNum"] = fire_num
        config["NPCNum"] = npc_num
        if extra:
            config.update(extra)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info("✅ SceneConfig updated")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to update SceneConfig: {e}")
        return False


def start_docker_container(cfg: Dict[str, Any]) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    global container_name
    docker = cfg["docker"]
    name = f"tongsim_auto_{int(time.time())}_{attempted_runs}"
    container_name = name
    cmd = ["docker"] + list(docker["base_args"]) + ["--name", name, docker["image"]]
    logger.info(f"Starting Docker container: {name}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    time.sleep(2)
    if not _docker_ps_has_container(name):
        logger.error("Failed to start container")
        return None, None
    logger.info(f"Container {name} started, PID: {process.pid}")
    return process, name


def start_ue_in_container(name: str, container_ue_script: str) -> bool:
    ue_cmd = f"cd /home/ue5/Linux && ./{Path(container_ue_script).name}"
    logger.info(f"Starting UE in container: {ue_cmd}")
    exec_result = subprocess.run(
        ["docker", "exec", "-d", name, "bash", "-c", ue_cmd],
        capture_output=True,
        text=True,
    )
    if exec_result.returncode != 0:
        logger.error(f"Failed to start UE: {exec_result.stderr}")
        return False
    return True


def run_benchmark_subprocess(
    cfg: Dict[str, Any],
    task_id: str,
    module: str,
    argv: List[str],
    backend: str,
    llm_env: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, str]:
    py_timeout = cfg["runtime"]["python_timeout"]
    cwd = str(_PYTHON_CLIENT_ROOT.resolve())
    env = os.environ.copy()
    data_root = benchmark_data_save_dir(backend, task_id)
    log_dir = data_root / "memory_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env["EMBODIED_BENCHMARK_DATA_ROOT"] = str(data_root)
    # Sibling to llm_tokens under DATA_ROOT; agents write memory / timing logs here
    env["EMBODIED_BENCHMARK_LOG_DIR"] = str(log_dir.resolve())
    perception_dir = data_root / "explore_perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    env["perception_evaluation_DIR"] = str(perception_dir.resolve())
    if llm_env:
        env.update({k: str(v) for k, v in llm_env.items()})

    if subprocess.run(["which", "uv"], capture_output=True).returncode == 0:
        cmd: List[str] = ["uv", "run", "python", "-m", module, *argv]
    else:
        cmd = ["python3", "-m", module, *argv]

    logger.info(f"Running: {' '.join(cmd)} (cwd={cwd})")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=py_timeout if py_timeout > 0 else None,
            cwd=cwd,
            env=env,
        )
        if result.returncode != 0:
            se = result.stderr or ""
            so = result.stdout or ""
            chunk = 2000
            if len(se) > chunk * 2:
                err = (
                    f"stderr has {len(se)} characters, showing first/last {chunk} characters\n"
                    f"--- head ---\n{se[:chunk]}\n--- tail ---\n{se[-chunk:]}"
                )
            else:
                err = se[-1000:] or so[-1000:] or "Unknown error"
            logger.error(f"benchmark failed, return code: {result.returncode}")
            logger.error(f"error details: {err}")
            return False, f"script returned non-zero exit code: {result.returncode}"
        logger.info("benchmark subprocess executed successfully")
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"control script execution timeout (exceeds {py_timeout} seconds)"
    except Exception as e:
        return False, f"error occurred while running control script: {e}"


def run_single_iteration(
    cfg: Dict[str, Any],
    task: Dict[str, Any],
    llm_profiles: Mapping[str, Mapping[str, Any]],
) -> bool:
    global container_process, container_name
    scene_id = int(task["scene_id"])
    task_id = str(task["task_id"])
    fire_num = int(task["fire_num"])
    npc_num = int(task["npc_num"])
    extra = task_scene_extra(task)
    paths = cfg["paths"]
    rt = cfg["runtime"]
    mod, argv = resolve_runner_argv(cfg, task, task_id, scene_id, fire_num)
    llm_env = resolve_task_llm_env(task, llm_profiles)

    logger.info("=" * 70)
    logger.info(
        f"Starting {attempted_runs}th run | task_id={task_id} | SceneID={scene_id} | {mod}"
    )
    logger.info("=" * 70)
    try:
        container_process, name = start_docker_container(cfg)
        if not container_process or not name:
            raise RuntimeError("Failed to start Docker container")
        container_name = name

        if not update_host_scene_config(
            paths["host_scene_config"],
            scene_id,
            fire_num,
            npc_num,
            extra,
        ):
            raise RuntimeError("Failed to update scene config")

        if not start_ue_in_container(name, paths["container_ue_script"]):
            raise RuntimeError("Failed to start UE")

        logger.info(f"Waiting for UE to start... ({rt['ue_startup_wait']} seconds)")
        time.sleep(float(rt["ue_startup_wait"]))

        if not _docker_ps_has_container(name):
            raise RuntimeError("Container is unhealthy or exited prematurely")

        logger.info("UE environment is ready")
        ok, err = run_benchmark_subprocess(
            cfg, task_id, mod, argv, task_backend(task), llm_env=llm_env
        )
        if not ok:
            raise RuntimeError(f"benchmark failed: {err}")

        logger.info("✅ Single run process completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error occurred during run: {e}")
        return False
    finally:
        cleanup()
        time.sleep(3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TongSim Docker automation (JSON config; per-task_id output directories)"
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=(
            "JSON config path. If benchmark_tasks.local.json exists and this flag is "
            f"omitted, that file is used by default; otherwise defaults to "
            f"{_BENCHMARK_TASKS_JSON.name}"
        ),
    )
    return p.parse_args()


def main() -> None:
    global attempted_runs
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    llm_profiles_path = _DEFAULT_LLM_PROFILES_PATH
    llm_profiles = load_llm_profiles(llm_profiles_path)

    setup_pre_task_logging()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    runs_per_task = int(cfg["runtime"]["runs_per_task"])
    max_err = int(cfg["runtime"]["max_consecutive_errors"])
    tasks = cfg["tasks"]

    logger.info("=" * 70)
    logger.info("TongSim Docker automation")
    logger.info(f"Config file: {config_path}")
    logger.info(f"LLM profile file: {llm_profiles_path} | number of entries: {len(llm_profiles)}")
    logger.info(
        f"Task directory root: {_EXPERIMENT_RUNNING_DIR / _TASKS_DIRNAME}"
        f" (<backend>/<task_id>/ below; backend from each task)"
    )
    logger.info(f"Number of tasks: {len(tasks)} | Number of successful runs per task: {runs_per_task}")
    logger.info("=" * 70)

    if subprocess.run(["docker", "version"], capture_output=True).returncode != 0:
        logger.error("Error: Docker is not available")
        sys.exit(1)

    task_success: Dict[str, int] = {str(t["task_id"]): 0 for t in tasks}

    for idx, task in enumerate(tasks):
        task_id = str(task["task_id"])
        benchmark_data_save_dir(task_backend(task), task_id)
        setup_task_logging(task_backend(task), task_id)

        logger.info("\n" + "=" * 80)
        logger.info(
            f"🚀 Task {idx + 1}/{len(tasks)} | task_id={task_id} | "
            f"backend={task_backend(task)} | scene_id={task['scene_id']}"
        )
        logger.info("=" * 80)

        consecutive_errors = 0
        success_count = 0

        while success_count < runs_per_task:
            if consecutive_errors >= max_err:
                logger.error(f"❌ Task {task_id} consecutive errors exceeded, skipping")
                break

            attempted_runs += 1
            success = run_single_iteration(cfg, task, llm_profiles)

            if success:
                success_count += 1
                task_success[task_id] += 1
                consecutive_errors = 0
                logger.info(
                    f"✅ Task {task_id} current run successful: {success_count}/{runs_per_task}"
                )
            else:
                consecutive_errors += 1
                logger.warning(
                    f"⚠️ Task {task_id} current run failed, consecutive errors: {consecutive_errors}"
                )

            time.sleep(3)

        logger.info(
            f"\n🏁 Task {task_id} completed: successful {success_count}/{runs_per_task}"
        )

    setup_pre_task_logging()
    logger.info("\n" + "=" * 80)
    logger.info("🎉 All tasks completed")
    for tid, cnt in task_success.items():
        logger.info(f"  {tid} successful: {cnt}/{runs_per_task}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
