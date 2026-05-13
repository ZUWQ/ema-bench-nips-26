#!/usr/bin/env python3
"""
TongSim Docker automation: runs ``tasks`` from JSON in order. Each task writes outputs
under ``Human_Agent/Running/tasks/<backend>/<task_id>/`` (evaluation, ``memory_logs/``
for memory/timing, ``llm_tokens/``, and ``explore_perception/`` JSONL from
``Metric_Tool.perception_evaluation`` via ``perception_evaluation_DIR``).
``EMBODIED_BENCHMARK_DATA_ROOT`` / ``EMBODIED_BENCHMARK_LOG_DIR`` are set the same way.

This runner targets the Human DMAS baseline only: it always invokes
``EmbodiedMAS.Human_Agent.Running.DMAS_human_runner``.

Top level must include ``burn_time`` (seconds to sleep after ``start_to_burn``), shared
across tasks. Each task must include ``backend`` (for on-disk layout). Optional per-task
fields: ``start_euler_z_deg`` (float spawn yaw in degrees, default -180; X/Y fixed at 0,
passed as ``--start-euler 0 0 <z>``) and ``fire_num`` for SceneConfig and ``--num-fires``.

The committed ``benchmark_human_tasks.json`` uses placeholder paths and example image
names. Put real local paths in ``benchmark_human_tasks.local.json`` (listed in
``.gitignore`` — do not commit). If that file exists and ``--config`` is omitted, it is
preferred.

Privacy / repository hygiene: if history ever contained host-specific absolute paths or
private registry hostnames, rewrite Git history with ``git filter-repo`` or BFG before
pushing to a public remote.
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
# Running/ is nested under EmbodiedMAS/Human_Agent/Running, so repo root is three levels up.
_PYTHON_CLIENT_ROOT = _EXPERIMENT_RUNNING_DIR.parent.parent.parent
_LOCAL_BENCHMARK_CONFIG = _EXPERIMENT_RUNNING_DIR / "benchmark_human_tasks.local.json"
_BENCHMARK_HUMAN_TASKS_JSON = _EXPERIMENT_RUNNING_DIR / "benchmark_human_tasks.json"
_DEFAULT_CONFIG_PATH = (
    _LOCAL_BENCHMARK_CONFIG
    if _LOCAL_BENCHMARK_CONFIG.is_file()
    else _BENCHMARK_HUMAN_TASKS_JSON
)
_TASKS_DIRNAME = "tasks"

_DMAS_HUMAN_MODULE = "EmbodiedMAS.Human_Agent.Running.DMAS_human_runner"

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
    raise NotImplementedError("Automation_human_runner only supports DMAS human runner.")


def build_cmas_runner_argv(
    cfg: Mapping[str, Any],
    d: Mapping[str, Any],
    task: Mapping[str, Any],
    task_id: str,
    scene_id: int,
    fire_num: int,
) -> List[str]:
    raise NotImplementedError("Automation_human_runner only supports DMAS human runner.")


def build_dmas_runner_argv(
    cfg: Mapping[str, Any],
    d: Mapping[str, Any],
    task: Mapping[str, Any],
    task_id: str,
    scene_id: int,
    fire_num: int,
) -> List[str]:
    rz = task_start_euler_z_deg(task)
    argv: List[str] = [
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

    # Optional web UI settings (pass-through to DMAS_human_runner -> DMAS_vl_human).
    if bool(d.get("no_web_ui")):
        argv.append("--no-web-ui")
    if isinstance(d.get("web_ui_host"), str) and d["web_ui_host"].strip():
        argv.extend(["--web-ui-host", str(d["web_ui_host"])])
    if d.get("web_ui_port") is not None:
        argv.extend(["--web-ui-port", str(int(d["web_ui_port"]))])
    return argv


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
    # Backward compatible: accept both "dmas_human" (new) and "dmas_benchmark" (old).
    d = task.get("dmas_human") or task.get("dmas_benchmark")
    if not isinstance(d, dict):
        raise ValueError("task must include 'dmas_human' (or legacy 'dmas_benchmark') dict")
    return _DMAS_HUMAN_MODULE, build_dmas_runner_argv(cfg, d, task, task_id, scene_id, fire_num)


def signal_handler(signum, frame):
    logger.info("\nInterrupt received, cleaning up...")
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
            logger.error(f"Error while cleaning up container: {e}")
        finally:
            container_process = None
    if container_name:
        try:
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                timeout=5,
            )
            logger.info(f"Forced stop of container: {container_name}")
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
        logger.error("Container failed to start")
        return None, None
    logger.info(f"Container {name} started, PID: {process.pid}")
    return process, name


def start_ue_in_container(name: str, container_ue_script: str) -> bool:
    ue_cmd = f"cd /home/ue5/Linux && ./{Path(container_ue_script).name}"
    logger.info(f"Starting UE inside container: {ue_cmd}")
    exec_result = subprocess.run(
        ["docker", "exec", "-d", name, "bash", "-c", ue_cmd],
        capture_output=True,
        text=True,
    )
    if exec_result.returncode != 0:
        logger.error(f"UE failed to start: {exec_result.stderr}")
        return False
    return True


def run_benchmark_subprocess(
    cfg: Dict[str, Any],
    task_id: str,
    module: str,
    argv: List[str],
    backend: str,
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
                    f"stderr has {len(se)} characters; showing first/last {chunk} chars\n"
                    f"--- head ---\n{se[:chunk]}\n--- tail ---\n{se[-chunk:]}"
                )
            else:
                err = se[-1000:] or so[-1000:] or "unknown error"
            logger.error(f"benchmark failed, exit code: {result.returncode}")
            logger.error(f"Error detail: {err}")
            return False, f"Child exited with non-zero code: {result.returncode}"
        logger.info("benchmark subprocess finished successfully")
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"Control script timed out (>{py_timeout}s)"
    except Exception as e:
        return False, f"Exception while running control script: {e}"


def run_single_iteration(cfg: Dict[str, Any], task: Dict[str, Any]) -> bool:
    global container_process, container_name
    scene_id = int(task["scene_id"])
    task_id = str(task["task_id"])
    fire_num = int(task["fire_num"])
    npc_num = int(task["npc_num"])
    extra = task_scene_extra(task)
    paths = cfg["paths"]
    rt = cfg["runtime"]
    mod, argv = resolve_runner_argv(cfg, task, task_id, scene_id, fire_num)

    logger.info("=" * 70)
    logger.info(
        f"Starting run {attempted_runs} | task_id={task_id} | SceneID={scene_id} | {mod}"
    )
    logger.info("=" * 70)
    try:
        container_process, name = start_docker_container(cfg)
        if not container_process or not name:
            raise RuntimeError("Docker container failed to start")
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
            raise RuntimeError("UE failed to start")

        logger.info(f"Waiting for UE startup... ({rt['ue_startup_wait']}s)")
        time.sleep(float(rt["ue_startup_wait"]))

        if not _docker_ps_has_container(name):
            raise RuntimeError("Container unhealthy or exited early")

        logger.info("UE environment ready")
        ok, err = run_benchmark_subprocess(
            cfg, task_id, mod, argv, task_backend(task)
        )
        if not ok:
            raise RuntimeError(f"benchmark failed: {err}")

        logger.info("✅ Single run completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error during run: {e}")
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
            "JSON config path. If benchmark_human_tasks.local.json exists and this flag "
            f"is omitted, that file is used by default; otherwise defaults to "
            f"{_BENCHMARK_HUMAN_TASKS_JSON.name}"
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

    setup_pre_task_logging()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    runs_per_task = int(cfg["runtime"]["runs_per_task"])
    max_err = int(cfg["runtime"]["max_consecutive_errors"])
    tasks = cfg["tasks"]

    logger.info("=" * 70)
    logger.info("TongSim Docker automation")
    logger.info(f"Config file: {config_path}")
    logger.info(
        f"Task root: {_EXPERIMENT_RUNNING_DIR / _TASKS_DIRNAME}"
        f" (<backend>/<task_id>/ below; backend from each task)"
    )
    logger.info(f"Tasks: {len(tasks)} | successful runs per task: {runs_per_task}")
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
                logger.error(f"❌ Task {task_id}: too many consecutive errors, skipping")
                break

            attempted_runs += 1
            success = run_single_iteration(cfg, task)

            if success:
                success_count += 1
                task_success[task_id] += 1
                consecutive_errors = 0
                logger.info(
                    f"✅ Task {task_id} succeeded this round: {success_count}/{runs_per_task}"
                )
            else:
                consecutive_errors += 1
                logger.warning(
                    f"⚠️ Task {task_id} failed this round, consecutive errors: {consecutive_errors}"
                )

            time.sleep(3)

        logger.info(
            f"\n🏁 Task {task_id} finished: successes {success_count}/{runs_per_task}"
        )

    setup_pre_task_logging()
    logger.info("\n" + "=" * 80)
    logger.info("🎉 All tasks finished")
    for tid, cnt in task_success.items():
        logger.info(f"  {tid}: successes {cnt}/{runs_per_task}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
