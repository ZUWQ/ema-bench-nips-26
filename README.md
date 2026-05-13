# EMA-Bench: A Benchmark for Embodied Multi-Agent Decision-Making in Dynamic Environments

[中文说明](README_zh.md)

![EMA-Bench overview: multi-robot fire and rescue in a dynamic indoor environment](ema-bench-frontend/src/picture/overview.jpg)

## Abstract

Embodied multi-agent systems are vital for high-risk disaster response, yet they struggle in dynamic environments characterized by rapid hazard escalation and path-dependent dynamics. The rapid compounding of hazards in these settings demands a shift from reactive execution to proactive reasoning to effectively anticipate environmental dynamics. Furthermore, the extreme time sensitivity of these scenarios makes multi-agent cooperation a functional necessity, as agents must coordinate their efforts to prevent the disaster from outpacing the team's capacity. To address this, we introduce **EMA-Bench**, a high-fidelity simulation platform designed to evaluate multi-agent coordination within self-progressing fire. EMA-Bench facilitates interactions where agent actions directly influence the environmental progression under strict temporal urgency and partial observability. We propose a structured evaluation framework spanning foundational task execution, environmental exploration, and collaborative efficiency. Our empirical analysis of state-of-the-art multimodal foundation model-based agents highlights a significant deficiency in their ability to handle time-sensitive trade-offs and irreversible state transitions. These findings reveal a substantial gap in current embodied intelligence and establish a rigorous foundation for future research in resilient multi-agent coordination.

---

## Resources

| Resource | Description |
|----------|-------------|
| [Platform guidelines (Google Doc)](https://drive.google.com/file/d/1wgos_ZZG2T6XC8TU00DXmWDLy1ua4tHv/view?usp=drive_link) | Experiment and collaboration guidelines for the EMA-Bench platform. |
| [Simulation configs (Hugging Face dataset)](https://huggingface.co/datasets/EMAS4Rescue/ema-bench-26) | Dataset and metadata for **simulation configuration parameters** (scenes, tasks, and related settings). |

---

## Repository overview

**PythonClient** ships the open-source **Python gRPC client** for the TongSim / Unreal-based simulator (`tongsim` under `src/tongsim/`) and the **EmbodiedMAS** codebase used to run and evaluate language-only (**OL**), vision-language (**VL** / **VLM**), centralized (**CMAS**), decentralized (**DMAS**), and human-in-the-loop agents against EMA-Bench-style scenarios.

---

## Repository layout

| Path | Role |
|------|--------|
| `src/tongsim/` | Core Python client: connection, entities, geometry helpers, version. |
| `EmbodiedMAS/Only_Language-base_Agent/` | Language-only agents and actions (OL). |
| `EmbodiedMAS/VLM_Agent/` | Vision-language / VLM-oriented agents (single, CMAS, DMAS). |
| `EmbodiedMAS/Human_Agent/` | Human baselines, web UI, and runners that mix human + agents. |
| `EmbodiedMAS/ExperimentRunning/` | CLI modules: `SA_benchmark_runner`, `CMAS_benchmark_runner`, `DMAS_benchmark_runner`, task JSON, automation helpers. |
| `EmbodiedMAS/prompt/` | Text prompts and action / observation space snippets. |
| `EmbodiedMAS/Metric_Tool/` | Token, timing, perception, and aggregate evaluation helpers. |
| `examples/` | Runnable demos (MAS / RL / capture / EXR, etc.). |

---

## Requirements

- **Python** `>=3.12` (see `pyproject.toml`).
- A **running TongSim / UE** build exposing the gRPC endpoint your scripts use (examples often assume a local port such as `5726`; adjust to your deployment).
- For **full benchmark automation**, edit `EmbodiedMAS/ExperimentRunning/benchmark_tasks.json`: replace Docker image placeholders, host paths to the shipping build, `SceneConfig.json`, and other paths so they match your machine or cluster.

---

## Installation

From the repository root (editable install so `EmbodiedMAS` and `examples` resolve imports as in the docs):

```bash
pip install -e .
```

If you use **uv**:

```bash
uv sync
```

Optional **dependency groups** (see `pyproject.toml`): `dev` (ruff, black, pre-commit), `docs` (MkDocs), `test` (pytest), `multi_agent` (gymnasium, psutil, numpy, etc.).

---

## Configuration

1. **LLM API**  
   Copy `EmbodiedMAS/llm_config.example.json` to `EmbodiedMAS/llm_config.json` and set `api_key`, `base_url`, `model`, and any `chat_completion_extra_kwargs`. You can rely on `OPENAI_API_KEY` instead of storing the key in the file if your code path supports it.

2. **Benchmark LLM profiles** (if you use multi-profile runs)  
   Start from `EmbodiedMAS/ExperimentRunning/llm_profiles.example.json` and adapt to your providers.

3. **Secrets**  
   Do not commit real keys or internal endpoints.

---

## Quick start

**1. Smoke-test the gRPC client** (with the simulator already running and the endpoint matching your script):

```bash
python examples/demo_mas_demo.py
```

Other demos under `examples/` cover RL, capture, voxel utilities, etc.; open each file for its assumptions and endpoint.

**2. Single-agent benchmark (OL backend)** from repo root:

```bash
python -m EmbodiedMAS.ExperimentRunning.SA_benchmark_runner --backend OL --agent-type FD
```

Use `--help` on that module for additional flags (other backends, agent types, timeouts). Comparable entry points exist for centralized and decentralized MAS (`CMAS_benchmark_runner`, `DMAS_benchmark_runner`, and variants such as `DMAS_benchmark_runner_wo`).

---

## Benchmark task JSON

`EmbodiedMAS/ExperimentRunning/benchmark_tasks.json` drives automation: Docker / GPU flags, mounts, UE startup script inside the container, host `SceneConfig.json`, per-task IDs, timeouts, and repeat counts. **All placeholder paths and image names must be updated** before unattended runs succeed on your infrastructure.

---

## Development

- **Lint / format**: `ruff`, `black` (see `[tool.ruff]` and dev dependency group).
- **Tests**: `pytest` (`[tool.pytest.ini_options]` enables asyncio defaults).
- **Pre-commit**: optional; install hooks if your team uses them.

---

## Path note (VL / VLM backends)

On disk, agent code for vision-language stacks often lives under `EmbodiedMAS/VLM_Agent/`. Some benchmark modules still reference legacy directory names such as `Vision_Language-base_Agent` or `Vision_VLM_Agent`. If `--backend VL` or `--backend VLM` fails with “directory not found”, align the path mapping in the corresponding `ExperimentRunning` module with your checkout, or add a symlink that matches the expected folder name.

---

## Citation

If you use EMA-Bench or this codebase in research, please cite the paper (full BibTeX will be linked here once available). Project packaging metadata lists maintainers in `pyproject.toml`.
