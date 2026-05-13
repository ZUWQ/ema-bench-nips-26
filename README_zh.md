# EMA-Bench: A Benchmark for Embodied Multi-Agent Decision-Making in Dynamic Environments

[English](README.md)

## 摘要

具身多智能体系统对高风险灾害响应至关重要，但在灾害快速升级、动力学具有路径依赖特征的动态环境中仍表现不足。危害的快速叠加要求从被动执行转向主动推理，以更好地预见环境演化。同时，极端的时间敏感性使多智能体协作成为必要条件：智能体必须协同行动，避免灾情发展超出团队处置能力。为此，我们提出 **EMA-Bench**——一个高保真仿真平台，用于在**动态蔓延火场**中评估多智能体协作。EMA-Bench 支持在严格时间压力与部分可观测条件下，让智能体行为直接影响环境演化。我们提出涵盖基础任务执行、环境探索与协作效率的结构化评测框架。对前沿多模态基础模型智能体的实证分析表明，其在时间敏感权衡与不可逆状态转移方面存在明显短板。上述结果表明当前具身智能仍存在显著差距，并为面向韧性多智能体协作的后续研究提供了严谨基础。

---

## 相关资源

| 资源 | 说明 |
|------|------|
| [方针平台说明（Google 文档）](https://docs.google.com/document/d/ema-bench) | EMA-Bench 平台相关的实验与协作方针说明。 |
| [仿真配置（Hugging Face 数据集）](https://huggingface.co/datasets/EMAS4Rescue/ema-bench-26) | **仿真配置参数**（场景、任务及相关设置）的数据集与元数据。 |

---

## 仓库概览

**PythonClient** 提供面向 TongSim / Unreal 仿真端的 **Python gRPC 客户端**（可安装包 `tongsim`，位于 `src/tongsim/`），以及用于在 EMA-Bench 类场景中运行与评测的 **EmbodiedMAS** 代码：支持纯语言（**OL**）、视觉语言（**VL** / **VLM**）、中心化（**CMAS**）、去中心化（**DMAS**）及人机协同智能体。

---

## 仓库结构

| 路径 | 作用 |
|------|------|
| `src/tongsim/` | 核心 Python 客户端：连接、实体、几何辅助、版本信息。 |
| `EmbodiedMAS/Only_Language-base_Agent/` | 纯语言智能体与动作（OL）。 |
| `EmbodiedMAS/VLM_Agent/` | 视觉语言 / VLM 向智能体（单智能体、CMAS、DMAS）。 |
| `EmbodiedMAS/Human_Agent/` | 人类基线、Web UI、人机混合运行脚本。 |
| `EmbodiedMAS/ExperimentRunning/` | 命令行入口：`SA_benchmark_runner`、`CMAS_benchmark_runner`、`DMAS_benchmark_runner` 及变体（如 `DMAS_benchmark_runner_wo`）、任务 JSON、自动化辅助。 |
| `EmbodiedMAS/prompt/` | 文本提示与动作 / 观测空间片段。 |
| `EmbodiedMAS/Metric_Tool/` | Token、耗时、感知与汇总类评测工具。 |
| `examples/` | 可运行演示（MAS、RL、采集、EXR 等）。 |

---

## 环境要求

- **Python** `>=3.12`（见 `pyproject.toml`）。
- 已启动并暴露 gRPC 的 **TongSim / UE** 构建；示例脚本中的端口需与你的部署一致（常见为本地 `5726` 等）。
- 若使用 **完整基准自动化**，请编辑 `EmbodiedMAS/ExperimentRunning/benchmark_tasks.json`：将 Docker 镜像、宿主机上的 Shipping 路径、`SceneConfig.json` 等占位符替换为你的实际环境。

---

## 安装

在仓库根目录（可编辑安装，便于与文档一致的导入路径）：

```bash
pip install -e .
```

若使用 **uv**：

```bash
uv sync
```

可选**依赖组**（见 `pyproject.toml`）：`dev`（ruff、black、pre-commit）、`docs`（MkDocs）、`test`（pytest）、`multi_agent`（gymnasium、psutil、numpy 等）。

---

## 配置

1. **大模型 API**  
   将 `EmbodiedMAS/llm_config.example.json` 复制为 `EmbodiedMAS/llm_config.json`，填写 `api_key`、`base_url`、`model` 及需要的 `chat_completion_extra_kwargs`。若代码支持，也可改用环境变量 `OPENAI_API_KEY` 等，避免在文件中写密钥。

2. **基准多模型配置**（若使用）  
   参考 `EmbodiedMAS/ExperimentRunning/llm_profiles.example.json` 并按实际供应商调整。

3. **密钥**  
   勿将真实密钥或内网地址提交到版本库。

---

## 快速开始

**1. 验证 gRPC 客户端**（需仿真已启动，且脚本中的端点一致）：

```bash
python examples/demo_mas_demo.py
```

`examples/` 下其他脚本覆盖 RL、采集、体素工具等；各文件内有端点与前置条件说明。

**2. 单智能体基准（OL 后端）**，在仓库根目录执行：

```bash
python -m EmbodiedMAS.ExperimentRunning.SA_benchmark_runner --backend OL --agent-type FD
```

对该模块使用 `--help` 可查看其余参数（其他后端、智能体类型、超时等）。中心化与去中心化入口见 `CMAS_benchmark_runner`、`DMAS_benchmark_runner` 及 `DMAS_benchmark_runner_wo` 等模块。

---

## 基准任务 JSON

`EmbodiedMAS/ExperimentRunning/benchmark_tasks.json` 用于驱动自动化：Docker / GPU 参数、卷挂载、容器内 UE 启动脚本、宿主机 `SceneConfig.json`、任务 ID、超时与重复次数等。**占位镜像名与路径必须全部替换**后，无人值守运行才能在目标机器或集群上成功执行。

---

## 开发说明

- **静态检查 / 格式化**：`ruff`、`black`（配置见 `[tool.ruff]` 与 dev 依赖组）。
- **测试**：`pytest`（`[tool.pytest.ini_options]` 中配置了 asyncio 相关默认项）。
- **Pre-commit**：按团队规范可选安装。

---

## 路径说明（VL / VLM 后端）

当前仓库中视觉语言相关实现多位于 `EmbodiedMAS/VLM_Agent/`。部分 `ExperimentRunning` 模块仍引用历史目录名（如 `Vision_Language-base_Agent`、`Vision_VLM_Agent`）。若使用 `--backend VL` 或 `--backend VLM` 时出现「目录不存在」错误，请对照对应 runner 内的路径映射与本地目录名是否一致，或通过符号链接对齐预期路径。

---

## 引用

若在研究中使用 EMA-Bench 或本仓库代码，请引用论文（正式 BibTeX 将在发表后补充至此）。打包元数据中的维护者信息见 `pyproject.toml`。
