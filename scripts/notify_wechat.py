#!/usr/bin/env python3
import os
import sys
from collections import defaultdict
from pathlib import Path

import requests

# ==== 配置 ====
WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=646873d6-0d38-4757-94f8-be51c2b3f5b7"
GITLAB_API = os.getenv("CI_API_V4_URL", "https://gitlab.bigai.ai/api/v4")
GITLAB_TOKEN = os.getenv("CI_JOB_TOKEN")

# === 基础 GitLab CI 环境变量 ===
project_id = os.getenv("CI_PROJECT_ID")
pipeline_id = os.getenv("CI_PIPELINE_ID")
project_name = os.getenv("CI_PROJECT_NAME", "unknown")
commit_message = os.getenv("CI_COMMIT_MESSAGE", "无提交信息")
commit_ref = os.getenv("CI_COMMIT_REF_NAME", "未知分支")
commit_url = (
    f"{os.getenv('CI_PROJECT_URL', '')}/-/commit/{os.getenv('CI_COMMIT_SHA', '')}"
)
job_url = os.getenv("CI_JOB_URL", os.getenv("CI_PROJECT_URL", ""))
current_stage = os.getenv("CI_JOB_STAGE", "").lower()

# === 状态 Emoji 映射（简化） ===
status_emoji = {
    "success": "✅",
    "failed": "❌",
    "canceled": "⚠️",
    "skipped": "➖",  # noqa: RUF001
    "pending": "⏳",
}


# === 获取各阶段状态和对应的最新 job URL ===
def get_stage_statuses(
    project_id: str, pipeline_id: str, excluded_stage: str
) -> tuple[dict[str, str], dict[str, str]]:
    url = f"{GITLAB_API}/projects/{project_id}/pipelines/{pipeline_id}/jobs"
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    resp = requests.get(url, headers=headers, timeout=5)
    jobs = resp.json()

    stage_status = {}
    stage_links = {}

    grouped = defaultdict(list)
    for job in jobs:
        stage = job["stage"]
        if stage.lower() == excluded_stage:
            continue
        grouped[stage].append(job)

    for stage, job_list in grouped.items():
        statuses = [job["status"] for job in job_list]
        urls = [job["web_url"] for job in job_list]

        # 最后一个 Job 作为链接展示（一般是最新）
        stage_links[stage] = urls[-1] if urls else ""

        if "failed" in statuses:
            stage_status[stage] = "failed"
        elif "success" in statuses:
            stage_status[stage] = "success"
        elif "manual" in statuses:
            stage_status[stage] = "manual"
        elif "skipped" in statuses:
            stage_status[stage] = "skipped"
        else:
            stage_status[stage] = statuses[-1]

    return stage_status, stage_links


# === 生成 Markdown 格式的阶段列表
def build_stage_summary(
    stage_status: dict[str, str], stage_links: dict[str, str]
) -> tuple[str, str]:
    lines = []
    pipeline_final_status = "success"
    for stage, status in stage_status.items():
        emoji = status_emoji.get(status, "❓")
        url = stage_links.get(stage)
        lines.append(f"- {emoji} [`{stage}`]({url})" if url else f"- {emoji} `{stage}`")
        if status == "failed":
            pipeline_final_status = "failed"
    return "\n".join(lines), pipeline_final_status


# === 获取 deploy 成功时的版本号 ===
def get_version_info(version_file: str = "src/tongsim/version.py") -> str | None:
    try:
        path = Path(version_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("VERSION"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        return None
    return None


# === 构建最终内容 ===
try:
    stage_status, stage_links = get_stage_statuses(
        project_id, pipeline_id, excluded_stage=current_stage
    )
    stage_lines, pipeline_status = build_stage_summary(stage_status, stage_links)
except Exception as e:
    stage_lines = f"⚠️ 无法获取阶段状态: {e}"
    pipeline_status = "unknown"

# === 构建正文内容 ===
lines = [
    f"### [{project_name}] CI 通知",
    "",
    f"{status_emoji.get(pipeline_status, '❓')} **流水线最终状态**: `{pipeline_status.upper()}`",
    f"**分支**: `{commit_ref}`",
    f"**提交信息**: {commit_message}",
    f"[🔗 查看提交详情]({commit_url})",
    "",
]

# === 如果 deploy 阶段成功，加入版本号信息 ===
if stage_status.get("deploy") == "success":
    version = get_version_info()
    if version:
        lines.insert(3, f"**部署版本**: `{version}`")

# === 加入阶段状态总结 ===
lines.append("**各阶段状态**:")
lines.append(stage_lines)

# === 发起通知 ===
payload = {
    "msgtype": "markdown",
    "markdown": {"content": "\n".join(lines)},
}

try:
    requests.post(WEBHOOK_URL, json=payload, timeout=5).raise_for_status()
except Exception as e:
    print(f"[ERROR] Failed to send wechat notification: {e}", file=sys.stderr)
