"""
tongsim 版本信息
"""

import platform
import sys

VERSION = "0.0.1"
"""tongsim 的主版本号。"""


def get_version_info() -> str:
    """
    获取 tongsim 及其关键依赖的版本信息、Python 版本、平台信息等。

    Returns:
        str: 多行格式化字符串，用于展示完整版本状态。
    """

    info = {
        "tongsim_lite version": VERSION,
        "python version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }

    return "\n".join(f"{k:<20}: {v}" for k, v in info.items())
