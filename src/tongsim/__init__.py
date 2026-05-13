"""
tongsim
"""

import typing
from importlib import import_module
from warnings import warn

from .version import VERSION

__version__ = VERSION

__all__ = (
    "AABB",
    "Pose",
    "Quaternion",
    "TongSim",
    "Transform",
    "CaptureAPI",
    "UnaryAPI",
    "Vector3",
    "__version__",
    "get_version_info",
    "initialize_logger",
    "math",
    "set_log_level",
)

if typing.TYPE_CHECKING:
    # 导入用于 IDE 提示和类型检查
    from . import math
    from .connection.grpc import CaptureAPI, UnaryAPI
    from .logger import initialize_logger, set_log_level
    from .math.geometry import AABB, Pose, Quaternion, Transform, Vector3
    from .tongsim import TongSim
    from .version import get_version_info

# 模块路径映射，基于 `__spec__.parent` 动态确定包路径
_dynamic_imports: dict[str, tuple[str, str]] = {
    # Math
    "Pose": (__spec__.parent, ".math.geometry"),
    "Quaternion": (__spec__.parent, ".math.geometry"),
    "Vector3": (__spec__.parent, ".math.geometry"),
    "AABB": (__spec__.parent, ".math.geometry"),
    "Transform": (__spec__.parent, ".math.geometry"),
    "math": (__spec__.parent, "."),
    # Core
    "TongSim": (__spec__.parent, ".tongsim"),
    # Logger
    "initialize_logger": (__spec__.parent, ".logger"),
    "set_log_level": (__spec__.parent, ".logger"),
    # gRPC
    "CaptureAPI": (__spec__.parent, ".connection.grpc"),
    "UnaryAPI": (__spec__.parent, ".connection.grpc"),
    # Version
    "get_version_info": (__spec__.parent, ".version"),
}

# Deprecated 动态导入，用于兼容旧版本
_deprecated_imports = {}


def __getattr__(attr_name: str) -> object:
    """
    动态导入模块成员，仅在首次访问时进行实际导入。
    """
    # 检查是否为废弃模块
    if attr_name in _deprecated_imports:
        warn(
            f"Importing `{attr_name}` from `tongsim` is deprecated and will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )

    # 检查有效动态导入成员
    dynamic_attr = _dynamic_imports.get(attr_name) or _deprecated_imports.get(attr_name)
    if dynamic_attr is None:
        raise AttributeError(f"Module 'tongsim' has no attribute '{attr_name}'")

    # 动态导入
    package, module_path = dynamic_attr
    module = import_module(module_path, package=package)
    result = getattr(module, attr_name)

    # 缓存到全局命名空间，避免重复导入
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """提供完整的模块成员列表，包括动态导入成员。"""
    return list(__all__)
