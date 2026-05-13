"""
tongsim.tongsim

对应单个 TongSim UE 实例, 内部依赖 WorldContext 管理连接与任务调度。
"""

from typing import Final

from tongsim.core.world_context import WorldContext
from tongsim.manager.utils import UtilFuncs

__all__ = ["TongSim"]


class TongSim:
    """
    TongSim 实例: 代表一个连接的 UE 实例，提供高级控制接口。
    所有方法为同步阻塞接口，便于在同步项目或脚本中使用。
    """

    def __init__(self, grpc_endpoint: str = "127.0.0.1:5726"):
        """
        初始化一个 TongSim 实例。

        Args:
            grpc_endpoint (str): gRPC 服务器地址，如 "localhost:5726"
        """
        self._context: Final[WorldContext] = WorldContext(grpc_endpoint)
        self._utils: Final[UtilFuncs] = UtilFuncs(self._context)

    @property
    def utils(self) -> UtilFuncs:
        """
        获取封装的实用工具函数接口。

        Returns:
            UtilFuncs: 提供实用工具函数方法。
        """
        return self._utils

    @property
    def context(self) -> WorldContext:
        """
        获取当前 TongSim 实例的运行时上下文。

        Returns:
            WorldContext: 管理连接、事件循环、任务派发等的上下文资源对象。
        """
        return self._context

    def close(self):
        """关闭当前实例并释放资源"""
        self._context.release()

    def __enter__(self):
        """支持 with 上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 上下文管理器"""
        self.close()
