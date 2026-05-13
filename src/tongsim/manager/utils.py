from tongsim.core.world_context import WorldContext
from tongsim.logger import get_logger

_logger = get_logger("utils")


class UtilFuncs:
    """
    通用工具函数集合，用于执行一些常见的模拟操作。
    """

    def __init__(self, world_context: WorldContext):
        self._context: WorldContext = world_context
