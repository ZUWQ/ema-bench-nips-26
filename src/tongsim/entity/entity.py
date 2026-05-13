from typing import Final, TypeVar

from tongsim.core.world_context import WorldContext
from tongsim.logger import get_logger

_logger = get_logger("entity")

T = TypeVar("T")


class Entity:
    """
    Entity 类代表一个 TongSim 世界中的对象

    类的职责包括:

    - 管理组件 ID，按组件类型分类
    - 提供能力（Ability）访问与转换机制

    注意: Entity 不直接持有组件数据，仅维护 component_id 结构。
    """

    __slots__ = ("_ability_cache", "_components", "_id", "_world_context")

    def __init__(
        self,
        entity_id: str,
        world_context: WorldContext,
    ):
        self._id: Final[str] = entity_id
        self._world_context: Final[WorldContext] = world_context
        self._ability_cache: dict[type, object] = {}  # 缓存已创建的 Impl 实例

    @property
    def id(self):
        return self._id

    @property
    def context(self):
        return self._world_context

    def __repr__(self) -> str:
        return f"Entity(id: {self._id})"
